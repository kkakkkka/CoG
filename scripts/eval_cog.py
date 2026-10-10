#!/usr/bin/env python3
"""Evaluate a CoG checkpoint on the BEAT English test split.

Predictions are generated in the training pose normalization, then converted
back to BVH rotations for L1, SRGR, and beat alignment. FGD uses the released
motion autoencoder on the same normalized windows.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["BEAT_ROOT"]) if os.environ.get("BEAT_ROOT") else REPO.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import beat_eval as base  # noqa: E402
import utils.textgrid as tg  # noqa: E402
from models.cog import CoG  # noqa: E402
from utils import config as cog_config  # noqa: E402


FULL_CONFIG = REPO / "configs" / "cog.yaml"
FULL_CACHE = DATA_ROOT / "datasets" / "BEAT" / "beat_cache" / "cog_4english_15_141_full"
FULL_EXPECTED_COUNTS = {"train": 392, "val": 64, "test": 64}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_vocab(cache_root: Path) -> Any:
    with (cache_root / "vocab.pkl").open("rb") as f:
        return pickle.load(f)


def cache_counts(cache_root: Path) -> dict[str, int]:
    report_path = cache_root / "preprocess_report.json"
    if report_path.exists():
        report = read_json(report_path)
        return {key: int(value) for key, value in report.get("counts", {}).items()}
    return {split: len(list((cache_root / split / "bvh_rot").glob("*.bvh"))) for split in ("train", "val", "test")}


def split_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    out = tuple(item.strip() for item in value.split(",") if item.strip())
    return out or default


def audit_cache(cache_root: Path, expected_counts: dict[str, int] | None, required_lmdb_splits: tuple[str, ...]) -> dict[str, Any]:
    counts = cache_counts(cache_root)
    vocab = load_vocab(cache_root)
    lmdb = {}
    for split in ("train", "val", "test"):
        lmdb_dir = cache_root / split / "bvh_rot_cache"
        data_mdb = lmdb_dir / "data.mdb"
        entries = None
        if data_mdb.exists() and data_mdb.stat().st_size > 0:
            import lmdb as lmdb_mod

            env = lmdb_mod.open(str(lmdb_dir), readonly=True, lock=False, readahead=False)
            try:
                with env.begin() as txn:
                    entries = int(txn.stat()["entries"])
            finally:
                env.close()
        lmdb[split] = {"path": str(lmdb_dir), "data_mdb_bytes": data_mdb.stat().st_size if data_mdb.exists() else 0, "entries": entries}
    ok = True
    issues: list[str] = []
    if expected_counts is not None:
        for split, expected in expected_counts.items():
            if counts.get(split) != expected:
                ok = False
                issues.append(f"{split}_count_expected_{expected}_got_{counts.get(split)}")
    for split in ("train", "val", "test"):
        if counts.get(split, 0) <= 0:
            ok = False
            issues.append(f"{split}_count_non_positive")
    for split in required_lmdb_splits:
        if not lmdb[split]["entries"] or int(lmdb[split]["entries"]) <= 0:
            ok = False
            issues.append(f"{split}_lmdb_entries_non_positive")
    if int(getattr(vocab, "n_words", 0)) <= 4:
        ok = False
        issues.append("vocab_only_special_tokens")
    return {
        "status": "ok" if ok else "failed",
        "cache_root": str(cache_root),
        "counts": counts,
        "expected_counts": expected_counts,
        "required_lmdb_splits": list(required_lmdb_splits),
        "vocab_words": int(getattr(vocab, "n_words", 0)),
        "lmdb": lmdb,
        "issues": issues,
    }


def load_eval_args(config_path: Path, cache_root: Path, vocab_words: int) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "eval_cog.py",
            "-c",
            str(config_path),
            "--word_index_num",
            str(vocab_words),
            "--batch_size",
            "1",
            "--loader_workers",
            "0",
            "--gpus",
        ]
        args = cog_config.parse_args()
    finally:
        sys.argv = old_argv
    args.is_train = False
    args.root_path = str(DATA_ROOT)
    args.train_data_path = f"/datasets/BEAT/beat_cache/{cache_root.name}/train/"
    args.val_data_path = f"/datasets/BEAT/beat_cache/{cache_root.name}/val/"
    args.test_data_path = f"/datasets/BEAT/beat_cache/{cache_root.name}/test/"
    args.mean_pose_path = f"/datasets/BEAT/beat_cache/{cache_root.name}/train/"
    args.std_pose_path = f"/datasets/BEAT/beat_cache/{cache_root.name}/train/"
    args.device = torch.device("cpu")
    args.ddp = False
    args.rank = 0
    args.world_size = 1
    args.gpus = []
    args.loader_workers = 0
    args.batch_size = 1
    return args


def load_checkpoint_state(checkpoint: Path) -> OrderedDict[str, torch.Tensor]:
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def strict_load_cog(args: argparse.Namespace, checkpoint: Path) -> tuple[CoG | None, dict[str, Any]]:
    model = CoG(args).cpu().eval()
    state = load_checkpoint_state(checkpoint)
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatch = [
        {"key": key, "model": list(model_state[key].shape), "checkpoint": list(state[key].shape)}
        for key in sorted(set(model_state) & set(state))
        if tuple(model_state[key].shape) != tuple(state[key].shape)
    ]
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_keys": len(state),
        "model_keys": len(model_state),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "strict_load_ok": False,
    }
    if not missing and not unexpected and not shape_mismatch:
        model.load_state_dict(state, strict=True)
        report["strict_load_ok"] = True
        return model, report
    return None, report


def text_tokens(textgrid_path: Path, frames: int, vocab: Any, fps: int = base.FPS) -> tuple[np.ndarray, dict[str, Any]]:
    grid = tg.TextGrid.fromFile(str(textgrid_path))
    tokens = []
    unk = int(getattr(vocab, "UNK_token", 3))
    pad = int(getattr(vocab, "PAD_token", 0))
    for i in range(frames):
        current_time = i / fps
        value = unk
        found = False
        for word in grid[0]:
            word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
            if word_s <= current_time <= word_e:
                value = pad if word_n == " " else int(vocab.get_word_index(word_n))
                found = True
                break
        tokens.append(value if found else unk)
    arr = np.asarray(tokens, dtype=np.int64)
    return arr, {
        "frames": int(frames),
        "unique_token_count": int(len(set(arr.tolist()))),
        "unk_frames": int(np.count_nonzero(arr == unk)),
        "pad_frames": int(np.count_nonzero(arr == pad)),
        "mode": "full_train_vocab_textgrid" if int(getattr(vocab, "n_words", 0)) > 4 else "degenerate_vocab",
    }


def load_clip_with_text(stem: str, frames: int, train_mean: np.ndarray, train_std: np.ndarray, vocab: Any) -> dict[str, Any]:
    clip = base.load_clip(stem, frames, train_mean, train_std)
    clip["gt_train_norm"] = clip["gt_norm"]
    n_frames = int(clip["gt_bvh"].shape[0])
    tokens, token_report = text_tokens(Path(clip["files"]["TextGrid"]), n_frames, vocab)
    clip["text_tokens"] = tokens
    clip["text_report"] = token_report
    return clip


def load_frozen_style(style_dir: Path, cache_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    emotion = np.load(style_dir / "test_emo.npy").astype(np.float32)
    speaker = np.load(style_dir / "test_spk.npy").astype(np.float32)
    stems = [path.stem for path in sorted((cache_root / "test" / "bvh_rot").glob("*.bvh"))]
    if len(stems) != len(emotion) or len(stems) != len(speaker):
        raise RuntimeError(f"frozen style rows {len(emotion)} do not match test bvh files {len(stems)}")
    return {stem: (emotion[index], speaker[index]) for index, stem in enumerate(stems)}


def rollout_cog(
    model: CoG,
    clip: dict[str, Any],
    frames: int,
    emotion_prob: np.ndarray | None = None,
    speaker_prob: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = np.zeros((frames, base.POSE_DIMS), dtype=np.float32)
    face_pred = None
    pred[: base.SEED_FRAMES] = clip["gt_train_norm"][: base.SEED_FRAMES]
    starts = base.model_windows(frames)
    window_reports = []
    for start in starts:
        end = start + base.WINDOW
        pre_seq = np.zeros((1, base.WINDOW, base.POSE_DIMS + 1), dtype=np.float32)
        if start == 0:
            seed = clip["gt_train_norm"][: base.SEED_FRAMES]
            write_from = 0
        else:
            seed = pred[start : start + base.SEED_FRAMES]
            write_from = base.SEED_FRAMES
        pre_seq[0, : base.SEED_FRAMES, : base.POSE_DIMS] = seed
        pre_seq[0, : base.SEED_FRAMES, base.POSE_DIMS] = 1.0
        sample0 = int(round(start / base.FPS * base.AUDIO_SR))
        sample1 = int(round(end / base.FPS * base.AUDIO_SR))
        need = int(round(base.WINDOW / base.FPS * base.AUDIO_SR))
        audio = clip["wav"][sample0:sample1]
        if audio.shape[0] < need:
            audio = np.pad(audio, (0, need - audio.shape[0]))
        elif audio.shape[0] > need:
            audio = audio[:need]
        tensors = {
            "pre_seq": torch.from_numpy(pre_seq),
            "in_audio": torch.from_numpy(audio[None, :].astype(np.float32)),
            "in_facial": torch.from_numpy(clip["face"][start:end][None, :, :].astype(np.float32)),
            "in_text": torch.from_numpy(clip["text_tokens"][start:end][None, :].astype(np.int64)),
            "in_id": torch.from_numpy(np.asarray([[clip["speaker_id"]]], dtype=np.int64)),
            "in_emo": torch.from_numpy(clip["emotion"][start:end][None, :].astype(np.int64)),
        }
        style_kwargs = {}
        if emotion_prob is not None and speaker_prob is not None:
            style_kwargs["emotion_prob"] = torch.from_numpy(emotion_prob[None, :].astype(np.float32))
            style_kwargs["speaker_prob"] = torch.from_numpy(speaker_prob[None, :].astype(np.float32))
        with torch.no_grad():
            out = model(
                tensors["pre_seq"],
                in_audio=tensors["in_audio"],
                in_facial=tensors["in_facial"],
                in_text=tensors["in_text"],
                in_id=tensors["in_id"],
                in_emo=tensors["in_emo"],
                is_test=True,
                return_dict=True,
                **style_kwargs,
            )
        out_np = out["pose"].detach().cpu().numpy().reshape(base.WINDOW, base.POSE_DIMS).astype(np.float32)
        face_np = out["face"].detach().cpu().numpy().reshape(base.WINDOW, -1).astype(np.float32)
        if face_pred is None:
            face_pred = np.zeros((frames, face_np.shape[1]), dtype=np.float32)
        if start == 0:
            pred[start:end] = out_np
            pred[: base.SEED_FRAMES] = clip["gt_train_norm"][: base.SEED_FRAMES]
            face_pred[start:end] = face_np
        else:
            pred[start + write_from : end] = out_np[write_from:]
            face_pred[start + write_from : end] = face_np[write_from:]
        window_reports.append({"start": int(start), "end": int(end), "write_from": int(write_from), "finite": bool(np.isfinite(out_np).all() and np.isfinite(face_np).all())})
    gt_face = clip["face"][:frames]
    face_mse = float(np.mean((face_pred - gt_face) ** 2))
    return pred, {
        "starts": [int(x) for x in starts],
        "windows": window_reports,
        "seed_frames_gt": base.SEED_FRAMES,
        "face_mse": face_mse,
        "face_dims": int(face_pred.shape[1]),
    }


def test_stems(cache_root: Path, max_clips: int) -> list[str]:
    stems = sorted(path.stem for path in (cache_root / "test" / "bvh_rot").glob("*.bvh"))
    return stems[:max_clips] if max_clips > 0 else stems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(FULL_CONFIG))
    parser.add_argument("--cache-root", default=str(FULL_CACHE))
    parser.add_argument("--out-dir", default=str(DATA_ROOT / "outputs" / "eval_cog"))
    parser.add_argument("--max-clips", type=int, default=0, help="0 means every clip in the cache test split")
    parser.add_argument("--frames", type=int, default=base.PREFIX_FRAMES)
    parser.add_argument("--full-clip", action="store_true", help="Score each test clip for its full available length instead of the first 8 seconds.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--protocol-label", default="cog_test")
    parser.add_argument("--expected-full-cache", action="store_true")
    parser.add_argument("--require-lmdb-splits", default=None, help="Comma-separated LMDB splits to require. Default: all splits for --expected-full-cache, otherwise train only.")
    parser.add_argument("--frozen-style-dir", default="", help="Directory with test_emo.npy and test_spk.npy aligned to sorted test BVH stems.")
    parser.add_argument("--mse-ae", default="", help="Optional cache-trained AE. FGD is computed in CoG train mean/std space.")
    args_cli = parser.parse_args()

    torch.set_num_threads(args_cli.threads)
    np.random.seed(2021)
    torch.manual_seed(2021)
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args_cli.checkpoint)
    config_path = Path(args_cli.config)
    cache_root = Path(args_cli.cache_root)

    expected_counts = FULL_EXPECTED_COUNTS if args_cli.expected_full_cache else None
    required_lmdb_splits = split_csv(args_cli.require_lmdb_splits, ("train", "val", "test") if args_cli.expected_full_cache else ("train",))
    cache_audit = audit_cache(cache_root, expected_counts, required_lmdb_splits)
    vocab = load_vocab(cache_root)
    eval_args = load_eval_args(config_path, cache_root, int(getattr(vocab, "n_words")))
    model, load_report = strict_load_cog(eval_args, checkpoint)

    report: dict[str, Any] = {
        "time_utc_started": utc_now(),
        "status": "initializing",
        "protocol": {
            "label": args_cli.protocol_label,
            "split": str(cache_root / "test" / "bvh_rot"),
            "frames": "per-clip full length" if args_cli.full_clip else int(args_cli.frames),
            "full_clip": bool(args_cli.full_clip),
            "fps": base.FPS,
            "window_frames": base.WINDOW,
            "stride_frames": base.GEN_STRIDE,
            "seed_policy": "first 4 frames use ground truth; later windows seed from previous generated frames",
        },
        "paths": {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "cache_root": str(cache_root),
            "out_dir": str(out_dir),
            "pose_mean": str(cache_root / "train" / "bvh_rot" / "bvh_mean.npy"),
            "pose_std": str(cache_root / "train" / "bvh_rot" / "bvh_std.npy"),
        },
        "cache_audit": cache_audit,
        "model_load": load_report,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "ld_library_path_set": bool(os.environ.get("LD_LIBRARY_PATH")),
            "threads": args_cli.threads,
        },
        "clips": [],
        "metrics": {},
    }
    if cache_audit["status"] != "ok":
        report["status"] = "cache_audit_failed"
        write_json(out_dir / "report.json", report)
        return 2
    if model is None:
        report["status"] = "strict_load_failed"
        write_json(out_dir / "report.json", report)
        return 3

    train_mean = np.load(cache_root / "train" / "bvh_rot" / "bvh_mean.npy").astype(np.float32)
    train_std = np.load(cache_root / "train" / "bvh_rot" / "bvh_std.npy").astype(np.float32)
    ae_args = SimpleNamespace(pose_length=base.WINDOW, pose_dims=base.POSE_DIMS, vae_length=300)
    mse_ae = None
    if args_cli.mse_ae:
        mse_ae, mse_report = base.build_ae(ae_args, Path(args_cli.mse_ae))
        report["mse_ae_evaluator"] = mse_report
        if mse_ae is None:
            report["status"] = "mse_ae_load_failed"
            write_json(out_dir / "report.json", report)
            return 4
    style_by_stem = {}
    if args_cli.frozen_style_dir:
        style_by_stem = load_frozen_style(Path(args_cli.frozen_style_dir), cache_root)
        report["frozen_style"] = {"dir": args_cli.frozen_style_dir, "clips": len(style_by_stem)}

    stems = test_stems(cache_root, args_cli.max_clips)
    all_gt_train_windows = []
    all_pred_train_windows = []
    per_clip_metrics = []
    start_time = time.time()
    for idx, stem in enumerate(stems):
        clip_entry: dict[str, Any] = {"stem": stem, "index": idx}
        try:
            n_frames = 0 if args_cli.full_clip else args_cli.frames
            clip = load_clip_with_text(stem, n_frames, train_mean, train_std, vocab)
            n_frames = int(clip["gt_bvh"].shape[0])
            emotion_prob = speaker_prob = None
            if style_by_stem:
                if stem not in style_by_stem:
                    raise KeyError(f"no frozen style for {stem}")
                emotion_prob, speaker_prob = style_by_stem[stem]
            pred_train_norm, rollout = rollout_cog(
                model, clip, n_frames, emotion_prob=emotion_prob, speaker_prob=speaker_prob
            )
            pred_bvh = pred_train_norm * train_std + train_mean
            finite = bool(np.isfinite(pred_bvh).all())
            metrics = {
                "stem": stem,
                "finite": finite,
                "frames": int(n_frames),
                "seconds": float(n_frames / base.FPS),
                "train_norm_l1_mean_abs": float(np.mean(np.abs(pred_train_norm - clip["gt_train_norm"]))),
                "bvh141_l1_mean_abs": float(np.mean(np.abs(pred_bvh - clip["gt_bvh"]))),
                "l1div_bvh141": base.l1div_value(pred_bvh),
                "srgr": base.srgr_value(
                    pred_bvh,
                    clip["gt_bvh"],
                    clip["semantic"],
                    semantic_mean=base.SRGR_FULL_SEMANTIC_MEAN if args_cli.full_clip else base.SRGR_PREFIX8_SEMANTIC_MEAN,
                ),
                "beat_align": base.beat_align_value(
                    clip["wav"], pred_bvh, 0, (n_frames + 0.5) / base.FPS, base.FPS
                ),
                "face_mse": rollout["face_mse"],
                "face_dims": rollout["face_dims"],
                "rollout": rollout,
                "text": clip["text_report"],
            }
            per_clip_metrics.append(metrics)
            if mse_ae is not None:
                all_gt_train_windows.append(base.collect_ae_windows(clip["gt_train_norm"]))
                all_pred_train_windows.append(base.collect_ae_windows(pred_train_norm))
            if args_cli.save_arrays or args_cli.max_clips > 0:
                base.save_arrays(out_dir, "cog", stem, pred_train_norm, pred_bvh, clip["gt_train_norm"], clip["gt_bvh"])
            clip_entry.update({"status": "ok", "frames_available": clip["frames_available"], "text": clip["text_report"], "finite": finite})
        except Exception as exc:  # noqa: BLE001
            clip_entry.update({"status": "failed", "error": repr(exc)})
        report["clips"].append(clip_entry)
        if args_cli.full_clip or idx == 0 or (idx + 1) % 8 == 0 or idx + 1 == len(stems):
            print(json.dumps({"time_utc": utc_now(), "completed_clips": idx + 1, "total": len(stems)}, ensure_ascii=False), flush=True)

    ok_metrics = [m for m in per_clip_metrics if m.get("finite")]
    beat_align_entries = [m.get("beat_align", {}) for m in ok_metrics]
    beat_align_ok = [entry for entry in beat_align_entries if entry.get("status") == "ok" and np.isfinite(entry.get("score", np.nan))]
    summary: dict[str, Any] = {
        "clip_count": len(per_clip_metrics),
        "finite_clip_count": len(ok_metrics),
        "train_norm_l1_mean_abs": base.aggregate(m["train_norm_l1_mean_abs"] for m in ok_metrics),
        "bvh141_l1_mean_abs": base.aggregate(m["bvh141_l1_mean_abs"] for m in ok_metrics),
        "l1div_bvh141": base.aggregate(m["l1div_bvh141"] for m in ok_metrics),
        "srgr": base.aggregate(m["srgr"] for m in ok_metrics),
        "face_mse": {
            "status": "ok",
            "space": "unnormalized 15 fps blendshape weights, 51 dims",
            "compared_to": "ground-truth facial weights",
            **base.aggregate(m.get("face_mse") for m in ok_metrics),
        },
        "beat_align": {
            "status": "ok" if beat_align_ok else "failed",
            "valid_clip_count": len(beat_align_ok),
            "failed_clip_count": len(beat_align_entries) - len(beat_align_ok),
            "score": base.aggregate(entry.get("score") for entry in beat_align_ok),
            "failed_examples": [entry for entry in beat_align_entries if entry.get("status") != "ok"][:5],
        },
        "per_clip_metrics": per_clip_metrics,
    }
    if mse_ae is not None and all_pred_train_windows and all_gt_train_windows:
        pred_windows = np.concatenate(all_pred_train_windows, axis=0)
        gt_windows = np.concatenate(all_gt_train_windows, axis=0)
        summary["fgd_mse_ae"] = {
            "status": "ok",
            "normalization": "cog train bvh_rot mean/std",
            "checkpoint": args_cli.mse_ae,
            "window_count": int(pred_windows.shape[0]),
            **base.frechet_distance(base.embed_windows(mse_ae, pred_windows), base.embed_windows(mse_ae, gt_windows)),
        }

    report["metrics"]["cog"] = summary
    ok_clip_count = sum(1 for clip in report["clips"] if clip.get("status") == "ok")
    report["clip_count_ok"] = ok_clip_count
    report["clip_count_total"] = len(stems)
    report["elapsed_sec_total"] = float(time.time() - start_time)
    report["time_utc_finished"] = utc_now()
    report["status"] = "ok" if ok_clip_count == len(stems) else "partial"
    write_json(out_dir / "report.json", report)
    marker_name = "COMPLETE.json" if report["status"] == "ok" else "PARTIAL.json"
    write_json(out_dir / marker_name, {"status": report["status"], "time_utc": utc_now(), "ok": ok_clip_count, "total": len(stems), "report": str(out_dir / "report.json")})
    print(json.dumps({"status": report["status"], "ok": ok_clip_count, "total": len(stems), "report": str(out_dir / "report.json")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
