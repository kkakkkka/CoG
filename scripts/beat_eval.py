#!/usr/bin/env python3
"""Clip loading and BEAT metrics used by the CoG evaluator.

Motion is 15 fps and 141-D. Full-length clips and the first 8 seconds share
the SRGR and beat-alignment constants in this file.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy import linalg, signal
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["BEAT_ROOT"]) if os.environ.get("BEAT_ROOT") else REPO.parent
PREPROCESS = REPO / "dataloaders" / "preprocess_beat.py"
RAW_ROOT = DATA_ROOT / "datasets" / "BEAT" / "raw" / "beat_english_v0.2.1" / "beat_english_v0.2.1"
FPS = 15
AUDIO_SR = 16000
POSE_DIMS = 141
WINDOW = 34
SEED_FRAMES = 4
GEN_STRIDE = WINDOW - SEED_FRAMES
PREFIX_SECONDS = 8.0
PREFIX_FRAMES = int(PREFIX_SECONDS * FPS)


class Vocab:  # required for pickle files that were saved from __main__.Vocab
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())




def load_literal(name: str) -> Any:
    tree = ast.parse(PREPROCESS.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(name)


JOINT_LIST = load_literal("JOINT_LIST")


def select_rot(data: np.ndarray) -> np.ndarray:
    out: List[float] = []
    for joint, width in JOINT_LIST["beat_141"].items():
        src = JOINT_LIST["beat_joints"][joint][1]
        out.extend(data[src - width : src])
    return np.asarray(out, dtype=np.float32)


def read_bvh_141(path: Path, target_fps: int = FPS) -> np.ndarray:
    reduce_factor = int(120 / target_fps)
    motion_start: Optional[int] = None
    lines = path.read_text(errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("frame time"):
            motion_start = idx + 1
            break
    if motion_start is None:
        raise RuntimeError(f"could not find BVH Frame Time line in {path}")
    rows = []
    for i, line in enumerate(lines[motion_start:]):
        if i % reduce_factor != 0:
            continue
        data = np.fromstring(line, dtype=np.float32, sep=" ")
        if data.size:
            rows.append(select_rot(data))
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != POSE_DIMS:
        raise RuntimeError(f"bad BVH141 shape: {arr.shape} in {path}")
    return arr


def read_wav_16k(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    sr, data = wavfile.read(path)
    source_dtype = str(data.dtype)
    data = np.asarray(data)
    source_min = float(np.min(data))
    source_max = float(np.max(data))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        data = data.astype(np.float32) / scale
        normalization = f"integer PCM divided by {scale:g}"
    else:
        data = data.astype(np.float32)
        normalization = "float PCM unchanged"
    if sr != AUDIO_SR:
        target_len = int(round(len(data) * AUDIO_SR / sr))
        data = signal.resample(data, target_len).astype(np.float32)
    stats = {
        "source_sr": int(sr),
        "source_dtype": source_dtype,
        "source_min": source_min,
        "source_max": source_max,
        "normalization": normalization,
        "samples_16k": int(data.shape[0]),
        "min_16k": float(np.min(data)),
        "max_16k": float(np.max(data)),
        "mean_16k": float(np.mean(data)),
        "std_16k": float(np.std(data)),
        "within_minus1_1": bool(np.max(np.abs(data)) <= 1.0001),
    }
    return data, stats


def read_facial_15fps(path: Path, target_fps: int = FPS) -> np.ndarray:
    data = json.loads(path.read_text())
    reduce_factor = int(60 / target_fps)
    weights = [frame["weights"] for idx, frame in enumerate(data["frames"]) if idx % reduce_factor == 0]
    arr = np.asarray(weights, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 51:
        raise RuntimeError(f"bad facial shape: {arr.shape} in {path}")
    return arr


def read_emotion(path: Path, frames: int, fps: int = FPS) -> np.ndarray:
    rows: List[Tuple[float, float, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                rows.append((float(row[1]), float(row[2]), int(float(row[4]))))
    values = []
    for i in range(frames):
        t = i / fps
        value = 0
        for start, end, score in rows:
            if start <= t <= end:
                value = score
                break
        values.append(value)
    return np.asarray(values, dtype=np.int64)


def read_semantic_scores(path: Path, frames: int, fps: int = FPS) -> np.ndarray:
    rows: List[Tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = raw.strip().split("\t")
            if len(parts) < 5:
                parts = raw.strip().split()
            if len(parts) >= 5:
                try:
                    rows.append((float(parts[1]), float(parts[2]), float(parts[4])))
                except ValueError:
                    continue
    sem = np.zeros((frames,), dtype=np.float32)
    for i in range(frames):
        t = i / fps
        for start, end, score in rows:
            if start <= t <= end:
                sem[i] = score
                break
    return sem






def strip_module_prefix(state: Dict[str, torch.Tensor]) -> OrderedDict:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def load_checkpoint_state(checkpoint: Path) -> OrderedDict:
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
    return strip_module_prefix(state)


def strict_load_report(model: torch.nn.Module, checkpoint: Path) -> Dict[str, Any]:
    state = load_checkpoint_state(checkpoint)
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatch = [
        {"key": key, "model": list(model_state[key].shape), "checkpoint": list(state[key].shape)}
        for key in sorted(set(model_state) & set(state))
        if tuple(model_state[key].shape) != tuple(state[key].shape)
    ]
    report: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_keys": len(state),
        "model_keys": len(model_state),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "strict_load_ok": False,
    }
    if "text_encoder.embedding.weight" in state:
        report["text_embedding_shape"] = list(state["text_encoder.embedding.weight"].shape)
    if not missing and not unexpected and not shape_mismatch:
        model.load_state_dict(state, strict=True)
        report["strict_load_ok"] = True
    return report






def raw_files_for_stem(stem: str) -> Dict[str, Path]:
    speaker = stem.split("_", 1)[0]
    base = RAW_ROOT / speaker
    return {ext: base / f"{stem}.{ext}" for ext in ("wav", "bvh", "json", "TextGrid", "csv", "txt")}


def load_clip(stem: str, frames: int, mean: np.ndarray, std: np.ndarray) -> Dict[str, Any]:
    files = raw_files_for_stem(stem)
    missing = {ext: str(path) for ext, path in files.items() if not path.exists()}
    if missing:
        raise FileNotFoundError({"stem": stem, "missing": missing})
    bvh = read_bvh_141(files["bvh"])
    wav, wav_stats = read_wav_16k(files["wav"])
    face = read_facial_15fps(files["json"])
    audio_frames = int(wav.shape[0] / AUDIO_SR * FPS)
    limit = min(int(bvh.shape[0]), int(face.shape[0]), audio_frames)
    if frames <= 0:
        frames = limit
    if bvh.shape[0] < frames or face.shape[0] < frames or audio_frames < frames:
        raise RuntimeError(f"{stem} too short: bvh={bvh.shape}, face={face.shape}, audio_frames={audio_frames}, need={frames}")
    audio_need = int(round(frames / FPS * AUDIO_SR))
    if wav.shape[0] < audio_need:
        raise RuntimeError(f"{stem} audio too short: samples={wav.shape[0]}, need={audio_need}")
    gt = bvh[:frames].astype(np.float32)
    gt_norm = ((gt - mean) / std).astype(np.float32)
    clip = {
        "stem": stem,
        "speaker_id": int(stem.split("_", 1)[0]) - 1,
        "files": {k: str(v) for k, v in files.items()},
        "gt_bvh": gt,
        "gt_norm": gt_norm,
        "wav": wav[:audio_need].astype(np.float32),
        "wav_stats": wav_stats,
        "face": face[:frames].astype(np.float32),
        "emotion": read_emotion(files["csv"], frames),
        "semantic": read_semantic_scores(files["txt"], frames),
        "frames_available": {"bvh15": int(bvh.shape[0]), "face15": int(face.shape[0]), "audio16k_samples": int(wav.shape[0])},
    }
    return clip


def model_windows(total_frames: int) -> List[int]:
    starts = list(range(0, max(total_frames - WINDOW + 1, 1), GEN_STRIDE))
    last = max(0, total_frames - WINDOW)
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))






def l1div_value(results: np.ndarray) -> float:
    mean = np.mean(results, axis=0)
    return float(np.sum(np.abs(results - mean)) / results.shape[0])


# Frame-mean semantic score. Prefix eval uses the first 8 seconds.
# Full-clip diagnostic uses the full official-test64 frame mean.
SRGR_PREFIX8_SEMANTIC_MEAN = 0.150495
SRGR_FULL_SEMANTIC_MEAN = 0.163921


def srgr_value(
    results: np.ndarray,
    targets: np.ndarray,
    semantic: np.ndarray,
    threshold: float = 4.0,
    joints: int = 47,
    semantic_mean: float = SRGR_PREFIX8_SEMANTIC_MEAN,
) -> float:
    res = results.reshape(-1, joints, 3)
    tar = targets.reshape(-1, joints, 3)
    sem = semantic.reshape(-1)
    diff = np.sum(np.abs(res - tar), axis=2)
    success = np.where(diff < threshold, 1.0, 0.0)
    scale = 1.0 / semantic_mean
    for i in range(success.shape[0]):
        success[i, :] *= sem[i] * scale
    return float(np.sum(success) / (success.shape[0] * success.shape[1]))


# Both-wrist raw mean on official test64, first 8 seconds, sigma 0.3, order 2.
# Dividing by this makes that ground-truth average 1.
BEAT_ALIGN_GT_MEAN = 0.959


def beat_align_value(wave: np.ndarray, pose: np.ndarray, t_start: int = 0, t_end: int = 8, pose_fps: int = FPS, sigma: float = 0.3, order: int = 2) -> Dict[str, Any]:
    """Beat alignment between audio onsets and wrist velocity minima."""

    try:
        import librosa  # noqa: WPS433
    except ModuleNotFoundError:
        lmm_sites = []
        for root in (DATA_ROOT, REPO.parent):
            lmm_sites.extend(sorted((root / ".venv-lmm" / "lib").glob("python*/site-packages")))
        for lmm_site in lmm_sites:
            if str(lmm_site) not in sys.path:
                sys.path.append(str(lmm_site))
        try:
            import librosa  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}

    try:
        short_y = wave[int(t_start * AUDIO_SR) : int(t_end * AUDIO_SR)]
        oenv = librosa.onset.onset_strength(y=short_y, sr=AUDIO_SR)
        onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False)
        onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)
        spectrogram = np.abs(librosa.stft(y=short_y))
        rms = librosa.feature.rms(S=spectrogram)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])

        data_each_file = np.asarray(
            [np.concatenate([line_data_np[30:39], line_data_np[112:121]], 0) for line_data_np in pose],
            dtype=np.float32,
        )
        if data_each_file.shape[0] < 2:
            return {"status": "failed", "error_type": "TooFewPoseFrames", "error": f"{data_each_file.shape[0]} pose frames"}
        vel = data_each_file[1:, :] - data_each_file[:-1, :]
        vel_right_shoulder = np.linalg.norm(np.array([vel[:, 0], vel[:, 1], vel[:, 2]]), axis=0)
        vel_right_arm = np.linalg.norm(np.array([vel[:, 3], vel[:, 4], vel[:, 5]]), axis=0)
        vel_right_wrist = np.linalg.norm(np.array([vel[:, 6], vel[:, 7], vel[:, 8]]), axis=0)
        vel_left_shoulder = np.linalg.norm(np.array([vel[:, 9], vel[:, 10], vel[:, 11]]), axis=0)
        vel_left_arm = np.linalg.norm(np.array([vel[:, 12], vel[:, 13], vel[:, 14]]), axis=0)
        vel_left_wrist = np.linalg.norm(np.array([vel[:, 15], vel[:, 16], vel[:, 17]]), axis=0)

        span = slice(int(t_start * pose_fps), int(t_end * pose_fps))
        beat_right_arm = signal.argrelextrema(vel_right_arm[span], np.less, order=order)
        beat_right_shoulder = signal.argrelextrema(vel_right_shoulder[span], np.less, order=order)
        beat_right_wrist = signal.argrelextrema(vel_right_wrist[span], np.less, order=order)
        beat_left_arm = signal.argrelextrema(vel_left_arm[span], np.less, order=order)
        beat_left_shoulder = signal.argrelextrema(vel_left_shoulder[span], np.less, order=order)
        beat_left_wrist = signal.argrelextrema(vel_left_wrist[span], np.less, order=order)

        # Audio onsets stay as in alignment.calculate_align. Pose beats are the
        # union of both wrists; either hand can match an onset.
        audio_bt = librosa.frames_to_time(onset_bt_rms)
        wrist_idx = [beat_right_wrist[0], beat_left_wrist[0]]
        wrist_idx = [idx for idx in wrist_idx if len(idx)]
        pose_bt = np.unique(np.concatenate(wrist_idx)) / pose_fps + t_start if wrist_idx else np.zeros(0)
        if len(audio_bt) == 0:
            return {
                "status": "failed",
                "error_type": "EmptyAudioBeats",
                "error": "onset_bt_rms is empty",
                "onset_raw_count": int(len(onset_raw)),
                "onset_backtrack_count": int(len(onset_bt)),
                "onset_backtrack_rms_count": int(len(onset_bt_rms)),
                "beat_counts": [int(len(x[0])) for x in [beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist]],
            }
        score = 0.0
        for audio_each in audio_bt:
            if len(pose_bt) == 0:
                min_dist = np.inf
            else:
                min_dist = float(np.min(np.abs(pose_bt - audio_each)))
            score += math.exp(-((min_dist**2) / (2 * sigma**2)))
        score /= len(audio_bt)
        raw_score = float(score)
        score = raw_score / BEAT_ALIGN_GT_MEAN
        return {
            "status": "ok",
            "score": float(score),
            "score_raw": raw_score,
            "gt_mean_reference": BEAT_ALIGN_GT_MEAN,
            "wrists": "both",
            "onset_raw_count": int(len(onset_raw)),
            "onset_backtrack_count": int(len(onset_bt)),
            "onset_backtrack_rms_count": int(len(onset_bt_rms)),
            "beat_counts": [int(len(x[0])) for x in [beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist]],
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> Dict[str, Any]:
    mu1 = np.atleast_1d(np.asarray(feats_a, dtype=np.float64).mean(axis=0))
    mu2 = np.atleast_1d(np.asarray(feats_b, dtype=np.float64).mean(axis=0))
    sigma1 = np.atleast_2d(np.cov(np.asarray(feats_a, dtype=np.float64), rowvar=False))
    sigma2 = np.atleast_2d(np.cov(np.asarray(feats_b, dtype=np.float64), rowvar=False))
    diff = mu1 - mu2
    covmean, info = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    used_offset = False
    if not np.isfinite(covmean).all():
        eps = 1e-6
        used_offset = True
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    imag_max = 0.0
    diag_imag_max = 0.0
    if np.iscomplexobj(covmean):
        imag_max = float(np.max(np.abs(covmean.imag)))
        diag_imag_max = float(np.max(np.abs(np.diagonal(covmean).imag)))
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            return {
                "status": "failed",
                "value": None,
                "error": f"Imaginary component {imag_max}",
                "sqrtm_info": float(info),
                "sqrtm_imag_max": imag_max,
                "sqrtm_diag_imag_max": diag_imag_max,
                "used_diagonal_offset": used_offset,
            }
        covmean = covmean.real
    value = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    return {
        "status": "ok",
        "value": value,
        "sqrtm_info": float(info),
        "sqrtm_imag_max": imag_max,
        "sqrtm_diag_imag_max": diag_imag_max,
        "used_diagonal_offset": used_offset,
    }


def build_ae(args: SimpleNamespace, checkpoint: Path) -> Tuple[Optional[torch.nn.Module], Dict[str, Any]]:
    from models.motion_autoencoder import HalfEmbeddingNet  # noqa: WPS433

    ae = HalfEmbeddingNet(args).cpu().eval()
    report = strict_load_report(ae, checkpoint)
    if not report.get("strict_load_ok"):
        return None, report
    return ae, report




def embed_windows(ae: torch.nn.Module, motions: np.ndarray, batch_size: int = 128) -> np.ndarray:
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, motions.shape[0], batch_size):
            batch = torch.from_numpy(motions[i : i + batch_size].astype(np.float32))
            feats.append(ae(batch).detach().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)




def collect_ae_windows(seq: np.ndarray) -> np.ndarray:
    starts = list(range(0, seq.shape[0] - WINDOW + 1, 10))
    return np.stack([seq[s : s + WINDOW] for s in starts], axis=0).astype(np.float32)


def aggregate(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def save_arrays(out_dir: Path, model_name: str, stem: str, pred_norm: np.ndarray, pred_bvh: np.ndarray, gt_norm: np.ndarray, gt_bvh: np.ndarray) -> None:
    sub = out_dir / "clips" / model_name / stem
    sub.mkdir(parents=True, exist_ok=True)
    np.save(sub / "pred_bvh141_norm.npy", pred_norm)
    np.save(sub / "pred_bvh141.npy", pred_bvh)
    np.save(sub / "gt_bvh141_norm.npy", gt_norm)
    np.save(sub / "gt_bvh141.npy", gt_bvh)




