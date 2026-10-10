import argparse
import json
import math
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.textgrid as tg
from dataloaders.build_vocab import Vocab


SPEAKER_NAMES = [
    "wayne", "scott", "solomon", "lawrence", "stewart", "carla", "sophie", "catherine", "miranda", "kieks",
    "nidal", "zhao", "lu", "zhang", "carlos", "jorge", "itoi", "daiki", "jaime", "li",
    "ayana", "luqi", "hailing", "kexin", "goto", "reamey", "yingqing", "tiffnay", "hanieh", "katya",
]

JOINT_LIST = {
    "beat_joints": {
        "Hips": [6, 6], "Spine": [3, 9], "Spine1": [3, 12], "Spine2": [3, 15], "Spine3": [3, 18],
        "Neck": [3, 21], "Neck1": [3, 24], "Head": [3, 27], "HeadEnd": [3, 30],
        "RShoulder": [3, 33], "RArm": [3, 36], "RArm1": [3, 39], "RHand": [3, 42],
        "RHandM1": [3, 45], "RHandM2": [3, 48], "RHandM3": [3, 51], "RHandM4": [3, 54],
        "RHandR": [3, 57], "RHandR1": [3, 60], "RHandR2": [3, 63], "RHandR3": [3, 66], "RHandR4": [3, 69],
        "RHandP": [3, 72], "RHandP1": [3, 75], "RHandP2": [3, 78], "RHandP3": [3, 81], "RHandP4": [3, 84],
        "RHandI": [3, 87], "RHandI1": [3, 90], "RHandI2": [3, 93], "RHandI3": [3, 96], "RHandI4": [3, 99],
        "RHandT1": [3, 102], "RHandT2": [3, 105], "RHandT3": [3, 108], "RHandT4": [3, 111],
        "LShoulder": [3, 114], "LArm": [3, 117], "LArm1": [3, 120], "LHand": [3, 123],
        "LHandM1": [3, 126], "LHandM2": [3, 129], "LHandM3": [3, 132], "LHandM4": [3, 135],
        "LHandR": [3, 138], "LHandR1": [3, 141], "LHandR2": [3, 144], "LHandR3": [3, 147], "LHandR4": [3, 150],
        "LHandP": [3, 153], "LHandP1": [3, 156], "LHandP2": [3, 159], "LHandP3": [3, 162], "LHandP4": [3, 165],
        "LHandI": [3, 168], "LHandI1": [3, 171], "LHandI2": [3, 174], "LHandI3": [3, 177], "LHandI4": [3, 180],
        "LHandT1": [3, 183], "LHandT2": [3, 186], "LHandT3": [3, 189], "LHandT4": [3, 192],
        "RUpLeg": [3, 195], "RLeg": [3, 198], "RFoot": [3, 201], "RFootF": [3, 204], "RToeBase": [3, 207],
        "RToeBaseEnd": [3, 210], "LUpLeg": [3, 213], "LLeg": [3, 216], "LFoot": [3, 219],
        "LFootF": [3, 222], "LToeBase": [3, 225], "LToeBaseEnd": [3, 228],
    },
    "beat_141": {
        "Spine": 3, "Neck": 3, "Neck1": 3, "RShoulder": 3, "RArm": 3, "RArm1": 3, "RHand": 3,
        "RHandM1": 3, "RHandM2": 3, "RHandM3": 3, "RHandR": 3, "RHandR1": 3, "RHandR2": 3, "RHandR3": 3,
        "RHandP": 3, "RHandP1": 3, "RHandP2": 3, "RHandP3": 3, "RHandI": 3, "RHandI1": 3, "RHandI2": 3,
        "RHandI3": 3, "RHandT1": 3, "RHandT2": 3, "RHandT3": 3, "LShoulder": 3, "LArm": 3, "LArm1": 3,
        "LHand": 3, "LHandM1": 3, "LHandM2": 3, "LHandM3": 3, "LHandR": 3, "LHandR1": 3, "LHandR2": 3,
        "LHandR3": 3, "LHandP": 3, "LHandP1": 3, "LHandP2": 3, "LHandP3": 3, "LHandI": 3, "LHandI1": 3,
        "LHandI2": 3, "LHandI3": 3, "LHandT1": 3, "LHandT2": 3, "LHandT3": 3,
    },
}

SPLIT = {
    "train": [
        "0_9_9", "0_10_10", "0_11_11", "0_12_12", "0_13_13", "0_14_14", "0_15_15", "0_16_16",
        "0_17_17", "0_18_18", "0_19_19", "0_20_20", "0_21_21", "0_22_22", "0_23_23", "0_24_24",
        "0_25_25", "0_26_26", "0_27_27", "0_28_28", "0_29_29", "0_30_30", "0_31_31", "0_32_32",
        "0_33_33", "0_34_34", "0_35_35", "0_36_36", "0_37_37", "0_38_38", "0_39_39", "0_40_40",
        "0_41_41", "0_42_42", "0_43_43", "0_44_44", "0_45_45", "0_46_46", "0_47_47", "0_48_48",
        "0_49_49", "0_50_50", "0_51_51", "0_52_52", "0_53_53", "0_54_54", "0_55_55", "0_56_56",
        "0_66_66", "0_67_67", "0_68_68", "0_69_69", "0_70_70", "0_71_71",
        "0_74_74", "0_75_75", "0_76_76", "0_77_77", "0_78_78", "0_79_79",
        "0_82_82", "0_83_83", "0_84_84", "0_85_85",
        "0_88_88", "0_89_89", "0_90_90", "0_91_91", "0_92_92", "0_93_93",
        "0_96_96", "0_97_97", "0_98_98", "0_99_99", "0_100_100", "0_101_101",
        "0_104_104", "0_105_105", "0_106_106", "0_107_107", "0_108_108", "0_109_109",
        "0_112_112", "0_113_113", "0_114_114", "0_115_115", "0_116_116", "0_117_117",
        "1_2_2", "1_3_3", "1_4_4", "1_5_5", "1_6_6", "1_7_7", "1_8_8", "1_9_9", "1_10_10", "1_11_11",
    ],
    "val": ["0_57_57", "0_58_58", "0_59_59", "0_60_60", "0_61_61", "0_62_62", "0_63_63", "0_64_64",
            "0_72_72", "0_80_80", "0_86_86", "0_94_94", "0_102_102", "0_110_110", "0_118_118", "1_12_12"],
    "test": ["0_1_1", "0_2_2", "0_3_3", "0_4_4", "0_5_5", "0_6_6", "0_7_7", "0_8_8",
             "0_65_65", "0_73_73", "0_81_81", "0_87_87", "0_95_95", "0_103_103", "0_111_111", "1_1_1"],
}


def ensure_dirs(cache_root):
    for split in ("train", "val", "test"):
        for folder in ("wave16k", "bvh_rot", "bvh_full", "bvh_rot_vis", "facial52", "text", "emo", "sem"):
            (cache_root / split / folder).mkdir(parents=True, exist_ok=True)


def read_wav_16k(path):
    sr, data = wavfile.read(path)
    source_dtype = str(data.dtype)
    source_min = float(np.min(data))
    source_max = float(np.max(data))
    data = np.asarray(data)
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
    if sr != 16000:
        target_len = int(round(len(data) * 16000 / sr))
        data = signal.resample(data, target_len).astype(np.float32)
    return data, {
        "path": str(path),
        "source_sr": int(sr),
        "source_dtype": source_dtype,
        "source_min": source_min,
        "source_max": source_max,
        "normalization": normalization,
        "samples_16k": int(data.shape[0]),
        "min_16k": float(np.min(data)),
        "max_16k": float(np.max(data)),
        "within_minus1_1": bool(np.max(np.abs(data)) <= 1.0001),
    }


def select_rot(data):
    out = []
    for joint, width in JOINT_LIST["beat_141"].items():
        src = JOINT_LIST["beat_joints"][joint][1]
        out.extend(data[src - width:src])
    return np.asarray(out, dtype=np.float32)


def _rewrite_bvh_header(header_lines, frame_count, target_fps):
    out = []
    for line in header_lines:
        stripped = line.strip().lower()
        if stripped.startswith("frames:"):
            out.append(f"Frames: {frame_count}\n")
        elif stripped.startswith("frame time:"):
            out.append(f"Frame Time: {1.0 / target_fps:.7f}\n")
        else:
            out.append(line if line.endswith("\n") else line + "\n")
    return out


def write_bvh_variants(raw_bvh, out_rot, out_full, out_vis, target_fps):
    reduce_factor_bvh = int(120 / target_fps)
    selected_full = []
    all_rot = []
    lines = Path(raw_bvh).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    frame_time_idx = next((i for i, line in enumerate(lines) if line.strip().lower().startswith("frame time")), None)
    if frame_time_idx is None:
        raise RuntimeError(f"could not find BVH Frame Time line in {raw_bvh}")
    first_motion_idx = frame_time_idx + 1
    motion_rows = []
    vis_rows = []
    for motion_idx, line in enumerate(lines[first_motion_idx:]):
        if motion_idx % reduce_factor_bvh != 0:
            continue
        data = np.fromstring(line, dtype=float, sep=" ")
        if data.size == 0:
            continue
        selected = select_rot(data)
        trainable = np.zeros_like(data)
        for joint, width in JOINT_LIST["beat_141"].items():
            src_idx = JOINT_LIST["beat_joints"][joint][1]
            trainable[src_idx - width:src_idx] = data[src_idx - width:src_idx]
        selected_full.append(data)
        all_rot.append(selected)
        motion_rows.append(np.array2string(data, max_line_width=np.inf, precision=6, separator=" ")[1:-1] + "\n")
        vis_rows.append(np.array2string(trainable, max_line_width=np.inf, precision=6, separator=" ")[1:-1] + "\n")

    header = _rewrite_bvh_header(lines[:first_motion_idx], len(all_rot), target_fps)
    with open(out_rot, "w") as rot, open(out_full, "w") as full, open(out_vis, "w") as vis:
        full.writelines(header)
        vis.writelines(header)
        for selected, full_line, vis_line in zip(all_rot, motion_rows, vis_rows):
            full.write(full_line)
            vis.write(vis_line)
            rot.write(np.array2string(selected, max_line_width=np.inf, precision=6, separator=" ")[1:-1] + "\n")
    return (
        np.asarray(selected_full),
        np.asarray(all_rot),
        {
            "path": str(raw_bvh),
            "frame_time_line_zero_based": int(frame_time_idx),
            "first_motion_line_zero_based": int(first_motion_idx),
            "source_fps_assumed": 120,
            "target_fps": int(target_fps),
            "reduce_factor": int(reduce_factor_bvh),
            "sample_phase": "motion_index % reduce_factor == 0",
            "selected_frames": int(len(all_rot)),
            "source_motion_lines": int(len(lines) - first_motion_idx),
        },
    )


def write_facial(raw_json, out_json, target_fps):
    reduce_factor_json = int(60 / target_fps)
    with open(raw_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = data.get("names", [])
    all_weights = []
    frames = []
    for idx, frame in enumerate(data["frames"]):
        if len(names) and len(frame["weights"]) != len(names):
            raise RuntimeError(f"facial weight/name mismatch in {raw_json}: {len(frame['weights'])} vs {len(names)}")
        all_weights.append(frame["weights"])
        if idx % reduce_factor_json == 0:
            frames.append(frame)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"names": data["names"], "frames": frames}, f)
    return np.asarray(all_weights, dtype=np.float32), {
        "path": str(raw_json),
        "names": int(len(names)),
        "weights": int(len(frames[0]["weights"])) if frames else 0,
        "target_fps": int(target_fps),
        "reduce_factor": int(reduce_factor_json),
    }


def build_vocab(cache_root, split="train"):
    vocab = Vocab("beat_4english_15_141")
    for text_file in sorted((cache_root / split / "text").glob("*.TextGrid")):
        grid = tg.TextGrid.fromFile(text_file)
        if not grid:
            continue
        for interval in grid[0]:
            word = interval.mark.replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ")
            vocab.index_word(word)
    vocab.word_embedding_weights = None
    with open(cache_root / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    return vocab.n_words


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def split_entry_to_stem(entry, raw_stems):
    if entry in raw_stems:
        return entry
    parts = entry.split("_")
    for end in range(len(parts) - 1, 3, -1):
        candidate = "_".join(parts[:end])
        if candidate in raw_stems:
            return candidate
    return "_".join(parts[:-1]) if len(parts) > 5 and parts[-1].isdigit() else entry


def load_official_stem_splits(split_dir, raw_root):
    split_dir = Path(split_dir)
    raw_stems = {path.stem for path in raw_root.glob("*/*.wav")}
    out = {}
    source_counts = {}
    duplicates = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"official_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        stems = [split_entry_to_stem(entry, raw_stems) for entry in entries]
        out[split] = set(stems)
        source_counts[split] = len(entries)
        duplicates[split] = len(stems) - len(out[split])
    overlap = {
        "train_val": sorted(out["train"] & out["val"]),
        "train_test": sorted(out["train"] & out["test"]),
        "val_test": sorted(out["val"] & out["test"]),
    }
    return out, {
        "split_dir": str(split_dir),
        "source_counts": source_counts,
        "unique_stem_counts": {key: len(value) for key, value in out.items()},
        "duplicate_entries_after_stem_normalization": duplicates,
        "overlap_counts": {key: len(value) for key, value in overlap.items()},
        "overlap_examples": {key: value[:10] for key, value in overlap.items()},
    }


def fallback_tail_split(stem, speaker):
    tail = stem.split(f"{speaker}_{SPEAKER_NAMES[speaker - 1]}_", 1)[-1]
    return next((key for key, values in SPLIT.items() if tail in values), None)


def process(args):
    raw_root = Path(args.raw_root)
    cache_root = Path(args.cache_root)
    if args.clean and cache_root.exists():
        shutil.rmtree(cache_root)
    ensure_dirs(cache_root)

    stats = {"wave": [], "facial": [], "bvh": [], "seconds": []}
    missing = []
    processed = []
    wav_reports = []
    bvh_reports = []
    facial_reports = []
    speakers = [int(x) for x in args.speakers.split(",")]
    if args.split_dir:
        stem_splits, split_report = load_official_stem_splits(args.split_dir, raw_root)
    else:
        stem_splits, split_report = None, {"split_dir": None, "source": "built-in tail SPLIT"}
    per_speaker_split_counts = {}
    all_processed_stems = {"train": set(), "val": set(), "test": set()}

    for speaker in speakers:
        name = SPEAKER_NAMES[speaker - 1]
        speaker_root = raw_root / str(speaker)
        wav_files = sorted(speaker_root.glob("*.wav"))
        for wav_path in tqdm(wav_files, desc=f"speaker {speaker}"):
            stem = wav_path.stem
            if stem_splits is not None:
                split = next((k for k, values in stem_splits.items() if stem in values), None)
            else:
                split = fallback_tail_split(stem, speaker)
            if split is None:
                continue
            count_key = f"{speaker}:{split}"
            if args.max_files_per_speaker_split > 0 and per_speaker_split_counts.get(count_key, 0) >= args.max_files_per_speaker_split:
                continue
            required = {ext: speaker_root / f"{stem}.{ext}" for ext in ("bvh", "json", "TextGrid", "csv", "txt")}
            if any(not p.exists() for p in required.values()):
                missing.append(stem)
                continue

            out_base = cache_root / split
            audio, wav_report = read_wav_16k(wav_path)
            np.save(out_base / "wave16k" / f"{stem}.npy", audio)
            shutil.copy(required["TextGrid"], out_base / "text" / f"{stem}.TextGrid")
            shutil.copy(required["csv"], out_base / "emo" / f"{stem}.csv")
            shutil.copy(required["txt"], out_base / "sem" / f"{stem}.txt")
            facial, facial_report = write_facial(required["json"], out_base / "facial52" / f"{stem}.json", args.target_fps)
            _, bvh_rot, bvh_report = write_bvh_variants(
                required["bvh"],
                out_base / "bvh_rot" / f"{stem}.bvh",
                out_base / "bvh_full" / f"{stem}.bvh",
                out_base / "bvh_rot_vis" / f"{stem}.bvh",
                args.target_fps,
            )
            per_speaker_split_counts[count_key] = per_speaker_split_counts.get(count_key, 0) + 1
            all_processed_stems[split].add(stem)
            processed.append({"stem": stem, "speaker": int(speaker), "split": split, "frames": int(bvh_rot.shape[0]), "seconds": float(len(audio) / 16000)})
            if len(wav_reports) < args.report_sample_limit:
                wav_reports.append(wav_report)
            if len(bvh_reports) < args.report_sample_limit:
                bvh_reports.append(bvh_report)
            if len(facial_reports) < args.report_sample_limit:
                facial_reports.append(facial_report)
            if split == "train":
                stats["wave"].append(audio)
                stats["facial"].append(facial)
                stats["bvh"].append(bvh_rot)
                stats["seconds"].append(len(audio) / 16000)

    wave = np.concatenate(stats["wave"]) if stats["wave"] else np.asarray([0.0], dtype=np.float32)
    facial = np.concatenate(stats["facial"], axis=0) if stats["facial"] else np.zeros((1, 51), dtype=np.float32)
    bvh = np.concatenate(stats["bvh"], axis=0) if stats["bvh"] else np.zeros((1, 141), dtype=np.float32)
    np.save(cache_root / "train" / "wave16k" / "npy_mean.npy", np.mean(wave, axis=0))
    np.save(cache_root / "train" / "wave16k" / "npy_std.npy", np.std(wave, axis=0) + 1e-8)
    np.save(cache_root / "train" / "facial52" / "json_mean.npy", np.mean(facial, axis=0))
    np.save(cache_root / "train" / "facial52" / "json_std.npy", np.std(facial, axis=0) + 1e-8)
    np.save(cache_root / "train" / "bvh_rot" / "bvh_mean.npy", np.mean(bvh, axis=0))
    np.save(cache_root / "train" / "bvh_rot" / "bvh_std.npy", np.std(bvh, axis=0) + 1e-8)
    n_words = build_vocab(cache_root, split="train")

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = len(list((cache_root / split / "bvh_rot").glob("*.bvh")))
    processed_overlap = {
        "train_val": sorted(all_processed_stems["train"] & all_processed_stems["val"]),
        "train_test": sorted(all_processed_stems["train"] & all_processed_stems["test"]),
        "val_test": sorted(all_processed_stems["val"] & all_processed_stems["test"]),
    }
    report = {
        "status": "ok",
        "created_utc": utc_now(),
        "cache_root": str(cache_root),
        "raw_root": str(raw_root),
        "speakers": speakers,
        "target_fps": int(args.target_fps),
        "counts": counts,
        "vocab_words": int(n_words),
        "vocab_source": "train split only",
        "stats_source": "train split only",
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "split_report": split_report,
        "processed_overlap_counts": {key: len(value) for key, value in processed_overlap.items()},
        "processed_overlap_examples": {key: value[:10] for key, value in processed_overlap.items()},
        "max_files_per_speaker_split": int(args.max_files_per_speaker_split),
        "processed_examples": processed[: args.report_sample_limit],
        "wav_examples": wav_reports,
        "bvh_examples": bvh_reports,
        "facial_examples": facial_reports,
    }
    with open(cache_root / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print({"cache_root": str(cache_root), "counts": counts, "vocab_words": n_words, "missing": len(missing)})
    if missing:
        print("missing examples:", missing[:20])


def main():
    parser = argparse.ArgumentParser()
    data_root = os.environ.get("BEAT_ROOT", "")
    parser.add_argument(
        "--raw-root",
        default=os.path.join(data_root, "datasets/BEAT/raw/beat_english_v0.2.1/beat_english_v0.2.1") if data_root else "",
    )
    parser.add_argument(
        "--cache-root",
        default=os.path.join(data_root, "datasets/BEAT/beat_cache/cog_4english_15_141_full") if data_root else "",
    )
    parser.add_argument("--speakers", default="2,4,6,8")
    parser.add_argument("--split-dir", default="")
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--max-files-per-speaker-split", type=int, default=0)
    parser.add_argument("--report-sample-limit", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    process(parser.parse_args())


if __name__ == "__main__":
    main()
