#!/usr/bin/env python3
"""Cache emotion and speaker distributions in LMDB order.

Emotion probabilities are the mean of the GoEmotions sentence-attention head
and the MPNet chunk-mean head, averaged across the four speakers of one script.
Speaker probabilities are a one-hot of the speaker id in the clip filename.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import train_emotion as words

POSE_LEN = 34
STRIDE = 10
CACHE = words.CACHE
OUT = CACHE / "frozen_style"


def lmdb_count(split: str) -> int:
    import lmdb

    env = lmdb.open(str(CACHE / split / "bvh_rot_cache"), readonly=True, lock=False)
    with env.begin() as txn:
        count = txn.stat()["entries"]
    env.close()
    return count


def first_frames(split: str, mean: np.ndarray, std: np.ndarray):
    import lmdb
    import pyarrow

    env = lmdb.open(str(CACHE / split / "bvh_rot_cache"), readonly=True, lock=False)
    frames = []
    with env.begin() as txn:
        count = txn.stat()["entries"]
        for index in range(count):
            sample = pyarrow.deserialize(txn.get(f"{index:05d}".encode("ascii")))
            pose = np.asarray(sample[0], dtype=np.float32)
            frames.append(pose[0] * std + mean)
    env.close()
    return frames


def emotion_probs(device, emotion_path: Path):
    go_tok = AutoTokenizer.from_pretrained(str(words.HUB / "roberta-base-go_emotions"))
    go = AutoModelForSequenceClassification.from_pretrained(str(words.HUB / "roberta-base-go_emotions")).to(device).eval()
    mp_tok = AutoTokenizer.from_pretrained(str(words.HUB / "all-mpnet-base-v2"))
    mp = AutoModel.from_pretrained(str(words.HUB / "all-mpnet-base-v2")).to(device).eval()
    sentence_state, chunk_state = words.load_emotion_heads(emotion_path)
    go_head = words.AttnHead(768)
    go_head.load_state_dict(sentence_state)
    go_head.to(device).eval()
    mp_head = words.VectorHead(768)
    mp_head.load_state_dict(chunk_state)
    mp_head.to(device).eval()
    by_stem = {}
    for split in ("train", "val", "test"):
        rows = words.load_split(split)
        _, chunks, _ = words.encode_sentences(go, go_tok, rows, device, 512, True)
        means, _, _ = words.encode_documents(mp, mp_tok, rows, device, 384, 192, False)
        data, mask = words.batch_chunks(chunks, device)
        with torch.no_grad():
            logits = (go_head(data, mask) + mp_head(torch.from_numpy(means).to(device))).cpu() / 2
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["script"]].append(index)
        pooled = logits.clone()
        for indexes in groups.values():
            pooled[indexes] = logits[indexes].mean(dim=0)
        prob = torch.softmax(pooled, dim=-1).numpy().astype(np.float32)
        for index, row in enumerate(rows):
            by_stem[row["stem"]] = prob[index]
        logger.info(f"emotion {split} {len(rows)}")
    del go, mp
    torch.cuda.empty_cache()
    return by_stem


def speaker_prob(stem: str) -> np.ndarray:
    prob = np.zeros(30, dtype=np.float32)
    prob[int(stem.split("_", 1)[0]) - 1] = 1.0
    return prob


def build_split(split: str, emotion, mean: np.ndarray, std: np.ndarray):
    frames = first_frames(split, mean, std)
    files = [(path.stem, np.loadtxt(path)) for path in sorted((CACHE / split / "bvh_rot").glob("*.bvh"))]
    emo_rows = []
    spk_rows = []
    cursor = 0
    for stem, pose in files:
        if cursor >= len(frames):
            break
        src = 0
        owned = False
        limit = pose.shape[0] if split == "test" else pose.shape[0] - POSE_LEN + 1
        while cursor < len(frames) and src < limit:
            if np.mean(np.abs(pose[src] - frames[cursor])) > 1e-3:
                src += STRIDE
                continue
            emo_rows.append(emotion[stem])
            spk_rows.append(speaker_prob(stem))
            owned = True
            src += STRIDE
            cursor += 1
            if split == "test":
                break
        if not owned:
            continue
    if cursor != len(frames):
        raise RuntimeError(f"{split} aligned {cursor} of {len(frames)} lmdb rows")
    emo = np.stack(emo_rows).astype(np.float32)
    spk = np.stack(spk_rows).astype(np.float32)
    np.save(OUT / f"{split}_emo.npy", emo)
    np.save(OUT / f"{split}_spk.npy", spk)
    logger.info(f"{split} cached {len(emo)} windows")
    return len(emo)


def main() -> int:
    global CACHE, OUT
    data = Path(os.environ["BEAT_ROOT"]) if os.environ.get("BEAT_ROOT") else Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(data / "datasets/BEAT/beat_cache/cog_4english_15_141_full"))
    parser.add_argument("--hub", default=str(data / "datasets/BEAT/hub"))
    parser.add_argument("--emotion", default=str(Path(__file__).resolve().parents[1] / "ckpts" / "emotion.pt"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    CACHE = Path(args.cache)
    OUT = Path(args.out) if args.out else CACHE / "frozen_style"
    words.configure(CACHE, Path(args.hub), Path(args.out) if args.out else OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    emotion = emotion_probs(device, Path(args.emotion))
    mean = np.load(CACHE / "train/bvh_rot/bvh_mean.npy")
    std = np.load(CACHE / "train/bvh_rot/bvh_std.npy")
    counts = {split: build_split(split, emotion, mean, std) for split in ("train", "val", "test")}
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "emotion": "go_sent_attn + mp_chunk_mean, speaker-pooled logits",
                "speaker": "one-hot of filename speaker id minus 1",
                "counts": counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
