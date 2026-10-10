#!/usr/bin/env python3
"""Train the utterance speaker classifier.

wav2vec2 stays frozen. Each clip is a bag of 2-second segments at 16 kHz.
An attention pool reads the bag and predicts one of 30 speaker classes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

WINDOW = 32000
HOP = 16000
N_CLASSES = 30


class AttentionPool(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, in_dim: int = 768):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.cls = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, n_classes))

    def forward(self, x, mask):
        score = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        attn = torch.softmax(score, dim=1)
        pooled = (attn.unsqueeze(-1) * x).sum(dim=1)
        return self.cls(pooled)


def segments(audio: np.ndarray) -> list[np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.shape[0] < WINDOW:
        padded = np.zeros(WINDOW, dtype=np.float32)
        padded[: audio.shape[0]] = audio
        return [padded]
    starts = range(0, audio.shape[0] - WINDOW + 1, HOP)
    return [audio[start : start + WINDOW] for start in starts]


def normalize(audio: torch.Tensor) -> torch.Tensor:
    mean = audio.mean(dim=-1, keepdim=True)
    std = audio.std(dim=-1, keepdim=True).clamp_min(1e-5)
    return (audio - mean) / std


def clip_label(stem: str) -> int:
    return int(stem.split("_", 1)[0]) - 1


@torch.no_grad()
def encode_segments(encoder, audio: np.ndarray, device: torch.device) -> torch.Tensor:
    chunks = segments(audio)
    batch = normalize(torch.from_numpy(np.stack(chunks)).to(device))
    hidden = []
    for start in range(0, batch.shape[0], 16):
        hidden.append(encoder(batch[start : start + 16]).last_hidden_state.mean(dim=1))
    return torch.cat(hidden, dim=0)


@torch.no_grad()
def predict_speaker(wav: np.ndarray, wav2vec_dir: Path, checkpoint: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from transformers import Wav2Vec2Model

    encoder = Wav2Vec2Model.from_pretrained(str(wav2vec_dir)).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    head = AttentionPool().to(device).eval()
    state = torch.load(checkpoint, map_location="cpu")
    head.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    feats = encode_segments(encoder, wav, device).unsqueeze(0)
    mask = torch.ones(feats.shape[:2], dtype=torch.bool, device=device)
    logits = head(feats, mask)
    prob = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32)
    one_hot = np.zeros(N_CLASSES, dtype=np.float32)
    one_hot[int(prob.argmax())] = 1.0
    return one_hot, prob


@torch.no_grad()
def extract(cache: Path, split: str, encoder, device: torch.device):
    stems = sorted(path.stem for path in (cache / split / "wave16k").glob("*.npy"))
    features = []
    offsets = [0]
    labels = []
    for index, stem in enumerate(stems):
        if index % 50 == 0:
            logger.info(f"extract {split} {index}/{len(stems)}")
        audio = np.load(cache / split / "wave16k" / f"{stem}.npy").astype(np.float32)
        encoded = encode_segments(encoder, audio, device).cpu()
        features.append(encoded)
        offsets.append(offsets[-1] + encoded.shape[0])
        labels.append(clip_label(stem))
    return torch.cat(features).numpy().astype(np.float32), np.asarray(offsets, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def load_or_extract(cache: Path, split: str, encoder, device: torch.device, out_dir: Path):
    path = out_dir / f"speaker_{split}.npz"
    if path.exists():
        saved = np.load(path)
        logger.info(f"reuse {path}")
        return saved["features"], saved["offsets"], saved["labels"]
    features, offsets, labels = extract(cache, split, encoder, device)
    np.savez(path, features=features, offsets=offsets, labels=labels)
    return features, offsets, labels


def bags(features, offsets, labels, indices):
    return [torch.from_numpy(features[offsets[i] : offsets[i + 1]]) for i in indices], torch.from_numpy(labels[indices]).long()


def collate(items):
    xs, ys = zip(*items)
    lengths = [x.shape[0] for x in xs]
    maxlen = max(lengths)
    batch = xs[0].new_zeros((len(xs), maxlen, xs[0].shape[1]))
    mask = torch.zeros(len(xs), maxlen, dtype=torch.bool)
    for row, (x, length) in enumerate(zip(xs, lengths)):
        batch[row, :length] = x
        mask[row, :length] = True
    return batch, mask, torch.stack(ys)


def macro_recall(pred, target, n_classes):
    recalls = []
    for cls in range(n_classes):
        chosen = target == cls
        if int(chosen.sum()) == 0:
            continue
        recalls.append(float((pred[chosen] == cls).float().mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def class_weights(labels: np.ndarray, n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(weights).to(device)


def run_epoch(model, features, offsets, labels, indices, optimizer, device, train: bool, weight: torch.Tensor):
    model.train(train)
    order = indices.copy()
    if train:
        np.random.shuffle(order)
    total_loss = 0.0
    preds = []
    targets = []
    for start in range(0, len(order), 16):
        chunk = order[start : start + 16]
        xs, ys = bags(features, offsets, labels, chunk)
        x, mask, y = collate(list(zip(xs, ys)))
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x, mask)
            loss = F.cross_entropy(logits, y, weight=weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * y.shape[0]
        preds.append(logits.detach().argmax(-1).cpu())
        targets.append(y.cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    return total_loss / max(len(order), 1), float((pred == target).float().mean()), macro_recall(pred, target, N_CLASSES)


def train_task(train_pack, val_pack, device, out_dir: Path):
    model = AttentionPool().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weight = class_weights(train_pack[2], N_CLASSES, device)
    train_idx = np.arange(len(train_pack[2]))
    val_idx = np.arange(len(val_pack[2]))
    best = -1.0
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(60):
        train_loss, train_acc, train_ua = run_epoch(model, *train_pack, train_idx, optimizer, device, True, weight)
        val_loss, val_acc, val_ua = run_epoch(model, *val_pack, val_idx, optimizer, device, False, weight)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_ua": train_ua,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_ua": val_ua,
            }
        )
        logger.info(f"epoch {epoch:02d} val_loss {val_loss:.4f} val_acc {val_acc:.3f} val_ua {val_ua:.3f} train_acc {train_acc:.3f}")
        if val_ua > best + 0.005:
            best = val_ua
            best_epoch = epoch
            stale = 0
            torch.save(
                {"task": "speaker", "epoch": epoch, "val_acc": val_acc, "val_ua": val_ua, "model": model.state_dict()},
                out_dir / "speaker.pt",
            )
        else:
            stale += 1
        if epoch + 1 >= 12 and stale >= 6:
            logger.info(f"unweighted recall stable at {best:.3f} since epoch {best_epoch}")
            break
    report = {"task": "speaker", "best_val_ua": best, "best_epoch": best_epoch, "history": history}
    (out_dir / "speaker_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    data = Path(os.environ["BEAT_ROOT"]) if os.environ.get("BEAT_ROOT") else Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(data / "datasets/BEAT/beat_cache/cog_4english_15_141_full"))
    parser.add_argument("--wav2vec", default=str(data / "datasets/BEAT/hub/wav2vec2-base"))
    parser.add_argument("--out", default=str(data / "outputs/speaker"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    from transformers import Wav2Vec2Model

    logger.info(f"load frozen wav2vec {args.wav2vec}")
    encoder = Wav2Vec2Model.from_pretrained(args.wav2vec).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    packs = [load_or_extract(cache, split, encoder, device, out_dir) for split in ("train", "val")]
    logger.info(f"clips train {len(packs[0][2])} val {len(packs[1][2])}")
    train_task(packs[0], packs[1], device, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
