#!/usr/bin/env python3
"""Train text emotion classifiers on the TextGrid words tier.

Each clip is encoded by a frozen GoEmotions RoBERTa and by MPNet. Heads pool
those features by chunk mean, chunk attention, or 20-word sentence attention.
The run writes one checkpoint per head and selects by validation unweighted recall.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

def default_data_root() -> Path:
    if os.environ.get("BEAT_ROOT"):
        return Path(os.environ["BEAT_ROOT"])
    return Path(__file__).resolve().parents[2]


def configure(cache: Path, hub: Path, out: Path) -> None:
    global CACHE, HUB, OUT
    CACHE = cache
    HUB = hub
    OUT = out


_DATA = default_data_root()
CACHE = _DATA / "datasets/BEAT/beat_cache/cog_4english_15_141_full"
HUB = _DATA / "datasets/BEAT/hub"
OUT = _DATA / "outputs/emotion"
N_CLASSES = 8
NAMES = ["neutral", "happiness", "anger", "sadness", "contempt", "surprise", "fear", "disgust"]
SENT_WORDS = 20


def _state_dict(value):
    if isinstance(value, dict) and isinstance(value.get("model"), dict):
        return value["model"]
    return value


def load_emotion_heads(path):
    """Load the two text emotion heads stored together in emotion.pt."""
    path = Path(path)
    if path.is_dir():
        path = path / "emotion.pt"
    saved = torch.load(path, map_location="cpu")
    return _state_dict(saved["sentence_attention"]), _state_dict(saved["chunk_mean"])


def words_of(path: Path):
    text = path.read_text(errors="ignore")
    tiers = list(re.finditer(r'name = "(.*)"', text))
    if not tiers:
        return []
    chosen = 0
    for i, tier in enumerate(tiers):
        if tier.group(1) == "words":
            chosen = i
            break
    end = tiers[chosen + 1].start() if chosen + 1 < len(tiers) else len(text)
    block = text[tiers[chosen].start() : end]
    return [item.strip() for item in re.findall(r'text = "(.*)"', block) if item.strip()]


def load_split(split: str):
    rows = []
    for path in sorted((CACHE / split / "text").glob("*.TextGrid")):
        csv_path = CACHE / split / "emo" / f"{path.stem}.csv"
        if not csv_path.exists():
            continue
        words = words_of(path)
        score = int(csv_path.read_text().strip().splitlines()[0].split(",")[-1])
        parts = path.stem.split("_")
        rows.append(
            {
                "split": split,
                "stem": path.stem,
                "script": f"{parts[-3]}_{parts[-2]}_{parts[-1]}",
                "label": score,
                "words": words,
                "text": " ".join(words),
            }
        )
    return rows


def sentences(words):
    if not words:
        return ["empty"]
    return [" ".join(words[start : start + SENT_WORDS]) for start in range(0, len(words), SENT_WORDS)]


def metrics(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    acc = float((pred == target).mean()) if len(target) else 0.0
    recalls = []
    for cls in range(N_CLASSES):
        mask = target == cls
        if np.any(mask):
            recalls.append(float((pred[mask] == cls).mean()))
    return {"acc": acc, "ua": float(np.mean(recalls)) if recalls else 0.0, "recall": recalls}


def script_metrics(rows, pred):
    buckets = defaultdict(list)
    for row, label in zip(rows, pred):
        buckets[row["script"]].append((int(label), row["label"]))
    chosen, truth = [], []
    for items in buckets.values():
        chosen.append(Counter(pred for pred, _ in items).most_common(1)[0][0])
        truth.append(items[0][1])
    out = metrics(chosen, truth)
    out["n_scripts"] = len(truth)
    return out


def class_weights(labels):
    counts = Counter(int(y) for y in labels)
    return torch.tensor(
        [len(labels) / (N_CLASSES * counts[c]) for c in range(N_CLASSES)],
        dtype=torch.float32,
    )


def script_groups(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["script"]].append(i)
    return [groups[script] for script in sorted(groups)]


class VectorHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, N_CLASSES)

    def forward(self, features):
        return self.fc(features)


class SpreadAttn(nn.Module):
    """Attention over sentences, mixed with the sentence mean.

    The mix and the entropy term keep a few emotional sentences from
    deciding a long neutral transcript by themselves.
    """

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.fc = nn.Linear(dim, N_CLASSES)
        self.mix = nn.Parameter(torch.zeros(()))

    def forward(self, chunks, mask):
        scores = self.score(chunks).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        sent_logits = self.fc(chunks)
        attended = torch.sum(sent_logits * weights.unsqueeze(-1), dim=1)
        mean_w = mask.float()
        mean_w = mean_w / mean_w.sum(dim=-1, keepdim=True).clamp(min=1.0)
        averaged = torch.sum(sent_logits * mean_w.unsqueeze(-1), dim=1)
        alpha = torch.sigmoid(self.mix)
        return alpha * attended + (1.0 - alpha) * averaged, weights


class AttnHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.fc = nn.Linear(dim, N_CLASSES)

    def forward(self, chunks, mask):
        scores = self.score(chunks).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(chunks * weights.unsqueeze(-1), dim=1)
        return self.fc(pooled)


def batch_chunks(chunk_sets, device):
    width = max(len(chunk) for chunk in chunk_sets)
    dim = chunk_sets[0].shape[1]
    data = torch.zeros((len(chunk_sets), width, dim), device=device)
    mask = torch.zeros((len(chunk_sets), width), dtype=torch.bool, device=device)
    for i, chunk in enumerate(chunk_sets):
        data[i, : len(chunk)] = torch.from_numpy(chunk)
        mask[i, : len(chunk)] = True
    return data, mask


def fit_vector(name, train_x, train_y, val_x, val_rows, test_x, test_rows, device):
    model = VectorHead(train_x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weight = class_weights(train_y).to(device)
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(np.asarray(train_y, dtype=np.int64)).to(device)
    bundles = {
        "val": (torch.from_numpy(val_x).to(device), val_rows),
        "test": (torch.from_numpy(test_x).to(device), test_rows),
    }
    best, best_epoch, stale = -1.0, -1, 0
    best_state, history = None, []
    vy = np.array([row["label"] for row in val_rows])
    for epoch in range(120):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tx), ty, weight=weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(bundles["val"][0]).argmax(dim=-1).cpu().numpy()
        scored = metrics(pred, vy)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), **{k: scored[k] for k in ("acc", "ua")}})
        if scored["ua"] > best + 0.005:
            best, best_epoch, stale = scored["ua"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= 20 and stale >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    report = {"scheme": name, "best_epoch": best_epoch, "history": history, "logits": {}}
    with torch.no_grad():
        for split, (feats, rows) in bundles.items():
            logits = model(feats).cpu().numpy()
            pred = logits.argmax(axis=-1)
            target = np.array([row["label"] for row in rows])
            report[split] = metrics(pred, target)
            report[f"{split}_script"] = script_metrics(rows, pred)
            report["logits"][split] = logits
    logger.info(
        f"[{name}] epoch {best_epoch} val_acc {report['val']['acc']:.3f} "
        f"val_ua {report['val']['ua']:.3f} test_ua {report['test']['ua']:.3f}"
    )
    torch.save({"scheme": name, "model": best_state}, OUT / f"{name}.pt")
    return report


def fit_attn(name, train_chunks, train_y, val_chunks, val_rows, test_chunks, test_rows, device):
    model = AttnHead(train_chunks[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weight = class_weights(train_y).to(device)
    ty = torch.from_numpy(np.asarray(train_y, dtype=np.int64)).to(device)
    bundles = {"val": (val_chunks, val_rows), "test": (test_chunks, test_rows)}
    best, best_epoch, stale = -1.0, -1, 0
    best_state, history = None, []
    vy = np.array([row["label"] for row in val_rows])
    for epoch in range(120):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(*batch_chunks(train_chunks, device)), ty, weight=weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(*batch_chunks(val_chunks, device)).argmax(dim=-1).cpu().numpy()
        scored = metrics(pred, vy)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), **{k: scored[k] for k in ("acc", "ua")}})
        if scored["ua"] > best + 0.005:
            best, best_epoch, stale = scored["ua"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= 20 and stale >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    report = {"scheme": name, "best_epoch": best_epoch, "history": history, "logits": {}}
    with torch.no_grad():
        for split, (chunks, rows) in bundles.items():
            logits = model(*batch_chunks(chunks, device)).cpu().numpy()
            pred = logits.argmax(axis=-1)
            report[split] = metrics(pred, np.array([row["label"] for row in rows]))
            report[f"{split}_script"] = script_metrics(rows, pred)
            report["logits"][split] = logits
    logger.info(
        f"[{name}] epoch {best_epoch} val_acc {report['val']['acc']:.3f} "
        f"val_ua {report['val']['ua']:.3f} test_ua {report['test']['ua']:.3f}"
    )
    torch.save({"scheme": name, "model": best_state}, OUT / f"{name}.pt")
    return report


def fit_spread(name, train_chunks, train_rows, val_chunks, val_rows, test_chunks, test_rows, device):
    model = SpreadAttn(train_chunks[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    script_labels = []
    seen = set()
    for row in train_rows:
        if row["script"] in seen:
            continue
        seen.add(row["script"])
        script_labels.append(row["label"])
    class_weight = class_weights(script_labels).to(device)
    counts = Counter(row["script"] for row in train_rows)
    sample_weight = torch.tensor([1.0 / counts[row["script"]] for row in train_rows], dtype=torch.float32, device=device)
    ty = torch.tensor([row["label"] for row in train_rows], dtype=torch.long, device=device)
    bundles = {"val": (val_chunks, val_rows), "test": (test_chunks, test_rows)}
    best, best_epoch, stale = -1.0, -1, 0
    best_state, history = None, []
    vy = np.array([row["label"] for row in val_rows])
    for epoch in range(120):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits, weights = model(*batch_chunks(train_chunks, device))
        ce = F.cross_entropy(logits, ty, weight=class_weight, reduction="none")
        ce = (ce * sample_weight).sum() / sample_weight.sum()
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        loss = ce - 0.02 * entropy
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(*batch_chunks(val_chunks, device))[0].argmax(dim=-1).cpu().numpy()
        scored = metrics(pred, vy)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "alpha": float(torch.sigmoid(model.mix).detach()), **{k: scored[k] for k in ("acc", "ua")}})
        if scored["ua"] > best + 0.005:
            best, best_epoch, stale = scored["ua"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= 20 and stale >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    report = {"scheme": name, "best_epoch": best_epoch, "history": history, "logits": {}, "alpha": float(torch.sigmoid(model.mix).detach())}
    with torch.no_grad():
        for split, (chunks, rows) in bundles.items():
            logits = model(*batch_chunks(chunks, device))[0].cpu().numpy()
            pred = logits.argmax(axis=-1)
            report[split] = metrics(pred, np.array([row["label"] for row in rows]))
            report[f"{split}_script"] = script_metrics(rows, pred)
            report["logits"][split] = logits
    logger.info(
        f"[{name}] epoch {best_epoch} alpha {report['alpha']:.3f} val_acc {report['val']['acc']:.3f} "
        f"val_ua {report['val']['ua']:.3f} test_ua {report['test']['ua']:.3f}"
    )
    torch.save({"scheme": name, "model": best_state}, OUT / f"{name}.pt")
    return report


def speaker_pool(name, report, val_rows, test_rows):
    pooled = {"scheme": name, "source": report["scheme"]}
    for split, rows in (("val", val_rows), ("test", test_rows)):
        logits = report["logits"][split]
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["script"]].append(index)
        merged = logits.copy()
        for indexes in groups.values():
            merged[indexes] = logits[indexes].mean(axis=0)
        pred = merged.argmax(axis=-1)
        pooled[split] = metrics(pred, np.array([row["label"] for row in rows]))
        pooled[f"{split}_script"] = script_metrics(rows, pred)
        pooled.setdefault("logits", {})[split] = merged
    logger.info(
        f"[{name}] val_acc {pooled['val']['acc']:.3f} val_ua {pooled['val']['ua']:.3f} "
        f"test_ua {pooled['test']['ua']:.3f}"
    )
    return pooled


def token_windows(tokenizer, text, length, stride):
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if not ids:
        ids = [tokenizer.unk_token_id]
    inner = length - 2
    if len(ids) <= inner:
        starts = [0]
    else:
        starts = list(range(0, len(ids) - inner + 1, stride))
        last = len(ids) - inner
        if starts[-1] != last:
            starts.append(last)
    return [tokenizer.build_inputs_with_special_tokens(ids[start : start + inner]) for start in starts]


@torch.no_grad()
def encode_windows(model, tokenizer, pieces, device, want_logits):
    pad_id = tokenizer.pad_token_id
    hidden_parts, logit_parts = [], []
    for start in range(0, len(pieces), 16):
        batch = pieces[start : start + 16]
        width = max(len(piece) for piece in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for i, piece in enumerate(batch):
            input_ids[i, : len(piece)] = torch.tensor(piece, device=device)
            mask[i, : len(piece)] = 1
        kwargs = {"input_ids": input_ids, "attention_mask": mask}
        if want_logits:
            kwargs["output_hidden_states"] = True
        out = model(**kwargs)
        hidden = out.hidden_states[-1] if want_logits else out.last_hidden_state
        weight = mask.unsqueeze(-1).float()
        pooled = (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        hidden_parts.append(pooled.detach().cpu())
        if want_logits:
            logit_parts.append(out.logits.detach().cpu())
    chunks = torch.cat(hidden_parts, dim=0).numpy().astype(np.float32)
    probs = torch.cat(logit_parts, dim=0).numpy().astype(np.float32) if logit_parts else None
    return chunks, probs


def encode_documents(model, tokenizer, rows, device, length, stride, want_logits):
    means, chunks, max_probs = [], [], []
    for i, row in enumerate(rows):
        pieces = token_windows(tokenizer, row["text"], length, stride)
        hidden, probs = encode_windows(model, tokenizer, pieces, device, want_logits)
        means.append(hidden.mean(axis=0))
        chunks.append(hidden)
        if probs is not None:
            max_probs.append(probs.max(axis=0))
        if (i + 1) % 80 == 0:
            logger.info(f"docs {i + 1}/{len(rows)}")
    return (
        np.stack(means).astype(np.float32),
        chunks,
        np.stack(max_probs).astype(np.float32) if max_probs else None,
    )


def encode_sentences(model, tokenizer, rows, device, length, want_logits):
    owners, pieces = [], []
    for index, row in enumerate(rows):
        for sentence in sentences(row["words"]):
            owners.append(index)
            pieces.append(tokenizer(sentence, add_special_tokens=True, truncation=True, max_length=length)["input_ids"])
    hidden, probs = encode_windows(model, tokenizer, pieces, device, want_logits)
    grouped_h = [[] for _ in rows]
    grouped_p = [[] for _ in rows]
    for owner, vector, prob in zip(owners, hidden, probs if probs is not None else [None] * len(hidden)):
        grouped_h[owner].append(vector)
        if prob is not None:
            grouped_p[owner].append(prob)
    chunks = [np.stack(item).astype(np.float32) for item in grouped_h]
    means = np.stack([item.mean(axis=0) for item in chunks]).astype(np.float32)
    max_probs = np.stack([np.stack(item).max(axis=0) for item in grouped_p]).astype(np.float32) if grouped_p[0] else None
    return means, chunks, max_probs


def train_from_features(name_prefix, means, chunks, max_probs, train, val, test, device, reports):
    groups = script_groups(train)
    train_y = np.array([train[idxs[0]]["label"] for idxs in groups])
    train_mean = np.stack([means["train"][idxs].mean(axis=0) for idxs in groups]).astype(np.float32)
    train_chunks = [chunks["train"][idxs[0]] for idxs in groups]
    reports.append(fit_vector(f"{name_prefix}_mean", train_mean, train_y, means["val"], val, means["test"], test, device))
    reports.append(fit_attn(f"{name_prefix}_attn", train_chunks, train_y, chunks["val"], val, chunks["test"], test, device))
    if max_probs is not None:
        train_max = np.stack([max_probs["train"][idxs].mean(axis=0) for idxs in groups]).astype(np.float32)
        reports.append(
            fit_vector(f"{name_prefix}_maxprob", train_max, train_y, max_probs["val"], val, max_probs["test"], test, device)
        )


def ensemble(name, parts, val_rows, test_rows):
    report = {"scheme": name, "parts": [part["scheme"] for part in parts]}
    for split, rows in (("val", val_rows), ("test", test_rows)):
        logits = np.mean([part["logits"][split] for part in parts], axis=0)
        pred = logits.argmax(axis=-1)
        report[split] = metrics(pred, np.array([row["label"] for row in rows]))
        report[f"{split}_script"] = script_metrics(rows, pred)
        report.setdefault("logits", {})[split] = logits
    logger.info(
        f"[{name}] val_acc {report['val']['acc']:.3f} val_ua {report['val']['ua']:.3f} "
        f"test_ua {report['test']['ua']:.3f}"
    )
    return report


def slim(report):
    return {
        key: value
        for key, value in report.items()
        if key not in {"history", "logits"}
    }


def main() -> int:
    data = default_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(data / "datasets/BEAT/beat_cache/cog_4english_15_141_full"))
    parser.add_argument("--hub", default=str(data / "datasets/BEAT/hub"))
    parser.add_argument("--out", default=str(data / "outputs/emotion"))
    args = parser.parse_args()
    configure(Path(args.cache), Path(args.hub), Path(args.out))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logger.add(OUT / "logs" / "train.log")
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    lengths = [len(row["words"]) for row in train]
    logger.info(
        f"clips train {len(train)} val {len(val)} test {len(test)} "
        f"train words min {min(lengths)} median {sorted(lengths)[len(lengths)//2]} max {max(lengths)}"
    )
    device = torch.device("cuda:0")
    reports = []
    specs = [
        ("go", HUB / "roberta-base-go_emotions", AutoModelForSequenceClassification, 512, 256, True),
        ("mp", HUB / "all-mpnet-base-v2", AutoModel, 384, 192, False),
    ]
    stored = {}
    for prefix, path, loader, length, stride, want_logits in specs:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = loader.from_pretrained(str(path)).to(device).eval()
        logger.info(f"encode {prefix} documents")
        doc = {}
        sent = {}
        for split, rows in (("train", train), ("val", val), ("test", test)):
            doc[split] = encode_documents(model, tokenizer, rows, device, length, stride, want_logits)
            logger.info(f"encode {prefix} sentences {split}")
            sent[split] = encode_sentences(model, tokenizer, rows, device, length, want_logits)
        del model
        torch.cuda.empty_cache()
        means = {split: doc[split][0] for split in doc}
        chunks = {split: doc[split][1] for split in doc}
        max_probs = {split: doc[split][2] for split in doc} if want_logits else None
        train_from_features(f"{prefix}_chunk", means, chunks, max_probs, train, val, test, device, reports)
        s_means = {split: sent[split][0] for split in sent}
        s_chunks = {split: sent[split][1] for split in sent}
        s_probs = {split: sent[split][2] for split in sent} if want_logits else None
        train_from_features(f"{prefix}_sent", s_means, s_chunks, s_probs, train, val, test, device, reports)
        if prefix == "go":
            reports.append(fit_spread("go_sent_spread", s_chunks["train"], train, s_chunks["val"], val, s_chunks["test"], test, device))
        stored[prefix] = {item["scheme"]: item for item in reports if item["scheme"].startswith(prefix)}
        (OUT / "summary.json").write_text(json.dumps([slim(item) for item in reports], indent=2))

    by_name = {item["scheme"]: item for item in reports}
    for name, parts in (
        ("ens_go_sent_spread_mp_chunk_mean", ["go_sent_spread", "mp_chunk_mean"]),
        ("ens_go_sent_attn_mp_chunk_mean", ["go_sent_attn", "mp_chunk_mean"]),
        ("ens_go_chunk_mean_mp_chunk_mean", ["go_chunk_mean", "mp_chunk_mean"]),
        ("ens_go_sent_attn_mp_sent_attn", ["go_sent_attn", "mp_sent_attn"]),
    ):
        if all(part in by_name for part in parts):
            reports.append(ensemble(name, [by_name[part] for part in parts], val, test))
    by_name = {item["scheme"]: item for item in reports}
    for source in ("go_sent_spread", "ens_go_sent_spread_mp_chunk_mean", "ens_go_sent_attn_mp_chunk_mean"):
        if source in by_name:
            reports.append(speaker_pool(f"{source}_spk", by_name[source], val, test))
    ranked = sorted(reports, key=lambda item: item["val"]["ua"], reverse=True)
    logger.info("best " + ", ".join(f"{item['scheme']} val {item['val']['ua']:.3f} test {item['test']['ua']:.3f}" for item in ranked))
    (OUT / "summary.json").write_text(json.dumps([slim(item) for item in reports], indent=2))
    chosen = ranked[0]["scheme"]
    (OUT / "selected.json").write_text(json.dumps({"selected_by": "val_ua", "scheme": chosen, **slim(ranked[0])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
