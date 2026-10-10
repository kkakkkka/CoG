#!/usr/bin/env python3
"""Generate a gesture from one waveform and its TextGrid.

The first four frames are the training-mean pose. Later windows continue from
the motion already generated. Emotion probabilities come from the text. Speaker
probabilities come from the utterance classifier on the waveform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import beat_eval as base  # noqa: E402
import eval_cog  # noqa: E402
import train_emotion as words  # noqa: E402
import train_speaker  # noqa: E402


def emotion_from_words(textgrid_path: Path, hub: Path, emotion_path: Path, device: torch.device) -> np.ndarray:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    word_list = words.words_of(textgrid_path)
    row = {"words": word_list, "text": " ".join(word_list)}
    words.configure(words.CACHE, hub, emotion_path.parent)
    go_tok = AutoTokenizer.from_pretrained(str(hub / "roberta-base-go_emotions"))
    go = AutoModelForSequenceClassification.from_pretrained(str(hub / "roberta-base-go_emotions")).to(device).eval()
    mp_tok = AutoTokenizer.from_pretrained(str(hub / "all-mpnet-base-v2"))
    mp = AutoModel.from_pretrained(str(hub / "all-mpnet-base-v2")).to(device).eval()
    sentence_state, chunk_state = words.load_emotion_heads(emotion_path)
    go_head = words.AttnHead(768)
    go_head.load_state_dict(sentence_state)
    go_head.to(device).eval()
    mp_head = words.VectorHead(768)
    mp_head.load_state_dict(chunk_state)
    mp_head.to(device).eval()
    _, chunks, _ = words.encode_sentences(go, go_tok, [row], device, 512, True)
    means, _, _ = words.encode_documents(mp, mp_tok, [row], device, 384, 192, False)
    data, mask = words.batch_chunks(chunks, device)
    with torch.no_grad():
        logits = (go_head(data, mask) + mp_head(torch.from_numpy(means).to(device))) / 2
        prob = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)
    return prob


def rollout(
    model: torch.nn.Module,
    wav: np.ndarray,
    tokens: np.ndarray,
    frames: int,
    emotion_prob: np.ndarray,
    speaker_prob: np.ndarray,
    speaker_id: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros((frames, base.POSE_DIMS), dtype=np.float32)
    face_pred = np.zeros((frames, 51), dtype=np.float32)
    for start in base.model_windows(frames):
        end = min(start + base.WINDOW, frames)
        window = base.WINDOW
        pre_seq = np.zeros((1, window, base.POSE_DIMS + 1), dtype=np.float32)
        if start == 0:
            write_from = 0
        else:
            pre_seq[0, : base.SEED_FRAMES, : base.POSE_DIMS] = pred[start : start + base.SEED_FRAMES]
            pre_seq[0, : base.SEED_FRAMES, base.POSE_DIMS] = 1.0
            write_from = base.SEED_FRAMES
        if start == 0:
            pre_seq[0, : base.SEED_FRAMES, base.POSE_DIMS] = 1.0
        sample0 = int(round(start / base.FPS * base.AUDIO_SR))
        need = int(round(window / base.FPS * base.AUDIO_SR))
        audio = wav[sample0 : sample0 + need]
        if audio.shape[0] < need:
            audio = np.pad(audio, (0, need - audio.shape[0]))
        text = np.zeros((window,), dtype=np.int64)
        text[: end - start] = tokens[start:end]
        emo = torch.zeros((1, window), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                torch.from_numpy(pre_seq).to(device),
                in_audio=torch.from_numpy(audio[None, :].astype(np.float32)).to(device),
                in_facial=torch.zeros((1, window, 51), dtype=torch.float32, device=device),
                in_text=torch.from_numpy(text[None, :]).to(device),
                in_id=torch.tensor([[speaker_id - 1]], dtype=torch.long, device=device),
                in_emo=emo,
                is_test=True,
                return_dict=True,
                emotion_prob=torch.from_numpy(emotion_prob[None, :]).to(device),
                speaker_prob=torch.from_numpy(speaker_prob[None, :]).to(device),
            )
        pose = out["pose"].detach().cpu().numpy().reshape(window, base.POSE_DIMS).astype(np.float32)
        face = out["face"].detach().cpu().numpy().reshape(window, -1).astype(np.float32)
        if start == 0:
            pred[base.SEED_FRAMES:end] = pose[base.SEED_FRAMES : end - start]
            face_pred[:end] = face[: end - start]
        else:
            pred[start + write_from : end] = pose[write_from : end - start]
            face_pred[start + write_from : end] = face[write_from : end - start]
    return pred, face_pred


def main() -> int:
    data_root = eval_cog.DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--textgrid", required=True)
    parser.add_argument("--checkpoint", default=str(REPO / "ckpts" / "cog.bin"))
    parser.add_argument("--config", default=str(eval_cog.FULL_CONFIG))
    parser.add_argument("--cache-root", default=str(eval_cog.FULL_CACHE))
    parser.add_argument("--hub", default=str(data_root / "datasets" / "BEAT" / "hub"))
    parser.add_argument("--wav2vec", default=str(data_root / "datasets" / "BEAT" / "hub" / "wav2vec2-base"))
    parser.add_argument("--emotion", default=str(REPO / "ckpts" / "emotion.pt"))
    parser.add_argument("--speaker-checkpoint", default=str(REPO / "ckpts" / "speaker.pt"))
    parser.add_argument("--speaker", type=int, default=0, help="Optional BEAT speaker id 1-30. 0 uses the utterance classifier.")
    parser.add_argument("--out-dir", default=str(data_root / "outputs" / "infer"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    audio_path = Path(args.audio)
    textgrid_path = Path(args.textgrid)
    wav, wav_stats = base.read_wav_16k(audio_path)
    frames = int(round(len(wav) / base.AUDIO_SR * base.FPS))
    frames = max(frames, base.WINDOW)
    cache_root = Path(args.cache_root)
    vocab = eval_cog.load_vocab(cache_root)
    tokens, token_report = eval_cog.text_tokens(textgrid_path, frames, vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speaker_distribution = None
    if args.speaker:
        speaker = int(args.speaker)
        speaker_prob = np.zeros(30, dtype=np.float32)
        speaker_prob[speaker - 1] = 1.0
        speaker_source = "given"
    else:
        speaker_prob, speaker_distribution = train_speaker.predict_speaker(
            wav, Path(args.wav2vec), Path(args.speaker_checkpoint), device
        )
        speaker = int(speaker_prob.argmax()) + 1
        speaker_source = "classifier"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    emotion = emotion_from_words(textgrid_path, Path(args.hub), Path(args.emotion), device)
    eval_args = eval_cog.load_eval_args(Path(args.config), cache_root, int(getattr(vocab, "n_words")))
    model, load_report = eval_cog.strict_load_cog(eval_args, Path(args.checkpoint))
    if model is None:
        raise SystemExit(f"checkpoint did not load: {load_report}")
    model = model.to(device).eval()
    pred, face = rollout(model, wav, tokens, frames, emotion, speaker_prob, speaker, device)
    mean = np.load(cache_root / "train" / "bvh_rot" / "bvh_mean.npy").astype(np.float32)
    std = np.load(cache_root / "train" / "bvh_rot" / "bvh_std.npy").astype(np.float32)
    pose = pred * std + mean
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = textgrid_path.stem
    np.save(out_dir / f"{stem}_pose.npy", pose.astype(np.float32))
    np.save(out_dir / f"{stem}_face.npy", face.astype(np.float32))
    report = {
        "audio": str(audio_path),
        "textgrid": str(textgrid_path),
        "checkpoint": str(args.checkpoint),
        "frames": int(frames),
        "speaker": int(speaker),
        "speaker_source": speaker_source,
        "speaker_prob": [float(x) for x in (speaker_distribution if speaker_distribution is not None else speaker_prob)],
        "emotion_names": words.NAMES,
        "emotion_prob": [float(x) for x in emotion],
        "emotion": words.NAMES[int(emotion.argmax())],
        "text": token_report,
        "audio_stats": wav_stats,
        "seed": "first 4 frames are the training-mean pose",
        "pose": str(out_dir / f"{stem}_pose.npy"),
        "face": str(out_dir / f"{stem}_face.npy"),
        "load": {key: load_report[key] for key in ("strict_load_ok", "loaded") if key in load_report},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pose": report["pose"], "face": report["face"], "emotion": report["emotion"], "speaker": speaker, "frames": frames}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
