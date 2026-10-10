# 🎙️ CoG

> *This repo includes an improved version of [Chain of Generation: Multi-Modal Gesture Synthesis via Cascaded Conditional Control](https://arxiv.org/abs/2312.15900). CoG generates face, body, and hand motion from speech.*



## 🛠️ Installation

```shell
git clone https://github.com/kkakkkka/CoG.git -b release
cd CoG

conda create -n cog python=3.9
conda activate cog

pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers huggingface_hub
```

Install PyTorch after `requirements.txt`.

## 📥 Weights

```shell
pip install "huggingface_hub[cli]"
huggingface-cli download thuteam/CoG --local-dir ckpts
```

```text
./ckpts/
├── cog.bin
├── ae_beat141.pt
├── emotion.pt
└── speaker.pt
```

`emotion.pt` has two keys, `sentence_attention` and `chunk_mean`. Also place these encoders under `$BEAT_ROOT/datasets/BEAT/hub/`: `roberta-base-go_emotions`, `all-mpnet-base-v2`, and `wav2vec2-base`.

## 📂 Data

Download BEAT English from [H-Liu1997/BEAT](https://huggingface.co/datasets/H-Liu1997/BEAT), then build the cache. Set `root_path` in `configs/cog.yaml` to `$BEAT_ROOT`, and link `ckpts/ae_beat141.pt` to `$BEAT_ROOT/weights/ae_beat141.pt`.

```shell
export BEAT_ROOT=/path/to/data
python dataloaders/preprocess_beat.py \
  --raw-root "$BEAT_ROOT/datasets/BEAT/raw/beat_english_v0.2.1/beat_english_v0.2.1" \
  --cache-root "$BEAT_ROOT/datasets/BEAT/beat_cache/cog_4english_15_141_full" \
  --speakers 2,4,6,8 \
  --clean

python scripts/build_style_cache.py \
  --cache "$BEAT_ROOT/datasets/BEAT/beat_cache/cog_4english_15_141_full" \
  --hub "$BEAT_ROOT/datasets/BEAT/hub" \
  --emotion ckpts/emotion.pt
```

## 🏋️ Training

```shell
python train.py --config configs/cog.yaml
```

## 📊 Inference

```shell
python scripts/eval_cog.py \
  --checkpoint ckpts/cog.bin \
  --config configs/cog.yaml \
  --cache-root "$BEAT_ROOT/datasets/BEAT/beat_cache/cog_4english_15_141_full" \
  --frozen-style-dir "$BEAT_ROOT/datasets/BEAT/beat_cache/cog_4english_15_141_full/frozen_style" \
  --mse-ae ckpts/ae_beat141.pt \
  --full-clip \
  --out-dir "$BEAT_ROOT/outputs/eval"
```

The report is `report.json`.

## 🌍 Your Own Speech

### 📝 TextGrid

CoG reads the words tier of a TextGrid. Use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) to make one from a recording. Put each wav and a same-name transcript in `./data` first:

```text
./data/
├── clip.wav
└── clip.txt
```

```shell
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
conda install -c conda-forge kalpy
pip install pgvector Bio

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

mfa align ./data english_us_arpa english_us_arpa ./data/textgrid
```

The aligned file is `./data/textgrid/clip.TextGrid`. Pass that path to `--textgrid`.

### 🚀 Run

```shell
python scripts/infer.py \
  --audio ./data/clip.wav \
  --textgrid ./data/textgrid/clip.TextGrid \
  --checkpoint ckpts/cog.bin \
  --cache-root "$BEAT_ROOT/datasets/BEAT/beat_cache/cog_4english_15_141_full" \
  --hub "$BEAT_ROOT/datasets/BEAT/hub" \
  --emotion ckpts/emotion.pt \
  --speaker-checkpoint ckpts/speaker.pt \
  --wav2vec "$BEAT_ROOT/datasets/BEAT/hub/wav2vec2-base" \
  --out-dir "$BEAT_ROOT/outputs/infer"
```

Outputs are `*_pose.npy`, `*_face.npy`, and a json file.

## 📖 Citation

If you find CoG useful for your research, please consider citing:

```bibtex
@inproceedings{xu2024chain,
  title={Chain of generation: Multi-modal gesture synthesis via cascaded conditional control},
  author={Xu, Zunnan and Zhang, Yachao and Yang, Sicheng and Li, Ronghui and Li, Xiu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6387--6395},
  year={2024}
}

```

## 📄 License

[CC BY-NC 4.0](LICENSE.md)
