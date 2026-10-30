# eptfe-structure-strength

**Can SEM images of ePTFE microstructure predict suture retention strength, on production lots the model has never seen?**

Suture retention strength is measured with a destructive pull test. Each sample in the
development cohort also has two SEM images, one of the inner surface and one of the outer.
This repository contains the code that tests whether those images, or features measured
from them, can predict strength. It evaluates every model on held-out production lots,
because that is how a model would be used.

> **No data lives here.** The development packet, SEM images, embeddings and results are
> confidential and stay on approved machines. `.gitignore` blocks `*.csv`, `*.tif`, `*.npz`,
> `*.pdf`, `data/` and `results/`. Keep it that way, and stage files by name, not with `git add .`.

## Contents
- [Why lot-aware evaluation](#why-lot-aware-evaluation)
- [What's implemented](#whats-implemented)
- [Setup](#setup)
- [Running the analyses](#running-the-analyses)
- [Image models, including VGG](#image-models-including-vgg)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Roadmap](#roadmap)

## Why lot-aware evaluation
Samples from the same production lot share raw material and process conditions, so they
look alike under the microscope and test alike. If training and test sets share lots, a
model can score well just by recognising the lot. Every script therefore reports two
schemes side by side:

| Scheme | Split | Answers |
|---|---|---|
| `shuffled_5fold` | Random sample folds; same lots on both sides | How well does it fill in samples from lots it already knows? |
| `leave_one_lot_out` | Train on all lots but one, predict the held-out lot | **How well does it predict a lot it has never seen?** |

Rules applied in every fold:
- Imputation, scaling, PCA, monotonic sign learning, feature selection and early stopping are fit on the **training fold only**.
- Inputs are imaging only. Lot IDs, sample IDs and mechanical outcomes never enter a model.
- Results are reported against a **training-mean baseline**. Beating zero R² is not enough under leave-one-lot-out.

Metrics: R², MAE, median absolute error, Spearman, **within-lot** Spearman (does it rank
samples inside a lot?), **lot-level** Spearman (does it rank lots?), and a **lot-identification
probe** (how much lot identity a representation encodes).

## What's implemented
| Track | Models / representations |
|---|---|
| Tabular, on the 48 supplied features | Ridge, lasso, PLS, random forest, monotonic random forest, monotonic gradient boosting, scikit-learn MLP |
| Tabular neural networks (PyTorch) | MLP, inner/outer two-tower, monotone network, lot-adversarial MLP; seed ensembles |
| Images | Frozen ResNet-50, DINOv2 ViT-S/14, **VGG16 / VGG19 texture statistics**; heads: ridge on pooled embeddings, ridge on embeddings + 48 features, attention-based multiple-instance learning over tiles |

## Setup
Python 3.10 or later.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export EPTFE_PACKET_DIR=/path/to/sem-suture-development-packet   # or pass --packet to each script
```

## Running the analyses
Write all outputs under `results/`, which git ignores.

| Step | Command |
|---|---|
| Exploratory analysis | `python scripts/run_eda.py --out results/eda` |
| Classical models | `python scripts/run_classical.py --out results/classical.csv` |
| Tabular neural networks | `python scripts/run_tabular_nn.py --out results/tabular_nn.csv` |
| Image embeddings | `python scripts/image_embed.py --images /path/to/tiffs --backbone vgg16 --crop-bottom 64 --preview results/preview --out results/emb_vgg16.npz` |
| Image models | `python scripts/image_eval.py --emb results/emb_vgg16.npz --out results/image_vgg16.csv` |

The tabular scripts take a few minutes on a laptop CPU. Image embedding uses Apple MPS or
CUDA when available.

## Image models, including VGG
`image_embed.py` reads the **original** TIFFs. It looks for each one at
`<images>/<relative_path>` or `<images>/<sha256>.tif`, then:

1. **Verifies** each file against the SHA-256 recorded in the packet and stops on any mismatch.
2. **Crops** the SEM data bar (`--crop-bottom`) and normalises contrast per image.
3. **Tiles** each image into a centred grid (`--tile`, default 224 px ≈ 44 µm).
4. **Embeds** every tile with a frozen backbone and caches the result to `.npz`.

| `--backbone` | Per-tile output | Weights download |
|---|---|---|
| `resnet50` | 2,048-d pooled features | ~100 MB |
| `dinov2_vits14` | 384-d class token | ~85 MB |
| `vgg16` | 2,560-d texture statistics | ~530 MB |
| `vgg19` | 2,560-d texture statistics | ~550 MB |
| `random`, `vgg_random` | untrained ResNet-18 / VGG-11, **dry runs only** | none |

**VGG texture statistics.** For VGG the pipeline does not use the classifier output. It
taps the last ReLU of convolutional blocks 3, 4 and 5 and records each channel's spatial
mean and standard deviation. This is a compact form of the Gram-matrix texture
representation, and pretrained VGG features are an established descriptor for
microstructure images. The cache records which columns come from which block, so depth can
be compared without re-embedding:

```bash
python scripts/image_eval.py --emb results/emb_vgg16.npz --blocks block3          # fine texture
python scripts/image_eval.py --emb results/emb_vgg16.npz --blocks block4 block5   # coarser structure
```

Backbones stay frozen. With a few hundred samples from a handful of lots, fine-tuning a
network of this size would memorise lots rather than learn structure.

**Before the first real run:**
- Open the `--preview` PNGs. Set `--crop-bottom` so the data bar is gone, and check that no burned-in labels or calipers fall inside the tile grid.
- Compare `image_eval.py`'s lot-probe accuracy with the 48-feature baseline. A gain that comes with a higher probe score is lot recognition, not a strength signal.

## Repository layout
```
src/eptfe/
  data.py               load and validate the packet; variance-explained-by-lot helper
  cv.py                 shuffled / leave-one-lot-out splits; lot-grouped inner split
  metrics.py            sample-, within-lot- and lot-level metrics
  models_classical.py   linear, PLS, tree and monotonic models
  models_nn.py          PyTorch tabular networks
  image.py              hash check, crop, tiling, backbones (incl. VGG texture), attention-MIL
scripts/                runnable entry points (see tables above)
tests/                  synthetic-data tests
docs/                   research framework: questions, protocol and plan
```

## Tests
```bash
pytest -q
```
The tests build a small synthetic packet in a temporary folder, with fake features and fake
TIFFs. They never touch real data and need no weight downloads. They cover:
- packet validation
- lot separation under cross-validation
- metrics
- every model, and that the monotone network really is monotone
- the image pipeline end to end, including rejection of files with the wrong hash
- the VGG texture statistics and block selection

They are safe for CI. Never run the real-data scripts in CI.

## Roadmap
1. Run the image backbones (VGG16/19, ResNet-50, DINOv2) on the original TIFFs under leave-one-lot-out, with the lot probe.
2. Classical image-processing features: GLCM, LBP, Gabor, FFT spacing, structure-tensor orientation, and node / fibril / pore geometry from segmentation.
3. A non-image baseline (wall thickness, geometry, test direction) once those covariates are available.
4. Mixed-effects models to separate within-lot from between-lot effects.
5. Conformal prediction intervals evaluated on held-out lots.

See [docs/research_framework.md](docs/research_framework.md) for the full plan.
