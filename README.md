# eptfe-structure-strength

Can SEM images of ePTFE microstructure predict **suture retention strength**?

Each sample has one inner-surface and one outer-surface SEM image. An upstream
morphology pipeline turns each pair into 48 features (node spacing, fibril
tortuosity and length, opening geometry, foreground fraction, node-axis coherence;
mean / SD / higher moments, inner and outer kept separate). This repo evaluates
classical models, tabular neural networks and image-based neural networks that
predict strength from those features or from the raw images.

> **No data in this repo.** The development packet, SEM images, embeddings and all
> results are confidential and stay local. `.gitignore` blocks `*.csv`, `*.tif`,
> `*.npz`, `data/` and `results/`. Don't override it.

## Evaluation principle
Samples from the same production lot share both microstructure and strength, so a
random sample split lets a model succeed just by recognising the lot. Every script
reports two schemes side by side:

- `shuffled_5fold`: random sample folds (optimistic; same lots on both sides)
- `leave_one_lot_out`: train on all lots but one, test on the held-out lot

Imputation, scaling, monotonic sign learning, feature selection and early stopping
are all fit inside the training fold. Metrics: R², MAE, median absolute error,
Spearman, **within-lot** Spearman and **lot-level** Spearman.
See [docs/research_framework.md](docs/research_framework.md).

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export EPTFE_PACKET_DIR=/path/to/sem-suture-development-packet   # or pass --packet
```

## Run
| Step | Command |
|---|---|
| Exploratory analysis | `python scripts/run_eda.py --out results/eda` |
| Classical models | `python scripts/run_classical.py --out results/classical.csv` |
| Tabular neural nets | `python scripts/run_tabular_nn.py --out results/tabular_nn.csv` |
| Image embeddings | `python scripts/image_embed.py --images /path/to/tiffs --backbone resnet50 --crop-bottom 64 --preview results/preview --out results/emb_resnet50.npz` |
| Image models | `python scripts/image_eval.py --emb results/emb_resnet50.npz --out results/image.csv` |

Before the image step, look at the `--preview` PNGs. Set `--crop-bottom` so the SEM
data bar is removed, and check that no burned-in labels sit inside the tile grid.
Images are verified against their recorded SHA-256 and the run stops on any mismatch.
Pretrained weights (ResNet-50 or DINOv2) download on first use.

Runtime on a laptop: the tabular scripts take a few minutes on CPU. Image embedding
uses Apple MPS or CUDA when available.

## Layout
```
src/eptfe/
  data.py               load + validate the packet, lot eta² helper
  cv.py                 shuffled / leave-one-lot-out splits, inner early-stopping split
  metrics.py            sample-, within-lot- and lot-level metrics
  models_classical.py   ridge, lasso, PLS, RF, monotonic RF / gradient boosting, sklearn MLP
  models_nn.py          PyTorch MLP, inner/outer two-tower, monotone net, lot-adversarial MLP
  image.py              hash check, crop, tiling, frozen backbones, attention-MIL head
scripts/                runnable entry points (above)
tests/                  synthetic-data tests; never touch real data
docs/                   research framework (methods and plan)
```

## Tests
```bash
pytest -q
```
The tests build a small synthetic packet (including fake TIFFs) in a temp folder. They
are safe to run in CI. Never run the real-data scripts in CI.
