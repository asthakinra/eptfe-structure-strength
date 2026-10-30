"""Evaluate heads on cached tile embeddings under shuffled 5-fold and LOLO CV, plus a
lot probe (how well each representation identifies the lot; compare with chance).

    python scripts/image_eval.py --packet PACKET_DIR --emb results/emb_resnet50.npz [--out results/image.csv]

For VGG caches, --blocks block4 block5 restricts the evaluation to those layers.
"""
import argparse, warnings
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eptfe.cv import SCHEMES, out_of_fold
from eptfe.data import load_packet, packet_dir
from eptfe.image import mil_fit_predict
from eptfe.metrics import regression_metrics

warnings.filterwarnings('ignore'); torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument('--packet'); ap.add_argument('--emb', required=True)
ap.add_argument('--out'); ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--blocks', nargs='+', help='VGG blocks to use, e.g. block3 block5 (default: all)')
a = ap.parse_args()
pk = load_packet(packet_dir(a.packet)); H, y, lots = pk.X, pk.y, pk.lots
z = np.load(a.emb)
assert list(z['sample_id']) == pk.samples.sample_id.tolist(), 'embedding cache is for a different sample order'
Ti, To = z['inner'].astype(np.float32), z['outer'].astype(np.float32)
blocks = {s.split(':')[0]: tuple(map(int, s.split(':')[1:])) for s in z['blocks']} if 'blocks' in z.files else {}
if a.blocks:
    missing = set(a.blocks) - set(blocks)
    if missing: raise SystemExit(f'--blocks {sorted(missing)} not in this cache (available: {sorted(blocks) or "none"})')
    cols = np.concatenate([np.arange(*blocks[b]) for b in a.blocks])
    Ti, To = Ti[:, :, cols], To[:, :, cols]
P = np.concatenate([Ti.mean(1), To.mean(1)], 1)

def pre(pca=None, n_fit=None):
    """Impute + scale, then PCA capped by the fold size so small folds still work."""
    k = min(pca, n_fit - 1) if pca and n_fit else pca
    return [SimpleImputer(strategy='median'), StandardScaler()] + ([PCA(k, random_state=0)] if k else [])

def ridge(X, pca=None):
    def f(tr, te):
        m = make_pipeline(*pre(pca, len(tr)), RidgeCV(alphas=np.logspace(-1, 5, 30))).fit(X[tr], np.log(y[tr]))
        return np.exp(m.predict(X[te]))
    return f

HP = np.c_[P, H]
models = {
    'mean': lambda s: lambda tr, te: np.full(len(te), y[tr].mean()),
    'ridge_handcrafted': lambda s: ridge(H),
    'ridge_embed_pca20': lambda s: ridge(P, 20),
    'ridge_embed_full': lambda s: ridge(P),
    'ridge_embed+handcrafted': lambda s: ridge(HP, 30),
    'attention_mil': lambda s: lambda tr, te: np.exp(np.mean(
        [np.log(mil_fit_predict(Ti, To, y, lots, tr, te, s, k)) for k in range(a.seeds)], 0)),
}
rows = []
for scheme in SCHEMES:
    for name, mk in models.items():
        p = out_of_fold(mk(scheme), P, y, lots, scheme)
        rows.append(dict(cv=scheme, model=name, **regression_metrics(y, p, lots))); print(rows[-1], flush=True)
res = pd.DataFrame(rows).round(3)

probe = lambda X, pca: cross_val_score(make_pipeline(*pre(pca, int(len(X) * 0.8)), LogisticRegression(C=0.1, max_iter=5000)), X, lots,
                                       cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
print(f'\nlot probe accuracy: handcrafted {probe(H, None):.3f} | embeddings {probe(P, 20):.3f} | '
      f'majority class {pd.Series(lots).value_counts(normalize=True).max():.3f}')
print(f'backbone={z["backbone"]} blocks={a.blocks or sorted(blocks) or "-"} tile={z["tile"]} crop_bottom={z["crop_bottom"]} tiles/image={Ti.shape[1]} dim={Ti.shape[2]}')
print(res.to_string(index=False))
if a.out: res.to_csv(a.out, index=False)
