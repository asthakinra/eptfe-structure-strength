"""Exploratory data analysis: target and lot structure, how much each feature
encodes lot, pooled vs within-lot association, redundancy, lot-level permutation test.

    python scripts/run_eda.py --packet PACKET_DIR [--out results/eda]
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from scipy import stats
from eptfe.data import load_packet, lot_eta2, packet_dir

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('--packet'); ap.add_argument('--out'); ap.add_argument('--perms', type=int, default=500)
a = ap.parse_args()
pk = load_packet(packet_dir(a.packet)); s, F, y, lots = pk.samples, pk.features, pk.samples.target_gf, pk.lots

print('target:', y.describe().round(1).to_dict(), '| skew', round(stats.skew(y), 2))
lot_tab = s.groupby('lot_id').target_gf.agg(['count', 'mean', 'std', 'min', 'max']).round(0).sort_values('mean')
print(lot_tab.to_string())
print('target variance explained by lot (eta^2):', round(lot_eta2(y, lots), 3))

rows = []
for f in F:
    d = s[[f, 'target_gf', 'lot_id']].dropna()
    dm_x = d[f] - d.groupby('lot_id')[f].transform('mean')
    dm_y = d.target_gf - d.groupby('lot_id').target_gf.transform('mean')
    rows.append(dict(feature=f, rho_pooled=stats.spearmanr(d[f], d.target_gf)[0],
                     rho_within_lot=stats.spearmanr(dm_x, dm_y)[0], eta2_lot=lot_eta2(s[f], lots),
                     missing=s[f].isna().mean()))
feat = pd.DataFrame(rows).set_index('feature')
print('\nfeatures by |pooled rho|:'); print(feat.reindex(feat.rho_pooled.abs().sort_values(ascending=False).index).head(12).round(2).to_string())
print('median feature eta^2 by lot:', round(feat.eta2_lot.median(), 3))

c = s[F].corr(method='spearman').abs().values.copy(); np.fill_diagonal(c, 0)
print('feature pairs with |rho| > 0.8:', int((c > 0.8).sum() / 2))
io = [stats.spearmanr(s[f], s['outer__' + f[7:]], nan_policy='omit')[0] for f in F if f.startswith('inner__')]
print('median inner-vs-outer rho of the same feature:', round(np.median(io), 2))

# lot level (n = number of lots): best feature vs permutation null of the max |rho|
L = s.groupby('lot_id')[F + ['target_gf']].median()
lot_rho = pd.Series({f: stats.spearmanr(L[f], L.target_gf)[0] for f in F})
rng = np.random.default_rng(0)
null = [max(abs(stats.spearmanr(L[f], yp)[0]) for f in F) for yp in (rng.permutation(L.target_gf.values) for _ in range(a.perms))]
print(f'\nlot-level (n={len(L)}): best |rho| {lot_rho.abs().max():.2f} ({lot_rho.abs().idxmax()}), '
      f'permutation 95th pct of max |rho| {np.quantile(null, 0.95):.2f}')

im = pk.images
print('\nacquisition settings (unique values):', {c: sorted(im[c].dropna().unique().tolist()) for c in ('pixel_size_nm', 'magnification_x', 'width_px', 'height_px') if c in im})
print('review flags:', im.review_flags_json.fillna('[]').value_counts().to_dict())
print('linkage grade:', im.linkage_grade.value_counts().to_dict())

if a.out:
    os.makedirs(a.out, exist_ok=True)
    feat.assign(rho_lot_level=lot_rho).to_csv(f'{a.out}/feature_associations.csv'); lot_tab.to_csv(f'{a.out}/lot_summary.csv')
    print('wrote', a.out)
