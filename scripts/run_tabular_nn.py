"""PyTorch networks on the 48 tabular features vs mean and ridge baselines.

    python scripts/run_tabular_nn.py --packet PACKET_DIR [--seeds 5] [--out results/tabular_nn.csv]
"""
import argparse, warnings
import pandas as pd, torch
from eptfe.cv import SCHEMES, out_of_fold
from eptfe.data import load_packet, packet_dir
from eptfe.metrics import regression_metrics
from eptfe import models_classical as mc, models_nn as mn

warnings.filterwarnings('ignore'); torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument('--packet'); ap.add_argument('--out')
ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--models', nargs='+', default=['mean', 'ridge_log', *mn.KINDS])
a = ap.parse_args()
pk = load_packet(packet_dir(a.packet)); X, y, lots = pk.X, pk.y, pk.lots
views = dict(inner_idx=pk.view_index('inner'), outer_idx=pk.view_index('outer'))

def runner(name, scheme):
    if name == 'mean': return lambda tr, te: mc.fit_predict('mean', X[tr], y[tr], X[te])
    if name == 'ridge_log': return lambda tr, te: mc.fit_predict('ridge', X[tr], y[tr], X[te], log_target=True)
    return lambda tr, te: mn.ensemble_predict(name, X[tr], y[tr], lots[tr], X[te], scheme, seeds=a.seeds, **views)

rows = []
for scheme in SCHEMES:
    for name in a.models:
        p = out_of_fold(runner(name, scheme), X, y, lots, scheme)
        rows.append(dict(cv=scheme, model=name, **regression_metrics(y, p, lots))); print(rows[-1], flush=True)
res = pd.DataFrame(rows).round(3); print(res.to_string(index=False))
if a.out: res.to_csv(a.out, index=False)
