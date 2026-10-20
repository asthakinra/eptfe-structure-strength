"""Classical models under shuffled 5-fold and leave-one-lot-out CV, raw and log target.

    python scripts/run_classical.py --packet PACKET_DIR [--out results/classical.csv]
"""
import argparse, warnings
import pandas as pd
from eptfe.cv import SCHEMES, out_of_fold
from eptfe.data import load_packet, packet_dir
from eptfe.metrics import regression_metrics
from eptfe.models_classical import MODELS, fit_predict

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('--packet'); ap.add_argument('--out')
ap.add_argument('--models', nargs='+', default=['mean'] + list(MODELS))
a = ap.parse_args()
pk = load_packet(packet_dir(a.packet)); X, y, lots = pk.X, pk.y, pk.lots

rows = []
for scheme in SCHEMES:
    for name in a.models:
        for log in ([False] if name == 'mean' else [False, True]):
            p = out_of_fold(lambda tr, te: fit_predict(name, X[tr], y[tr], X[te], log), X, y, lots, scheme)
            rows.append(dict(cv=scheme, model=name, target='log' if log else 'raw', **regression_metrics(y, p, lots)))
            print(rows[-1], flush=True)
res = pd.DataFrame(rows).round(3); print(res.to_string(index=False))
if a.out: res.to_csv(a.out, index=False)
