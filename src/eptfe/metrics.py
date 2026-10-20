import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score


def regression_metrics(y, p, lots, min_lot=8):
    """Sample-level metrics plus within-lot rank agreement (mean Spearman over lots
    with >= min_lot samples) and lot-level agreement (Spearman of lot medians)."""
    df = pd.DataFrame({'y': y, 'p': p, 'lot': lots})
    within = [stats.spearmanr(g.y, g.p)[0] for _, g in df.groupby('lot')
              if len(g) >= min_lot and g.p.nunique() > 1]
    lm = df.groupby('lot')[['y', 'p']].median()
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), MedAE=float(np.median(abs(y - p))),
                spearman=stats.spearmanr(y, p)[0],
                within_lot_rho=float(np.nanmean(within)) if within else np.nan,
                lot_level_rho=stats.spearmanr(lm.y, lm.p)[0] if lm.p.nunique() > 1 else np.nan)
