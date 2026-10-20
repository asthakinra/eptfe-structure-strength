"""Cross-validation schemes. Every data-adaptive step must happen inside `fit_predict`,
which only ever sees the training fold."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, KFold, LeaveOneGroupOut, ShuffleSplit

SCHEMES = ('shuffled_5fold', 'leave_one_lot_out')


def splitter(scheme, seed=42):
    if scheme == 'shuffled_5fold': return KFold(5, shuffle=True, random_state=seed)
    if scheme == 'leave_one_lot_out': return LeaveOneGroupOut()
    raise ValueError(scheme)


def inner_split(scheme, lots_tr, seed):
    """Early-stopping / tuning split inside a training fold: grouped by lot unless the
    outer scheme is itself shuffled. Returns positions into the training fold."""
    sp = (ShuffleSplit(1, test_size=0.2, random_state=seed) if scheme == 'shuffled_5fold'
          else GroupShuffleSplit(1, test_size=0.2, random_state=seed))
    return next(sp.split(np.zeros(len(lots_tr)), groups=lots_tr))


def out_of_fold(fit_predict, X, y, lots, scheme):
    """fit_predict(tr_idx, te_idx) -> predictions for te_idx. Returns pooled OOF predictions."""
    p = np.full(len(y), np.nan)
    for tr, te in splitter(scheme).split(X, y, lots):
        if scheme == 'leave_one_lot_out': assert not set(lots[tr]) & set(lots[te])
        p[te] = fit_predict(tr, te)
    assert not np.isnan(p).any()
    return p
