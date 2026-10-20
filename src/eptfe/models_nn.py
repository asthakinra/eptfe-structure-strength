"""Small PyTorch networks on the 48 tabular features.

All are trained on a standardised log target with Huber loss, AdamW and early
stopping on an inner split of the training fold (grouped by lot under LOLO).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .cv import inner_split

KINDS = ('mlp', 'two_tower', 'monotone', 'lot_adversarial')


class MLP(nn.Module):
    def __init__(self, p, h=(64, 32), drop=0.2):
        super().__init__()
        layers, k = [], p
        for w in h: layers += [nn.Linear(k, w), nn.GELU(), nn.Dropout(drop)]; k = w
        self.body = nn.Sequential(*layers); self.head = nn.Linear(k, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head(z).squeeze(-1), z


class TwoTower(nn.Module):
    """Separate inner/outer encoders, fused late (the two views correlate only weakly)."""
    def __init__(self, inner_idx, outer_idx, drop=0.2):
        super().__init__()
        self.ii, self.oi = list(inner_idx), list(outer_idx)
        enc = lambda n: nn.Sequential(nn.Linear(n, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 16), nn.GELU())
        self.ei, self.eo = enc(len(self.ii)), enc(len(self.oi))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        z = torch.cat([self.ei(x[:, self.ii]), self.eo(x[:, self.oi])], 1)
        return self.head(z).squeeze(-1), z


class PosLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i) * 0.1 - 1.5); self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x): return x @ nn.functional.softplus(self.w).T + self.b


class Monotone(nn.Module):
    """Inputs pre-multiplied by training-fold Spearman signs; non-negative weights and
    bounded monotone (tanh) activations make the output non-decreasing in every signed input."""
    def __init__(self, p, signs):
        super().__init__()
        self.register_buffer('sg', torch.tensor(signs, dtype=torch.float32))
        self.net = nn.Sequential(PosLinear(p, 32), nn.Tanh(), PosLinear(32, 16), nn.Tanh(), PosLinear(16, 1))

    def forward(self, x): return self.net(x * self.sg).squeeze(-1), None


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


def fit_predict(kind, Xtr, ytr, lots_tr, Xte, seed, scheme, inner_idx=None, outer_idx=None,
                max_epochs=1500, patience=60):
    imp = SimpleImputer(strategy='median').fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
    A = np.clip(sc.transform(imp.transform(Xtr)), -5, 5); B = np.clip(sc.transform(imp.transform(Xte)), -5, 5)
    t = np.log(ytr); mu, sd = t.mean(), t.std(); t = (t - mu) / sd
    tr, va = inner_split(scheme, lots_tr, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    p = A.shape[1]
    adv = None
    if kind == 'mlp': m = MLP(p)
    elif kind == 'two_tower': m = TwoTower(inner_idx, outer_idx)
    elif kind == 'monotone':
        m = Monotone(p, [1.0 if (stats.spearmanr(A[:, j], t)[0] or 0) >= 0 else -1.0 for j in range(p)])
    elif kind == 'lot_adversarial':
        m = MLP(p); lot_ix = {l: i for i, l in enumerate(np.unique(lots_tr))}
        adv = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, len(lot_ix)))
        glab = torch.tensor([lot_ix[l] for l in lots_tr])
    else: raise ValueError(kind)
    params = list(m.parameters()) + (list(adv.parameters()) if adv is not None else [])
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-2)
    At, tt = torch.tensor(A, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)
    best, best_state, wait = np.inf, None, 0
    for ep in range(max_epochs):
        m.train(); opt.zero_grad()
        out, z = m(At[tr]); loss = nn.functional.huber_loss(out, tt[tr], delta=1.0)
        if adv is not None:
            lam = min(1.0, ep / 200) * 0.5
            loss = loss + nn.functional.cross_entropy(adv(GradRev.apply(z, lam)), glab[tr])
        loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = nn.functional.mse_loss(m(At[va])[0], tt[va]).item()
        if v < best - 1e-4: best, wait, best_state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            wait += 1
            if wait > patience: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad(): q = m(torch.tensor(B, dtype=torch.float32))[0].numpy()
    return np.exp(q * sd + mu)


def ensemble_predict(kind, Xtr, ytr, lots_tr, Xte, scheme, seeds=5, **kw):
    """Seed ensemble, averaged on the log scale."""
    return np.exp(np.mean([np.log(fit_predict(kind, Xtr, ytr, lots_tr, Xte, s, scheme, **kw))
                           for s in range(seeds)], 0))
