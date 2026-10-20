"""Raw SEM TIFF -> tile embeddings with a frozen pretrained backbone, plus an
attention-MIL head trained on those tile embeddings.

Originals are verified against images.csv SHA-256 before use; previews must not be
used for features. The SEM data bar is cropped and intensity is normalised per image
to reduce session/contrast drift.
"""
import hashlib
import os

import numpy as np

from .cv import inner_split

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def resolve(root, row):
    """Find an image at ROOT/<relative_path> or ROOT/<sha256>.tif."""
    for p in (os.path.join(root, row.relative_path), os.path.join(root, f'{row.sha256}.tif')):
        if os.path.isfile(p): return p
    raise FileNotFoundError(f'{row.sample_id} {row.side}: not found under {root}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def read_gray(path, crop_bottom=0):
    import tifffile
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 3: a = a[..., :3].mean(-1) if a.shape[-1] in (3, 4) else a[0]
    if crop_bottom: a = a[:-crop_bottom]
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def tile_grid(a, size):
    """Centred non-overlapping grid. Returns tiles (n, size, size) and (oy, ox, ny, nx)."""
    ny, nx = a.shape[0] // size, a.shape[1] // size
    oy, ox = (a.shape[0] - ny * size) // 2, (a.shape[1] - nx * size) // 2
    return np.stack([a[oy + i * size: oy + (i + 1) * size, ox + j * size: ox + (j + 1) * size]
                     for i in range(ny) for j in range(nx)]), (oy, ox, ny, nx)


def save_preview(g, grid, size, path):
    from PIL import Image, ImageDraw
    oy, ox, ny, nx = grid
    img = Image.fromarray((g * 255).astype(np.uint8)).convert('RGB'); d = ImageDraw.Draw(img)
    for i in range(ny + 1): d.line([(ox, oy + i * size), (ox + nx * size, oy + i * size)], fill=(255, 0, 0))
    for j in range(nx + 1): d.line([(ox + j * size, oy), (ox + j * size, oy + ny * size)], fill=(255, 0, 0))
    img.save(path)


def load_backbone(name, device):
    """resnet50 (torchvision ImageNet), dinov2_vits14 (torch.hub) or random (untrained
    resnet18, for dry runs). Pretrained weights download on first use."""
    import torch, torchvision
    if name == 'resnet50':
        m = torchvision.models.resnet50(weights='IMAGENET1K_V2'); m.fc = torch.nn.Identity()
    elif name == 'random':
        torch.manual_seed(0); m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
    elif name == 'dinov2_vits14':
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    else: raise ValueError(name)
    return m.eval().to(device)


def embed_tiles(net, T, device):
    import torch
    x = (np.repeat(T[:, None], 3, 1) - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    with torch.no_grad():
        return net(torch.tensor(x, dtype=torch.float32, device=device)).float().cpu().numpy()


def mil_fit_predict(Ti, To, y, lots, tr, te, scheme, seed, hidden=32, max_epochs=800, patience=40):
    """Gated-attention MIL with separate pooling over inner and outer tiles.
    Ti, To: (n_samples, n_tiles, dim) tile embeddings for all samples."""
    import torch, torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    D = Ti.shape[2]
    sc = StandardScaler().fit(np.concatenate([Ti[tr], To[tr]], 1).reshape(-1, D))
    f = lambda T: torch.tensor(sc.transform(T.reshape(-1, D)).reshape(T.shape), dtype=torch.float32)
    Ai, Ao = f(Ti), f(To)
    t = np.log(y[tr]); mu, sd = t.mean(), t.std(); tt = torch.tensor((t - mu) / sd, dtype=torch.float32)
    torch.manual_seed(seed)

    class Pool(nn.Module):
        def __init__(self):
            super().__init__(); self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(D, hidden), nn.GELU())
            self.v, self.u, self.w = nn.Linear(hidden, 16), nn.Linear(hidden, 16), nn.Linear(16, 1)

        def forward(self, x):
            e = self.proj(x); w = self.w(torch.tanh(self.v(e)) * torch.sigmoid(self.u(e))).softmax(1)
            return (w * e).sum(1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.pi, self.po = Pool(), Pool()
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 1))

        def forward(self, xi, xo): return self.head(torch.cat([self.pi(xi), self.po(xo)], 1)).squeeze(-1)

    m = Net(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=5e-2)
    jtr, jva = inner_split(scheme, lots[tr], seed)
    itr, iva = tr[jtr], tr[jva]
    best, state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        m.train(); opt.zero_grad()
        nn.functional.huber_loss(m(Ai[itr], Ao[itr]), tt[jtr]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = nn.functional.mse_loss(m(Ai[iva], Ao[iva]), tt[jva]).item()
        if v < best - 1e-4: best, wait, state = v, 0, {k: x.clone() for k, x in m.state_dict().items()}
        else:
            wait += 1
            if wait > patience: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): q = m(Ai[te], Ao[te]).numpy()
    return np.exp(q * sd + mu)
