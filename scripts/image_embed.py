"""Embed every SEM tile with a frozen backbone and cache to .npz.

    python scripts/image_embed.py --packet PACKET_DIR --images TIFF_ROOT --backbone resnet50 \
        --crop-bottom 64 --out results/emb_resnet50.npz [--preview results/preview]

TIFF_ROOT holds originals at TIFF_ROOT/<relative_path> or TIFF_ROOT/<sha256>.tif.
Inspect --preview PNGs first to set --crop-bottom (SEM data bar) and to check for
burned-in labels, which must not reach the model.
"""
import argparse, os, sys
import numpy as np, torch
from eptfe.data import load_packet, packet_dir
from eptfe import image as I

ap = argparse.ArgumentParser()
ap.add_argument('--packet'); ap.add_argument('--images', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--backbone', default='resnet50', choices=['resnet50', 'dinov2_vits14', 'random'])
ap.add_argument('--tile', type=int, default=224); ap.add_argument('--crop-bottom', type=int, default=0)
ap.add_argument('--preview'); ap.add_argument('--preview-n', type=int, default=6)
ap.add_argument('--no-verify', action='store_true', help='skip SHA-256 check (synthetic dry runs only)')
a = ap.parse_args()
pk = load_packet(packet_dir(a.packet))
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
net = I.load_backbone(a.backbone, device)
if a.preview: os.makedirs(a.preview, exist_ok=True)
E, bad = {}, []
for k, row in enumerate(pk.images.itertuples()):
    path = I.resolve(a.images, row)
    if not a.no_verify and I.sha256(path) != row.sha256: bad.append((row.sample_id, row.side)); continue
    g = I.read_gray(path, a.crop_bottom); T, grid = I.tile_grid(g, a.tile)
    if a.preview and k < a.preview_n:
        I.save_preview(g, grid, a.tile, f'{a.preview}/{row.sample_id.replace(":", "_")}_{row.side}.png')
    E[(row.sample_id, row.side)] = I.embed_tiles(net, T, device).astype(np.float16)
    if k % 50 == 0: print(f'{k}/{len(pk.images)} images, {len(T)} tiles each', flush=True)
if bad: sys.exit(f'SHA-256 mismatch for {len(bad)} images, refusing to continue: {bad[:5]}')
ids = pk.samples.sample_id.tolist()
np.savez_compressed(a.out, sample_id=np.array(ids), inner=np.stack([E[(i, 'in')] for i in ids]),
                    outer=np.stack([E[(i, 'out')] for i in ids]), backbone=a.backbone, tile=a.tile, crop_bottom=a.crop_bottom)
print('wrote', a.out)
