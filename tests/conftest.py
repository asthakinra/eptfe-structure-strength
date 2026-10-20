"""Synthetic stand-in for the confidential packet: same file names and columns,
random values. Tests never read real data."""
import numpy as np
import pandas as pd
import pytest

VIEWS = ('inner', 'outer')
METRICS = ('ind__mean', 'ind__std', 'tortuosity__mean', 'fibrilLength__mean')


def make_packet(d, n_lots=6, per_lot=8, seed=0, tiffs=False):
    rng = np.random.default_rng(seed)
    rows, imgs = [], []
    for li in range(n_lots):
        lot = f'L{li:02d}'; shift = rng.normal(0, 1)
        for k in range(1, per_lot + 1):
            sid = f'{lot}:{k}'
            feats = {f'{v}__{m}': rng.normal(shift, 1) for v in VIEWS for m in METRICS}
            rows.append(dict(lot_id=lot, sample_number=k, sample_id=sid, model_split='train',
                             target_gf=float(np.exp(5.8 + 0.3 * shift + rng.normal(0, 0.1))), **feats))
            for side in ('in', 'out'):
                imgs.append(dict(lot_id=lot, sample_id=sid, side=side, sha256=f'{li:02d}{k:03d}{side}'.ljust(64, '0'),
                                 relative_path=f'{lot}/{k}_{side}.tif', pixel_size_nm=198.4, magnification_x=500,
                                 width_px=320, height_px=256, linkage_grade='annotation_corroborated',
                                 review_flags_json='[]'))
    s, im = pd.DataFrame(rows), pd.DataFrame(imgs)
    s.loc[0, 'inner__ind__std'] = np.nan  # blanks mean missing, as in the real packet
    if tiffs:
        import tifffile, hashlib
        for i, r in im.iterrows():
            p = d / r.relative_path; p.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(p, rng.integers(0, 255, (r.height_px, r.width_px), dtype=np.uint8))
            im.loc[i, 'sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    s.to_csv(d / 'dataset_summary.csv', index=False); im.to_csv(d / 'images.csv', index=False)
    return d


@pytest.fixture
def packet(tmp_path):
    return make_packet(tmp_path)


@pytest.fixture
def packet_with_tiffs(tmp_path):
    return make_packet(tmp_path, n_lots=4, per_lot=5, tiffs=True)
