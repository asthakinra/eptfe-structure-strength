"""Load the SEM -> suture-strength development packet.

The packet itself is confidential and never lives in this repo. Point the code at
it with --packet or the EPTFE_PACKET_DIR environment variable.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ID_DTYPES = {'lot_id': str, 'sample_id': str, 'sha256': str}


def packet_dir(arg=None):
    d = arg or os.environ.get('EPTFE_PACKET_DIR')
    if not d:
        raise SystemExit('Set --packet or EPTFE_PACKET_DIR to the development packet folder.')
    if not os.path.isfile(os.path.join(d, 'dataset_summary.csv')):
        raise SystemExit(f'{d} does not look like the packet (dataset_summary.csv missing).')
    return d


@dataclass
class Packet:
    samples: pd.DataFrame   # one row per sample: features, target, IDs
    images: pd.DataFrame    # two rows per sample (side = in / out)
    features: list          # the 48 inner__/outer__ predictor columns

    @property
    def X(self): return self.samples[self.features].values.astype(float)

    @property
    def y(self): return self.samples.target_gf.values.astype(float)

    @property
    def lots(self): return self.samples.lot_id.values

    def view_index(self, view):
        return [i for i, c in enumerate(self.features) if c.startswith(f'{view}__')]


def load_packet(d):
    s = pd.read_csv(os.path.join(d, 'dataset_summary.csv'), dtype=ID_DTYPES)
    im = pd.read_csv(os.path.join(d, 'images.csv'), dtype=ID_DTYPES)
    feats = [c for c in s if c.startswith(('inner__', 'outer__'))]
    assert s.sample_id.is_unique, 'duplicate sample_id'
    assert s.target_gf.notna().all(), 'missing target'
    assert im.groupby('sample_id').size().eq(2).all(), 'every sample needs exactly one inner and one outer image'
    assert set(im.sample_id) == set(s.sample_id), 'images.csv and dataset_summary.csv cover different samples'
    return Packet(s, im, feats)


def lot_eta2(values, lots):
    """Share of a variable's variance explained by lot means (eta squared)."""
    v = pd.Series(values); m = v.notna()
    g = v[m].groupby(np.asarray(lots)[m.values]).transform('mean')
    return 1 - ((v[m] - g) ** 2).sum() / ((v[m] - v[m].mean()) ** 2).sum()
