import subprocess, sys
from pathlib import Path

import numpy as np
import pytest

from eptfe.cv import out_of_fold, splitter
from eptfe.data import load_packet, lot_eta2
from eptfe.metrics import regression_metrics
from eptfe import models_classical as mc, models_nn as mn

ROOT = Path(__file__).resolve().parents[1]


def test_load_packet(packet):
    pk = load_packet(packet)
    assert len(pk.samples) == 48 and len(pk.features) == 8
    assert pk.view_index('inner') and pk.view_index('outer')


def test_lolo_keeps_lots_apart(packet):
    pk = load_packet(packet)
    for tr, te in splitter('leave_one_lot_out').split(pk.X, pk.y, pk.lots):
        assert len(set(pk.lots[te])) == 1 and not set(pk.lots[tr]) & set(pk.lots[te])


def test_eta2_bounds():
    assert lot_eta2([1, 1, 5, 5], ['a', 'a', 'b', 'b']) == pytest.approx(1.0)
    assert lot_eta2([1, 5, 1, 5], ['a', 'a', 'b', 'b']) == pytest.approx(0.0)


def test_metrics_perfect():
    y = np.array([1., 2, 3, 4]); m = regression_metrics(y, y, ['a', 'a', 'b', 'b'], min_lot=2)
    assert m['R2'] == 1 and m['MAE'] == 0 and m['within_lot_rho'] == pytest.approx(1)


@pytest.mark.parametrize('name', ['mean', 'ridge', 'mono_rf', 'mono_hgb'])
def test_classical_models_run(packet, name):
    pk = load_packet(packet)
    p = out_of_fold(lambda tr, te: mc.fit_predict(name, pk.X[tr], pk.y[tr], pk.X[te], name != 'mean'),
                    pk.X, pk.y, pk.lots, 'leave_one_lot_out')
    assert np.isfinite(p).all()


@pytest.mark.parametrize('kind', mn.KINDS)
def test_tabular_nn_runs(packet, kind):
    pk = load_packet(packet); tr = np.where(pk.lots != 'L00')[0]; te = np.where(pk.lots == 'L00')[0]
    p = mn.fit_predict(kind, pk.X[tr], pk.y[tr], pk.lots[tr], pk.X[te], 0, 'leave_one_lot_out',
                       inner_idx=pk.view_index('inner'), outer_idx=pk.view_index('outer'), max_epochs=30)
    assert p.shape == (len(te),) and np.isfinite(p).all() and (p > 0).all()


def test_monotone_net_is_monotone(packet):
    import torch
    torch.manual_seed(0); signs = [1.0, -1.0, 1.0]
    m = mn.Monotone(3, signs).eval(); x = torch.zeros(50, 3)
    for j, sg in enumerate(signs):
        xs = x.clone(); xs[:, j] = torch.linspace(-3, 3, 50)
        d = np.diff(m(xs)[0].detach().numpy())
        assert (sg * d >= -1e-6).all()


def run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_image_embed_and_eval_end_to_end(packet_with_tiffs, tmp_path):
    emb = tmp_path / 'emb.npz'
    run('image_embed.py', '--packet', packet_with_tiffs, '--images', packet_with_tiffs, '--backbone', 'random',
        '--tile', 64, '--crop-bottom', 32, '--out', emb, '--preview', tmp_path / 'prev', '--preview-n', 2)
    z = np.load(emb); assert z['inner'].shape[:2] == (20, 15)
    out = run('image_eval.py', '--packet', packet_with_tiffs, '--emb', emb, '--seeds', 1)
    assert 'lot probe accuracy' in out


def test_image_embed_rejects_hash_mismatch(packet_with_tiffs, tmp_path):
    import pandas as pd
    im = pd.read_csv(packet_with_tiffs / 'images.csv', dtype=str); im.loc[0, 'sha256'] = '0' * 64
    im.to_csv(packet_with_tiffs / 'images.csv', index=False)
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'image_embed.py'), '--packet', str(packet_with_tiffs),
                        '--images', str(packet_with_tiffs), '--backbone', 'random', '--tile', '64',
                        '--out', str(tmp_path / 'e.npz')], capture_output=True, text=True)
    assert r.returncode != 0 and 'SHA-256 mismatch' in r.stderr
