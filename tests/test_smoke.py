"""Smoke tests: verify imports and minimal forward-pass correctness."""

import numpy as np
import torch


def test_imports():
    from vaesde.model         import SequenceVAE
    from vaesde.normalisation import denormalise, renormalise
    from vaesde.corruption    import (corrupt_speed_scale, corrupt_position_jump,
                                       corrupt_random_walk_velocity, corrupt_sudden_turn,
                                       corrupt_stationary)
    from vaesde.kinematics    import compute_kinematics, compute_flags
    from vaesde.inference     import compute_recon_mse
    from vaesde.training      import SequenceDataset, vae_loss, run_epoch
    from vaesde.io_utils      import load_config


def test_sequence_vae_forward():
    from vaesde.model     import SequenceVAE
    from vaesde.constants import SEQ_LEN, N_FEAT, FLAT_DIM, LATENT_DIM
    model = SequenceVAE(FLAT_DIM, LATENT_DIM)
    model.eval()
    x = torch.zeros(4, SEQ_LEN, N_FEAT)
    with torch.no_grad():
        recon, mu, logvar = model(x)
    assert recon.shape == (4, SEQ_LEN, N_FEAT)
    assert mu.shape    == (4, LATENT_DIM)
    assert logvar.shape == (4, LATENT_DIM)


def test_corruption_functions():
    from vaesde.corruption import (corrupt_speed_scale, corrupt_position_jump,
                                    corrupt_random_walk_velocity, corrupt_sudden_turn,
                                    corrupt_stationary)
    from vaesde.constants import SEQ_LEN, N_FEAT
    rng  = np.random.default_rng(0)
    phys = rng.standard_normal((8, SEQ_LEN, N_FEAT))
    for fn, args in [
        (corrupt_speed_scale,          (phys, 1.5)),
        (corrupt_position_jump,        (phys, 500.0)),
        (corrupt_random_walk_velocity, (phys, rng)),
        (corrupt_sudden_turn,          (phys,)),
        (corrupt_stationary,           (phys,)),
    ]:
        out = fn(*args)
        assert out.shape == phys.shape
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()


def test_compute_kinematics():
    from vaesde.kinematics import compute_kinematics
    from vaesde.constants  import SEQ_LEN, N_FEAT
    rng  = np.random.default_rng(1)
    phys = rng.standard_normal((16, SEQ_LEN, N_FEAT))
    kin  = compute_kinematics(phys)
    expected_keys = {"mean_speed_mps", "max_speed_mps", "mean_accel_mps2", "max_accel_mps2",
                     "mean_pv_error_m", "max_pv_error_m", "total_displacement_m",
                     "mean_turn_rate_degps", "max_turn_rate_degps"}
    assert expected_keys == set(kin.keys())
    for v in kin.values():
        assert v.shape == (16,)


def test_vae_loss():
    from vaesde.training  import vae_loss
    from vaesde.constants import SEQ_LEN, N_FEAT
    x      = torch.randn(8, SEQ_LEN, N_FEAT)
    recon  = torch.randn_like(x)
    mu     = torch.randn(8, 16)
    logvar = torch.randn(8, 16)
    total, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=0.001)
    assert total.item() >= 0
