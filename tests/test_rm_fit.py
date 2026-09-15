import numpy as np

from faradaymr.analysis import estimate_rm_lsq


def test_recupera_rm_sin_ruido():
    wavelengths_m = np.array([0.10, 0.16, 0.22, 0.31])
    rm_true = 83.5
    psi_0_true = -0.27

    psi_obs_rad = psi_0_true + rm_true * wavelengths_m**2

    rm, psi_0 = estimate_rm_lsq(wavelengths_m, psi_obs_rad)

    assert np.isclose(rm, rm_true, atol=1e-12)
    assert np.isclose(psi_0, psi_0_true, atol=1e-12)


def test_dos_frecuencias_determinan_la_recta():
    wavelengths_m = np.array([0.12, 0.28])
    rm_true = -41.0
    psi_0_true = 0.9

    psi_obs_rad = psi_0_true + rm_true * wavelengths_m**2

    rm, psi_0 = estimate_rm_lsq(wavelengths_m, psi_obs_rad)

    assert np.isclose(rm, rm_true, atol=1e-12)
    assert np.isclose(psi_0, psi_0_true, atol=1e-12)


def test_multiples_frecuencias_recuperan_rm_con_ruido():
    rng = np.random.default_rng(25)
    wavelengths_m = np.linspace(0.08, 0.32, 20)
    rm_true = 120.0
    psi_0_true = 0.15

    psi_obs_rad = psi_0_true + rm_true * wavelengths_m**2
    psi_obs_rad += rng.normal(scale=0.01, size=wavelengths_m.size)

    rm, psi_0 = estimate_rm_lsq(wavelengths_m, psi_obs_rad)

    assert np.isclose(rm, rm_true, atol=0.5)
    assert np.isclose(psi_0, psi_0_true, atol=0.02)