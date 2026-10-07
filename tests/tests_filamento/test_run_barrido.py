import os
import tempfile
from unittest import mock

import astropy.units as u
import numpy as np

from examples.filamento_whim.run_barrido_theta import (
    analizar_perfiles,
    barrer_angulos,
    barrer_angulos_monte_carlo,
)
from faradaymr.analysis.fitting import beta_dispersion_model

_PARCHES = [
    mock.patch("examples.filamento_whim.config_fisica.N_BASE", 32),
    mock.patch("examples.filamento_whim.config_fisica.DX_BASE", 25.0 * u.kpc),
    mock.patch("examples.filamento_whim.config_fisica.RC", 60.0 * u.kpc),
    mock.patch("examples.filamento_whim.config_fisica.LONGITUD_FILAMENTO", 300.0 * u.kpc),
    mock.patch("examples.filamento_whim.config_fisica.DIST_MAX_AJUSTE", 180.0 * u.kpc),
    mock.patch("examples.filamento_whim.config_fisica.LAMBDA_MAX", 100.0 * u.kpc),
]


def _con_parches(funcion):
    for parche in reversed(_PARCHES):
        funcion = parche(funcion)
    return funcion


@_con_parches
def test_barrido_theta_smoke():
    """Malla chica (32^3) y filamento corto: el barrido corre de punta a punta,
    devuelve todas las llaves que usan plots.py/umbral_deteccion.py y se
    puede guardar como .npz."""
    with tempfile.TemporaryDirectory() as temp_dir:
        res = barrer_angulos_monte_carlo(
            [0, 45, 90], temp_dir, n_semillas=3, use_gpu=False, n_bins=6, n_bootstrap=20,
        )
        for llave in ["theta_grados", "centros_kpc", "perfil_apilado", "perfil_apilado_err",
                      "perfil_esperado", "mc_sigma0", "mc_sigma0_err", "mc_p", "mc_p_err",
                      "mc_hwhm", "esp_p", "esp_sigma0", "mc_w_gauss", "hist_z",
                      "una_realizacion_sigma0_std", "p_frontal", "p_lateral"]:
            assert llave in res, llave
        assert res["perfiles_rms"].shape == (3, 3, 5)
        assert np.all(np.isfinite(res["mc_sigma0"]))
        # La dispersión cae al alejarse del eje en el perfil esperado.
        assert np.all(res["perfil_esperado"][:, 0] > res["perfil_esperado"][:, -1])
        archivo = os.path.join(temp_dir, "salida.npz")
        np.savez(archivo, **res)
        assert os.path.exists(archivo)


@_con_parches
def test_barrer_angulos_una_semilla_no_tiene_errores_bootstrap():
    with tempfile.TemporaryDirectory() as temp_dir:
        res = barrer_angulos([0, 90], temp_dir, use_gpu=False, n_bins=6, seed=7)
    assert res["perfiles_rms"].shape[0] == 1
    assert np.all(np.isnan(res["mc_sigma0_err"]))


def test_analizar_perfiles_recupera_p_de_perfiles_sinteticos():
    """Sin física: perfiles beta sintéticos con ruido multiplicativo pequeño.
    El apilado + ajuste con r_c fijo debe recuperar p y sigma0, y el bootstrap
    debe dar errores finitos y chicos."""
    rng = np.random.default_rng(0)
    centros = np.linspace(40, 860, 11)
    rc = 300.0
    p_verdad = np.array([1.0, 0.75])
    esperados = np.array([beta_dispersion_model(centros, 0.05, rc, p) for p in p_verdad])
    perfiles = esperados[None] * (1 + 0.05 * rng.standard_normal((40, 2, 11)))

    res = analizar_perfiles(perfiles, centros, esperados, rc, n_bootstrap=50)

    np.testing.assert_allclose(res["esp_p"], p_verdad, rtol=1e-4)
    np.testing.assert_allclose(res["mc_p"], p_verdad, rtol=0.03)
    np.testing.assert_allclose(res["mc_sigma0"], 0.05, rtol=0.03)
    assert np.all(np.isfinite(res["mc_p_err"])) and np.all(res["mc_p_err"] < 0.05)
    # El HWHM es mayor para el perfil más plano (vista lateral).
    assert res["esp_hwhm"][1] > res["esp_hwhm"][0]
