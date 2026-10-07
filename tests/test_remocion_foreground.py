"""
`remover_foreground_polarizado_planck` y `signo_rm_por_region` sobre cielos
sintéticos: Planck se reemplaza por un múltiplo conocido de la plantilla
del modelo, ya en la grilla (l, b), para saber qué tiene que salir.
"""

from __future__ import annotations

import numpy as np
import pytest

from faradaymr import observational as obs

N_L, N_B = 72, 37


def _grilla():
    l_grid = np.linspace(-np.pi, np.pi, N_L, endpoint=False)
    b_grid = np.linspace(-np.radians(85.0), np.radians(85.0), N_B)
    return l_grid, b_grid


def _ensamble(rng, n=4):
    base_q = rng.normal(size=(N_L, N_B))
    base_u = rng.normal(size=(N_L, N_B))
    ruido = 0.05
    q = np.stack([base_q + ruido * rng.normal(size=base_q.shape) for _ in range(n)])
    u = np.stack([base_u + ruido * rng.normal(size=base_u.shape) for _ in range(n)])
    return q, u


@pytest.fixture
def planck_sintetico(monkeypatch):
    """Instala un Planck en la grilla con Q, U = escala · plantilla."""

    def instalar(q_cielo, u_cielo, sigma=1e-6):
        datos = {
            "q_k_rj": q_cielo,
            "u_k_rj": u_cielo,
            "sigma_q_k_rj": np.full_like(q_cielo, sigma),
            "sigma_u_k_rj": np.full_like(u_cielo, sigma),
            "referencia": "Planck sintético",
        }
        monkeypatch.setattr(obs, "load_planck_030ghz_map", lambda *a, **k: datos)
        monkeypatch.setattr(obs, "project_healpix_to_grid", lambda mapa, l, b: mapa)

    return instalar


def test_plantilla_exacta_se_remueve_por_completo(planck_sintetico):
    l_grid, b_grid = _grilla()
    q_ens, u_ens = _ensamble(np.random.default_rng(0))
    escala = 2.5
    planck_sintetico(escala * q_ens.mean(axis=0), escala * u_ens.mean(axis=0))

    resultado = obs.remover_foreground_polarizado_planck(l_grid, b_grid, q_ens, u_ens)

    for region in resultado["regiones"].values():
        assert np.isclose(region["amplitud"], escala, rtol=1e-6)
        assert region["fraccion_removida_modelo"] > 0.999
    assert resultado["fraccion_removida_sin_franja_amplitud_global"] > 0.999


def test_plantilla_anticorrelacionada_no_remueve_nada(planck_sintetico):
    l_grid, b_grid = _grilla()
    q_ens, u_ens = _ensamble(np.random.default_rng(1))
    planck_sintetico(-q_ens.mean(axis=0), -u_ens.mean(axis=0))

    resultado = obs.remover_foreground_polarizado_planck(l_grid, b_grid, q_ens, u_ens)

    for region in resultado["regiones"].values():
        assert region["amplitud"] == 0.0
        assert np.isclose(region["fraccion_removida_modelo"], 0.0, atol=1e-9)


def test_signo_rm_por_region_usa_la_media_del_ensamble():
    l_grid, b_grid = _grilla()
    rm_ensamble = np.stack([np.full((N_L, N_B), v) for v in (30.0, 10.0, -10.0)])
    rm_observado = np.full((N_L, N_B), 5.0)

    resultado = obs.signo_rm_por_region(l_grid, b_grid, rm_ensamble, rm_observado)

    for region in resultado.values():
        assert np.isclose(region["rm_modelo_media"], 10.0)
        assert np.isclose(region["rm_modelo_dispersion"], 20.0)
        assert region["signo_correcto"]
