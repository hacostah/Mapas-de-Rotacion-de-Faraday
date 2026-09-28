"""
Verifica la suposición central de `calibrar_amplitud_campo.py`: con una
semilla de campo turbulento fija, escalar `B0_REGULAR` y `B0_TURBULENTO`
por el mismo factor `alpha` escala el mapa de RM resultante por
EXACTAMENTE ese `alpha` -por eso el script de calibración puede resolver
el factor óptimo con una fórmula cerrada a partir de UNA sola corrida ya
guardada, sin tener que volver a correr el ray tracing para cada `alpha`
candidato.

Por qué es cierto (no solo una aproximación numérica): RM es una integral
LINEAL en el campo magnético a lo largo de la línea de visión
(`faradaymr.los.rotation_measure`); `bx_reg` es linealmente proporcional a
`B0_REGULAR` (`LogarithmicSpiralField.sample`), y `bx_turb` -para una
semilla fija- es exactamente proporcional a `B0_TURBULENTO`
(`GaussianRandomVectorField.normalize_to_rms` reescala el mismo patrón
direccional a la RMS pedida). Escalar ambos por el mismo `alpha` escala
`bx = bx_reg + bx_turb` (y por lo tanto RM) por ese mismo `alpha`, exacto
salvo error de redondeo de punto flotante.

Se usa una malla y una grilla de cielo mucho más chicas que las de
`config_fisica.py` (más rápido) y se pasa `N_BASE`/`DX_BASE_KPC`/etc.
monkeypateados sobre el propio módulo `config` -no un escenario aparte de
juguete- para probar la suposición sobre el pipeline REAL
(`model.construir_escenario`) que usa `calibrar_amplitud_campo.py`, no
sobre una reimplementación paralela que podría divergir del código real.
"""

from __future__ import annotations

import numpy as np

import config
import model
from faradaymr import los_raytrace as lr


def test_rm_escala_linealmente_con_b0_regular_y_turbulento(monkeypatch):
    monkeypatch.setattr(config, "N_BASE", 16)
    monkeypatch.setattr(config, "N_L", 12)
    monkeypatch.setattr(config, "N_B", 7)

    alpha = 0.35

    def _correr(b0_regular, b0_turbulento, semilla):
        monkeypatch.setattr(config, "B0_REGULAR_MG", b0_regular)
        monkeypatch.setattr(config, "B0_TURBULENTO_MG", b0_turbulento)
        rng = np.random.RandomState(semilla)
        bx, by, bz, ne, ne_rel, observer_pos, box_size, dx = model.construir_escenario(
            use_gpu=False, rng=rng, arm_contrast=True
        )
        l_grid = np.linspace(-np.pi, np.pi, config.N_L, endpoint=False)
        b_grid = np.linspace(-np.radians(config.B_MAX_DEG), np.radians(config.B_MAX_DEG), config.N_B)
        rm_map, _i, _q, _u = lr.sky_map(
            bx, by, bz, ne, ne_rel, observer_pos, dx, box_size, l_grid, b_grid,
            dl=config.DL_KPC, frequency=config.NU_HZ, wavelength=config.LAMBDA_ONDA_M,
            p_index=config.P_SPEC, length_unit_pc=config.KPC_A_PC,
        )
        return rm_map

    rm_base = _correr(2.0, 3.0, semilla=0)
    rm_escalado = _correr(2.0 * alpha, 3.0 * alpha, semilla=0)

    np.testing.assert_allclose(rm_escalado, alpha * rm_base, rtol=1e-5, atol=1e-8)


def test_semillas_distintas_no_dan_la_misma_realizacion_turbulenta(monkeypatch):
    # Contraprueba de que la linealidad de arriba viene de la semilla FIJA,
    # no de que el campo turbulento sea, de por sí, insensible a B0: con
    # semillas distintas el mapa de RM no tiene por qué escalar linealmente
    # (cada uno trae su propia realización turbulenta independiente).
    monkeypatch.setattr(config, "N_BASE", 16)
    monkeypatch.setattr(config, "N_L", 12)
    monkeypatch.setattr(config, "N_B", 7)

    def _correr(semilla):
        monkeypatch.setattr(config, "B0_REGULAR_MG", 2.0)
        monkeypatch.setattr(config, "B0_TURBULENTO_MG", 3.0)
        rng = np.random.RandomState(semilla)
        bx, by, bz, ne, ne_rel, observer_pos, box_size, dx = model.construir_escenario(
            use_gpu=False, rng=rng, arm_contrast=True
        )
        l_grid = np.linspace(-np.pi, np.pi, config.N_L, endpoint=False)
        b_grid = np.linspace(-np.radians(config.B_MAX_DEG), np.radians(config.B_MAX_DEG), config.N_B)
        rm_map, _i, _q, _u = lr.sky_map(
            bx, by, bz, ne, ne_rel, observer_pos, dx, box_size, l_grid, b_grid,
            dl=config.DL_KPC, frequency=config.NU_HZ, wavelength=config.LAMBDA_ONDA_M,
            p_index=config.P_SPEC, length_unit_pc=config.KPC_A_PC,
        )
        return rm_map

    rm_semilla_0 = _correr(semilla=0)
    rm_semilla_1 = _correr(semilla=1)

    assert not np.allclose(rm_semilla_0, rm_semilla_1, rtol=1e-3)
