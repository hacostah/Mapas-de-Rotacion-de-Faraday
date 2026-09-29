"""
Validación del estimador contra el valor esperado analítico.

Es el test más importante de la física del proyecto: el sigma_RM medido
en los mapas simulados (Monte Carlo) tiene que converger a la predicción
analítica exacta de `faradaymr.analysis.expected` para el mismo campo y la
misma densidad. Si alguien cambia el generador del campo, la integración de
línea de visión o el estimador, y deja de converger, este test lo detecta.
"""
import numpy as np

from faradaymr import BetaModel
from faradaymr.analysis.expected import (
    expected_rm_dispersion_map,
    los_correlation_bz,
    longitud_correlacion_los,
)
from faradaymr.fields import GaussianRandomVectorField
from faradaymr.los import rotation_measure

N, DX, B0, DL_PC = 32, 25.0, 0.01, 25e3


def _campo():
    return GaussianRandomVectorField(n=N, dx=DX, spectral_index=3.0,
                                     scale_min=DX, scale_max=200.0)


def _densidad_cilindro():
    eje = (np.arange(N) - N // 2) * DX
    xx, yy, _ = np.meshgrid(eje, eje, eje, indexing="ij")
    return BetaModel(n0=1e-5, r_core=100.0, beta=2 / 3).density(np.hypot(xx, yy))


def test_correlacion_isotropa_en_el_origen():
    c = los_correlation_bz(_campo(), B0)
    assert np.isclose(c[0], B0**2 / 3, rtol=1e-10)
    assert 0 < longitud_correlacion_los(_campo()) < N * DX


def test_monte_carlo_converge_al_valor_esperado():
    campo, ne = _campo(), _densidad_cilindro()
    esperado = expected_rm_dispersion_map(ne, campo, B0, DL_PC)
    acumulado = np.zeros((N, N))
    n_semillas = 200
    for s in range(n_semillas):
        bx, by, bz = campo.sample(use_gpu=False, rng=np.random.RandomState(s))
        _, _, bz = GaussianRandomVectorField.normalize_to_rms(bx, by, bz, B0)
        acumulado += rotation_measure(ne, bz, DL_PC) ** 2
    medido = np.sqrt(acumulado / n_semillas)
    # Comparación global (promedio sobre el mapa, para no depender del ruido
    # por píxel) y en el eje del filamento.
    assert np.isclose(np.sqrt((medido**2).mean()), np.sqrt((esperado**2).mean()), rtol=0.05)
    assert np.isclose(medido[N // 2, N // 2], esperado[N // 2, N // 2], rtol=0.15)
