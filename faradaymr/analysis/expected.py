"""
Dispersión de RM de ensamble (promedio sobre realizaciones) de un campo
turbulento gaussiano sobre una densidad dada:

    <RM^2>(x, y) = (0.812 dl)^2  sum_{a,b} n_e(x,y,a) n_e(x,y,b) C_zz(a - b)

con C_zz la correlación de B_z a lo largo de la línea de visión, calculada
del mismo espectro que usa `GaussianRandomVectorField`.

Usos: validación del Monte Carlo (test_expected.py), normalización de RM por
píxel en el histograma del barrido y perfil de ensamble por theta (llaves
`esp_*` del barrido).
"""

from __future__ import annotations

import numpy as np

from ..backend import backend_de, to_numpy
from ..los import FARADAY_CONSTANT_CGS


def los_correlation_bz(field, b_rms: float, xp=None):
    """
    Función de correlación C_zz(Δz), en µG², para los desplazamientos
    Δz = 0, dx, 2dx, ..., (n-1)dx de la malla periódica de `field` (una
    instancia de `GaussianRandomVectorField`), normalizada a un campo con
    <Bx^2 + By^2 + Bz^2> = b_rms^2 (lo mismo que `normalize_to_rms`).

    Por isotropía C_zz(0) = b_rms^2 / 3.
    """
    if xp is None:
        xp = np
    kx, ky, kz, sigma_k = field._grid_y_espectro(xp)
    potencia_z = (kx**2 + ky**2) * sigma_k**2
    potencia_total = 2.0 * (kx**2 + ky**2 + kz**2) * sigma_k**2
    espectro_1d = potencia_z.sum(axis=(0, 1))
    n = espectro_1d.shape[0]
    correlacion = xp.real(xp.fft.ifft(espectro_1d)) * n
    return correlacion * (b_rms**2) / potencia_total.sum()


def longitud_correlacion_los(field, b_rms: float = 1.0) -> float:
    """
    Longitud integral de correlación de B_z a lo largo de la línea de
    visión, Λ_corr = sum_Δ C_zz(Δ) dx / C_zz(0), en las unidades de `field.dx`.
    Es la "celda" efectiva del paseo aleatorio: sigma_RM ≈ 0.812 n_e
    (B/sqrt(3)) sqrt(Λ_corr L).
    """
    c = to_numpy(los_correlation_bz(field, b_rms))
    return float(c.sum() * field.dx / c[0])


def expected_rm_dispersion_map(ne, field, b_rms: float, dl_pc: float, xp=None):
    """
    Mapa 2D de sigma_RM esperado, sqrt(<RM^2>), en rad/m², para la densidad
    3D `ne` (cm^-3, línea de visión en el último eje, mismo tamaño que la
    malla de `field`) y un campo turbulento de `field` con RMS `b_rms` (µG).

    Se evalúa con FFT a lo largo de la línea de visión (la malla del campo
    es periódica): sum_{a,b} n_a C_{a-b} n_b = (1/n) sum_q |ñ_q|^2 Ĉ_q.
    """
    if xp is None:
        xp = backend_de(ne)
    n = ne.shape[-1]
    if n != field.n:
        raise ValueError(
            f"La densidad tiene {n} celdas en la línea de visión y el campo {field.n}."
        )
    correlacion = los_correlation_bz(field, b_rms, xp=xp)
    c_hat = xp.real(xp.fft.fft(correlacion))
    ne_q = xp.fft.fft(ne.astype(xp.float64), axis=-1)
    varianza = (xp.abs(ne_q) ** 2 * c_hat).sum(axis=-1) / n
    varianza = xp.clip(varianza, 0.0, None)  # redondeo numérico
    return FARADAY_CONSTANT_CGS * dl_pc * xp.sqrt(varianza)
