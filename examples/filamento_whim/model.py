from __future__ import annotations

import numpy as np
import sys
import os
import importlib

from faradaymr import BetaModel, DensityProfile, GaussianRandomVectorField, get_backend
from faradaymr.simulation.geometry import axial_projection, cylindrical_radius

# Import absoluto para asegurar que se carga la configuración correcta del filamento
from examples.filamento_whim import config as cfg
# Forzamos la recarga para eludir silenciosos errores de caché en sys.modules durante los tests
importlib.reload(cfg)


def construir_escenario(
    n_spec: float,
    b0_microgauss: float,
    density_profile: DensityProfile | None = None,
    use_gpu=None,
    rng=None,
    axis_direction=(0, 0, 1),
    longitud_filamento_kpc: float | None = None,
    campo_b: tuple | None = None,
    dx_kpc: float | None = None,
):
    """
    Construye la malla 3D de campo magnético y densidad electrónica de un
    filamento cilíndrico de longitud finita.

    `longitud_filamento_kpc`: longitud del filamento a lo largo de su eje
    (por defecto `config.LONGITUD_FILAMENTO_KPC`); la densidad se trunca en
    |s| > L/2. `float("inf")` da un cilindro infinito.

    `campo_b`: tupla `(bx, by, bz)` ya normalizada a `b0_microgauss`. Si se
    da, se reutiliza en vez de generar un campo nuevo (el campo no depende
    de `axis_direction`).

    `dx_kpc`: tamaño de celda; por defecto `config.DX_BASE_KPC`.
    """

    xp = get_backend(use_gpu)

    # `dx_kpc` explícito permite otra resolución sin tocar config.py.
    if dx_kpc is None:
        dx_kpc = cfg.DX_BASE_KPC

    if density_profile is None:
        # El BetaModel intacto, calculando la forma funcional correcta
        density_profile = BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA)

    if longitud_filamento_kpc is None:
        longitud_filamento_kpc = cfg.LONGITUD_FILAMENTO_KPC

    if campo_b is not None:
        bx, by, bz = campo_b
        # Tamaño de malla tomado del campo dado (no de cfg.N_BASE).
        n_grid = bx.shape[0]
    else:
        campo = GaussianRandomVectorField(
            n=cfg.N_BASE,
            dx=dx_kpc,
            spectral_index=n_spec,
            scale_min=cfg.LAMBDA_MIN_KPC,
            scale_max=cfg.LAMBDA_MAX_KPC,
        )
        bx, by, bz = campo.sample(use_gpu=use_gpu, rng=rng)
        bx, by, bz = GaussianRandomVectorField.normalize_to_rms(
            bx, by, bz, b0_microgauss, xp=xp
        )
        n_grid = cfg.N_BASE

    # Malla con paso exacto dx y convención de píxel-centro.
    eje = (xp.arange(n_grid) - n_grid // 2) * dx_kpc
    xx, yy, zz = xp.meshgrid(eje, eje, eje, indexing="ij")

    # Reemplazamos la métrica esférica por la cilíndrica
    r = cylindrical_radius(xx, yy, zz, axis_direction, xp=xp)

    ne = density_profile.density(r, xp=xp).astype(xp.float32)

    if longitud_filamento_kpc is not None:
        # Corte axial en |s| <= L/2, suavizado sobre ~1 celda (tanh).
        s = axial_projection(xx, yy, zz, axis_direction, xp=xp)
        media_longitud = longitud_filamento_kpc / 2.0
        ancho_borde = max(dx_kpc, 1e-6)
        mascara_axial = 0.5 * (
            1.0 - xp.tanh((xp.abs(s) - media_longitud) / ancho_borde)
        )
        ne = ne * mascara_axial.astype(xp.float32)

    ne_rel = ne * 0.01

    return bx, by, bz, ne, ne_rel, r