from __future__ import annotations

import numpy as np
import sys
import os
import importlib

from faradaymr import BetaModel, DensityProfile, GaussianRandomVectorField, get_backend
from faradaymr.simulation.geometry import cylindrical_radius

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
    axis_direction=(0, 0, 1), # 2. Añadimos la dirección del eje con valor por defecto
):
    xp = get_backend(use_gpu)

    if density_profile is None:
        # El BetaModel intacto, calculando la forma funcional correcta
        density_profile = BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA)

    campo = GaussianRandomVectorField(
        n=cfg.N_BASE,
        dx=cfg.DX_BASE_KPC,
        spectral_index=n_spec,
        scale_min=cfg.LAMBDA_MIN_KPC,
        scale_max=cfg.LAMBDA_MAX_KPC,
    )
    bx, by, bz = campo.sample(use_gpu=use_gpu, rng=rng)
    bx, by, bz = GaussianRandomVectorField.normalize_to_rms(
        bx, by, bz, b0_microgauss, xp=xp
    )

    eje = xp.linspace(-cfg.N_BASE / 2, cfg.N_BASE / 2, cfg.N_BASE) * cfg.DX_BASE_KPC
    xx, yy, zz = xp.meshgrid(eje, eje, eje, indexing="ij")
    
    # Reemplazamos la métrica esférica por la cilíndrica
    r = cylindrical_radius(xx, yy, zz, axis_direction, xp=xp)

    ne = density_profile.density(r, xp=xp).astype(xp.float32)

    ne_rel = ne * 0.01

    return bx, by, bz, ne, ne_rel, r