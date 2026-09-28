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
    axis_direction=(0, 0, 1), # 2. Añadimos la dirección del eje con valor por defecto
    longitud_filamento_kpc: float | None = None,
):
    """
    Construye la malla 3D de campo magnético y densidad electrónica de un
    filamento cilíndrico de longitud FINITA.

    `longitud_filamento_kpc` (por defecto, `config.LONGITUD_FILAMENTO_KPC`)
    trunca la densidad más allá de esa distancia a lo largo del propio eje
    del filamento. Esto importa físicamente sobre todo para theta cercano a
    0 (filamento visto "de frente", casi paralelo a la línea de visión):
    sin el corte axial, un observador end-on integraría a lo largo de TODA
    la profundidad de la caja a densidad aproximadamente constante, y el
    RM resultante (y su dispersión) dependería del tamaño de la caja/malla
    en vez de la longitud física real del objeto. Para theta cercano a 90°
    (filamento "de lado") el corte apenas importa, porque la línea de
    visión solo atraviesa el perfil radial en un tramo del orden de unos
    pocos r_core, muy por debajo de la longitud total.

    Pasar `longitud_filamento_kpc=None` explícitamente (o un valor >= al
    lado de la caja) reproduce el comportamiento de cilindro infinito.
    """
    xp = get_backend(use_gpu)

    if density_profile is None:
        # El BetaModel intacto, calculando la forma funcional correcta
        density_profile = BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA)

    if longitud_filamento_kpc is None:
        longitud_filamento_kpc = cfg.LONGITUD_FILAMENTO_KPC

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

    if longitud_filamento_kpc is not None:
        # Corte axial "duro": el filamento existe solo dentro de
        # |proyección sobre el eje| <= L/2. Se suaviza sobre un ancho de
        # ~1 celda (tanh) únicamente para evitar un escalón discontinuo
        # de un solo píxel; no representa un borde físico difuso real.
        s = axial_projection(xx, yy, zz, axis_direction, xp=xp)
        media_longitud = longitud_filamento_kpc / 2.0
        ancho_borde = max(cfg.DX_BASE_KPC, 1e-6)
        mascara_axial = 0.5 * (
            1.0 - xp.tanh((xp.abs(s) - media_longitud) / ancho_borde)
        )
        ne = ne * mascara_axial.astype(xp.float32)

    ne_rel = ne * 0.01

    return bx, by, bz, ne, ne_rel, r