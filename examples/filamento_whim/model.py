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
    campo_b: tuple | None = None,
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

    OJO: `longitud_filamento_kpc=None` NO da un cilindro infinito -cae al
    default de `config.LONGITUD_FILAMENTO_KPC`, que es finito (un docstring
    anterior de esta función decía lo contrario; era falso, `None` nunca
    llegaba a saltarse el corte axial). Para reproducir el comportamiento
    de cilindro infinito, pasar explícitamente `longitud_filamento_kpc=float("inf")`:
    con L=inf, `media_longitud` es inf y la máscara tanh vale 1.0 en todo
    punto finito, sin necesidad de una rama de código aparte.

    `campo_b`: opcional, tupla `(bx, by, bz)` ya generada y normalizada a
    `b0_microgauss` (ver `GaussianRandomVectorField.sample` +
    `normalize_to_rms`). Si se da, se reutiliza tal cual en vez de generar
    un campo turbulento nuevo -el campo NO depende de `axis_direction` (se
    genera siempre en el marco de la caja, independiente de cómo se orienta
    el filamento dentro de ella), así que en un barrido en theta con la
    misma semilla es exactamente el mismo campo para cada ángulo. Antes
    `run_barrido_theta.py` lo regeneraba desde cero en cada iteración del
    barrido (misma semilla -> mismo resultado, pero recalculado ~10 veces
    de más); pasar `campo_b` una sola vez por semilla evita ese trabajo
    redundante.
    """
    xp = get_backend(use_gpu)

    if density_profile is None:
        # El BetaModel intacto, calculando la forma funcional correcta
        density_profile = BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA)

    if longitud_filamento_kpc is None:
        longitud_filamento_kpc = cfg.LONGITUD_FILAMENTO_KPC

    if campo_b is not None:
        bx, by, bz = campo_b
        # El tamaño de malla de la densidad se toma del campo YA generado,
        # no de `cfg.N_BASE`: así se evita un posible desajuste de forma si
        # `cfg.N_BASE` (que pasa por `config.py`, cacheado a nivel de módulo
        # con un `importlib.reload` que solo corre una vez por proceso, ver
        # arriba) no coincidiera con el `N_BASE` real usado para generar
        # `campo_b` -por ejemplo, en un test que mockea `config_fisica.N_BASE`
        # después de que este módulo ya se importó una vez.
        n_grid = bx.shape[0]
    else:
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
        n_grid = cfg.N_BASE

    # Malla con paso EXACTO de cfg.DX_BASE_KPC. `linspace(-N/2, N/2, N)` (la
    # versión anterior) da un paso real de N/(N-1)*dx, no dx -con N=128 eso
    # es ~20.16 kpc en vez de 20, un desajuste de ~0.8% respecto al `dl` que
    # de verdad se usa para integrar la línea de visión. `arange` con offset
    # entero coincide además con la convención de píxel-centro que ya usa
    # `faradaymr.simulation.geometry.projected_axis_distance` para el mapa 2D.
    eje = (xp.arange(n_grid) - n_grid // 2) * cfg.DX_BASE_KPC
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