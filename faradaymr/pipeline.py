"""
Pipeline observacional: de una caja 3D de plasma magnetizado a lo que un
radiotelescopio realmente mediría.

Toda esta cadena repite el mismo patrón sin importar el objeto astrofísico
que se simule (cúmulo de galaxias, halo de una galaxia, resto de supernova):
1) se tiene un campo magnético y una densidad de electrones en una caja 3D,
2) se integra a lo largo de la línea de visión para obtener lo que de verdad
   llega como radiación (RM, I, Q, U, ver `faradaymr.los`),
3) se le aplica la respuesta del instrumento -resolución finita (beam) y
   ruido- porque comparar el mapa "verdadero" con datos reales sin pasar por
   este paso es comparar peras con radiotelescopios (ver
   `faradaymr.instrument`).

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from . import los
from .instrument import (
    add_gaussian_noise,
    apply_beam,
    apply_beam_stokes,
    depolarization,
    polarization_fraction,
)
from .logging_config import medir_tiempo_kernel

_logger = logging.getLogger(__name__)


@dataclass
class ObservationConfig:
    """Parámetros puramente instrumentales/observacionales de una corrida."""

    pixel_size: float  # tamaño de celda de la caja, en la misma unidad que dl
    dl: float  # tamaño de celda a lo largo del LOS, en pc (para RM)
    frequency: float  # frecuencia de observación
    wavelength: float  # longitud de onda correspondiente a `frequency`
    p_index: float  # índice de energía de los electrones relativistas
    beam_fwhm: Optional[float] = None  # None -> no se aplica beam
    noise_sigma: Optional[float] = None  # fracción del piso de ruido respecto
    # a la señal más fuerte de cada mapa (p.ej. 0.01 = 1%); None -> sin ruido


@dataclass
class ObservationResult:
    """Todos los mapas 2D que produce el pipeline, sin unidades: cada uno
    documenta en `faradaymr.los` / `faradaymr.instrument` en qué unidad está."""

    rm_map: object
    i_map: object
    q_map: object
    u_map: object
    rm_beam: object = None
    i_beam: object = None
    q_beam: object = None
    u_beam: object = None
    p_beam: object = None
    frac_pol_beam: object = None
    depolarization_map: object = None


@dataclass
class ObservationPipeline:
    """
    Toma campos 3D (Bx, By, Bz, n_e, n_rel) y produce los mapas 2D
    observables, con y sin respuesta instrumental.

    El eje de línea de visión se asume el último (`axis=-1`) por convención;
    quien arme la caja 3D debe respetar ese orden (x, y, línea_de_visión).
    """

    config: ObservationConfig
    xp: object = field(default=None)

    def __post_init__(self):
        if self.xp is None:
            import numpy as xp

            self.xp = xp

    @medir_tiempo_kernel
    def run(self, bx, by, bz, ne, ne_rel) -> ObservationResult:
        xp = self.xp
        cfg = self.config
        _logger.info("Integrando línea de visión (RM, I, Q, U)...")

        # B_perp = |B|*sin(alpha), con alpha el ángulo entre el campo local
        # y la línea de visión (eje z de la caja), calculado explícitamente
        # en vez de solo tomar sqrt(bx^2+by^2) -ver `los.inclination_angle`
        # y `los.perpendicular_field_magnitude` para la justificación.
        b_perp = los.perpendicular_field_magnitude(bx, by, bz, xp=xp)
        j_nu = los.synchrotron_emissivity(
            b_perp, ne_rel, cfg.frequency, cfg.p_index, xp=xp
        )
        i_map = los.synchrotron_intensity(j_nu, cfg.pixel_size, xp=xp)

        psi_0 = los.polarization_angle_intrinsic(bx, by, xp=xp)
        # La RM acumulada (rotación de Faraday dentro de la fuente) lleva la
        # misma constante 0.812 que el mapa de RM, que exige el paso en pc:
        # hay que usar `cfg.dl`, no `cfg.pixel_size`. Con pixel_size en kpc
        # (caso del filamento) la rotación interna salía 1000 veces menor.
        rm_cumulative = los.rotation_measure_cumulative(ne, bz, cfg.dl, xp=xp)
        q_map, u_map = los.stokes_qu(
            j_nu,
            psi_0,
            rm_cumulative,
            cfg.wavelength,
            cfg.p_index,
            cfg.pixel_size,
            xp=xp,
        )

        rm_map = los.rotation_measure(ne, bz, cfg.dl, xp=xp)

        result = ObservationResult(rm_map=rm_map, i_map=i_map, q_map=q_map, u_map=u_map)

        if cfg.beam_fwhm is not None:
            self._apply_instrument(result, cfg)

        return result

    def _apply_instrument(self, result: ObservationResult, cfg: ObservationConfig):
        _logger.info(
            "Aplicando respuesta instrumental (beam=%.4g, ruido=%s)...",
            cfg.beam_fwhm,
            cfg.noise_sigma is not None,
        )
        xp = self.xp
        pixel_size = cfg.pixel_size

        rm_beam = apply_beam(result.rm_map, cfg.beam_fwhm, pixel_size, xp=xp)
        i_beam, q_beam, u_beam = apply_beam_stokes(
            result.i_map, result.q_map, result.u_map, cfg.beam_fwhm, pixel_size, xp=xp
        )

        if cfg.noise_sigma is not None:

            sigma_i = cfg.noise_sigma * float(xp.max(i_beam))
            sigma_rm = cfg.noise_sigma * float(xp.std(rm_beam))
            rm_beam = add_gaussian_noise(rm_beam, sigma_rm, xp=xp)
            i_beam = add_gaussian_noise(i_beam, sigma_i, xp=xp)
            q_beam = add_gaussian_noise(q_beam, sigma_i, xp=xp)
            u_beam = add_gaussian_noise(u_beam, sigma_i, xp=xp)

        p_beam, frac_pol_beam = polarization_fraction(i_beam, q_beam, u_beam, xp=xp)
        dp = depolarization(
            result.i_map, result.q_map, result.u_map, i_beam, q_beam, u_beam, xp=xp
        )

        result.rm_beam = rm_beam
        result.i_beam = i_beam
        result.q_beam = q_beam
        result.u_beam = u_beam
        result.p_beam = p_beam
        result.frac_pol_beam = frac_pol_beam
        result.depolarization_map = dp
