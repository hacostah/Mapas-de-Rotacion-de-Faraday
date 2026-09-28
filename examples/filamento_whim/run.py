from __future__ import annotations

import os
import sys
import numpy as np
import importlib
import astropy.units as u

from examples.filamento_whim import config as cfg
from examples.filamento_whim import config_fisica
from examples.filamento_whim.model import construir_escenario
from examples.filamento_whim.validacion import verificar_caja_suficiente

importlib.reload(cfg)
importlib.reload(config_fisica)

from faradaymr import ObservationConfig, ObservationPipeline, get_backend
from faradaymr.io import save_maps
from faradaymr.logging_config import configurar_logging, generar_id_simulacion
from faradaymr.simulation.geometry import filament_axis_from_viewing_angle

RUTA_RESULTADOS = os.path.join(
    os.path.dirname(__file__), "results", "mapas_poster"
)
RUTA_LOGS = os.path.join(os.path.dirname(__file__), "results", "logs")

def ejecutar_corrida_angular(
    theta_rad: float,
    n_spec: float,
    b0_microgauss: float,
    ruta_destino: str,
    use_gpu=True,
    seed: int | None = 42,
):
    id_simulacion = generar_id_simulacion()
    logger = configurar_logging(directorio_logs=RUTA_LOGS, id_simulacion=id_simulacion)
    xp = get_backend(use_gpu)
    eje_filamento = filament_axis_from_viewing_angle(theta_rad)
    rng = np.random.RandomState(seed) if seed is not None else None

    logger.info("Generando mapa para póster: theta=%.1f°", np.degrees(theta_rad))

    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec=n_spec,
        b0_microgauss=b0_microgauss,
        use_gpu=use_gpu,
        rng=rng,
        axis_direction=eje_filamento,
    )

    observacion = ObservationConfig(
        pixel_size=config_fisica.DX_BASE.to_value(u.kpc),
        dl=config_fisica.DX_BASE.to_value(u.pc),
        frequency=config_fisica.NU.to_value(u.Hz),
        wavelength=config_fisica.LAMBDA_ONDA.to_value(u.m),
        p_index=config_fisica.P_SPEC,
    )
    resultado = ObservationPipeline(config=observacion, xp=xp).run(bx, by, bz, ne, ne_rel)

    ruta_theta = os.path.join(ruta_destino, f"theta_{np.degrees(theta_rad):02.0f}deg")
    save_maps(
        ruta_theta,
        {
            "rm_mapa": resultado.rm_map,
            "intensidad": resultado.i_map,
            "stokes_q": resultado.q_map,
            "stokes_u": resultado.u_map,
        },
    )
    return resultado.rm_map


def generar_mapas_poster(use_gpu=False):
    # Ángulos clave para visualización: Frente, Oblicuo, Lado
    angulos_grados = [0, 15, 30]
    angulos_rad = np.deg2rad(angulos_grados)
    
    # Extraemos valores para la validación geométrica
    n_base = config_fisica.N_BASE
    dx_base_kpc = config_fisica.DX_BASE.to_value(u.kpc)
    longitud_filamento_kpc = config_fisica.LONGITUD_FILAMENTO.to_value(u.kpc)
    
    logger = configurar_logging(directorio_logs=RUTA_LOGS)
    
    # Validamos que la caja contenga los mapas visuales sin cortarlos
    if not verificar_caja_suficiente(n_base, dx_base_kpc, longitud_filamento_kpc, angulos_grados, logger):
        logger.error("La caja recortará los mapas visuales. Ajusta config_fisica.py")
        sys.exit(1)

    for theta in angulos_rad:
        ejecutar_corrida_angular(
            theta_rad=theta,
            n_spec=config_fisica.N_SPEC,
            b0_microgauss=config_fisica.B0.to_value(u.microgauss),
            ruta_destino=RUTA_RESULTADOS,
            use_gpu=use_gpu,
            seed=42 # Mantener la semilla fija es crucial para que el patrón turbulento sea visualmente comparable
        )

if __name__ == "__main__":
    generar_mapas_poster()