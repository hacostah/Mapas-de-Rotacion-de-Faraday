from __future__ import annotations

import os
import sys

_RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _RAIZ_REPO not in sys.path:
    sys.path.insert(0, _RAIZ_REPO)

import config as cfg
from model import construir_escenario
from plots import generar_graficos_estudio

from faradaymr import (
    BetaModel,
    DensityProfile,
    ObservationConfig,
    ObservationPipeline,
    get_backend,
)
from faradaymr.io import save_maps
from faradaymr.logging_config import configurar_logging, generar_id_simulacion

RUTA_RESULTADOS = os.path.join(
    os.path.dirname(__file__), "results", "estudio_parametrico_icm"
)
RUTA_LOGS = os.path.join(os.path.dirname(__file__), "results", "logs")


def ejecutar_corrida(
    n_spec: float,
    b0_microgauss: float,
    ruta_destino: str,
    use_gpu=True,
    density_profile: DensityProfile | None = None,
):
    # Un archivo de log por corrida, nombrado con su propio id de
    # simulación: en un estudio paramétrico (muchos n_spec, muchos B0) esto
    # es lo que permite después ir de "esta corrida se comportó raro" al
    # registro exacto de qué pasó en ella, sin mezclarlo con el de las
    # demás corridas del barrido.
    id_simulacion = generar_id_simulacion()
    logger = configurar_logging(directorio_logs=RUTA_LOGS, id_simulacion=id_simulacion)

    if density_profile is None:
        density_profile = BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA)

    xp = get_backend(use_gpu)

    ruta_absoluta = os.path.abspath(ruta_destino)

    logger.info(
        "Corrida %s: n_spec=%.3g, B0=%.3g uG, GPU=%s",
        id_simulacion,
        n_spec,
        b0_microgauss,
        use_gpu,
    )

    logger.info(
        "Generando plasma magnetizado (campo turbulento + perfil de densidad)..."
    )
    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec, b0_microgauss, density_profile=density_profile, use_gpu=use_gpu
    )

    observacion = ObservationConfig(
        pixel_size=cfg.DX_BASE_KPC,
        dl=cfg.DX_BASE_PC,
        frequency=cfg.NU_HZ,
        wavelength=cfg.LAMBDA_ONDA_M,
        p_index=cfg.P_SPEC,
        beam_fwhm=cfg.BEAM_FWHM_KPC,
        noise_sigma=cfg.DESV_EST_RUIDO if cfg.AGREGAR_RUIDO else None,
    )
    pipeline = ObservationPipeline(config=observacion, xp=xp)

    resultado = pipeline.run(bx, by, bz, ne, ne_rel)

    logger.info("Guardando mapas en %s ...", ruta_absoluta)

    save_maps(
        ruta_destino,
        {
            "rm_mapa": resultado.rm_map,
            "intensidad": resultado.i_map,
            "stokes_q": resultado.q_map,
            "stokes_u": resultado.u_map,
            "rm_mapa_beam": resultado.rm_beam,
            "i_beam": resultado.i_beam,
            "q_beam": resultado.q_beam,
            "u_beam": resultado.u_beam,
            "intensidad_de_polaridad": resultado.p_beam,
            "fraccion_de_polarizacion": resultado.frac_pol_beam,
            "despolarizacion": resultado.depolarization_map,
        },
    )

    generar_graficos_estudio(ruta_destino, n_spec=n_spec, b0_microgauss=b0_microgauss)
    logger.info("Corrida %s completa. Todo quedó en: %s", id_simulacion, ruta_absoluta)


if __name__ == "__main__":
    ejecutar_corrida(
        n_spec=cfg.N_SPEC,
        b0_microgauss=cfg.B0_MG,
        ruta_destino=RUTA_RESULTADOS,
    )
