"""
Barrido angular: dispersión de RM transversal en función del ángulo entre
la línea de visión y el eje del filamento.

Perspectiva de físico: un filamento del WHIM no tiene una orientación
privilegiada respecto al observador -a diferencia del ICM esférico, aquí
"de frente" (mirando a lo largo del filamento) y "de lado" (mirando
perpendicular a él) son observaciones físicamente distintas, con distinto
camino óptico atravesado y distinta firma de RM. Este script recorre ese
ángulo.

Cómo se logra sin tocar `faradaymr.los`: ese módulo integra siempre sobre
el último eje de la caja (`axis=-1`), asumiendo una malla cúbica regular
con un `dl` constante, no para ray tracing con un observador interior. 
En vez de inclinar la línea de visión (lo que exigiría
resolver esa integral con paso variable por celda), se inclina el objeto:
`filament_axis_from_viewing_angle(theta)` gira el eje del filamento dentro
de la misma caja cúbica regular. La caja y la integración no cambian en
absoluto entre corridas de este barrido; solo cambia la orientación del
filamento dentro de ella.
"""

from __future__ import annotations

import os
import numpy as np
import importlib

# Imports absolutos usando la ruta completa del paquete
from examples.filamento_whim import config as cfg
from examples.filamento_whim.model import construir_escenario

# Blindaje adicional sugerido por la revisión
importlib.reload(cfg)

from faradaymr import ObservationConfig, ObservationPipeline, get_backend, to_numpy
from faradaymr.io import save_maps
from faradaymr.logging_config import configurar_logging, generar_id_simulacion
from faradaymr.simulation.geometry import filament_axis_from_viewing_angle

RUTA_RESULTADOS = os.path.join(
    os.path.dirname(__file__), "results", "barrido_angular_filamento"
)
RUTA_LOGS = os.path.join(os.path.dirname(__file__), "results", "logs")

# theta=0 (filamento "de frente", paralelo a la LoS) hasta theta=pi/2
# (filamento "de lado", perpendicular a la LoS).
ANGULOS_THETA_RAD = np.linspace(0.0, np.pi / 2, 7)


def ejecutar_corrida_angular(
    theta_rad: float,
    n_spec: float,
    b0_microgauss: float,
    ruta_destino: str,
    use_gpu=False,
    seed: int | None = 42,
):
    """
    Una corrida del filamento con su eje inclinado `theta_rad` respecto a
    la línea de visión. El pipeline observacional (`faradaymr.los`, vía
    `ObservationPipeline`) es exactamente el mismo que usa el ejemplo del
    ICM: no se le pasa nada relacionado con el ángulo, porque no necesita
    saberlo -sigue integrando sobre el eje Z de la caja igual que siempre.
    Lo único que "sabe" del ángulo es `construir_escenario`, a través de
    `axis_direction`.
    """
    id_simulacion = generar_id_simulacion()
    logger = configurar_logging(directorio_logs=RUTA_LOGS, id_simulacion=id_simulacion)

    xp = get_backend(use_gpu)
    eje_filamento = filament_axis_from_viewing_angle(theta_rad)
    rng = np.random.RandomState(seed) if seed is not None else None

    logger.info(
        "Corrida %s: theta=%.1f° (eje del filamento=%s), n_spec=%.3g, B0=%.3g uG",
        id_simulacion,
        np.degrees(theta_rad),
        np.round(eje_filamento, 3).tolist(),
        n_spec,
        b0_microgauss,
    )

    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec=n_spec,
        b0_microgauss=b0_microgauss,
        use_gpu=use_gpu,
        rng=rng,
        axis_direction=eje_filamento,
    )

    observacion = ObservationConfig(
        pixel_size=cfg.DX_BASE_KPC,
        dl=cfg.DX_BASE_PC,
        frequency=cfg.NU_HZ,
        wavelength=cfg.LAMBDA_ONDA_M,
        p_index=cfg.P_SPEC,
        beam_fwhm=None,
        noise_sigma=None,
    )
    # `ObservationPipeline.run` llama a `faradaymr.los` sin ninguna
    # modificación ni parámetro nuevo: no tiene forma de saber que el
    # filamento está inclinado, y no necesita saberlo.
    resultado = ObservationPipeline(config=observacion, xp=xp).run(bx, by, bz, ne, ne_rel)

    sigma_rm = float(xp.std(resultado.rm_map))

    ruta_theta = os.path.join(ruta_destino, f"theta_{np.degrees(theta_rad):05.1f}deg")
    save_maps(
        ruta_theta,
        {
            "rm_mapa": resultado.rm_map,
            "intensidad": resultado.i_map,
            "stokes_q": resultado.q_map,
            "stokes_u": resultado.u_map,
        },
    )

    logger.info(
        "Corrida %s completa (sigma_RM=%.4g). Resultados en: %s",
        id_simulacion,
        sigma_rm,
        ruta_theta,
    )
    return {
        "theta_rad": theta_rad,
        "eje_filamento": eje_filamento,
        "ne": ne,
        "resultado": resultado,
        "sigma_rm": sigma_rm,
    }


def ejecutar_barrido_angular(
    angulos_rad=ANGULOS_THETA_RAD,
    n_spec: float = None,
    b0_microgauss: float = None,
    ruta_destino: str = RUTA_RESULTADOS,
    use_gpu=False,
    seed: int | None = 42,
):
    """
    Repite `ejecutar_corrida_angular` para cada ángulo de `angulos_rad`,
    usando la MISMA semilla de turbulencia en todas las corridas del
    barrido: así la única variable física que cambia entre corridas es la
    orientación del filamento, no una realización distinta del campo
    turbulento (que también movería sigma_RM por su cuenta y confundiría
    la dependencia angular que se quiere aislar).
    """
    n_spec = cfg.N_SPEC if n_spec is None else n_spec
    b0_microgauss = cfg.B0_MG if b0_microgauss is None else b0_microgauss

    resultados = []
    for theta in angulos_rad:
        resultados.append(
            ejecutar_corrida_angular(
                theta_rad=theta,
                n_spec=n_spec,
                b0_microgauss=b0_microgauss,
                ruta_destino=ruta_destino,
                use_gpu=use_gpu,
                seed=seed,
            )
        )

    thetas_grados = [np.degrees(r["theta_rad"]) for r in resultados]
    sigmas = [r["sigma_rm"] for r in resultados]
    np.savetxt(
        os.path.join(ruta_destino, "sigma_rm_vs_theta.csv"),
        np.column_stack([thetas_grados, sigmas]),
        delimiter=",",
        header="theta_deg,sigma_rm",
        comments="",
    )
    return resultados


if __name__ == "__main__":
    ejecutar_barrido_angular()