"""
Estudio del foreground galáctico: mapa de cielo (l, b) de RM, intensidad
sincrotrón y polarización, generado por un observador dentro del disco de
la Vía Láctea de juguete (Proyecto III).

Sigue el mismo patrón que `examples/icm_faraday_rotation/run.py`
(config.py/model.py/run.py/plots.py, logging por corrida, mapas a disco),
con una diferencia central: en vez de `ObservationPipeline` (que asume un
observador externo mirando la caja de frente a lo largo de `axis=-1`, ver
su docstring) se usa `faradaymr.los_raytrace.sky_map`, la única pieza del
framework que sabe integrar líneas de visión desde un observador *dentro*
de la caja -exactamente el caso de la Vía Láctea vista desde el Sol.
"""

from __future__ import annotations

import os
import time

import numpy as np

import config as cfg
from model import construir_escenario
from plots import generar_graficos_estructura_3d, generar_graficos_estudio

from faradaymr import GaussianRandomVectorField, get_backend, los_raytrace, to_numpy
from faradaymr.calibration import (
    dm_hacia_direccion,
    rms_rm_alta_latitud,
    validar_dm_polo,
    validar_rms_rm_alta_latitud,
)
from faradaymr.io import save_maps
from faradaymr.logging_config import configurar_logging, generar_id_simulacion

RUTA_RESULTADOS = os.path.join(
    os.path.dirname(__file__), "results", "foreground_galactico"
)
RUTA_LOGS = os.path.join(os.path.dirname(__file__), "results", "logs")


def radios_campo_regular():
    """Radios con significado en el campo regular del disco, para la figura de líneas de campo."""
    radios = []
    if cfg.RADIO_SIN_CAMPO_B_KPC is not None:
        radios.append((cfg.RADIO_SIN_CAMPO_B_KPC, f"R = {cfg.RADIO_SIN_CAMPO_B_KPC:g} kpc: sin campo de disco (barra); dentro, solo el halo débil"))
    for r_min, r_max in cfg.ANILLOS_INVERSION_CAMPO_KPC:
        radios.append((r_max, f"R = {r_max:g} kpc: dentro, campo invertido"))
    return radios


def figuras_estructura(
    ruta_destino, bx, by, bz, ne, observer_pos, box_size, dx,
    componentes_regular, componentes_turbulento, use_gpu=None, seed=0,
):
    """
    Figuras de estructura 3D del escenario (disco, perfiles, líneas de
    campo, turbulencia). Aparte de `ejecutar_corrida` para poder rehacerlas
    sin volver a integrar el cielo: `python run.py --solo-estructura`.
    """
    xp = get_backend(use_gpu)
    # Misma densidad pero sin brazos: referencia para separar, en las
    # figuras, el aporte de los brazos de la caída radial del disco.
    ne_axisimetrico = construir_escenario(
        use_gpu=use_gpu, rng=np.random.RandomState(0), arm_contrast=False
    )[3]
    # El mismo generador de turbulencia sin envolvente, normalizado al RMS
    # efectivo de la vecindad solar: para comprobar en la figura que el
    # generador es gaussiano y tiene el espectro impuesto (la envolvente
    # mezcla amplitudes y oculta ambas cosas).
    turbulento_homogeneo = GaussianRandomVectorField.normalize_to_rms(
        *GaussianRandomVectorField(
            n=cfg.N_BASE,
            dx=cfg.DX_BASE_KPC,
            spectral_index=cfg.SPECTRAL_INDEX,
            scale_min=cfg.LAMBDA_MIN_KPC,
            scale_max=cfg.LAMBDA_MAX_KPC,
        ).sample(use_gpu=use_gpu, rng=np.random.RandomState(0 if seed is None else seed)),
        cfg.B0_TURBULENTO_MG,
        xp=xp,
    )
    generar_graficos_estructura_3d(
        ruta_destino,
        to_numpy(ne),
        to_numpy(bx),
        to_numpy(by),
        to_numpy(bz),
        dx,
        box_size,
        to_numpy(observer_pos),
        cfg.R_SOLAR_KPC,
        cfg.PITCH_ANGLE_RAD,
        cfg.N_ARMS,
        phase0=cfg.ARM_PHASE0_RAD,
        arm_r_min=cfg.ARM_R_MIN_KPC,
        componentes_regular=tuple(to_numpy(c) for c in componentes_regular),
        componentes_turbulento=tuple(to_numpy(c) for c in componentes_turbulento),
        ne_axisimetrico=to_numpy(ne_axisimetrico),
        escalas_b={
            "b0": cfg.B0_REGULAR_MG * cfg.FACTOR_CAMPO_REGULAR,
            "escala_radial_b": cfg.SCALE_RADIAL_B_KPC,
            "escala_altura_b": cfg.SCALE_HEIGHT_B_KPC,
            "ancho_vertical_b": cfg.ANCHO_VERTICAL_B_KPC,
            "radio_nucleo": cfg.RADIO_NUCLEO_B_KPC,
            "radio_sin_campo": cfg.RADIO_SIN_CAMPO_B_KPC,
            # El halo toroidal norte gira al revés que el disco local (es lo
            # que da RM > 0 al norte en el primer cuadrante): B_phi pasa por
            # cero entre los dos y |B| tiene un mínimo.
            "nota_vertical": "$B_\\phi$ cambia de signo:\ndisco horario →\nhalo norte antihorario",
        },
        spectral_index=cfg.SPECTRAL_INDEX,
        # `power_law_spectrum` usa k = pi / escala como corte
        k_banda=(np.pi / cfg.LAMBDA_MAX_KPC, np.pi / cfg.LAMBDA_MIN_KPC),
        turbulento_homogeneo=tuple(to_numpy(c) for c in turbulento_homogeneo),
        radios_campo=radios_campo_regular(),
    )



def solo_figuras_estructura(ruta_destino: str = RUTA_RESULTADOS, use_gpu=None, seed: int | None = 0):
    """Reconstruye el escenario (misma semilla que `ejecutar_corrida`) y rehace solo sus figuras."""
    rng = np.random.RandomState(seed) if seed is not None else None
    (bx, by, bz, ne, _ne_rel, observer_pos, box_size, dx, regular, turbulento) = construir_escenario(
        use_gpu=use_gpu, rng=rng, return_components=True
    )
    figuras_estructura(ruta_destino, bx, by, bz, ne, observer_pos, box_size, dx, regular, turbulento,
                       use_gpu=use_gpu, seed=seed)


def ejecutar_corrida(
    ruta_destino: str = RUTA_RESULTADOS,
    use_gpu=None,
    seed: int | None = 0,
    arm_contrast: bool = True,
):
    id_simulacion = generar_id_simulacion()
    logger = configurar_logging(directorio_logs=RUTA_LOGS, id_simulacion=id_simulacion)

    xp = get_backend(use_gpu)
    if cfg.N_BASE >= 256 and xp.__name__ == "numpy":
        logger.warning(
            "Perfil %s (malla %d³) sin GPU: corre en numpy, tarda decenas de "
            "minutos y necesita ~5 GB de RAM. Instalar cupy para usar la GPU, "
            "o FARADAYMR_PERFIL_RESOLUCION=rapido para una prueba rápida.",
            cfg.cfg_units.PERFIL_RESOLUCION, cfg.N_BASE,
        )
    rng = np.random.RandomState(seed) if seed is not None else None
    ruta_absoluta = os.path.abspath(ruta_destino)

    logger.info(
        "Corrida %s: foreground galáctico (N=%d, dx=%.2f kpc, brazos=%s, GPU=%s)",
        id_simulacion,
        cfg.N_BASE,
        cfg.DX_BASE_KPC,
        arm_contrast,
        use_gpu,
    )

    logger.info(
        "Generando plasma magnetizado (disco + brazos + campo regular + "
        "turbulencia)..."
    )
    (
        bx, by, bz, ne, ne_rel, observer_pos, box_size, dx,
        componentes_regular, componentes_turbulento,
    ) = construir_escenario(
        use_gpu=use_gpu, rng=rng, arm_contrast=arm_contrast, return_components=True
    )

    logger.info("Generando gráficos de estructura 3D (disco + campo)...")
    figuras_estructura(
        ruta_destino, bx, by, bz, ne, observer_pos, box_size, dx,
        componentes_regular, componentes_turbulento, use_gpu=use_gpu, seed=seed,
    )

    l_grid = xp.linspace(-xp.pi, xp.pi, cfg.N_L, endpoint=False)
    b_max = xp.radians(cfg.B_MAX_DEG)
    b_grid = xp.linspace(-b_max, b_max, cfg.N_B)

    logger.info(
        "Integrando líneas de visión desde el observador interior "
        "(observer_pos=%s kpc, %d x %d píxeles (l,b), lote de %d píxeles, "
        "backend=%s)...",
        np.array2string(to_numpy(observer_pos), precision=2),
        cfg.N_L,
        cfg.N_B,
        cfg.PIXEL_CHUNK_SIZE,
        xp.__name__,
    )
    def integrar_cielo(frecuencia_hz, longitud_onda_m):
        return los_raytrace.sky_map(
            bx,
            by,
            bz,
            ne,
            ne_rel,
            observer_pos,
            dx,
            box_size,
            l_grid,
            b_grid,
            dl=cfg.DL_KPC,
            frequency=frecuencia_hz,
            wavelength=longitud_onda_m,
            p_index=cfg.P_SPEC,
            xp=xp,
            pixel_chunk_size=cfg.PIXEL_CHUNK_SIZE,
            length_unit_pc=cfg.KPC_A_PC,
        )

    t0 = time.perf_counter()
    rm_map, i_map, q_map, u_map = integrar_cielo(cfg.NU_HZ, cfg.LAMBDA_ONDA_M)
    # `xp.cuda.Stream.null.synchronize()` (implícito en cualquier
    # conversión a CPU, ver más abajo) es lo que de verdad marca cuándo
    # terminó el cómputo en GPU: CUDA lanza kernels de forma asíncrona, así
    # que sin forzar la sincronización este tiempo mediría solo cuánto
    # tardó en *encolar* el trabajo, no en ejecutarlo.
    if xp.__name__ != "numpy":
        xp.cuda.Stream.null.synchronize()
    logger.info(
        "sky_map completado en %.2f s (%d píxeles, %s).",
        time.perf_counter() - t0,
        cfg.N_L * cfg.N_B,
        xp.__name__,
    )

    # Mismo cielo a la frecuencia de Planck LFI 30 GHz, para la validación
    # de `compare_observaciones.py`: ahí Q/U casi no sufren rotación de
    # Faraday, y la comparación de ángulos contra Planck es directa.
    logger.info("Integrando el mismo cielo a %.1f GHz (Planck LFI)...", cfg.NU_PLANCK_HZ / 1e9)
    _, i_map_30, q_map_30, u_map_30 = integrar_cielo(cfg.NU_PLANCK_HZ, cfg.LAMBDA_PLANCK_M)

    # Varianza galáctica (Planck Int. XLII 2016, sec. 3.4.1): el cielo real
    # es UNA realización de la turbulencia, así que la comparación con
    # Planck necesita saber cuánto varía el modelo entre realizaciones. Se
    # integran otras semillas a 28.4 GHz; la primera es la corrida de arriba.
    # La RM no depende de la frecuencia: sale del mismo trazado de rayos y
    # sirve para dar el signo de la RM por región como media ± dispersión
    # entre realizaciones, en vez de depender de una sola semilla.
    q_ensamble, u_ensamble = [to_numpy(q_map_30)], [to_numpy(u_map_30)]
    rm_ensamble = [to_numpy(rm_map)]
    semilla_base = 0 if seed is None else seed
    for k in range(1, cfg.N_REALIZACIONES_VARIANZA_GALACTICA):
        logger.info(
            "Realización %d/%d para la varianza galáctica...",
            k + 1,
            cfg.N_REALIZACIONES_VARIANZA_GALACTICA,
        )
        bx_k, by_k, bz_k, ne_k, ne_rel_k, *_ = construir_escenario(
            use_gpu=use_gpu,
            rng=np.random.RandomState(semilla_base + k),
            arm_contrast=arm_contrast,
        )
        rm_k, _, q_k, u_k = los_raytrace.sky_map(
            bx_k, by_k, bz_k, ne_k, ne_rel_k, observer_pos, dx, box_size, l_grid, b_grid,
            dl=cfg.DL_KPC, frequency=cfg.NU_PLANCK_HZ, wavelength=cfg.LAMBDA_PLANCK_M,
            p_index=cfg.P_SPEC, xp=xp, pixel_chunk_size=cfg.PIXEL_CHUNK_SIZE,
            length_unit_pc=cfg.KPC_A_PC,
        )
        q_ensamble.append(to_numpy(q_k))
        u_ensamble.append(to_numpy(u_k))
        rm_ensamble.append(to_numpy(rm_k))

    logger.info("Guardando mapas en %s ...", ruta_absoluta)
    save_maps(
        ruta_destino,
        {
            "rm_mapa": rm_map,
            "intensidad": i_map,
            "stokes_q": q_map,
            "stokes_u": u_map,
            "intensidad_030ghz": i_map_30,
            "stokes_q_030ghz": q_map_30,
            "stokes_u_030ghz": u_map_30,
            "stokes_q_030ghz_ensamble": np.stack(q_ensamble),
            "stokes_u_030ghz_ensamble": np.stack(u_ensamble),
            "rm_mapa_ensamble": np.stack(rm_ensamble),
            "l_grid": l_grid,
            "b_grid": b_grid,
        },
    )

    # `to_numpy` (no `np.asarray`) es obligatorio acá: un arreglo de cupy
    # (backend GPU) no se puede convertir con `np.asarray` -numpy rechaza
    # explícitamente esa conversión implícita-, así que con `use_gpu=True`
    # esta llamada fallaba siempre antes de llegar a graficar/guardar nada.
    # `to_numpy` ya sabe hacer `cupy.asnumpy` cuando corresponde (y no hace
    # nada distinto de `np.asarray` cuando el backend ya era numpy), así
    # que es la única función seguro-para-ambos-backends para traer un
    # resultado de vuelta a CPU antes de graficar con matplotlib (que no
    # entiende arreglos de GPU).
    generar_graficos_estudio(
        ruta_destino,
        to_numpy(l_grid),
        to_numpy(b_grid),
        to_numpy(rm_map),
        to_numpy(i_map),
        to_numpy(q_map),
        to_numpy(u_map),
        to_numpy(i_map_30),
        to_numpy(q_map_30),
        to_numpy(u_map_30),
        p_index=cfg.P_SPEC,
    )

    logger.info("Corrida %s completa. Todo quedó en: %s", id_simulacion, ruta_absoluta)

    # Chequeos de calibración de orden de magnitud (Fases A/B del Proyecto
    # III en `plan_faradaymr.md`): antes esta corrida solo *reportaba* la
    # RM hacia el polo sin decir si era razonable. Ahora se compara contra
    # el rango publicado (ver `faradaymr.calibration` para las
    # referencias) tanto para RM (que depende de B y n_e) como para DM
    # (que depende solo de n_e, un chequeo independiente del campo
    # magnético que aísla si el problema -de haberlo- está en la densidad
    # o en el campo).
    # La DM sí es una sola línea de visión (el píxel más cercano a l=0,
    # b=b_max: columna de electrones hacia el polo, casi determinista). La
    # RM no: hacia un solo píxel depende de la realización de la
    # turbulencia, así que se valida su dispersión en toda la calota polar
    # (antes se leía rm_map[0, -1], que además era l=-180 y no l=0.)
    idx_l_polo = int(np.argmin(np.abs(to_numpy(l_grid))))
    direccion_polo = los_raytrace.direction_from_galactic(
        l_grid[idx_l_polo], b_grid[-1], xp=xp
    )
    dm_polo = dm_hacia_direccion(
        ne, observer_pos, direccion_polo, cfg.DL_KPC, dx, box_size, xp=xp
    )
    resultado_rm = validar_rms_rm_alta_latitud(
        rms_rm_alta_latitud(to_numpy(rm_map), to_numpy(b_grid))
    )
    resultado_dm = validar_dm_polo(dm_polo)
    logger.info(resultado_rm.mensaje())
    logger.info(resultado_dm.mensaje())
    if not (resultado_rm.dentro_de_tolerancia and resultado_dm.dentro_de_tolerancia):
        logger.warning(
            "Al menos un chequeo de calibración quedó fuera del rango "
            "publicado (ver mensajes arriba): revisar NE0_CM3/FACTOR_CAMPO_REGULAR "
            "en config_fisica.py antes de usar esta corrida como resultado "
            "final (ver plan_faradaymr.md, Fase A/B del Proyecto III)."
        )

    return rm_map, i_map, q_map, u_map


if __name__ == "__main__":
    import sys

    if "--solo-estructura" in sys.argv:
        solo_figuras_estructura()
    else:
        ejecutar_corrida()
