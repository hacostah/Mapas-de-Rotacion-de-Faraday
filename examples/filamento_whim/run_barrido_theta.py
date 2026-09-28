from __future__ import annotations

import os
import numpy as np
import astropy.units as u
import logging

# Importaciones del proyecto (rutas relativas a examples/filamento_whim)
from examples.filamento_whim.model import construir_escenario
from examples.filamento_whim import config
from examples.filamento_whim import config_fisica
from examples.filamento_whim.validacion import verificar_caja_suficiente

# Importaciones del framework faradaymr
from faradaymr import ObservationConfig, ObservationPipeline, get_backend
from faradaymr.fields import GaussianRandomVectorField
from faradaymr.simulation.geometry import sky_footprint_mask, filament_body_mask
from faradaymr.analysis.spatial_stats import transverse_rm_dispersion
from faradaymr.analysis.fitting import (
    fit_transverse_dispersion,
    fit_beta_dispersion,
    p_random_walk_cilindro,
)
from faradaymr.logging_config import configurar_logging

# Logger de este módulo, colgado del árbol "faradaymr" (en vez de
# `logging.getLogger(__name__)`, que quedaba fuera de esa jerarquía y no
# tenía handlers propios): así sus mensajes salen por los mismos handlers
# (consola + archivo) que configura `configurar_logging`, en vez de perderse
# en silencio -antes ningún `.info(...)` de este módulo llegaba a verse.
_logger_modulo = logging.getLogger("faradaymr.filamento_whim.barrido")


def barrer_angulos(thetas_grados, ruta_resultados, use_gpu=True, n_bins=None, seed=0, logger=None):
    """
    Corre un barrido en ángulo de visión theta para UNA semilla del campo
    turbulento, y ajusta el perfil de dispersión transversal de RM en cada
    ángulo (con las dos formas funcionales: gaussiana y beta, esta última
    tanto libre como con `p` fijado a su valor físico esperado).

    `logger`: si se da (uso normal desde `barrer_angulos_monte_carlo`, que
    configura un único logger para las 50 semillas), se usa tal cual. Si es
    None (uso standalone de esta función sola), se configura uno aquí mismo
    para que la función siga siendo utilizable de forma independiente.
    """
    if logger is None:
        os.makedirs(os.path.join(ruta_resultados, "logs"), exist_ok=True)
        logger = configurar_logging(directorio_logs=os.path.join(ruta_resultados, "logs"))

    if n_bins is None:
        n_bins = config.N_BORDES_PERFIL

    xp = get_backend(use_gpu)

    # 1. Extracción de valores físicos con unidades
    n_base = config_fisica.N_BASE
    dx_base_kpc = config_fisica.DX_BASE.to_value(u.kpc)
    dx_base_pc = config_fisica.DX_BASE.to_value(u.pc)
    longitud_filamento_kpc = config_fisica.LONGITUD_FILAMENTO.to_value(u.kpc)
    b0_microgauss = config_fisica.B0.to_value(u.microgauss)
    nu_hz = config_fisica.NU.to_value(u.Hz)
    lambda_onda_m = config_fisica.LAMBDA_ONDA.to_value(u.m)
    dist_max_ajuste_kpc = config_fisica.DIST_MAX_AJUSTE.to_value(u.kpc)
    lambda_min_kpc = config_fisica.LAMBDA_MIN.to_value(u.kpc)
    lambda_max_kpc = config_fisica.LAMBDA_MAX.to_value(u.kpc)

    # 2. Validación de la geometría
    caja_ok = verificar_caja_suficiente(
        n_base=n_base,
        dx_base_kpc=dx_base_kpc,
        longitud_filamento_kpc=longitud_filamento_kpc,
        thetas_grados=thetas_grados,
        logger=logger
    )
    if not caja_ok:
        logger.error("Abortando: Dimensiones de caja insuficientes para el barrido.")
        raise ValueError(
            "Caja insuficiente para el barrido angular: el filamento se trunca "
            "en la línea de visión para algún theta del barrido (ver el warning "
            "de verificar_caja_suficiente para el valor exacto). Aumentar N_BASE, "
            "DX_BASE, acortar LONGITUD_FILAMENTO o el rango de theta antes de "
            "continuar -ignorar este error invalida la comparación entre ángulos."
        )

    # Ventana de ajuste FIJA (independiente de N_BASE/DX_BASE): usar hasta la
    # mitad de la caja solo si esta resulta más angosta que la ventana física
    # por defecto (4*RC). Ver la nota en config_fisica.DIST_MAX_AJUSTE.
    distancia_media_caja = (n_base / 2) * dx_base_kpc
    distancia_max = min(dist_max_ajuste_kpc, distancia_media_caja)
    if dist_max_ajuste_kpc > distancia_media_caja:
        logger.warning(
            "DIST_MAX_AJUSTE (%.0f kpc) excede la mitad de la caja (%.0f kpc); "
            "se recorta la ventana de ajuste a %.0f kpc.",
            dist_max_ajuste_kpc, distancia_media_caja, distancia_max,
        )

    logger.info(f"Iniciando barrido para {len(thetas_grados)} ángulos: {thetas_grados}")
    logger.info(
        "Ventana de ajuste: 0-%.0f kpc en %d bordes de bin (fija, no depende "
        "del tamaño de caja; compartida con plots.fig2 vía config.N_BORDES_PERFIL).",
        distancia_max, n_bins,
    )

    # El campo turbulento NO depende de axis_direction (se genera siempre en
    # el marco fijo de la caja; es la densidad del filamento la que se
    # reorienta dentro de ella, ver `model.construir_escenario`), así que
    # para una semilla dada es idéntico en todos los theta del barrido.
    # Antes se regeneraba (con la misma semilla, o sea el mismo resultado)
    # en cada iteración del `for theta_deg in thetas_grados` de abajo -aquí
    # se genera una sola vez y se reutiliza, evitando recalcular la FFT del
    # campo ~len(thetas_grados) veces de más por semilla.
    generador_campo = GaussianRandomVectorField(
        n=n_base,
        dx=dx_base_kpc,
        spectral_index=config_fisica.N_SPEC,
        scale_min=lambda_min_kpc,
        scale_max=lambda_max_kpc,
    )
    rng = np.random.RandomState(seed)
    bx0, by0, bz0 = generador_campo.sample(use_gpu=use_gpu, rng=rng)
    bx0, by0, bz0 = GaussianRandomVectorField.normalize_to_rms(
        bx0, by0, bz0, b0_microgauss, xp=xp
    )
    campo_b = (bx0, by0, bz0)

    # Valor de `p` físicamente esperado para el ajuste beta con p fijo (ver
    # `p_random_walk_cilindro`): la comprobación de validación real de si el
    # modelo recupera r_c, no el ajuste de 3 parámetros libres (casi
    # degenerado entre r_c y p en el rango de distancias de este proyecto).
    p_fijo = p_random_walk_cilindro(config_fisica.BETA)

    anchos, anchos_err, sigma0s, r2_gauss = [], [], [], []
    rc_beta, rc_beta_err, p_beta, sigma0_beta, r2_beta = [], [], [], [], []
    rc_beta_fijo, rc_beta_fijo_err, sigma0_beta_fijo, r2_beta_fijo = [], [], [], []

    for theta_deg in thetas_grados:
        theta = np.deg2rad(theta_deg)
        axis_direction = [np.sin(theta), 0.0, np.cos(theta)]

        # 3. Construcción del escenario físico (filamento de longitud finita),
        # reutilizando el campo turbulento generado arriba para esta semilla.
        bx, by, bz, ne, ne_rel, r = construir_escenario(
            n_spec=config_fisica.N_SPEC,
            b0_microgauss=b0_microgauss,
            axis_direction=axis_direction,
            use_gpu=use_gpu,
            longitud_filamento_kpc=longitud_filamento_kpc,
            campo_b=campo_b,
        )

        # 4. Observación (Mock de telescopio)
        observacion = ObservationConfig(
            pixel_size=dx_base_kpc,
            dl=dx_base_pc,
            frequency=nu_hz,
            wavelength=lambda_onda_m,
            p_index=config_fisica.P_SPEC,
        )
        resultado = ObservationPipeline(config=observacion, xp=xp).run(bx, by, bz, ne, ne_rel)

        # Máscara de huella real del filamento finito sobre el mapa 2D: sin
        # esto, los píxeles más allá de las puntas proyectadas del filamento
        # (RM≈0 en toda su línea de visión) se mezclan en los bins de
        # distancia transversal y diluyen el sigma_RM medido -tanto más
        # cuanto más corta es la huella proyectada (peor a theta chico). Ver
        # `faradaymr.simulation.geometry.sky_footprint_mask`.
        footprint = sky_footprint_mask(ne, xp=xp) & filament_body_mask(
            resultado.rm_map.shape, axis_direction, dx_base_kpc,
            longitud_filamento_kpc, config_fisica.RC.to_value(u.kpc), xp=xp,
        )

        # 5. Análisis de dispersión transversal (ventana fija, ver arriba)
        bordes = np.linspace(0.0, distancia_max, n_bins)
        centros, dispersion = transverse_rm_dispersion(
            resultado.rm_map, axis_direction, dx_base_kpc, bordes,
            footprint_mask=footprint, xp=xp,
        )

        # 6a. Ajuste gaussiano (hipótesis original del proyecto)
        ajuste_g = fit_transverse_dispersion(centros, dispersion)
        anchos.append(ajuste_g.width)
        anchos_err.append(ajuste_g.width_err)
        sigma0s.append(ajuste_g.sigma0)
        r2_gauss.append(ajuste_g.r_squared)

        # 6b. Ajuste con la forma derivada del perfil beta, p LIBRE
        # (comparación exploratoria: no se asume a ciegas que la gaussiana
        # es la forma correcta). r_c y p quedan casi degenerados en el rango
        # de distancias típico -ver 6c para la comprobación de validación.
        ajuste_b = fit_beta_dispersion(centros, dispersion)
        rc_beta.append(ajuste_b.r_c)
        rc_beta_err.append(ajuste_b.r_c_err)
        p_beta.append(ajuste_b.p)
        sigma0_beta.append(ajuste_b.sigma0)
        r2_beta.append(ajuste_b.r_squared)

        # 6c. Ajuste beta con p FIJO al valor físico esperado (2 parámetros
        # libres: sigma0, r_c). Esta es la comprobación real de si el
        # pipeline recupera el r_c de entrada (config_fisica.RC).
        ajuste_b_fijo = fit_beta_dispersion(centros, dispersion, p_fijo=p_fijo)
        rc_beta_fijo.append(ajuste_b_fijo.r_c)
        rc_beta_fijo_err.append(ajuste_b_fijo.r_c_err)
        sigma0_beta_fijo.append(ajuste_b_fijo.sigma0)
        r2_beta_fijo.append(ajuste_b_fijo.r_squared)

        logger.info(
            "theta=%.1f°: sigma0=%.3g rad/m2 | gauss: width=%.3g±%.3g kpc R2=%.3f "
            "| beta(p libre): rc=%.3g±%.3g kpc p=%.2f R2=%.3f "
            "| beta(p=%.2f fijo): rc=%.3g±%.3g kpc R2=%.3f",
            theta_deg, ajuste_g.sigma0,
            ajuste_g.width, ajuste_g.width_err, ajuste_g.r_squared,
            ajuste_b.r_c, ajuste_b.r_c_err, ajuste_b.p, ajuste_b.r_squared,
            p_fijo, ajuste_b_fijo.r_c, ajuste_b_fijo.r_c_err, ajuste_b_fijo.r_squared,
        )

    return {
        "theta_grados": np.array(thetas_grados),
        # Ajuste gaussiano (nombres históricos, se conservan por compatibilidad)
        "width_kpc": np.array(anchos),
        "width_err_kpc": np.array(anchos_err),
        "sigma0": np.array(sigma0s),
        "r_squared_gauss": np.array(r2_gauss),
        # Ajuste con la forma beta, p libre (ver faradaymr.analysis.fitting.beta_dispersion_model)
        "rc_beta_kpc": np.array(rc_beta),
        "rc_beta_err_kpc": np.array(rc_beta_err),
        "p_beta": np.array(p_beta),
        "sigma0_beta": np.array(sigma0_beta),
        "r_squared_beta": np.array(r2_beta),
        # Ajuste con la forma beta, p FIJO al valor físico esperado (validación)
        "rc_beta_fijo_kpc": np.array(rc_beta_fijo),
        "rc_beta_fijo_err_kpc": np.array(rc_beta_fijo_err),
        "sigma0_beta_fijo": np.array(sigma0_beta_fijo),
        "r_squared_beta_fijo": np.array(r2_beta_fijo),
        "p_fijo": p_fijo,
    }


def barrer_angulos_monte_carlo(
    thetas_grados,
    ruta_resultados,
    n_semillas=50,
    use_gpu=True,
    n_bins=None,
):
    """
    Orquesta múltiples realizaciones aleatorias (Monte Carlo) del barrido angular.

    Configura el logging UNA sola vez aquí (un solo archivo de log para las
    `n_semillas` corridas) y se lo pasa a `barrer_angulos` en cada
    iteración -antes `barrer_angulos` llamaba a `configurar_logging` en cada
    semilla, generando 50 archivos de log por corrida del Monte Carlo (el
    mismo patrón que ya causó el incidente de logs commiteados por
    accidente, ver `.gitignore`).
    """
    os.makedirs(os.path.join(ruta_resultados, "logs"), exist_ok=True)
    logger = configurar_logging(directorio_logs=os.path.join(ruta_resultados, "logs"))

    logger.info("Iniciando Monte Carlo: %d semillas x %d ángulos = %d corridas totales.",
                n_semillas, len(thetas_grados), n_semillas * len(thetas_grados))

    n_theta = len(thetas_grados)
    nan_col = np.full(n_theta, np.nan)

    anchos_por_theta = []
    sigma0_por_theta = []
    r2_gauss_por_theta = []
    rc_beta_por_theta = []
    r2_beta_por_theta = []
    rc_beta_fijo_por_theta = []
    r2_beta_fijo_por_theta = []
    p_fijo = None

    for semilla in range(n_semillas):
        logger.info("--- Ejecutando semilla Monte Carlo %d/%d ---", semilla + 1, n_semillas)
        resultado = barrer_angulos(
            thetas_grados,
            ruta_resultados,
            use_gpu=use_gpu,
            n_bins=n_bins,
            seed=semilla,
            logger=logger,
        )
        anchos_por_theta.append(resultado["width_kpc"])
        sigma0_por_theta.append(resultado["sigma0"])
        # .get(...) con relleno de NaN: mantiene compatibilidad con quien
        # mockee `barrer_angulos` devolviendo solo las llaves históricas.
        r2_gauss_por_theta.append(resultado.get("r_squared_gauss", nan_col))
        rc_beta_por_theta.append(resultado.get("rc_beta_kpc", nan_col))
        r2_beta_por_theta.append(resultado.get("r_squared_beta", nan_col))
        rc_beta_fijo_por_theta.append(resultado.get("rc_beta_fijo_kpc", nan_col))
        r2_beta_fijo_por_theta.append(resultado.get("r_squared_beta_fijo", nan_col))
        if p_fijo is None:
            p_fijo = resultado.get("p_fijo", np.nan)

    anchos_por_theta = np.array(anchos_por_theta)
    sigma0_por_theta = np.array(sigma0_por_theta)
    r2_gauss_por_theta = np.array(r2_gauss_por_theta)
    rc_beta_por_theta = np.array(rc_beta_por_theta)
    r2_beta_por_theta = np.array(r2_beta_por_theta)
    rc_beta_fijo_por_theta = np.array(rc_beta_fijo_por_theta)
    r2_beta_fijo_por_theta = np.array(r2_beta_fijo_por_theta)

    logger.info("Monte Carlo finalizado exitosamente.")

    return {
        "theta_grados": np.array(thetas_grados),
        "width_medio_kpc": anchos_por_theta.mean(axis=0),
        "width_std_kpc": anchos_por_theta.std(axis=0),
        "sigma0_medio": sigma0_por_theta.mean(axis=0),
        "sigma0_std": sigma0_por_theta.std(axis=0),
        "r_squared_gauss_medio": np.nanmean(r2_gauss_por_theta, axis=0),
        "rc_beta_medio_kpc": np.nanmean(rc_beta_por_theta, axis=0),
        "rc_beta_std_kpc": np.nanstd(rc_beta_por_theta, axis=0),
        "r_squared_beta_medio": np.nanmean(r2_beta_por_theta, axis=0),
        "rc_beta_fijo_medio_kpc": np.nanmean(rc_beta_fijo_por_theta, axis=0),
        "rc_beta_fijo_std_kpc": np.nanstd(rc_beta_fijo_por_theta, axis=0),
        "r_squared_beta_fijo_medio": np.nanmean(r2_beta_fijo_por_theta, axis=0),
        "p_fijo": p_fijo if p_fijo is not None else np.nan,
        "n_semillas": n_semillas,
    }


if __name__ == "__main__":
    # Leemos los ángulos definidos en config.py, si están (si no, un barrido
    # por defecto de 0 a 85 grados en 10 pasos).
    angulos_numpy = config.THETAS_BARRIDO if hasattr(config, "THETAS_BARRIDO") else np.linspace(0, 85, 10)
    angulos = [float(x) for x in angulos_numpy]

    ruta_salida = os.path.join(os.path.dirname(__file__), "results", "barrido_theta")
    os.makedirs(ruta_salida, exist_ok=True)

    try:
        resultados = barrer_angulos_monte_carlo(
            thetas_grados=angulos,
            ruta_resultados=ruta_salida,
            n_semillas=50,  # Decidido explícitamente para el congreso
            use_gpu=False,
        )

        archivo_npz = os.path.join(ruta_salida, "barrido_theta_mc.npz")
        np.savez(archivo_npz, **resultados)
        print(f"Barrido Monte Carlo finalizado correctamente. Resultados guardados en {archivo_npz}")

    except Exception as e:
        print(f"Error durante el barrido Monte Carlo: {e}")
        import sys
        sys.exit(1)
