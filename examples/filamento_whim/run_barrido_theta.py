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
from faradaymr.analysis.spatial_stats import transverse_rm_dispersion
from faradaymr.analysis.fitting import fit_transverse_dispersion, fit_beta_dispersion
from faradaymr.logging_config import configurar_logging


def barrer_angulos(thetas_grados, ruta_resultados, use_gpu=True, n_bins=12, seed=0):
    os.makedirs(os.path.join(ruta_resultados, "logs"), exist_ok=True)
    logger = configurar_logging(directorio_logs=os.path.join(ruta_resultados, "logs"))
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
        "Ventana de ajuste: 0-%.0f kpc en %d bins (fija, no depende del tamaño de caja).",
        distancia_max, n_bins,
    )

    anchos, anchos_err, sigma0s, r2_gauss = [], [], [], []
    rc_beta, rc_beta_err, p_beta, sigma0_beta, r2_beta = [], [], [], [], []

    for theta_deg in thetas_grados:
        theta = np.deg2rad(theta_deg)
        axis_direction = [np.sin(theta), 0.0, np.cos(theta)]

        # 3. Construcción del escenario físico (filamento de longitud finita)
        bx, by, bz, ne, ne_rel, r = construir_escenario(
            n_spec=config_fisica.N_SPEC,
            b0_microgauss=b0_microgauss,
            axis_direction=axis_direction,
            use_gpu=use_gpu,
            rng=np.random.RandomState(seed),
            longitud_filamento_kpc=longitud_filamento_kpc,
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

        # 5. Análisis de dispersión transversal (ventana fija, ver arriba)
        bordes = np.linspace(0.0, distancia_max, n_bins)
        centros, dispersion = transverse_rm_dispersion(
            resultado.rm_map, axis_direction, dx_base_kpc, bordes, xp=xp
        )

        # 6a. Ajuste gaussiano (hipótesis original del proyecto)
        ajuste_g = fit_transverse_dispersion(centros, dispersion)
        anchos.append(ajuste_g.width)
        anchos_err.append(ajuste_g.width_err)
        sigma0s.append(ajuste_g.sigma0)
        r2_gauss.append(ajuste_g.r_squared)

        # 6b. Ajuste con la forma derivada del perfil beta (comparación: no se
        # asume a ciegas que la gaussiana es la forma correcta, se verifica
        # con R^2 contra una alternativa físicamente motivada).
        ajuste_b = fit_beta_dispersion(centros, dispersion)
        rc_beta.append(ajuste_b.r_c)
        rc_beta_err.append(ajuste_b.r_c_err)
        p_beta.append(ajuste_b.p)
        sigma0_beta.append(ajuste_b.sigma0)
        r2_beta.append(ajuste_b.r_squared)

        logger.info(
            "theta=%.1f°: sigma0=%.3g rad/m2 | gauss: width=%.3g±%.3g kpc R2=%.3f "
            "| beta: rc=%.3g±%.3g kpc p=%.2f R2=%.3f",
            theta_deg, ajuste_g.sigma0,
            ajuste_g.width, ajuste_g.width_err, ajuste_g.r_squared,
            ajuste_b.r_c, ajuste_b.r_c_err, ajuste_b.p, ajuste_b.r_squared,
        )

    return {
        "theta_grados": np.array(thetas_grados),
        # Ajuste gaussiano (nombres históricos, se conservan por compatibilidad)
        "width_kpc": np.array(anchos),
        "width_err_kpc": np.array(anchos_err),
        "sigma0": np.array(sigma0s),
        "r_squared_gauss": np.array(r2_gauss),
        # Ajuste con la forma beta (ver faradaymr.analysis.fitting.beta_dispersion_model)
        "rc_beta_kpc": np.array(rc_beta),
        "rc_beta_err_kpc": np.array(rc_beta_err),
        "p_beta": np.array(p_beta),
        "sigma0_beta": np.array(sigma0_beta),
        "r_squared_beta": np.array(r2_beta),
    }


def barrer_angulos_monte_carlo(
    thetas_grados,
    ruta_resultados,
    n_semillas=50,
    use_gpu=True,
    n_bins=12
):
    """
    Orquesta múltiples realizaciones aleatorias (Monte Carlo) del barrido angular.
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando Monte Carlo: %d semillas x %d ángulos = %d corridas totales.",
                n_semillas, len(thetas_grados), n_semillas * len(thetas_grados))

    n_theta = len(thetas_grados)
    nan_col = np.full(n_theta, np.nan)

    anchos_por_theta = []
    sigma0_por_theta = []
    r2_gauss_por_theta = []
    rc_beta_por_theta = []
    r2_beta_por_theta = []

    for semilla in range(n_semillas):
        logger.info("--- Ejecutando semilla Monte Carlo %d/%d ---", semilla + 1, n_semillas)
        resultado = barrer_angulos(
            thetas_grados,
            ruta_resultados,
            use_gpu=use_gpu,
            n_bins=n_bins,
            seed=semilla
        )
        anchos_por_theta.append(resultado["width_kpc"])
        sigma0_por_theta.append(resultado["sigma0"])
        # .get(...) con relleno de NaN: mantiene compatibilidad con quien
        # mockee `barrer_angulos` devolviendo solo las llaves históricas.
        r2_gauss_por_theta.append(resultado.get("r_squared_gauss", nan_col))
        rc_beta_por_theta.append(resultado.get("rc_beta_kpc", nan_col))
        r2_beta_por_theta.append(resultado.get("r_squared_beta", nan_col))

    anchos_por_theta = np.array(anchos_por_theta)
    sigma0_por_theta = np.array(sigma0_por_theta)
    r2_gauss_por_theta = np.array(r2_gauss_por_theta)
    rc_beta_por_theta = np.array(rc_beta_por_theta)
    r2_beta_por_theta = np.array(r2_beta_por_theta)

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
