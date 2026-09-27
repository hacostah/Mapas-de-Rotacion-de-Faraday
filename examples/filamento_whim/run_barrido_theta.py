from __future__ import annotations

import os
import sys
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
from faradaymr.analysis.fitting import fit_transverse_dispersion
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
        # raise ValueError("Caja insuficiente para el barrido angular.")

    logger.info(f"Iniciando barrido para {len(thetas_grados)} ángulos: {thetas_grados}")

    anchos, anchos_err, sigma0s = [], [], []

    for theta_deg in thetas_grados:
        theta = np.deg2rad(theta_deg)
        axis_direction = [np.sin(theta), 0.0, np.cos(theta)]

        # 3. Construcción del escenario físico
        bx, by, bz, ne, ne_rel, r = construir_escenario(
            n_spec=config_fisica.N_SPEC,
            b0_microgauss=b0_microgauss,
            axis_direction=axis_direction,
            use_gpu=use_gpu,
            rng=np.random.RandomState(seed),
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

        # 5. Análisis de dispersión transversal
        distancia_max = (n_base / 2) * dx_base_kpc
        bordes = np.linspace(0.0, distancia_max, n_bins)
        centros, dispersion = transverse_rm_dispersion(
            resultado.rm_map, axis_direction, dx_base_kpc, bordes, xp=xp
        )

        # 6. Ajuste
        ajuste = fit_transverse_dispersion(centros, dispersion)
        anchos.append(ajuste.width)
        anchos_err.append(ajuste.width_err)
        sigma0s.append(ajuste.sigma0)

        logger.info(
            "theta=%.1f°: sigma0=%.3g rad/m2, width=%.3g±%.3g kpc, R2=%.3f",
            theta_deg, ajuste.sigma0, ajuste.width, ajuste.width_err, ajuste.r_squared,
        )

    return {
        "theta_grados": np.array(thetas_grados),
        "width_kpc": np.array(anchos),
        "width_err_kpc": np.array(anchos_err),
        "sigma0": np.array(sigma0s),
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
    
    anchos_por_theta = []
    sigma0_por_theta = []
    
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

    # Convertir a matrices de NumPy, forma: (n_semillas, n_thetas)
    anchos_por_theta = np.array(anchos_por_theta)
    sigma0_por_theta = np.array(sigma0_por_theta)
    
    logger.info("Monte Carlo finalizado exitosamente.")

    return {
        "theta_grados": np.array(thetas_grados),
        "width_medio_kpc": anchos_por_theta.mean(axis=0),
        "width_std_kpc": anchos_por_theta.std(axis=0),
        "sigma0_medio": sigma0_por_theta.mean(axis=0),
        "sigma0_std": sigma0_por_theta.std(axis=0),
    }

if __name__ == "__main__":
    import config
    
    # Leemos los ángulos definidos en config.py
    angulos_numpy = config.THETAS_BARRIDO if hasattr(config, "THETAS_BARRIDO") else np.linspace(0, 85, 10)
    angulos = [float(x) for x in angulos_numpy]
    
    ruta_salida = os.path.join(os.path.dirname(__file__), "results", "barrido_theta")
    os.makedirs(ruta_salida, exist_ok=True)
    
    try:
        # Reemplazamos la llamada a barrer_angulos por la versión Monte Carlo
        resultados = barrer_angulos_monte_carlo(
            thetas_grados=angulos,
            ruta_resultados=ruta_salida,
            n_semillas=50,  # Decidido explícitamente para el congreso
            use_gpu=True,
        )
        
        archivo_npz = os.path.join(ruta_salida, "barrido_theta_mc.npz")
        np.savez(archivo_npz, **resultados)
        print(f"Barrido Monte Carlo finalizado correctamente. Resultados guardados en {archivo_npz}")
        
    except Exception as e:
        print(f"Error durante el barrido Monte Carlo: {e}")
        import sys
        sys.exit(1)