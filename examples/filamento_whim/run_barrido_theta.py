from __future__ import annotations

import os
import sys
import numpy as np
import astropy.units as u

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


def barrer_angulos(thetas_grados, ruta_resultados, use_gpu=False, n_bins=12, seed=0):
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
        raise ValueError("Caja insuficiente para el barrido angular.")

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


if __name__ == "__main__":
    import os
    import numpy as np
    
    ruta_salida = os.path.join(os.path.dirname(__file__), "results", "barrido_theta")
    os.makedirs(ruta_salida, exist_ok=True)
    
    try:
        resultados = barrer_angulos(
            # Definimos los ángulos 
            thetas_grados=np.linspace(0, 85, 10), 
            ruta_resultados=ruta_salida,
            use_gpu=False,
        )
        np.savez(os.path.join(ruta_salida, "barrido_theta.npz"), **resultados)
        print("Barrido finalizado correctamente.")
    except Exception as e:
        print(f"Error durante el barrido: {e}")
        import sys
        sys.exit(1)