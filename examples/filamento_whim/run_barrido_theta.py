"""
Barrido en ángulo de visión theta del filamento WHIM (resultado central del
Proyecto II).

Qué hace
--------
1. Para cada theta construye la densidad del filamento finito y calcula:
   - la máscara del cuerpo del filamento sobre el cielo,
   - el perfil transversal ESPERADO sigma_RM(d) (analítico, sin ruido; ver
     `faradaymr.analysis.expected`), que es la predicción del toy model.
2. Para cada semilla genera UN campo turbulento (no depende de theta) y mide
   el perfil sigma_RM(d) de cada mapa simulado con el estimador RMS
   (insesgado, ver `transverse_rm_dispersion`).
3. Apila las semillas (promedio de RM^2 por bin), ajusta el perfil apilado y
   estima errores con bootstrap sobre semillas. También guarda la dispersión
   de UNA sola realización (lo que vería un observador con un filamento).

Ajustes por theta (tanto al perfil simulado como al esperado):
- forma beta con r_c fijo (modo principal): sigma0(theta), p(theta) y el
  semiancho a media altura d_1/2(theta);
- forma beta libre (comprobación: r_c tiene que salir ~ RC);
- gaussiana (la hipótesis original de la propuesta, para comparar su R^2).

Uso (desde la raíz del repo):
    python -m examples.filamento_whim.run_barrido_theta
"""
from __future__ import annotations

import logging
import os

import astropy.units as u
import numpy as np

from examples.filamento_whim import config_fisica
from examples.filamento_whim.model import construir_escenario
from examples.filamento_whim.validacion import (
    verificar_caja_suficiente,
    verificar_profundidad_radial,
)
from faradaymr import BetaModel, get_backend
from faradaymr.analysis.expected import (
    expected_rm_dispersion_map,
    longitud_correlacion_los,
)
from faradaymr.analysis.fitting import (
    fit_beta_dispersion,
    fit_transverse_dispersion,
    p_vista_frontal,
    p_vista_lateral,
    semiancho_media_altura,
)
from faradaymr.analysis.spatial_stats import radial_profile, transverse_rm_dispersion
from faradaymr.backend import to_numpy
from faradaymr.fields import GaussianRandomVectorField
from faradaymr.logging_config import configurar_logging
from faradaymr.los import rotation_measure
from faradaymr.simulation.geometry import (
    filament_axis_from_viewing_angle,
    filament_body_mask,
    projected_axis_distance,
    sky_footprint_mask,
)

_logger_modulo = logging.getLogger("faradaymr.filamento_whim.barrido")

# Histograma de RM/sigma_esperada (para verificar que RM es gaussiano).
BORDES_HIST_Z = np.linspace(-5.0, 5.0, 51)


def _parametros():
    """Valores desnudos de config_fisica (leídos en cada llamada para que los
    tests puedan mockear config_fisica)."""
    return dict(
        n_base=int(config_fisica.N_BASE),
        dx_kpc=config_fisica.DX_BASE.to_value(u.kpc),
        dl_pc=config_fisica.DX_BASE.to_value(u.pc),
        longitud_kpc=config_fisica.LONGITUD_FILAMENTO.to_value(u.kpc),
        rc_kpc=config_fisica.RC.to_value(u.kpc),
        n0_cm3=config_fisica.N0.to_value(u.cm**-3),
        beta=float(config_fisica.BETA),
        b0_ug=config_fisica.B0.to_value(u.microgauss),
        dist_max_kpc=config_fisica.DIST_MAX_AJUSTE.to_value(u.kpc),
        lambda_min_kpc=config_fisica.LAMBDA_MIN.to_value(u.kpc),
        lambda_max_kpc=config_fisica.LAMBDA_MAX.to_value(u.kpc),
        n_spec=float(config_fisica.N_SPEC),
    )


def _validar_caja(par, thetas_grados, logger):
    ok_longitud = verificar_caja_suficiente(
        n_base=par["n_base"], dx_base_kpc=par["dx_kpc"],
        longitud_filamento_kpc=par["longitud_kpc"],
        thetas_grados=thetas_grados, logger=logger,
    )
    ok_radial = verificar_profundidad_radial(
        n_base=par["n_base"], dx_base_kpc=par["dx_kpc"], r_core_kpc=par["rc_kpc"],
        beta=par["beta"], dist_max_ajuste_kpc=par["dist_max_kpc"], logger=logger,
    )
    if not (ok_longitud and ok_radial):
        raise ValueError(
            "Caja insuficiente para el barrido angular (ver el warning de "
            "validacion.py): el filamento o su cola radial quedan truncados "
            "en la línea de visión, lo que sesga la comparación entre ángulos."
        )


def preparar_geometria(thetas_grados, bordes, generador, par, xp):
    """
    Para cada theta: densidad `ne` (en el backend `xp`), máscara del cuerpo del
    filamento, mapa de distancia al eje proyectado, mapa de sigma_RM esperado
    y perfil esperado por bin. No depende de la semilla: se calcula una vez.
    """
    n = par["n_base"]
    vacio = xp.zeros((n, n, n), dtype=xp.float32)
    geometria = []
    for theta_deg in thetas_grados:
        eje = filament_axis_from_viewing_angle(np.deg2rad(theta_deg))
        _, _, _, ne, _, _ = construir_escenario(
            n_spec=par["n_spec"], b0_microgauss=par["b0_ug"], axis_direction=eje,
            use_gpu=(xp is not np), longitud_filamento_kpc=par["longitud_kpc"],
            campo_b=(vacio, vacio, vacio), dx_kpc=par["dx_kpc"],
            density_profile=BetaModel(n0=par["n0_cm3"], r_core=par["rc_kpc"], beta=par["beta"]),
        )
        mascara = to_numpy(sky_footprint_mask(ne, xp=xp)) & filament_body_mask(
            (n, n), eje, par["dx_kpc"], par["longitud_kpc"], par["rc_kpc"],
        )
        distancia = projected_axis_distance((n, n), eje, par["dx_kpc"])
        sigma_esp = to_numpy(
            expected_rm_dispersion_map(ne, generador, par["b0_ug"], par["dl_pc"], xp=xp)
        )
        _, var_esp = radial_profile(
            sigma_esp**2, distancia, bordes, statistic="mean", mascara=mascara,
        )
        en_ventana = mascara & (distancia <= bordes[-1])
        geometria.append(dict(
            theta=float(theta_deg), eje=eje, ne=ne, mascara=mascara,
            en_ventana=en_ventana, sigma_esperada=sigma_esp,
            perfil_esperado=np.sqrt(var_esp),
        ))
    return geometria


def _ajustes(centros, perfil, rc_kpc, errores=None):
    """Los tres ajustes de un perfil, como diccionario plano."""
    b = fit_beta_dispersion(centros, perfil, rc_fijo=rc_kpc, errores=errores)
    libre = fit_beta_dispersion(centros, perfil, errores=errores)
    g = fit_transverse_dispersion(centros, perfil, errores=errores)
    return dict(
        sigma0=b.sigma0, sigma0_err=b.sigma0_err, p=b.p, p_err=b.p_err,
        hwhm=semiancho_media_altura(rc_kpc, b.p), r2_beta=b.r_squared,
        rc_libre=libre.r_c, p_libre=libre.p, r2_libre=libre.r_squared,
        w_gauss=g.width, sigma0_gauss=g.sigma0, r2_gauss=g.r_squared,
    )


def analizar_perfiles(perfiles, centros, perfiles_esperados, rc_kpc,
                      n_bootstrap=300, semilla_bootstrap=12345):
    """
    Analiza los perfiles medidos, forma (n_semillas, n_theta, n_bins):

    - apila: sigma_apilado = sqrt(<sigma_s^2>_s) por bin (equivale a
      promediar RM^2 de todos los píxeles de todas las semillas);
    - errores por bin y de los parámetros con bootstrap sobre semillas;
    - ajusta el perfil apilado (ponderado) y el esperado;
    - dispersión de UNA realización: ajuste semilla por semilla.

    Función pura de numpy (sin física): la reusan los tests y plots.py.
    """
    perfiles = np.asarray(perfiles, dtype=float)
    n_semillas, n_theta, _ = perfiles.shape
    rng = np.random.default_rng(semilla_bootstrap)

    apilado = np.sqrt(np.nanmean(perfiles**2, axis=0))
    claves = ["sigma0", "p", "hwhm", "w_gauss"]
    boot_perfiles = np.full((n_bootstrap,) + apilado.shape, np.nan)
    boot_par = {k: np.full((n_bootstrap, n_theta), np.nan) for k in claves}
    if n_semillas > 1:
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_semillas, n_semillas)
            boot_perfiles[b] = np.sqrt(np.nanmean(perfiles[idx] ** 2, axis=0))
            for t in range(n_theta):
                try:
                    a = _ajustes(centros, boot_perfiles[b, t], rc_kpc)
                except (RuntimeError, ValueError):
                    continue
                for k in claves:
                    boot_par[k][b, t] = a[k]
        apilado_err = np.nanstd(boot_perfiles, axis=0, ddof=1)
    else:
        apilado_err = np.full_like(apilado, np.nan)

    salida = {
        "perfil_apilado": apilado,
        "perfil_apilado_err": apilado_err,
        "perfil_esperado": np.asarray(perfiles_esperados, dtype=float),
    }
    por_theta_mc, por_theta_esp = [], []
    for t in range(n_theta):
        errores = apilado_err[t] if n_semillas > 1 else None
        por_theta_mc.append(_ajustes(centros, apilado[t], rc_kpc, errores))
        por_theta_esp.append(_ajustes(centros, perfiles_esperados[t], rc_kpc))
    for k in por_theta_mc[0]:
        salida[f"mc_{k}"] = np.array([a[k] for a in por_theta_mc])
        salida[f"esp_{k}"] = np.array([a[k] for a in por_theta_esp])
    for k in claves:
        salida[f"mc_{k}_err"] = (
            np.nanstd(boot_par[k], axis=0, ddof=1) if n_semillas > 1
            else np.full(n_theta, np.nan)
        )

    # Una sola realización: dispersión entre semillas del ajuste individual.
    s0_ind = np.full((n_semillas, n_theta), np.nan)
    p_ind = np.full((n_semillas, n_theta), np.nan)
    for s in range(n_semillas):
        for t in range(n_theta):
            try:
                b = fit_beta_dispersion(centros, perfiles[s, t], rc_fijo=rc_kpc)
            except (RuntimeError, ValueError):
                continue
            s0_ind[s, t], p_ind[s, t] = b.sigma0, b.p
    salida["una_realizacion_sigma0_std"] = np.nanstd(s0_ind, axis=0)
    salida["una_realizacion_p_std"] = np.nanstd(p_ind, axis=0)
    return salida


def barrer_angulos_monte_carlo(
    thetas_grados,
    ruta_resultados,
    n_semillas=None,
    use_gpu=None,
    n_bins=None,
    n_bootstrap=300,
    semilla_inicial=0,
    logger=None,
):
    """
    Barrido Monte Carlo completo. Devuelve un diccionario listo para
    `np.savez` (ver el docstring del módulo para el contenido).
    """
    if logger is None:
        os.makedirs(os.path.join(ruta_resultados, "logs"), exist_ok=True)
        logger = configurar_logging(directorio_logs=os.path.join(ruta_resultados, "logs"))
    if n_semillas is None:
        n_semillas = int(config_fisica.N_SEMILLAS)
    if n_bins is None:
        n_bins = int(config_fisica.N_BORDES_PERFIL)
    thetas_grados = [float(t) for t in thetas_grados]

    par = _parametros()
    xp = get_backend(use_gpu)
    _validar_caja(par, thetas_grados, logger)

    dist_max = min(par["dist_max_kpc"], par["n_base"] / 2 * par["dx_kpc"])
    bordes = np.linspace(0.0, dist_max, n_bins)
    centros = 0.5 * (bordes[:-1] + bordes[1:])

    generador = GaussianRandomVectorField(
        n=par["n_base"], dx=par["dx_kpc"], spectral_index=par["n_spec"],
        scale_min=par["lambda_min_kpc"], scale_max=par["lambda_max_kpc"],
    )
    lambda_corr = longitud_correlacion_los(generador)
    logger.info(
        "Barrido: %d ángulos x %d semillas; ventana 0-%.0f kpc en %d bins; "
        "longitud de correlación de B_z en la LoS: %.0f kpc.",
        len(thetas_grados), n_semillas, dist_max, n_bins - 1, lambda_corr,
    )

    geometria = preparar_geometria(thetas_grados, bordes, generador, par, xp)

    n_theta, n_b = len(thetas_grados), len(centros)
    perfiles = np.full((n_semillas, n_theta, n_b), np.nan)
    hist_z = np.zeros((n_theta, len(BORDES_HIST_Z) - 1))
    momentos = np.zeros((n_theta, 5))  # n, sum z, z^2, z^3, z^4

    for k in range(n_semillas):
        bz = _campo_bz(generador, semilla_inicial + k, par["b0_ug"], xp)
        for t, geo in enumerate(geometria):
            rm = to_numpy(rotation_measure(geo["ne"], bz, par["dl_pc"], xp=xp))
            _, perfil = transverse_rm_dispersion(
                rm, geo["eje"], par["dx_kpc"], bordes,
                footprint_mask=geo["mascara"], statistic="rms",
            )
            perfiles[k, t] = to_numpy(perfil)
            z = rm[geo["en_ventana"]] / geo["sigma_esperada"][geo["en_ventana"]]
            hist_z[t] += np.histogram(z, bins=BORDES_HIST_Z)[0]
            momentos[t] += [z.size, z.sum(), (z**2).sum(), (z**3).sum(), (z**4).sum()]
        logger.info("Semilla %d/%d lista.", k + 1, n_semillas)

    analisis = analizar_perfiles(
        perfiles, centros, np.array([g["perfil_esperado"] for g in geometria]),
        par["rc_kpc"], n_bootstrap=n_bootstrap if n_semillas > 1 else 0,
    )

    n_z = momentos[:, 0]
    media = momentos[:, 1] / n_z
    m2 = momentos[:, 2] / n_z - media**2
    m3 = momentos[:, 3] / n_z - 3 * media * momentos[:, 2] / n_z + 2 * media**3
    m4 = (momentos[:, 4] / n_z - 4 * media * momentos[:, 3] / n_z
          + 6 * media**2 * momentos[:, 2] / n_z - 3 * media**4)

    resultados = dict(
        theta_grados=np.array(thetas_grados),
        bordes_kpc=bordes, centros_kpc=centros,
        perfiles_rms=perfiles,
        rc_kpc=par["rc_kpc"], beta=par["beta"], b0_ng=1e3 * par["b0_ug"],
        n0_cm3=par["n0_cm3"],
        longitud_kpc=par["longitud_kpc"], lambda_corr_kpc=lambda_corr,
        p_frontal=p_vista_frontal(par["beta"]), p_lateral=p_vista_lateral(par["beta"]),
        n_semillas=n_semillas,
        hist_bordes_z=BORDES_HIST_Z, hist_z=hist_z,
        z_media=media, z_std=np.sqrt(m2), z_asimetria=m3 / m2**1.5,
        z_exceso_curtosis=m4 / m2**2 - 3.0,
        **analisis,
    )
    for t, th in enumerate(thetas_grados):
        logger.info(
            "theta=%4.1f°: sigma0=%.4f±%.4f (esp %.4f) rad/m2 | p=%.3f±%.3f "
            "(esp %.3f) | d_1/2=%.0f kpc | R2 beta=%.3f gauss=%.3f",
            th, resultados["mc_sigma0"][t], resultados["mc_sigma0_err"][t],
            resultados["esp_sigma0"][t], resultados["mc_p"][t],
            resultados["mc_p_err"][t], resultados["esp_p"][t],
            resultados["mc_hwhm"][t], resultados["mc_r2_beta"][t],
            resultados["mc_r2_gauss"][t],
        )
    return resultados


def _campo_bz(generador, semilla, b0_ug, xp):
    """B_z de la realización `semilla`, normalizada a <B^2> = B0^2. Es el
    único componente que entra en RM (la LoS es el eje z de la caja) y no
    depende de theta: se genera una vez por semilla para todo el barrido."""
    bx, by, bz = generador.sample(
        use_gpu=(xp is not np), rng=np.random.RandomState(semilla)
    )
    _, _, bz = GaussianRandomVectorField.normalize_to_rms(bx, by, bz, b0_ug, xp=xp)
    return bz


def barrer_angulos(thetas_grados, ruta_resultados, use_gpu=None, n_bins=None,
                   seed=0, logger=None):
    """Barrido con UNA sola realización del campo (semilla `seed`): útil
    para pruebas rápidas. Sin bootstrap (los errores quedan en NaN)."""
    return barrer_angulos_monte_carlo(
        thetas_grados, ruta_resultados, n_semillas=1, use_gpu=use_gpu,
        n_bins=n_bins, n_bootstrap=0, semilla_inicial=seed, logger=logger,
    )


if __name__ == "__main__":
    ruta_salida = os.path.join(os.path.dirname(__file__), "results", "barrido_theta")
    os.makedirs(ruta_salida, exist_ok=True)
    resultados = barrer_angulos_monte_carlo(
        thetas_grados=config_fisica.THETAS_BARRIDO, ruta_resultados=ruta_salida,
    )
    archivo_npz = os.path.join(ruta_salida, "barrido_theta_mc.npz")
    np.savez(archivo_npz, **resultados)
    print(f"Barrido Monte Carlo terminado. Resultados en {archivo_npz}")
