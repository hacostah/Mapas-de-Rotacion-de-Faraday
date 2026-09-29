import numpy as np
import warnings


def verificar_caja_suficiente(
    n_base: int,
    dx_base_kpc: float,
    longitud_filamento_kpc: float,
    thetas_grados: list,
    logger=None,
) -> bool:
    """
    Verifica que la caja contenga, a lo largo de la línea de visión (eje z),
    la proyección del filamento de longitud L: L |cos(theta)| <= N dx. El
    peor caso es theta = 0°. Devuelve False (y avisa) si no alcanza.
    """

    # len() en vez de `not`: acepta listas y arrays de numpy.
    if len(thetas_grados) == 0:
        return True

    lado_caja_kpc = n_base * dx_base_kpc

    thetas_rad = np.deg2rad(np.asarray(thetas_grados, dtype=float))
    profundidad_por_theta = longitud_filamento_kpc * np.abs(np.cos(thetas_rad))
    idx_peor = int(np.argmax(profundidad_por_theta))
    profundidad_necesaria = float(profundidad_por_theta[idx_peor])
    theta_peor = float(np.asarray(thetas_grados, dtype=float)[idx_peor])

    if profundidad_necesaria > lado_caja_kpc:
        msg = (
            f"La caja ({lado_caja_kpc:.0f} kpc de lado) no alcanza para "
            f"theta={theta_peor:.1f}° sin truncar el filamento en la línea de "
            f"visión (se necesitan ~{profundidad_necesaria:.0f} kpc de "
            "profundidad, longitud_filamento * |cos(theta)|). Aumentar N_BASE "
            "o DX_BASE, acortar LONGITUD_FILAMENTO, o acotar el rango de theta "
            "lejos de 0°."
        )
        if logger is not None:
            logger.warning(msg)
        else:
            warnings.warn(msg)

    return bool(profundidad_necesaria <= lado_caja_kpc)

def fraccion_varianza_retenida(d_kpc, semiprofundidad_kpc, r_core_kpc, beta):
    """
    Fracción de la varianza de RM (vista lateral, paseo aleatorio:
    sigma_RM^2 ∝ integral de n_e^2 dl) que conserva una línea de visión
    truncada en |l| <= semiprofundidad, a distancia `d_kpc` del eje de un
    filamento con perfil beta, respecto de una línea de visión infinita.
    """
    from scipy.integrate import quad

    def integrando(l):
        return (1.0 + (d_kpc**2 + l**2) / r_core_kpc**2) ** (-3.0 * beta)

    truncada = quad(integrando, -semiprofundidad_kpc, semiprofundidad_kpc)[0]
    infinita = quad(integrando, -np.inf, np.inf)[0]
    return truncada / infinita


def verificar_profundidad_radial(
    n_base: int,
    dx_base_kpc: float,
    r_core_kpc: float,
    beta: float,
    dist_max_ajuste_kpc: float,
    tolerancia: float = 0.05,
    logger=None,
) -> bool:
    """
    Verifica que la caja contenga la cola radial del perfil a lo largo de la
    línea de visión (vista lateral): la fracción de la varianza de RM
    retenida en d = `dist_max_ajuste_kpc` debe ser >= 1 - `tolerancia`.
    Devuelve False (y avisa) si no.
    """

    semiprofundidad = n_base * dx_base_kpc / 2.0
    retenida = fraccion_varianza_retenida(
        dist_max_ajuste_kpc, semiprofundidad, r_core_kpc, beta
    )
    ok = retenida >= 1.0 - tolerancia
    if not ok:
        msg = (
            f"La caja (+/-{semiprofundidad:.0f} kpc) solo conserva el "
            f"{100 * retenida:.1f}% de la varianza de RM a d={dist_max_ajuste_kpc:.0f} kpc "
            f"en la vista lateral (tolerancia {100 * tolerancia:.0f}%). Aumentar "
            "N_BASE o DX_BASE, o reducir DIST_MAX_AJUSTE."
        )
        if logger is not None:
            logger.warning(msg)
        else:
            warnings.warn(msg)
    return bool(ok)
