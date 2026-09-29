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
    Verifica que la caja cúbica sea suficientemente profunda a lo largo de
    la línea de visión (siempre el eje z de la caja: `model.py` rota el
    OBJETO, no la línea de visión) para contener la proyección del
    filamento finito de longitud `longitud_filamento_kpc`.

    Nota de corrección física (versión anterior de esta función): un
    segmento de longitud L centrado en el origen, con su eje inclinado un
    ángulo theta respecto a z, tiene una extensión a lo largo de z de
    `L * |cos(theta)|` -no `L / cos(theta)`. El peor caso (mayor
    profundidad requerida) ocurre entonces en theta CERCANO A 0° (el
    filamento visto "de frente", casi paralelo a la línea de visión, donde
    cos(theta) es máximo), no cerca de 90° como asumía la versión anterior
    de este chequeo. A theta=90° (filamento "de lado") la profundidad
    requerida es ~0: la línea de visión solo atraviesa el perfil radial
    transversal, sobre una escala de unos pocos radios de núcleo, muy por
    debajo de la longitud total del filamento.

    Esto importa en la práctica porque `construir_escenario` (ver
    `model.py`) trunca la densidad más allá de +/- longitud/2 a lo largo
    del eje del filamento; si la caja es más angosta que esa proyección,
    la simulación además corta el filamento en la línea de visión de forma
    dependiente del tamaño de malla, contaminando la comparación entre
    corridas.
    """
    # `len(...) == 0` en vez de `not thetas_grados`: con un array de numpy
    # de más de un elemento, `not array` lanza ValueError ("truth value of
    # an array... is ambiguous"). El `__main__` actual de
    # run_barrido_theta.py convierte a lista antes de llamar aquí, así que
    # no se disparaba en la práctica, pero cualquier llamada directa con un
    # array (un notebook, un test) sí lo hacía.
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
    Complemento de `verificar_caja_suficiente`: aquella comprueba que quepa
    la LONGITUD del filamento; esta comprueba que quepa su COLA RADIAL.

    Con el filamento de lado, la línea de visión de un píxel a distancia d
    del eje atraviesa el perfil beta a radios sqrt(d^2 + l^2). Si la caja
    corta esa integral en |l| = N*dx/2, la varianza de RM en los bins
    exteriores sale subestimada y el perfil transversal queda más empinado
    de lo que es (p ajustado mayor que el valor analítico). Ocurre solo en
    las vistas oblicuas/laterales, así que además sesga la comparación entre
    ángulos.

    Devuelve False (y avisa) si en `dist_max_ajuste_kpc` se pierde más de
    `tolerancia` de la varianza.
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
