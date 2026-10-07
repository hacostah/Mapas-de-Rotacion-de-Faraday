"""
Umbral de detección de un exceso de dispersión de RM (región del filamento
contra región de control, N fuentes cada una):

    sigma_tot^2 = sigma_intr^2 + sigma_med^2
    N >= c k^2 (sigma_tot^2 / sigma0^2)^2,     c = 4 (con control) o 2 (sin)
    B_min = B_modelo * sqrt(k sigma_tot^2 sqrt(c/N)) / sigma0_modelo

sigma_intr = 7 rad/m² y sigma_med = 12 rad/m² (POSSUM), de Stuardi et al.
(2026). Supuestos: RM gaussianos por fuente, sigma0 del eje del filamento,
sin foreground galáctico.
"""

from __future__ import annotations

import numpy as np

SIGMA_INTRINSECA_RAD_M2 = 7.0   # Stuardi et al. 2026 (Oppermann et al.)
SIGMA_MEDICION_POSSUM_RAD_M2 = 12.0  # Stuardi et al. 2026, error típico POSSUM
N_FUENTES_STUARDI = 54          # fuentes detrás del filamento A3667/3651


def _sigma_total2(sigma_fondo, sigma_medicion):
    return sigma_fondo**2 + sigma_medicion**2


def n_fuentes_necesarias(
    sigma0_rad_m2: float,
    sigma_fondo_rad_m2: float = SIGMA_INTRINSECA_RAD_M2,
    k_sigma: float = 1.0,
    sigma_medicion_rad_m2: float = 0.0,
    muestra_control: bool = True,
) -> float:
    """
    N de fuentes (por muestra) para detectar un exceso de dispersión
    `sigma0_rad_m2` a `k_sigma` sigmas. Ver la derivación en el módulo.
    """
    if sigma0_rad_m2 <= 0:
        return float("inf")
    c = 4.0 if muestra_control else 2.0
    razon = _sigma_total2(sigma_fondo_rad_m2, sigma_medicion_rad_m2) / sigma0_rad_m2**2
    return c * k_sigma**2 * razon**2


def sigma0_minimo_detectable(
    n_fuentes,
    sigma_fondo_rad_m2: float = SIGMA_INTRINSECA_RAD_M2,
    k_sigma: float = 1.0,
    sigma_medicion_rad_m2: float = 0.0,
    muestra_control: bool = True,
):
    """Inverso de `n_fuentes_necesarias`: el sigma0 mínimo detectable con N fuentes."""
    c = 4.0 if muestra_control else 2.0
    n = np.asarray(n_fuentes, dtype=float)
    return np.sqrt(
        k_sigma * _sigma_total2(sigma_fondo_rad_m2, sigma_medicion_rad_m2) * np.sqrt(c / n)
    )


def campo_minimo_detectable(
    n_fuentes,
    sigma0_modelo_rad_m2: float,
    b_modelo: float,
    **kwargs,
):
    """
    Campo mínimo detectable (en las unidades de `b_modelo`) con N fuentes,
    usando que sigma0 es lineal en B: B_min = b_modelo * sigma0_min / sigma0_modelo.
    """
    return b_modelo * sigma0_minimo_detectable(n_fuentes, **kwargs) / sigma0_modelo_rad_m2


def resumen_umbral_deteccion(
    theta_grados: np.ndarray,
    sigma0_rad_m2: np.ndarray,
    sigma_fondo_rad_m2: float = SIGMA_INTRINSECA_RAD_M2,
    k_sigma: float = 1.0,
    sigma_medicion_rad_m2: float = 0.0,
    muestra_control: bool = True,
) -> dict:
    """
    Aplica `n_fuentes_necesarias` a un barrido en theta y devuelve el
    mínimo, el máximo y el valor por ángulo.
    """
    theta_grados = np.asarray(theta_grados, dtype=float)
    sigma0_rad_m2 = np.asarray(sigma0_rad_m2, dtype=float)
    n_por_theta = np.array([
        n_fuentes_necesarias(s0, sigma_fondo_rad_m2, k_sigma,
                             sigma_medicion_rad_m2, muestra_control)
        for s0 in sigma0_rad_m2
    ])
    idx_min = int(np.argmin(n_por_theta))
    idx_max = int(np.argmax(n_por_theta))
    return {
        "theta_grados": theta_grados,
        "sigma0_rad_m2": sigma0_rad_m2,
        "n_por_theta": n_por_theta,
        "sigma_fondo_rad_m2": sigma_fondo_rad_m2,
        "sigma_medicion_rad_m2": sigma_medicion_rad_m2,
        "k_sigma": k_sigma,
        "n_minimo": float(n_por_theta[idx_min]),
        "theta_n_minimo": float(theta_grados[idx_min]),
        "n_maximo": float(n_por_theta[idx_max]),
        "theta_n_maximo": float(theta_grados[idx_max]),
    }
