"""
Umbral de detección de un exceso de dispersión de RM vía stacking.

Este módulo responde a la pregunta que la propuesta del proyecto promete
("proporcionar un umbral teórico claro para las futuras campañas de
observación profunda") y que el resto del pipeline no calculaba: dado el
sigma0 que predice el modelo (la dispersión de RM sobre el eje del
filamento) y el nivel de dispersión de RM extragaláctica de fondo medido en
campañas reales (POSSUM/Stuardi et al. 2026 reporta ~7 rad/m² en el
filamento A3667/3651), ¿cuántas fuentes de fondo N hacen falta en un
experimento de stacking para poder distinguir ese exceso del ruido?
"""

from __future__ import annotations

import numpy as np


def n_fuentes_necesarias(
    sigma0_rad_m2: float,
    sigma_fondo_rad_m2: float = 7.0,
    k_sigma: float = 1.0,
) -> float:
    """
    Número de fuentes de fondo N necesarias, en un experimento de stacking,
    para detectar un exceso de varianza de RM de amplitud `sigma0_rad_m2`
    sobre un fondo de dispersión extragaláctica `sigma_fondo_rad_m2`, con
    significancia `k_sigma`.

    Derivación: para una variable gaussiana de varianza real sigma^2 medida
    con N muestras independientes, el error estándar de la varianza
    MUESTRAL es aproximadamente sigma^2 * sqrt(2/N) (resultado estándar para
    la varianza muestral de una normal). Para que el exceso de varianza que
    predice el modelo, `sigma0^2`, sea distinguible de ese error a un nivel
    `k_sigma`, se necesita:

        sigma0^2  >=  k_sigma * sigma_fondo^2 * sqrt(2/N)

    despejando:

        N  >=  2 * k_sigma^2 * (sigma_fondo^2 / sigma0^2)^2

    Es una estimación de orden de magnitud (asume ruido gaussiano e ignora
    sistemáticos, como la propia complejidad del foreground galáctico que
    POSSUM reporta como su limitación principal) pensada para responder
    "¿es esto detectable con un filamento individual o hace falta stacking
    de una muestra grande?", no para diseñar la campaña observacional real.

    Parámetros
    ----------
    sigma0_rad_m2 : float
        Amplitud del exceso de dispersión de RM que predice el modelo
        (sigma0 en el eje del filamento), en rad/m².
    sigma_fondo_rad_m2 : float
        Dispersión de RM extragaláctica de fondo (default: 7 rad/m², el
        valor típico citado por Stuardi et al. 2026 para fuentes de fondo a
        1.4 GHz, consistente con Oppermann et al. 2015 y Schnitzeler et al.
        2019).
    k_sigma : float
        Nivel de significancia deseado (1.0 = detección marginal a 1 sigma,
        el mínimo para que el exceso supere el error de la medición; 3.0
        para una detección convincente a 3 sigma, etc.).

    Devuelve
    --------
    N : float
        Número de fuentes necesario (no se redondea; quien llama decide si
        conviene `math.ceil` o dejarlo como orden de magnitud).
    """
    if sigma0_rad_m2 <= 0:
        return float("inf")
    razon = (sigma_fondo_rad_m2 ** 2) / (sigma0_rad_m2 ** 2)
    return 2.0 * (k_sigma ** 2) * (razon ** 2)


def resumen_umbral_deteccion(
    theta_grados: np.ndarray,
    sigma0_rad_m2: np.ndarray,
    sigma_fondo_rad_m2: float = 7.0,
    k_sigma: float = 1.0,
) -> dict:
    """
    Aplica `n_fuentes_necesarias` a un barrido completo en theta y devuelve
    un resumen (mínimo, máximo, y por ángulo) listo para reportar.
    """
    theta_grados = np.asarray(theta_grados, dtype=float)
    sigma0_rad_m2 = np.asarray(sigma0_rad_m2, dtype=float)
    n_por_theta = np.array([
        n_fuentes_necesarias(s0, sigma_fondo_rad_m2, k_sigma) for s0 in sigma0_rad_m2
    ])
    idx_min = int(np.argmin(n_por_theta))
    idx_max = int(np.argmax(n_por_theta))
    return {
        "theta_grados": theta_grados,
        "sigma0_rad_m2": sigma0_rad_m2,
        "n_por_theta": n_por_theta,
        "sigma_fondo_rad_m2": sigma_fondo_rad_m2,
        "k_sigma": k_sigma,
        "n_minimo": float(n_por_theta[idx_min]),
        "theta_n_minimo": float(theta_grados[idx_min]),
        "n_maximo": float(n_por_theta[idx_max]),
        "theta_n_maximo": float(theta_grados[idx_max]),
    }
