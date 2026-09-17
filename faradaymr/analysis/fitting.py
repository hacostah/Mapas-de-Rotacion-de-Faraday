"""
Módulo para el ajuste de funciones analíticas a perfiles estadísticos,
especialmente diseñado para la dispersión transversal de la Medida de Rotación (RM).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit

def gaussian_model(d: np.ndarray | float, sigma0: float, width: float) -> np.ndarray | float:
    """
    Forma funcional esperada para la dispersión transversal de RM asumiendo
    una caída idealizada: sigma_RM(d) = sigma0 * exp(-d^2 / (2*width^2)).

    Parámetros
    ----------
    d : np.ndarray o float
        Distancia transversal (impact parameter) desde el eje del filamento.
    sigma0 : float
        Dispersión de RM máxima sobre el eje del filamento (d=0).
    width : float
        Escala característica de caída. Define el "umbral teórico" de la 
        firma observacional del filamento.
    """
    return sigma0 * np.exp(-(d**2) / (2.0 * width**2))

@dataclass
class GaussianFitResult:
    sigma0: float
    width: float
    sigma0_err: float
    width_err: float
    r_squared: float

def fit_transverse_dispersion(centros: np.ndarray, sigma_rm: np.ndarray, p0: tuple[float, float] | None = None) -> GaussianFitResult:
    """
    Ajusta `gaussian_model` a un perfil sigma_RM(d) ya calculado.
    Filtra automáticamente los valores NaN (bins vacíos) antes de realizar el ajuste.

    Parámetros
    ----------
    centros : np.ndarray
        Arreglo con los centros de los bins (distancias).
    sigma_rm : np.ndarray
        Arreglo con la dispersión de RM medida en cada bin. Puede contener NaNs.
    p0 : tuple, opcional
        Estimación inicial para (sigma0, width). Si es None, se infiere de los datos.

    Retorna
    -------
    GaussianFitResult
        Objeto con los parámetros ajustados, sus errores y la bondad del ajuste (R^2).
    """
    centros = np.asarray(centros, dtype=float)
    sigma_rm = np.asarray(sigma_rm, dtype=float)
    
    # Filtrar bins sin datos (NaNs o infinitos)
    valido = np.isfinite(sigma_rm)
    if valido.sum() < 3:  # Se necesitan al menos 3 puntos para ajustar 2 parámetros con cierta validez
        raise ValueError(
            f"No hay suficientes bins válidos (no-NaN) para ajustar la gaussiana. "
            f"Se requieren al menos 3, pero se encontraron {valido.sum()}."
        )
        
    centros_v = centros[valido]
    sigma_v = sigma_rm[valido]

    if p0 is None:
        distancias_positivas = np.abs(centros_v[centros_v != 0])
        ancho_inicial = float(distancias_positivas.mean()) if distancias_positivas.size else 1.0
        p0 = (float(np.max(sigma_v)), ancho_inicial)

    # Forzar que los parámetros sean positivos (bounds)
    limites = (0.0, np.inf)

    parametros, covarianza = curve_fit(
        gaussian_model, 
        centros_v, 
        sigma_v, 
        p0=p0, 
        bounds=limites
    )
    
    sigma0, width = parametros
    errores = np.sqrt(np.diag(covarianza))

    # Cálculo de R cuadrado para evaluar la bondad del ajuste
    prediccion = gaussian_model(centros_v, *parametros)
    ss_res = np.sum((sigma_v - prediccion) ** 2)
    ss_tot = np.sum((sigma_v - np.mean(sigma_v)) ** 2)
    
    r_cuadrado = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return GaussianFitResult(
        sigma0=float(sigma0),
        width=float(width),
        sigma0_err=float(errores[0]),
        width_err=float(errores[1]),
        r_squared=float(r_cuadrado),
    )