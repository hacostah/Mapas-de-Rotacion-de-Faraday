"""
Módulo para el ajuste de funciones analíticas a perfiles estadísticos,
especialmente diseñado para la dispersión transversal de la Medida de Rotación (RM).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit

def _errores_validos(errores, valido):
    """Recorta los errores por bin a los bins válidos; None si no hay o no sirven."""
    if errores is None:
        return None
    if hasattr(errores, "get"):
        errores = errores.get()
    errores = np.asarray(errores, dtype=float)[valido]
    if not np.all(np.isfinite(errores)) or np.any(errores <= 0):
        return None
    return errores


def _chi2_reducido(sigma_v, prediccion, errores_v, n_parametros):
    """
    Chi^2 reducido, sum(((dato - modelo) / error)^2) / (N - n_par). NaN si
    no hay errores por bin. Es la métrica coherente con un ajuste ponderado
    (el R^2 de este módulo es sin pesos).
    """
    grados_libertad = sigma_v.size - n_parametros
    if errores_v is None or grados_libertad <= 0:
        return float("nan")
    return float(np.sum(((sigma_v - prediccion) / errores_v) ** 2) / grados_libertad)


def gaussian_model(d: np.ndarray | float, sigma0: float, width: float) -> np.ndarray | float:
    """
    Forma gaussiana: sigma_RM(d) = sigma0 * exp(-d^2 / (2*width^2)).

    Parámetros
    ----------
    d : np.ndarray o float
        Distancia transversal (impact parameter) desde el eje del filamento.
    sigma0 : float
        Dispersión de RM máxima sobre el eje del filamento (d=0).
    width : float
        Escala característica de caída.
    """
    return sigma0 * np.exp(-(d**2) / (2.0 * width**2))

@dataclass
class GaussianFitResult:
    sigma0: float
    width: float
    sigma0_err: float
    width_err: float
    r_squared: float
    chi2_red: float = float("nan")

def fit_transverse_dispersion(
    centros: np.ndarray,
    sigma_rm: np.ndarray,
    p0: tuple[float, float] | None = None,
    errores: np.ndarray | None = None,
) -> GaussianFitResult:
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
    errores : np.ndarray, opcional
        Incertidumbre de cada bin (p.ej. bootstrap sobre semillas). Si se da,
        el ajuste es por mínimos cuadrados PONDERADOS y los errores de los
        parámetros quedan en unidades absolutas.

    Retorna
    -------
    GaussianFitResult
        Objeto con los parámetros ajustados, sus errores y la bondad del
        ajuste: R^2 (sin pesos) y, si se dieron `errores`, chi^2 reducido.
        Para comparar formas funcionales tras un ajuste ponderado, usar
        `chi2_red`, no `r_squared`.
    """
    if hasattr(centros, 'get'):
        centros = centros.get()
    if hasattr(sigma_rm, 'get'):
        sigma_rm = sigma_rm.get()

    centros = np.asarray(centros, dtype=float)
    sigma_rm = np.asarray(sigma_rm, dtype=float)
    
    # Filtrar bins sin datos (NaNs o infinitos)
    valido = np.isfinite(sigma_rm)
    sigma_err_v = _errores_validos(errores, valido)
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
        bounds=limites,
        sigma=sigma_err_v,
        absolute_sigma=sigma_err_v is not None,
        maxfev=10000,
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
        chi2_red=_chi2_reducido(sigma_v, prediccion, sigma_err_v, 2),
    )


def beta_dispersion_model(
    d: np.ndarray | float, sigma0: float, r_c: float, p: float
) -> np.ndarray | float:
    """
    Forma tipo beta: sigma_RM(d) = sigma0 * (1 + d^2/r_c^2)^(-p).

    Parámetros
    ----------
    d : distancia transversal al eje.
    sigma0 : dispersión de RM en el eje (d=0).
    r_c : escala característica de caída (análoga al radio de núcleo).
    p : índice de la caída (límites en `p_vista_frontal` / `p_vista_lateral`).
    """
    return sigma0 * (1.0 + (d**2) / (r_c**2)) ** (-p)


@dataclass
class BetaFitResult:
    sigma0: float
    r_c: float
    p: float
    sigma0_err: float
    r_c_err: float
    p_err: float
    r_squared: float
    chi2_red: float = float("nan")


def p_vista_lateral(beta: float) -> float:
    """
    Límite de paseo aleatorio de p a theta = 90°: p = (3β - 1/2)/2
    (Murgia et al. 2004).
    """
    return (3.0 * beta - 0.5) / 2.0


def p_vista_frontal(beta: float) -> float:
    """
    Límite de paseo aleatorio de p a theta = 0°: p = 3β/2.
    """
    return 1.5 * beta


def p_random_walk_cilindro(beta: float) -> float:
    """Alias retrocompatible de `p_vista_lateral` (solo válido para theta=90°)."""
    return p_vista_lateral(beta)


def semiancho_media_altura(r_c: float, p: float) -> float:
    """
    Semiancho a media altura del perfil sigma0 (1+d^2/r_c^2)^(-p):
    d_1/2 = r_c sqrt(2^(1/p) - 1).
    """
    return float(r_c * np.sqrt(2.0 ** (1.0 / p) - 1.0))


def fit_beta_dispersion(
    centros: np.ndarray,
    sigma_rm: np.ndarray,
    p0: tuple | None = None,
    p_fijo: float | None = None,
    rc_fijo: float | None = None,
    errores: np.ndarray | None = None,
) -> BetaFitResult:
    """
    Ajusta `beta_dispersion_model` a un perfil sigma_RM(d) ya calculado.
    Misma interfaz y mismo tratamiento de NaNs que `fit_transverse_dispersion`,
    para poder comparar directamente ambas formas funcionales sobre el mismo
    perfil.

    Modos (a lo más uno de `p_fijo`/`rc_fijo`):
    - libre: (sigma0, r_c, p).
    - `rc_fijo`: (sigma0, p) con r_c dado. Modo principal del barrido.
    - `p_fijo`: (sigma0, r_c) con p dado.

    Los parámetros fijos se devuelven con error 0.0.
    `errores`: incertidumbre por bin para un ajuste ponderado. En ese caso
    `chi2_red` es la métrica de bondad de ajuste (`r_squared` es sin pesos).
    """
    if p_fijo is not None and rc_fijo is not None:
        raise ValueError("Se puede fijar p o r_c, no ambos.")

    if hasattr(centros, "get"):
        centros = centros.get()
    if hasattr(sigma_rm, "get"):
        sigma_rm = sigma_rm.get()

    centros = np.asarray(centros, dtype=float)
    sigma_rm = np.asarray(sigma_rm, dtype=float)

    valido = np.isfinite(sigma_rm)
    n_libres = 3 if (p_fijo is None and rc_fijo is None) else 2
    if valido.sum() < n_libres + 1:
        raise ValueError(
            "No hay suficientes bins válidos (no-NaN) para ajustar la forma beta. "
            f"Se requieren al menos {n_libres + 1}, pero se "
            f"encontraron {valido.sum()}."
        )

    centros_v = centros[valido]
    sigma_v = sigma_rm[valido]
    err_v = _errores_validos(errores, valido)

    distancias_positivas = np.abs(centros_v[centros_v != 0])
    r_c_inicial = float(distancias_positivas.mean()) if distancias_positivas.size else 1.0
    s0_inicial = float(np.max(sigma_v))

    if p_fijo is not None:
        modelo = lambda d, s0, rc: beta_dispersion_model(d, s0, rc, p_fijo)
        inicial = (s0_inicial, r_c_inicial) if p0 is None else (p0[0], p0[1])
    elif rc_fijo is not None:
        modelo = lambda d, s0, p: beta_dispersion_model(d, s0, rc_fijo, p)
        inicial = (s0_inicial, 1.0) if p0 is None else (p0[0], p0[-1])
    else:
        modelo = beta_dispersion_model
        inicial = (s0_inicial, r_c_inicial, 1.0) if p0 is None else p0

    parametros, covarianza = curve_fit(
        modelo, centros_v, sigma_v, p0=inicial, bounds=(0.0, np.inf),
        sigma=err_v, absolute_sigma=err_v is not None, maxfev=10000,
    )
    errores_par = np.sqrt(np.diag(covarianza))

    if p_fijo is not None:
        (sigma0, r_c), p = parametros, p_fijo
        sigma0_err, r_c_err, p_err = errores_par[0], errores_par[1], 0.0
    elif rc_fijo is not None:
        (sigma0, p), r_c = parametros, rc_fijo
        sigma0_err, p_err, r_c_err = errores_par[0], errores_par[1], 0.0
    else:
        sigma0, r_c, p = parametros
        sigma0_err, r_c_err, p_err = errores_par

    prediccion = beta_dispersion_model(centros_v, sigma0, r_c, p)
    ss_res = np.sum((sigma_v - prediccion) ** 2)
    ss_tot = np.sum((sigma_v - np.mean(sigma_v)) ** 2)
    r_cuadrado = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return BetaFitResult(
        sigma0=float(sigma0),
        r_c=float(r_c),
        p=float(p),
        sigma0_err=float(sigma0_err),
        r_c_err=float(r_c_err),
        p_err=float(p_err),
        r_squared=float(r_cuadrado),
        chi2_red=_chi2_reducido(sigma_v, prediccion, err_v, n_libres),
    )
