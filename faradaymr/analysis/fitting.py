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
    if hasattr(centros, 'get'):
        centros = centros.get()
    if hasattr(sigma_rm, 'get'):
        sigma_rm = sigma_rm.get()
        
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


def beta_dispersion_model(
    d: np.ndarray | float, sigma0: float, r_c: float, p: float
) -> np.ndarray | float:
    """
    Forma funcional alternativa a `gaussian_model`, derivada del propio
    perfil beta de densidad usado para construir el filamento
    (`ne(r) = n0 (1 + r^2/r_c^2)^(-3*beta/2)`, ver
    `examples/filamento_whim/config_fisica.py`).

    Para un cilindro con ese perfil, sigma_RM(d) ~ sqrt(integral ne^2 dl)
    es también una potencia de (1 + d^2/r_c^2); a diferencia de la
    gaussiana (que decae como exp(-d^2), mucho más rápido en las colas),
    esta forma tiene colas que decaen como una ley de potencia, y en las
    simulaciones de este proyecto ajusta sistemáticamente mejor (R^2 más
    alto) que la gaussiana -ver `fit_beta_dispersion`.

    Parámetros
    ----------
    d : distancia transversal al eje.
    sigma0 : dispersión de RM en el eje (d=0).
    r_c : escala característica de caída (análoga al radio de núcleo).
    p : índice de la caída (libre; no se fija a 3*beta/2 - 1/4 a priori
        para no imponer la hipótesis del perfil beta, solo inspirarse en
        su forma funcional).
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


def p_random_walk_cilindro(beta: float) -> float:
    """
    Valor de `p` esperado en `beta_dispersion_model` para un filamento
    cilíndrico con perfil de densidad beta (`ne(r) = n0 (1+r^2/r_c^2)^(-3*beta/2)`)
    y un campo magnético turbulento tipo random-walk a lo largo de la línea
    de visión (muchas celdas de correlación independientes, cada una
    contribuyendo su propia varianza a RM).

    Se deriva igual que la fórmula de brillo superficial en rayos X de un
    perfil beta (S_x ~ integral(ne^2 dl) ~ (1+r^2/r_c^2)^(-3*beta+1/2), ver
    p.ej. Murgia et al. 2004 Ec. 12): si sigma_RM(d)^2 ~ integral(ne^2 dl),
    entonces sigma_RM(d) ~ (1+d^2/r_c^2)^{-(3*beta-1/2)/2}, es decir
    p = (3*beta - 1/2)/2. Para beta=0.5 (valor de este proyecto) da p=0.5.

    Sirve como valor fijo de referencia para `fit_beta_dispersion(...,
    p_fijo=...)`: el ajuste de 3 parámetros libres (sigma0, r_c, p) tiene a
    r_c y p casi degenerados en el rango de distancias típico de este
    proyecto, así que un ajuste de 2 parámetros con p fijado a su valor
    físicamente esperado es la comprobación de validación real (¿recupera
    el r_c de entrada?), no el ajuste libre.
    """
    return (3.0 * beta - 0.5) / 2.0


def fit_beta_dispersion(
    centros: np.ndarray,
    sigma_rm: np.ndarray,
    p0: tuple[float, float, float] | None = None,
    p_fijo: float | None = None,
) -> BetaFitResult:
    """
    Ajusta `beta_dispersion_model` a un perfil sigma_RM(d) ya calculado.
    Misma interfaz y mismo tratamiento de NaNs que `fit_transverse_dispersion`,
    para poder comparar directamente ambas formas funcionales (su R^2)
    sobre el mismo perfil, en vez de asumir a ciegas cuál es la correcta.

    Parámetros
    ----------
    p_fijo : float, opcional
        Si se da, fija el índice `p` a este valor y ajusta solo (sigma0,
        r_c) -2 parámetros libres en vez de 3. Recomendado para validación:
        con los 3 parámetros libres, r_c y p quedan casi degenerados en el
        rango de distancias típico de este proyecto (barras de error
        grandes y con fuerte correlación cruzada), así que el ajuste libre
        no es una prueba confiable de si el modelo recupera el r_c real.
        Ver `p_random_walk_cilindro` para el valor físicamente motivado.
        Cuando se usa, `p_err` en el resultado es 0.0 (no es un parámetro
        ajustado, no tiene error estadístico propio).
    """
    if hasattr(centros, "get"):
        centros = centros.get()
    if hasattr(sigma_rm, "get"):
        sigma_rm = sigma_rm.get()

    centros = np.asarray(centros, dtype=float)
    sigma_rm = np.asarray(sigma_rm, dtype=float)

    valido = np.isfinite(sigma_rm)
    n_parametros_libres = 2 if p_fijo is not None else 3
    if valido.sum() < n_parametros_libres + 1:
        raise ValueError(
            "No hay suficientes bins válidos (no-NaN) para ajustar la forma beta. "
            f"Se requieren al menos {n_parametros_libres + 1}, pero se "
            f"encontraron {valido.sum()}."
        )

    centros_v = centros[valido]
    sigma_v = sigma_rm[valido]

    distancias_positivas = np.abs(centros_v[centros_v != 0])
    r_c_inicial = float(distancias_positivas.mean()) if distancias_positivas.size else 1.0
    limites = (0.0, np.inf)

    if p_fijo is not None:
        if p0 is None:
            p0_2 = (float(np.max(sigma_v)), r_c_inicial)
        else:
            p0_2 = (p0[0], p0[1])

        def modelo_p_fijo(d, sigma0, r_c):
            return beta_dispersion_model(d, sigma0, r_c, p_fijo)

        parametros, covarianza = curve_fit(
            modelo_p_fijo, centros_v, sigma_v, p0=p0_2, bounds=limites, maxfev=5000,
        )
        sigma0, r_c = parametros
        p = p_fijo
        errores = np.sqrt(np.diag(covarianza))
        sigma0_err, r_c_err = float(errores[0]), float(errores[1])
        p_err = 0.0
        prediccion = beta_dispersion_model(centros_v, sigma0, r_c, p)
    else:
        if p0 is None:
            p0 = (float(np.max(sigma_v)), r_c_inicial, 1.0)

        parametros, covarianza = curve_fit(
            beta_dispersion_model, centros_v, sigma_v, p0=p0, bounds=limites, maxfev=5000,
        )
        sigma0, r_c, p = parametros
        errores = np.sqrt(np.diag(covarianza))
        sigma0_err, r_c_err, p_err = float(errores[0]), float(errores[1]), float(errores[2])
        prediccion = beta_dispersion_model(centros_v, *parametros)

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
    )