"""Ajuste de la medida de rotación a partir de ángulos observados.

La rotación de Faraday obedece psi(lambda) = psi_0 + RM * lambda**2.
Por tanto, un ajuste lineal de psi contra lambda**2 recupera la
pendiente RM y la ordenada al origen psi_0.
"""

from __future__ import annotations
import itertools


def unwrap_polarization_angle(psi_obs_rad, axis=-1, xp=None):
    """Desenrolla saltos artificiales de pi en el ángulo de polarización.
    
    Aprovecha la simetría de 180° de la polarización multiplicando el 
    ángulo por 2, aplicando el unwrap estándar (2*pi) y dividiendo a la mitad.
    
    Parameters
    ----------
    psi_obs_rad : array-like
        Ángulos de polarización observados en radianes.
    axis : int, optional
        Eje a lo largo del cual desenrollar. Por defecto es el último (-1).
    xp : module, optional
        Backend de arreglos compatible con NumPy. Si se omite se usa NumPy.
        
    Returns
    -------
    array-like
        Ángulos de polarización desenrollados continuamente.
    """
    if xp is None:
        import numpy as xp
        
    psi_obs_rad = xp.asarray(psi_obs_rad)
    return xp.unwrap(2.0 * psi_obs_rad, axis=axis) / 2.0


def resolve_n_pi_ambiguity(wavelengths_m, psi_obs_rad, n_candidatos=range(-3, 4), xp=None):
    """Resuelve la ambigüedad n*pi minimizando el residuo del ajuste lineal.
    
    En configuraciones de frecuencias discretas y no uniformes en lambda^2 
    (como 1.4, 2.0 y 5.0 GHz), el unwrap estándar falla. Esta función utiliza 
    fuerza bruta sobre combinaciones enteras para recuperar la RM verdadera.
    
    Se emplea fuerza bruta en lugar de un optimizador continuo porque el 
    espacio de búsqueda es lo suficientemente pequeño para ser computacionalmente 
    trivial (ej. probando n entre -3 y 3 para 3 frecuencias, resultan en 
    7^3 = 343 combinaciones totales).
    
    Parameters
    ----------
    wavelengths_m : array-like, shape (n,)
        Longitudes de onda en metros.
    psi_obs_rad : array-like, shape (n,)
        Ángulos de polarización observados, en radianes.
    n_candidatos : iterable, optional
        Rango de enteros n a evaluar por cada frecuencia.
    xp : module, optional
        Backend de arreglos compatible con NumPy. Si se omite se usa NumPy.
        
    Returns
    -------
    mejor_rm : float
        La pendiente (Medida de Rotación) que produce el menor residuo.
    """
    if xp is None:
        import numpy as xp
        
    lambda2 = xp.asarray(wavelengths_m) ** 2
    psi_obs_rad = xp.asarray(psi_obs_rad)
    
    mejor_residuo = None
    mejor_rm = None
    
    for combinacion in itertools.product(n_candidatos, repeat=len(wavelengths_m)):
        psi_prueba = psi_obs_rad + xp.pi * xp.asarray(combinacion)
        pendiente, ordenada = xp.polyfit(lambda2, psi_prueba, 1)
        
        ajuste = pendiente * lambda2 + ordenada
        residuo = xp.sum((psi_prueba - ajuste) ** 2)
        
        if mejor_residuo is None or residuo < mejor_residuo:
            mejor_residuo = residuo
            mejor_rm = pendiente
            
    return mejor_rm


def estimate_rm_lsq(wavelengths_m, psi_obs_rad, xp=None):
    """Estima (RM, psi_0) mediante mínimos cuadrados lineales.

    Parameters
    ----------
    wavelengths_m : array-like, shape (n,)
        Longitudes de onda en metros.
    psi_obs_rad : array-like, shape (n,)
        Ángulos de polarización observados, en radianes y ya desenrollados
        respecto de la ambigüedad n*pi.
    xp : module, optional
        Backend de arreglos compatible con NumPy. Si se omite se usa NumPy.

    Returns
    -------
    rm : scalar
        Medida de rotación estimada, en rad/m^2.
    psi_0 : scalar
        Ángulo de polarización intrínseco estimado, en radianes.
    """
    if xp is None:
        import numpy as xp

    wavelengths_m = xp.asarray(wavelengths_m, dtype=float)
    psi_obs_rad = xp.asarray(psi_obs_rad, dtype=float)

    if wavelengths_m.ndim != 1 or psi_obs_rad.ndim != 1:
        raise ValueError("wavelengths_m y psi_obs_rad deben ser vectores 1D.")
    if wavelengths_m.size != psi_obs_rad.size:
        raise ValueError("wavelengths_m y psi_obs_rad deben tener la misma longitud.")
    if wavelengths_m.size < 2:
        raise ValueError("Se requieren al menos dos longitudes de onda.")
    if not bool(xp.all(xp.isfinite(wavelengths_m))) or not bool(
        xp.all(xp.isfinite(psi_obs_rad))
    ):
        raise ValueError("wavelengths_m y psi_obs_rad deben contener valores finitos.")
    if bool(xp.any(wavelengths_m < 0)):
        raise ValueError("wavelengths_m no puede contener longitudes de onda negativas.")

    wavelengths_squared = wavelengths_m**2

    if not bool(xp.all(xp.isfinite(wavelengths_squared))):
        raise ValueError("wavelengths_m**2 debe contener valores finitos.")
    if xp.unique(wavelengths_squared).size < 2:
        raise ValueError("Se requieren al menos dos valores distintos de lambda**2.")

    rm, psi_0 = xp.polyfit(wavelengths_squared, psi_obs_rad, deg=1)
    return rm, psi_0