"""Ajuste de la medida de rotación a partir de ángulos observados.

La rotación de Faraday obedece psi(lambda) = psi_0 + RM * lambda**2.
Por tanto, un ajuste lineal de psi contra lambda**2 recupera la
pendiente RM y la ordenada al origen psi_0.
"""

from __future__ import annotations


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