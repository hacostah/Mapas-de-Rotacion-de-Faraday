#Análisis espacial y estadístico de mapas 2D observacionales.

from __future__ import annotations

from scipy.stats import binned_statistic

from ..backend import to_numpy

def radial_profile(map2d, distance_map, bins, statistic="std", xp=None):
    """
    Calcula un perfil estadístico (dispersión, media, etc.) bindeado por distancia.

    Al delegar en `scipy.stats.binned_statistic`, esta función no asume ninguna
    geometría particular. `distance_map` puede representar una distancia radial
    desde un centro proyectado, o una distancia cilíndrica a un eje transversal 
    (necesario para evaluar la dispersión transversal de RM).

    Parámetros:
    -----------
    map2d : array-like
        Mapa bidimensional con los valores a analizar (ej. mapa de RM).
    distance_map : array-like
        Mapa de la misma dimensión que `map2d` que contiene la métrica de distancia
        evaluada en cada píxel.
    bins : int o secuencia de escalares
        Número de bins o los bordes exactos de los bins a utilizar.
    statistic : str o callable, opcional
        Estadística a calcular en cada bin ("std", "mean", "median", etc.). 
        Por defecto es "std" para obtener perfiles de dispersión.
    xp : module, opcional
        Backend de arreglos (numpy o cupy).

    Devuelve
    --------
    centros : array 1D
        Centros geométricos de los bins calculados.
    valores : array 1D
        Valor de la estadística calculada para cada bin.
    """
    # scipy.stats opera exclusivamente en CPU, por lo que es imperativo 
    # asegurar que los arreglos se traigan desde la GPU a memoria principal
    dist_cpu = to_numpy(distance_map).ravel()
    mapa_cpu = to_numpy(map2d).ravel()

    valores, bordes, _ = binned_statistic(
        x=dist_cpu,
        values=mapa_cpu,
        statistic=statistic,
        bins=bins,
    )
    
    centros = 0.5 * (bordes[:-1] + bordes[1:])

    # Como el resultado es un arreglo 1D pequeño (típicamente para plotting),
    # suele ser más práctico devolverlo en el backend que el usuario esté utilizando.
    if xp is not None and getattr(xp, "__name__", None) == "cupy":
        return xp.asarray(centros), xp.asarray(valores)

    return centros, valores

def transverse_rm_dispersion(rm_map, distance_map, bins, xp=None):
    """
    Perfil de dispersión transversal de RM (Proyecto II): sigma_RM(d), la
    desviación estándar de la Medida de Rotación en función de la distancia
    transversal al eje del filamento.

    Reutiliza `radial_profile` sin modificarla, fijando `statistic="std"`.
    """
    return radial_profile(rm_map, distance_map, bins, statistic="std", xp=xp)

def isotropic_energy_spectrum(bx, by, bz, dx, xp=None):
    """
    Espectro de energía magnética isotrópico E(k) de un campo vectorial 3D
    periódico, promediado en cascarones esféricos de |k|.

    Normalizado por Parseval: sum(E(k) * dk) = <|B|^2> (restringido a la
    esfera |k| <= k_Nyquist), de modo que la forma de E(k) se pueda
    comparar directo contra la ley de potencias que se impuso (Kolmogorov:
    E(k) ∝ k^{-5/3}; ver `faradaymr.fields.gaussian_random_field`).

    Los cascarones tienen ancho igual al modo fundamental k_f = 2*pi/(N*dx)
    y llegan hasta k_Nyquist = pi/dx; los modos de las esquinas del cubo
    (|k| > k_Nyquist) no llenan un cascarón completo y se descartan.

    Devuelve (k_centros, E_k): en unidades de 1/dx y [B]^2 * dx.
    """
    import numpy as np

    campos = [to_numpy(c) for c in (bx, by, bz)]
    n = campos[0].shape[0]
    k1d = np.fft.fftfreq(n, d=dx) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2).ravel()

    potencia = sum(np.abs(np.fft.fftn(c)) ** 2 for c in campos).ravel() / n**6

    k_f = 2.0 * np.pi / (n * dx)
    bordes = (np.arange(0, n // 2 + 1) + 0.5) * k_f
    bordes[0] = 0.0
    indice = np.digitize(k_mag, bordes) - 1
    valido = (indice >= 0) & (indice < len(bordes) - 1)

    suma = np.bincount(indice[valido], weights=potencia[valido], minlength=len(bordes) - 1)
    ancho = np.diff(bordes)
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    return centros[1:], (suma / ancho)[1:]  # se descarta el cascarón k=0
