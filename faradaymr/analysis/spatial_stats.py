#Análisis espacial y estadístico de mapas 2D observacionales.

from __future__ import annotations

from scipy.stats import binned_statistic

from ..backend import to_numpy
from ..simulation.geometry import projected_axis_distance

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

def transverse_rm_dispersion(rm_map, filament_axis_3d, pixel_size, bins, xp=None):
    """
    Perfil de dispersión transversal de RM (Proyecto II): sigma_RM(d) en
    función de la distancia al eje *proyectado* del filamento sobre el mapa.

    No tiene lógica propia de geometría ni de binning: `projected_axis_distance`
    resuelve la proyección del eje 3D al plano del mapa, y `radial_profile`
    resuelve el binning estadístico. Esta función solo los compone.
    """
    if xp is None:
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp

    distance_map = projected_axis_distance(
        rm_map.shape, filament_axis_3d, pixel_size, xp=xp
    )
    return radial_profile(rm_map, distance_map, bins, statistic="std", xp=xp)