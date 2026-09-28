from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from faradaymr.galactic_structure_plots import generar_graficos_estructura
from faradaymr.plotting_sky import (
    mapa_sintetico_estilo_hammurabi,
    mapas_individuales_estilo_hammurabi,
)


def fig1_mapa_de_cielo(ruta, l_grid, b_grid, rm_map, i_map, q_map, u_map):
    """
    Mapa de cielo completo -RM, intensidad total, intensidad polarizada y
    ángulo de polarización- para el disco galáctico de juguete visto desde
    el observador interior, en proyección de Mollweide (paneles ovalados
    de igual área).

    Mismo conjunto de observables, y el mismo estilo general de figura,
    que la Fig. 1 de Waelkens et al. (2008) -el paper del código
    hammurabi-: un panel por observable, título arriba, barra de color
    horizontal debajo. Reemplaza la versión anterior de esta función
    (un `imshow` rectangular en (l, b), que distorsiona el área cerca de
    los polos, ver docstring de `faradaymr.plotting_sky` para el porqué);
    el nombre de archivo y la firma de la función no cambian, para no
    romper a quien ya llama `generar_graficos_estudio`.
    """
    mapa_sintetico_estilo_hammurabi(
        ruta,
        l_grid,
        b_grid,
        rm_map,
        i_map,
        q_map,
        u_map,
        nombre_archivo="figura_1_mapa_de_cielo.png",
    )


def fig2_perfil_en_latitud(ruta, l_grid, b_grid, i_map):
    """
    Corte de intensidad en función de la latitud galáctica b, promediado
    sobre todas las longitudes l, en escala log.

    Es el chequeo cuantitativo más simple de que el disco de verdad se ve
    como un disco (pico angosto en b~0) y no como una nube esférica
    difusa: sirve como diagnóstico rápido antes de pasar a comparar contra
    Planck (fuera del alcance de esta corrida).
    """
    perfil_b = np.mean(i_map, axis=0)

    plt.figure(figsize=(6, 4.5))
    plt.semilogy(np.degrees(b_grid), perfil_b, "k-")
    plt.xlabel("b [grados]")
    plt.ylabel("Intensidad sincrotrón (promedio en l), u.a.")
    plt.title("Perfil en latitud galáctica")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta, "figura_2_perfil_latitud.png"), dpi=200)
    plt.close()


def fig3_mapas_individuales(ruta, l_grid, b_grid, rm_map, i_map, q_map, u_map):
    """
    Los mismos cuatro observables de `fig1_mapa_de_cielo`, pero cada uno en
    su propio archivo PNG (a tamaño completo, más detalle visible) en vez
    de cuatro paneles pequeños compartiendo una sola figura -pedido
    explícito: separar el mapa de cielo completo en varias gráficas
    distintas.
    """
    return mapas_individuales_estilo_hammurabi(
        ruta, l_grid, b_grid, rm_map, i_map, q_map, u_map, prefijo="figura_1"
    )


def fig4_histograma_rm(ruta, rm_map, nombre_archivo="figura_4_histograma_rm.png"):
    """
    Distribución de valores de RM del mapa sintético completo (todos los
    píxeles l, b), escala log en y: muestra de un vistazo si la
    distribución es aproximadamente simétrica en torno a cero (esperado,
    ver `faradaymr.plotting_sky.mollweide_panel(simetrico=True)`) y qué tan
    pesada es la cola hacia valores extremos -el mismo diagnóstico que
    `faradaymr.observational.comparar_con_oppermann` usa para señalar la
    sobreestimación de amplitud cerca del plano, pero calculable sin
    ningún dato externo.
    """
    datos = np.asarray(rm_map).ravel()
    datos = datos[np.isfinite(datos)]

    plt.figure(figsize=(6.5, 4.5))
    plt.hist(datos, bins=80, color="firebrick", alpha=0.85)
    plt.axvline(0, color="k", lw=1, ls="--")
    plt.yscale("log")
    plt.xlabel(r"RM [rad m$^{-2}$]")
    plt.ylabel("Número de píxeles (l, b)")
    plt.title(
        f"Distribución de RM sintética "
        f"(media={np.mean(datos):.1f}, RMS={np.sqrt(np.mean(datos**2)):.1f} rad/m$^2$)"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ruta, nombre_archivo), dpi=200)
    plt.close()


def generar_graficos_estudio(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map):
    fig1_mapa_de_cielo(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map)
    fig2_perfil_en_latitud(ruta_destino, l_grid, b_grid, i_map)
    fig3_mapas_individuales(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map)
    fig4_histograma_rm(ruta_destino, rm_map)


def generar_graficos_estructura_3d(
    ruta_destino, ne, bx, by, bz, dx, box_size, observer_pos,
    r0, pitch_angle, n_arms, phase0=0.0, componentes_regular=None,
):
    """
    Envoltorio delgado sobre `faradaymr.galactic_structure_plots` para que
    `run.py` tenga un único punto de entrada por tipo de figura (mapas de
    cielo -> `generar_graficos_estudio`, estructura 3D -> esta función),
    igual que ya hace el resto de este archivo.
    """
    return generar_graficos_estructura(
        ruta_destino, ne, bx, by, bz, dx, box_size, observer_pos,
        r0, pitch_angle, n_arms, phase0=phase0, componentes_regular=componentes_regular,
    )
