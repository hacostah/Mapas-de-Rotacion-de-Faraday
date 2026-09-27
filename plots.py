from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from faradaymr.plotting_sky import mapa_sintetico_estilo_hammurabi


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


def generar_graficos_estudio(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map):
    fig1_mapa_de_cielo(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map)
    fig2_perfil_en_latitud(ruta_destino, l_grid, b_grid, i_map)
