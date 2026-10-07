"""
Figuras de una corrida de `run.py`.

Salidas del mapa de cielo:
  - mapa_de_cielo.png:      I y P a 1.4 GHz, RM, y P a 28.4 GHz con la
                            orientación del campo magnético (Mollweide)
  - estadistica_mapa.png:   I(b) contra un disco plano-paralelo, y la
                            fracción de polarización contra |b| a 1.4 y
                            28.4 GHz (despolarización por Faraday)
Salidas de la estructura 3D: ver `faradaymr.galactic_structure_plots`.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from faradaymr import estilo_figuras as estilo
from faradaymr.galactic_structure_plots import generar_graficos_estructura
from faradaymr.plotting_sky import mapa_de_cielo_sintetico

BORDES_B_DEG = np.arange(0.0, 90.0, 10.0)


def fig_mapa_de_cielo(ruta, l_grid, b_grid, rm_map, i_map, q_map, u_map, q_map_30, u_map_30):
    """Los cuatro paneles de `faradaymr.plotting_sky.mapa_de_cielo_sintetico`."""
    return mapa_de_cielo_sintetico(
        ruta, l_grid, b_grid, rm_map, i_map, q_map, u_map, q_map_30, u_map_30,
        nombre_archivo="mapa_de_cielo.png",
    )


def _perfil_fraccion_polarizacion(b_grid, frac):
    """Mediana y rango 16-84 % de P/I en bandas de |b| de 10°."""
    b_abs = np.broadcast_to(np.abs(np.degrees(np.asarray(b_grid, dtype=float))), frac.shape)
    centros, med, bajo, alto = [], [], [], []
    for lo, hi in zip(BORDES_B_DEG[:-1], BORDES_B_DEG[1:]):
        v = frac[(b_abs >= lo) & (b_abs < hi)]
        v = v[np.isfinite(v)]
        if v.size:
            centros.append(0.5 * (lo + hi))
            bajo.append(np.percentile(v, 16))
            med.append(np.percentile(v, 50))
            alto.append(np.percentile(v, 84))
    return np.array(centros), np.array(med), np.array(bajo), np.array(alto)


def fig_estadistica_mapa(
    ruta, l_grid, b_grid, i_map, q_map, u_map, i_map_30, q_map_30, u_map_30, p_index=3.0,
    nombre_archivo="estadistica_mapa.png",
):
    """
    Dos chequeos físicos del cielo sintético:

    (a) I(b) promediada en l contra la ley 1/sin|b| de un disco
        plano-paralelo (la columna de emisores crece como 1/sin|b|), fijada
        en |b| = 30°. Por debajo de ~10° el disco no es infinito y la curva
        se aparta; a latitud alta la aparta la emisión del halo.
    (b) Fracción de polarización P/I contra |b| a 1.4 GHz y a 28.4 GHz. El
        tope físico es p_max = (p+1)/(p+7/3), el de un campo perfectamente
        uniforme. A 28.4 GHz la rotación de Faraday es despreciable y P/I
        solo baja por la turbulencia y por el cambio de dirección del campo
        a lo largo del rayo. A 1.4 GHz la rotación de Faraday cambia P/I:
        cerca del plano (|RM| ~ 90 rad/m², más de una vuelta) despolariza;
        a latitud alta (|RM| ~ 10 rad/m², ~25°) puede repolarizar un poco,
        porque el campo cambia de orientación a lo largo del rayo (disco ->
        halo) y una rotación diferencial pequeña realinea esas
        contribuciones (rotación de Faraday diferencial; Sokoloff et al. 1998).
    """
    estilo.aplicar()
    b = np.asarray(b_grid, dtype=float)
    b_deg = np.degrees(b)
    p_max = (p_index + 1.0) / (p_index + 7.0 / 3.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    perfil = np.asarray(i_map, dtype=float).mean(axis=0)
    perfil = perfil / perfil.max()
    ax.semilogy(b_deg, perfil, color=estilo.MODELO, label=estilo.ETIQUETA_MODELO + " (1.4 GHz)")
    ref = np.abs(b_deg) >= 3
    b_norm = np.argmin(np.abs(np.abs(b_deg) - 30.0))
    ax.semilogy(b_deg[ref], perfil[b_norm] * np.sin(np.radians(30.0)) / np.sin(np.radians(np.abs(b_deg[ref]))),
                "--", color=estilo.REFERENCIA, lw=1.4, label=r"disco plano-paralelo, $\propto 1/\sin|b|$")
    ax.set_xlabel("b [grados]")
    ax.set_ylabel(r"$\langle I\rangle_l$ / máximo")
    ax.set_title("(a) Intensidad sincrotrón contra latitud")
    ax.grid(True, which="both")
    ax.legend(loc="upper right")

    ax = axes[1]
    for i_m, q_m, u_m, etiqueta, alfa in [
        (i_map_30, q_map_30, u_map_30, "28.4 GHz (sin Faraday)", 0.18),
        (i_map, q_map, u_map, "1.4 GHz (con Faraday)", 0.18),
    ]:
        frac = np.hypot(q_m, u_m) / np.asarray(i_m, dtype=float)
        centros, med, bajo, alto = _perfil_fraccion_polarizacion(b, frac)
        color = estilo.OBSERVADO if "28.4" in etiqueta else estilo.MODELO
        ax.fill_between(centros, bajo, alto, color=color, alpha=alfa, lw=0)
        ax.plot(centros, med, "o-", color=color, ms=5, label=f"{etiqueta}: mediana y rango 16-84 %")
    ax.axhline(p_max, ls="--", color=estilo.REFERENCIA, lw=1.4,
               label=rf"tope de un campo uniforme, $(p+1)/(p+7/3) = {p_max:.2f}$")
    ax.set_xlabel("|b| [grados]")
    ax.set_ylabel("fracción de polarización P/I")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 80)
    ax.set_title("(b) Fracción de polarización a 1.4 y 28.4 GHz")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04,
            "cerca del plano, Faraday despolariza (|RM| grande);\n"
            "a latitud alta, |RM| pequeña: rotación de ~25° que puede\n"
            "realinear el campo del disco y el del halo (repolarización)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color=estilo.TEXTO_SECUNDARIO)

    fig.tight_layout()
    ruta_completa = os.path.join(ruta, nombre_archivo)
    fig.savefig(ruta_completa)
    plt.close(fig)
    return ruta_completa


def generar_graficos_estudio(
    ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map, i_map_30, q_map_30, u_map_30,
    p_index=3.0,
):
    fig_mapa_de_cielo(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map, q_map_30, u_map_30)
    fig_estadistica_mapa(ruta_destino, l_grid, b_grid, i_map, q_map, u_map, i_map_30, q_map_30,
                         u_map_30, p_index=p_index)


def generar_graficos_estructura_3d(ruta_destino, *args, **kwargs):
    """Punto de entrada único desde `run.py` para las figuras de estructura;
    los argumentos son los de `faradaymr.galactic_structure_plots.generar_graficos_estructura`."""
    return generar_graficos_estructura(ruta_destino, *args, **kwargs)
