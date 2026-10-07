"""
Figuras de una corrida de `run.py`.

Salidas del mapa de cielo:
  - mapa_de_cielo.png:            I, P, ángulo de polarización y RM (Mollweide)
  - estadistica_mapa.png:         I(b), y la fracción de polarización P/I
                                  (histograma y perfil en |b|)
Salidas de la estructura 3D: ver `faradaymr.galactic_structure_plots`.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from faradaymr.galactic_structure_plots import generar_graficos_estructura
from faradaymr.plotting_sky import mapa_sintetico_estilo_hammurabi

BINS_B_DEG = np.arange(0.0, 90.0, 10.0)


def fig_mapa_de_cielo(ruta, l_grid, b_grid, rm_map, i_map, q_map, u_map):
    """Panel I, P, ángulo y RM en proyección Mollweide (estilo Fig. 1 de
    Waelkens et al. 2008)."""
    return mapa_sintetico_estilo_hammurabi(
        ruta, l_grid, b_grid, rm_map, i_map, q_map, u_map,
        nombre_archivo="mapa_de_cielo.png",
    )


def fig_estadistica_mapa(
    ruta, l_grid, b_grid, i_map, q_map, u_map, p_index=3.0,
    nombre_archivo="estadistica_mapa.png",
):
    """
    Tres chequeos cuantitativos del mapa sintético:

    (a) I(b) promediada en l contra la ley 1/sin|b| de un disco plano-
        paralelo (la columna de emisores crece como 1/sin|b| mientras el
        rayo siga dentro del disco): cerca del plano debe seguirla, y a
        latitud alta se aparta cuando el rayo sale del disco.
    (b) Histograma de P/I (pesado por cos b, ángulo sólido). No puede pasar
        de p_max = (p+1)/(p+7/3), el grado de polarización de la
        sincrotrón en un campo perfectamente uniforme; los valores bajos
        miden cuánto despolariza la turbulencia y la rotación de Faraday
        interna.
    (c) Mediana y rango 16-84 % de P/I contra |b|.
    """
    i_map, q_map, u_map = (np.asarray(m, dtype=float) for m in (i_map, q_map, u_map))
    b = np.asarray(b_grid, dtype=float)
    b_deg = np.degrees(b)
    p_max = (p_index + 1.0) / (p_index + 7.0 / 3.0)
    frac = np.hypot(q_map, u_map) / i_map
    peso = np.broadcast_to(np.cos(b), frac.shape)
    b_abs = np.broadcast_to(np.abs(b_deg), frac.shape)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    perfil = i_map.mean(axis=0)
    ax.semilogy(b_deg, perfil, "k-", label="modelo")
    ref = np.abs(b_deg) >= 5
    b_norm = np.argmin(np.abs(np.abs(b_deg) - 30.0))
    ax.semilogy(
        b_deg[ref], perfil[b_norm] * np.sin(np.radians(30.0)) / np.sin(np.radians(np.abs(b_deg[ref]))),
        "r--", lw=1.2, label=r"$\propto 1/\sin|b|$ (disco plano-paralelo)",
    )
    ax.set_xlabel("b [grados]")
    ax.set_ylabel("I promedio en l [unid. arbitrarias]")
    ax.set_title("Intensidad sincrotrón vs latitud")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.hist(frac.ravel(), bins=np.linspace(0, 1, 60), weights=peso.ravel(), density=True,
            color="steelblue", alpha=0.8)
    ax.axvline(p_max, color="r", ls="--", label=rf"$p_{{\rm max}}=(p+1)/(p+7/3)={p_max:.2f}$")
    ax.set_xlabel("P/I")
    ax.set_ylabel("densidad de probabilidad (pesada por cos b)")
    ax.set_title(f"Fracción de polarización (máximo del mapa: {frac.max():.2f})")
    ax.legend(fontsize=8)

    ax = axes[2]
    centros, med, bajo, alto = [], [], [], []
    for lo, hi in zip(BINS_B_DEG[:-1], BINS_B_DEG[1:]):
        v = frac[(b_abs >= lo) & (b_abs < hi)]
        if v.size:
            centros.append(0.5 * (lo + hi))
            bajo.append(np.percentile(v, 16))
            med.append(np.percentile(v, 50))
            alto.append(np.percentile(v, 84))
    ax.plot(centros, med, "o-", color="steelblue", label="mediana")
    ax.fill_between(centros, bajo, alto, color="steelblue", alpha=0.25, label="16-84 %")
    ax.axhline(p_max, color="r", ls="--", label=r"$p_{\rm max}$")
    ax.set_xlabel("|b| [grados]")
    ax.set_ylabel("P/I")
    ax.set_ylim(0, 1)
    ax.set_title("Fracción de polarización vs |b|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    ruta_completa = os.path.join(ruta, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


def generar_graficos_estudio(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map, p_index=3.0):
    fig_mapa_de_cielo(ruta_destino, l_grid, b_grid, rm_map, i_map, q_map, u_map)
    fig_estadistica_mapa(ruta_destino, l_grid, b_grid, i_map, q_map, u_map, p_index=p_index)


def generar_graficos_estructura_3d(ruta_destino, *args, **kwargs):
    """Punto de entrada único desde `run.py` para las figuras de estructura;
    los argumentos son los de `faradaymr.galactic_structure_plots.generar_graficos_estructura`."""
    return generar_graficos_estructura(ruta_destino, *args, **kwargs)
