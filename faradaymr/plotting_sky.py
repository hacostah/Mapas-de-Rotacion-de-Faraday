"""
Mapas de cielo completo en proyección de Mollweide ("ovalados"), con el
mismo aspecto general que las Figs. 1, 2 y 4 de Waelkens et al. (2008, el
paper del código hammurabi): una elipse de igual área por panel, sin
marcas de longitud/latitud (solo el contorno de la proyección), título
arriba y una barra de color horizontal angosta debajo de cada panel.

Por qué Mollweide y no el `imshow` rectangular que ya usaba `plots.py`:
un mapa (l, b) en `imshow` rectangular distorsiona el área -un píxel cerca
del polo ocupa, en el dibujo, el mismo ancho angular en l que uno en el
plano galáctico, aunque represente muchísima menos área real del cielo
(el mismo motivo por el que un mapamundi rectangular exagera Groenlandia).
Mollweide es la proyección de igual área estándar en astronomía de cielo
completo (es la que usa HEALPix/hammurabi/Planck) precisamente porque
corrige esa distorsión: la estructura visual del disco galáctico
concentrada cerca de b=0 dcorresponde entonces al área real que ocupa en
el cielo, no a un artefacto de la proyección.

No se requiere `healpy`: la grilla nativa de este framework ya es
rectangular en (l, b) (ver `faradaymr.los_raytrace.sky_map`), y
`matplotlib` trae la proyección Mollweide integrada
(`projection="mollweide"`); alcanza con `pcolormesh` sobre esa grilla, sin
necesidad de re-pixelizar a HEALPix solo para graficar.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np


def _preparar_grilla_mollweide(l_grid, b_grid, mapa, mirror_l=True):
    """
    Convierte (l_grid, b_grid, mapa) -la grilla nativa de `sky_map`, con
    `l_grid` en radianes en [-pi, pi) y `b_grid` en radianes- a los
    arreglos 2D (Lon, Lat, datos) que `pcolormesh` necesita sobre unos ejes
    `projection="mollweide"` de matplotlib.

    `mirror_l=True` invierte el signo de la longitud antes de graficar: la
    convención galáctica estándar (y la de las figuras del paper) dibuja l
    creciente hacia la IZQUIERDA visto desde el centro, mientras que una
    proyección matemática estándar (y `los_raytrace.direction_from_galactic`)
    la define creciente hacia la derecha -sin este espejo, el mapa saldría
    con este del disco a la izquierda, invertido respecto a toda la
    literatura que se cita en el paper.

    Los ejes `mollweide` de matplotlib solo aceptan longitudes en
    [-pi, pi]; como `l_grid` puede no venir ordenada tras el espejo, se
    reordena antes de pasarla a `pcolormesh` (que requiere una grilla
    monótona).
    """
    l = np.asarray(l_grid, dtype=float)
    b = np.asarray(b_grid, dtype=float)
    datos = np.asarray(mapa, dtype=float)

    l_plot = -l if mirror_l else l
    l_plot = np.mod(l_plot + np.pi, 2 * np.pi) - np.pi  # envolver a [-pi, pi)
    orden = np.argsort(l_plot)

    lon, lat = np.meshgrid(l_plot[orden], b, indexing="ij")
    return lon, lat, datos[orden, :]


def _rotular_ejes_galacticos(ax, mirror_l=True):
    """
    Rotula meridianos y paralelos del Mollweide en grados galácticos. Con
    `mirror_l` el dibujo tiene l creciendo hacia la izquierda, así que la
    marca en la posición x lleva la etiqueta l = -x (l=180 en los bordes,
    l=0 en el centro, como en cualquier mapa galáctico publicado).
    """
    marcas_lon = np.arange(-120, 121, 60)
    ax.set_xticks(np.radians(marcas_lon))
    ax.set_xticklabels(
        [f"{(-x if mirror_l else x) % 360:.0f}°" for x in marcas_lon], fontsize=6
    )
    marcas_lat = np.arange(-60, 61, 30)
    ax.set_yticks(np.radians(marcas_lat))
    ax.set_yticklabels([f"{y:.0f}°" for y in marcas_lat], fontsize=6)


def mollweide_panel(
    ax,
    l_grid,
    b_grid,
    mapa,
    cmap="inferno",
    simetrico=False,
    vmin=None,
    vmax=None,
    mirror_l=True,
    escala="lineal",
    rango_dinamico=1e4,
):
    """
    Dibuja un único panel ovalado (Mollweide) sobre unos ejes `ax` que ya
    deben haberse creado con `projection="mollweide"`.

    `simetrico=True` centra la escala de color en cero (`vmin=-vmax`) -la
    opción correcta para RM, que cambia de signo según la orientación del
    campo a lo largo de la línea de visión (igual que el `RdBu_r` de la
    Fig. 1 del paper); para intensidad (siempre positiva) se deja en
    `False`, con `vmin`/`vmax` iguales al rango de datos salvo que se
    pasen explícitamente.

    `escala="log"` usa una escala logarítmica con `rango_dinamico` décadas
    (vmin = vmax / rango_dinamico): necesaria para intensidad sincrotrón,
    cuyo contraste centro/plano/alto-b abarca varios órdenes de magnitud y en
    escala lineal deja todo salvo el centro galáctico en negro.

    Devuelve el objeto `QuadMesh` de `pcolormesh` (para poder pasarlo a
    `fig.colorbar` con el formato que quiera quien llama).
    """
    lon, lat, datos = _preparar_grilla_mollweide(l_grid, b_grid, mapa, mirror_l=mirror_l)

    if simetrico:
        # Percentil 99 y no el máximo: unos pocos píxeles hacia el centro
        # galáctico (donde la RM es un orden de magnitud mayor que en el
        # resto del cielo) fijaban toda la escala y dejaban el cielo
        # medio-alto en un blanco uniforme. Los píxeles que saturan se
        # marcan con `extend` en la barra de color (ver quien llama).
        amplitud = vmax if vmax is not None else float(np.nanpercentile(np.abs(datos), 99))
        vmin_final, vmax_final = -amplitud, amplitud
    else:
        vmin_final = vmin if vmin is not None else float(np.nanmin(datos))
        vmax_final = vmax if vmax is not None else float(np.nanmax(datos))

    if escala == "log":
        from matplotlib.colors import LogNorm

        techo = vmax if vmax is not None else float(np.nanmax(datos))
        malla = ax.pcolormesh(
            lon,
            lat,
            np.clip(datos, techo / rango_dinamico, None),
            cmap=cmap,
            norm=LogNorm(vmin=techo / rango_dinamico, vmax=techo),
            shading="auto",
        )
    else:
        malla = ax.pcolormesh(
            lon, lat, datos, cmap=cmap, vmin=vmin_final, vmax=vmax_final, shading="auto"
        )
    _rotular_ejes_galacticos(ax, mirror_l)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return malla


def figura_estilo_hammurabi(
    ruta_destino: str,
    nombre_archivo: str,
    l_grid,
    b_grid,
    paneles: Sequence[tuple],
    titulo_figura: Optional[str] = None,
    mirror_l: bool = True,
    figsize_por_panel: tuple = (4.2, 3.2),
):
    """
    Figura de N paneles ovalados en una fila, con el mismo diseño general
    que las Figs. 1/2/4 del paper de hammurabi: un panel por observable,
    título arriba de cada uno y una barra de color horizontal angosta
    debajo -en vez del `imshow` rectangular que usaba la versión anterior
    de este módulo (ver docstring del módulo para la razón física de
    preferir Mollweide).

    `paneles`: secuencia de tuplas
        (mapa2d, titulo, etiqueta_barra, cmap, simetrico[, escala])
    (`escala` opcional: "lineal" por defecto o "log", ver `mollweide_panel`)
    una por panel, en el orden en que deben aparecer (izquierda a
    derecha) -por ejemplo, el mismo orden que usa la Fig. 1 del paper:
    intensidad total, intensidad polarizada, ángulo de polarización, RM.

    Guarda el archivo en `os.path.join(ruta_destino, nombre_archivo)` y
    devuelve esa ruta.
    """
    import matplotlib.pyplot as plt

    n_paneles = len(paneles)
    fig = plt.figure(figsize=(figsize_por_panel[0] * n_paneles, figsize_por_panel[1] + 0.6))

    for i, panel in enumerate(paneles):
        mapa, titulo, etiqueta, cmap, simetrico = panel[:5]
        escala = panel[5] if len(panel) > 5 else "lineal"
        opciones = panel[6] if len(panel) > 6 else {}
        ax = fig.add_subplot(1, n_paneles, i + 1, projection="mollweide")
        # Los píxeles NaN (p.ej. ángulo indefinido donde P ~ 0) quedan en
        # gris, distinguibles de un valor del mapa de color.
        ax.set_facecolor("0.8")
        malla = mollweide_panel(
            ax,
            l_grid,
            b_grid,
            mapa,
            cmap=cmap,
            simetrico=simetrico,
            mirror_l=mirror_l,
            escala=escala,
            **opciones,
        )
        ax.set_title(titulo, fontsize=10, pad=10)
        cbar = fig.colorbar(
            malla, ax=ax, orientation="horizontal", fraction=0.055, pad=0.06, aspect=28,
            extend="both" if simetrico else "neither",
        )
        cbar.set_label(etiqueta, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if titulo_figura:
        fig.suptitle(titulo_figura, fontsize=12, y=1.02)

    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


def angulo_polarizacion_enmascarado(q_map, u_map, umbral_pol: float = 1e-3):
    """
    Ángulo de polarización observado (0.5*arctan2(U,Q)), enmascarado a NaN
    donde la intensidad polarizada P cae por debajo de `umbral_pol` veces
    el máximo de P en el mapa: con P~0, (Q,U) es ruido numérico y su fase
    es aleatoria, no una medición real del ángulo. Extraído como función
    propia (antes vivía inline en `mapa_sintetico_estilo_hammurabi`) para
    poder probar la lógica de enmascarado en sí, sin tener que inspeccionar
    píxeles de un PNG ya renderizado.

    Devuelve (p_map, psi_obs_enmascarado).
    """
    q_map = np.asarray(q_map, dtype=float)
    u_map = np.asarray(u_map, dtype=float)
    p_map = np.sqrt(q_map**2 + u_map**2)
    psi_obs = 0.5 * np.arctan2(u_map, q_map)
    psi_obs = np.where(p_map >= umbral_pol * np.nanmax(p_map), psi_obs, np.nan)
    return p_map, psi_obs


def mapa_sintetico_estilo_hammurabi(
    ruta_destino: str,
    l_grid,
    b_grid,
    rm_map,
    i_map,
    q_map,
    u_map,
    nombre_archivo: str = "figura_1_mapa_de_cielo_mollweide.png",
    titulo_figura: str = (
        "Foreground galáctico sintético a 1.4 GHz (disco + brazos espirales + "
        "campo regular de disco y halo + turbulencia)"
    ),
    umbral_pol: float = 1e-3,
):
    """
    Reemplazo, en proyección Mollweide, de `plots.fig1_mapa_de_cielo`: los
    mismos cuatro observables que compara la Fig. 1 del paper de hammurabi
    (intensidad total, intensidad polarizada, ángulo de polarización, RM),
    calculados a partir de I, Q, U -no hace falta volver a integrar nada,
    P y PA son funciones algebraicas puntuales de Q y U que ya devuelve
    `los_raytrace.sky_map`.
    """
    p_map, psi_obs = angulo_polarizacion_enmascarado(q_map, u_map, umbral_pol=umbral_pol)
    # La emisividad no está calibrada en unidades físicas (ver
    # `faradaymr.observational`): se normaliza al máximo de I para que la
    # escala diga algo (contraste y P/I), no un número arbitrario.
    i_max = float(np.nanmax(i_map))

    paneles = [
        (np.asarray(i_map) / i_max, "Intensidad total", r"$I\,/\,I_{\rm max}$", "inferno", False, "log"),
        (p_map / i_max, "Intensidad polarizada", r"$P\,/\,I_{\rm max}$", "inferno", False, "log"),
        # Cíclico (twilight) y con rango fijo: el ángulo vale módulo 180°,
        # así que -90° y +90° son el mismo color.
        (np.degrees(psi_obs), "Ángulo de polarización (IAU)",
         rf"grados (gris: $P < 10^{{{np.log10(umbral_pol):.0f}}}\,P_{{\rm max}}$)", "twilight", False,
         "lineal", {"vmin": -90.0, "vmax": 90.0}),
        (rm_map, "Medida de Rotación (RM)", r"rad m$^{-2}$", "RdBu_r", True),
    ]
    return figura_estilo_hammurabi(
        ruta_destino,
        nombre_archivo,
        l_grid,
        b_grid,
        paneles,
        titulo_figura=titulo_figura,
    )
