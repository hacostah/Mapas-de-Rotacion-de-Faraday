"""
Estilo común de las figuras del proyecto (póster y README).

Un color por FUENTE de datos, el mismo en todas las figuras, para que el
lector aprenda la convención una vez: el modelo siempre en naranja, las
observaciones de referencia (Oppermann+2012, Planck, Haslam) en azul, el
catálogo NVSS en verde agua, y las referencias sin física (plantillas
triviales, leyes analíticas) en gris. Los colores categóricos son los de
una paleta validada para daltonismo (orden fijo, no se ciclan).

Mapas: magnitud positiva (intensidad, P) en `inferno` (secuencial); RM,
que tiene signo, en `RdBu_r` (divergente con blanco en cero); ángulos, que
son cíclicos, en `twilight`.
"""

from __future__ import annotations

MODELO = "#eb6834"
OBSERVADO = "#2a78d6"
NVSS = "#1baf7a"
REFERENCIA = "#8a8984"
SECUNDARIO = "#4a3aa7"
TEXTO_SECUNDARIO = "#52514e"

# Figuras de estructura (todo es modelo): una magnitud física por color.
DENSIDAD = "#2a78d6"
CAMPO_REGULAR = "#4a3aa7"
CAMPO_TURBULENTO = "#1baf7a"

MAPA_MAGNITUD = "inferno"
MAPA_SIGNO = "RdBu_r"
MAPA_ANGULO = "twilight"

ETIQUETA_MODELO = "Modelo"
ETIQUETA_OPPERMANN = "Oppermann & Enßlin 2012"
ETIQUETA_NVSS = "NVSS (Taylor+2009)"
ETIQUETA_PLANCK = "Planck 28.4 GHz"
ETIQUETA_HASLAM = "Haslam 408 MHz"


def aplicar():
    """Fija los rcParams de matplotlib comunes (llamar al inicio de cada figura)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.grid": True,
        "grid.color": "#d9d8d3",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.dpi": 100,
    })
