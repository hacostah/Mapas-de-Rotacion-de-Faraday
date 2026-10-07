"""
Descarga los conjuntos de datos observacionales reales que usa
`faradaymr.observational` para comparar el foreground galáctico sintético
(Proyecto III) contra el cielo real (issue #37).

No se versionan en git (`data/external/` está en `.gitignore`: ~530 MB de
FITS/TSV que nunca cambian de contenido, no tiene sentido inflar el repo
con ellos) -correr este script para obtenerlos en una copia nueva del
repositorio:

    python data/external/descargar_datos.py

Los nombres y rutas de destino coinciden EXACTAMENTE con las constantes
`RUTA_OPPERMANN2012`/`RUTA_TAYLOR2009`/`RUTA_HASLAM408`/`RUTA_PLANCK_030GHZ`
de `faradaymr.observational` -si se renombra un archivo acá, hay que
actualizar también esas constantes.
"""

from __future__ import annotations

import os
import urllib.request

_DIR_DESTINO = os.path.dirname(os.path.abspath(__file__))

FUENTES = {
    # Oppermann & Enßlin (2012, A&A 542, A93; arXiv:1111.6186): reconstrucción
    # Bayesiana de cielo completo de la Faraday depth galáctica (HEALPix,
    # nside=128, RING). Mismo archivo que en su día publicó el grupo IFT del
    # MPA (wwwmpa.mpa-garching.mpg.de/ift/faraday/), reempaquetado en Zenodo.
    "oppermann2012_galactic_faraday_depth.fits": (
        "https://zenodo.org/records/17963231/files/faraday.fits?download=1"
    ),
    # Taylor, Stil & Sunstrum (2009, ApJ 702, 1230): catálogo NVSS de RM de
    # 37543 fuentes puntuales extragalácticas, vía VizieR
    # (J/ApJ/702/1230/catalog), columnas RAJ2000/DEJ2000/RM/e_RM/Pk/Si/m.
    "taylor2009_nvss_rm_catalog.tsv": (
        "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
        "-source=J/ApJ/702/1230/catalog&-out.max=100000"
        "&-out=RAJ2000,DEJ2000,RM,e_RM,Pk,Si,m"
    ),
    # Haslam et al. (1982), 408 MHz, versión destriped/desourced (Remazeilles
    # et al. 2015), HEALPix nside=512 NESTED, servida por LAMBDA/NASA.
    "haslam408_dsds.fits": (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam/lambda_haslam408_dsds.fits"
    ),
    # Planck LFI 30 GHz, misión completa (PR3/DX12, "full"), I/Q/U en
    # K_CMB, HEALPix nside=1024 NESTED -el canal de Planck más dominado por
    # sincrotrón (el resto ya está dominado por polvo/CMB), la elección
    # natural para comparar contra el I/Q/U sintético de este framework
    # (ver `faradaymr.observational.comparar_con_planck_030ghz`). ~500 MB,
    # el archivo más pesado de los cuatro.
    "planck_030ghz_iqu.fits": (
        "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/"
        "maps/LFI_SkyMap_030_1024_R3.00_full.fits"
    ),
}


def descargar_todo(forzar: bool = False) -> None:
    """Descarga cada archivo de `FUENTES` a `data/external/`, salvo que ya
    exista (`forzar=True` para volver a bajarlo de todos modos)."""
    for nombre_archivo, url in FUENTES.items():
        destino = os.path.join(_DIR_DESTINO, nombre_archivo)
        if os.path.isfile(destino) and not forzar:
            print(f"Ya existe, se salta: {destino}")
            continue
        print(f"Descargando {nombre_archivo} <- {url}")
        urllib.request.urlretrieve(url, destino)
        print(f"  guardado en {destino} ({os.path.getsize(destino) / 1e6:.1f} MB)")


if __name__ == "__main__":
    descargar_todo()
