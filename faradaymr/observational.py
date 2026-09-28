"""
Datos observacionales reales para comparar el foreground galáctico
sintético (Proyecto III) contra el cielo real -issue #37: "generar una
réplica exacta de la contaminación galáctica para sustraerla de datos
reales" empieza, necesariamente, por poder comparar contra datos reales.

Tres conjuntos de datos, cada uno sirve para una comparación distinta
-ninguno solo alcanza para validar el modelo completo-:

1. Oppermann & Enßlin (2012, A&A 542, A93; arXiv:1111.6186) -LA MISMA
   referencia que ya cita `faradaymr.calibration` para el chequeo de RM
   hacia el polo-: reconstrucción de cielo completo de la Faraday depth
   galáctica (HEALPix, nside=128, RING), con su mapa de incertidumbre.
   Permite comparar estructura de gran escala (perfil en b, distribución
   de valores, contraste plano/polo) contra la mejor estimación
   observacional disponible del MISMO observable físico (RM), en las
   mismas unidades (rad/m^2) -sin reescalar nada.

2. Taylor, Stil & Sunstrum (2009, ApJ 702, 1230): catálogo de RM de 37543
   fuentes puntuales extragalácticas (NVSS, 1.4 GHz), parte de los datos
   crudos que alimentan la reconstrucción de Oppermann. Da una comparación
   independiente, punto a punto en el cielo, sin pasar por el suavizado/
   interpolación Gaussiana de la reconstrucción -al costo de que cada
   punto mezcla RM galáctica con una componente propia de la fuente de
   fondo y con ruido de medición (`e_RM`), que este modelo de juguete no
   simula (de ahí que la comparación se haga con estadística, no resta
   directa; ver más abajo).

3. Haslam et al. (1982) 408 MHz (mapa "dsds", HEALPix nside=512, NESTED):
   el estándar para trazar sincrotrón Galáctico de cielo completo. Se usa
   SOLO para comparación MORFOLÓGICA (forma espacial, normalizada), nunca
   cuantitativa: `faradaymr.los_raytrace.sky_map` devuelve I en unidades
   arbitrarias (no hay, en este framework, una población de electrones
   relativistas normalizada en unidades físicas de brillo), así que no
   hay forma honesta de comparar esas unidades contra los Kelvin de
   Haslam -solo la FORMA de la distribución espacial (ver
   `comparar_morfologia_sincrotron`).

Limitación compartida por las tres comparaciones (ver docstring de
`faradaymr.los_raytrace.direction_from_galactic`): la caja de este modelo
de juguete no está rotada a la orientación real de la Vía Láctea en el
cielo -l=0 apunta al centro galáctico (correcto), pero la FASE de los
brazos espirales del modelo en longitud galáctica es arbitraria (fijada
por `phase0=0` en `spiral_arm_density_factor`, no por dónde están de
verdad Norma/Scutum-Centaurus/Sagitario/Perseo). Por eso ninguna función
de este módulo resta punto a punto en (l,b) y reporta eso como "el error
del modelo": exigiría que el modelo reprodujera la posición real de cada
brazo, que no es su objetivo (un modelo de juguete de 4 brazos
logarítmicos idealizados, no un ajuste a la estructura espiral real).

Lo que SÍ es una comparación honesta -y lo que hace este módulo- es
estadística: perfiles promediados/dispersión en función de |b|
(insensibles a la fase en l), histogramas de la distribución global de
valores, el contraste plano/polo, y -para el catálogo puntual- una
comparación punto a punto que es válida precisamente porque no depende de
la fase en l: se interpola el modelo en la posición exacta de cada fuente
y se compara sin promediar, dejando que sea la propia dispersión
estadística (no una coincidencia de fase) la que hable.

Nota sobre la convención de longitud (verificado contra astropy, no dar
por sentado): `faradaymr.los_raytrace.direction_from_galactic` define
x=cos(b)cos(l), y=cos(b)sin(l), z=sin(b), que es EXACTAMENTE la
convención Cartesiana Galáctica real (right-handed, l=90° hacia +y, ver
`astropy.coordinates.Galactic(l=90*u.deg, b=0*u.deg).cartesian` ->
(0,1,0)). No hace falta ningún espejo en l para comparar contra datos
reales: el `mirror_l=True` de `faradaymr.plotting_sky` es solo un flip
izquierda/derecha del DIBUJO en Mollweide (para que en el papel se vea
como en la literatura), no una redefinición del ángulo. Todas las
funciones de este módulo usan l tal cual (sin espejo) en ambos lados de
cada comparación.
"""

from __future__ import annotations

import os

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ_PROYECTO = os.path.dirname(_AQUI)
_DIR_DATOS = os.path.join(_RAIZ_PROYECTO, "data", "external")

RUTA_OPPERMANN2012 = os.path.join(
    _DIR_DATOS, "oppermann2012_galactic_faraday_depth.fits"
)
RUTA_TAYLOR2009 = os.path.join(_DIR_DATOS, "taylor2009_nvss_rm_catalog.tsv")
RUTA_HASLAM408 = os.path.join(_DIR_DATOS, "haslam408_dsds.fits")
RUTA_PLANCK_030GHZ = os.path.join(_DIR_DATOS, "planck_030ghz_iqu.fits")


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------


def load_oppermann2012_map(path: str = RUTA_OPPERMANN2012):
    """
    Carga la reconstrucción de Faraday depth galáctica de Oppermann &
    Enßlin (2012) desde el FITS HEALPix original (6 extensiones, ver
    docstring del módulo). Las extensiones de interés son la 3 (`Faraday
    depth`, rad/m^2, índice de HDU 3 con `hdu=3` en `healpy.read_map`
    porque hay un HDU por extensión, no una sola tabla multi-columna) y la
    4 (su incertidumbre 1-sigma, también en rad/m^2).

    Devuelve un diccionario con el mapa HEALPix crudo (RING, nside=128) y
    su incertidumbre -listos para `project_healpix_to_grid`, no para
    graficar directamente (no son un arreglo (l, b) rectangular).
    """
    import healpy as hp

    rm_healpix = hp.read_map(path, hdu=3)
    rm_err_healpix = hp.read_map(path, hdu=4)
    return {
        "rm_healpix": rm_healpix,
        "rm_err_healpix": rm_err_healpix,
        "nside": hp.get_nside(rm_healpix),
        "nest": False,
        "referencia": "Oppermann & Enßlin (2012), A&A 542, A93; arXiv:1111.6186",
    }


def load_haslam408_map(path: str = RUTA_HASLAM408):
    """
    Carga el mapa de brillo sincrotrón a 408 MHz de Haslam et al. (1982)
    (versión "dsds" servida por LAMBDA/NASA), HEALPix nside=512. El archivo
    en disco está en orden NESTED (`ORDERING='NESTED'` en el header), pero
    `healpy.read_map` ya lo reordena a RING automáticamente antes de
    devolverlo (ver nota más abajo) -el arreglo que llega a quien llama
    esta función SIEMPRE está en RING, igual que el resto de los mapas de
    este módulo.
    """
    import healpy as hp

    # `healpy.read_map` con `nest=False` (el default, no explícito aquí a
    # propósito) NO devuelve el arreglo crudo del FITS tal cual: lee
    # `ORDERING` del header (NESTED en este archivo, ver docstring de la
    # función) y REORDENA automáticamente a RING antes de devolverlo -
    # verificado con un mapa sintético (`hp.reorder(..., n2r=True)` contra
    # el resultado de `read_map`, coinciden exactamente). El diccionario
    # marcaba antes `"nest": True`, lo cual habría hecho que
    # `project_healpix_to_grid(..., nest=True)` interpretara un arreglo ya
    # en RING como si fuera NESTED (doble manejo del reordenamiento),
    # devolviendo valores de píxeles equivocados en la interpolación.
    temp_k = hp.read_map(path, hdu=1)
    return {
        "temperatura_k": temp_k,
        "nside": hp.get_nside(temp_k),
        "nest": False,
        "referencia": "Haslam et al. (1982), reprocesado por LAMBDA/NASA",
    }


def load_planck_030ghz_map(path: str = RUTA_PLANCK_030GHZ):
    """
    Carga el mapa Planck LFI 30 GHz, misión completa (PR3/DX12, "full"):
    I/Q/U en K_CMB, HEALPix nside=1024 -el canal de frecuencia más baja de
    Planck, el más dominado por sincrotrón Galáctico de los 9 disponibles
    (a frecuencias más altas dominan polvo térmico y CMB; ver Planck 2018
    results IV, "Diffuse component separation"), y por eso el elegido para
    comparar contra el I/Q/U sintético de este framework.

    El archivo en disco está en orden NESTED (ver header,
    `ORDERING='NESTED'`); igual que en `load_haslam408_map`,
    `healpy.read_map` ya lo reordena a RING automáticamente antes de
    devolverlo (ver la nota en esa función).
    """
    import healpy as hp

    i_k, q_k, u_k = hp.read_map(path, hdu=1, field=(0, 1, 2))
    return {
        "i_k_cmb": i_k,
        "q_k_cmb": q_k,
        "u_k_cmb": u_k,
        "nside": hp.get_nside(i_k),
        "nest": False,
        "referencia": "Planck Collaboration, mapa LFI 30GHz de misión completa (PR3/DX12)",
    }


def _parse_sexagesimal_dec(token: str) -> float:
    """
    '+DD MM SS.s' o '-DD MM SS.s' -> grados decimales, con signo. No basta
    con `float(partes[0])` para detectar el signo: para declinaciones
    entre -1 y 0 grados ("-00 43 01.3") `float('-00')` da -0.0, y
    `-0.0 < 0` es `False` en Python -perdería el signo silenciosamente.
    Se lee el signo directamente del primer carácter del token en cambio.
    """
    partes = token.split()
    signo = -1.0 if partes[0].strip().startswith("-") else 1.0
    grados = abs(float(partes[0]))
    minutos = float(partes[1])
    segundos = float(partes[2])
    return signo * (grados + minutos / 60.0 + segundos / 3600.0)


def load_taylor2009_catalog(path: str = RUTA_TAYLOR2009):
    """
    Carga el catálogo de rotation measures de Taylor, Stil & Sunstrum
    (2009, ApJ 702, 1230) -37543 fuentes extragalácticas de NVSS a 1.4
    GHz-, descargado de VizieR (J/ApJ/702/1230/catalog) en formato TSV
    (columnas separadas por tab: RAJ2000, DEJ2000, RM, e_RM, Pk, Si, m; las
    dos primeras son strings sexagesimales "HH MM SS.ss"/"+DD MM SS.s", no
    números -de ahí el parseo manual en vez de `np.loadtxt`).

    Convierte RA/Dec (J2000, ICRS) a coordenadas galácticas (l, b) con
    astropy -el catálogo se publica en ecuatoriales, pero todo el resto
    del framework trabaja en (l, b) galácticas (ver
    `faradaymr.los_raytrace`).

    Devuelve un diccionario de arreglos (todos de longitud 37543):
    l_rad, b_rad (radianes, l envuelto a [-pi, pi) igual que `l_grid` de
    `los_raytrace.sky_map`), l_deg, b_deg, rm (rad/m^2), e_rm (rad/m^2,
    incertidumbre 1-sigma de la medición), pk_mjy (pico de intensidad
    polarizada), si_mjy (flujo Stokes I), frac_pol_pct (polarización
    fraccional, %).
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    with open(path, "r") as f:
        lineas = f.readlines()

    inicio = next(i for i, l in enumerate(lineas) if l.startswith("---")) + 1

    ra_deg, dec_deg = [], []
    rm, e_rm, pk, si, m = [], [], [], [], []
    for linea in lineas[inicio:]:
        linea = linea.rstrip("\n")
        if not linea.strip():
            continue
        cols = linea.split("\t")
        h, mi, s = cols[0].split()
        ra_deg.append(15.0 * (float(h) + float(mi) / 60.0 + float(s) / 3600.0))
        dec_deg.append(_parse_sexagesimal_dec(cols[1]))
        rm.append(float(cols[2]))
        e_rm.append(float(cols[3]))
        pk.append(float(cols[4]))
        si.append(float(cols[5]))
        m.append(float(cols[6]))

    coords = SkyCoord(
        ra=np.array(ra_deg) * u.deg, dec=np.array(dec_deg) * u.deg, frame="icrs"
    ).galactic
    l_deg = coords.l.wrap_at("180d").degree
    b_deg = coords.b.degree

    return {
        "l_rad": np.radians(l_deg),
        "b_rad": np.radians(b_deg),
        "l_deg": l_deg,
        "b_deg": b_deg,
        "rm": np.array(rm),
        "e_rm": np.array(e_rm),
        "pk_mjy": np.array(pk),
        "si_mjy": np.array(si),
        "frac_pol_pct": np.array(m),
        "referencia": "Taylor, Stil & Sunstrum (2009), ApJ 702, 1230 (NVSS)",
        "n_fuentes": len(rm),
    }


# --------------------------------------------------------------------------
# Regrillado a la grilla (l, b) nativa del framework
# --------------------------------------------------------------------------


def project_healpix_to_grid(healpix_map, l_grid, b_grid, nest=False):
    """
    Interpola un mapa HEALPix (1D, indexado por píxel) a la grilla
    rectangular (l, b) nativa de `faradaymr.los_raytrace.sky_map`
    -`l_grid`/`b_grid` en RADIANES, misma convención que el resto del
    framework (l=0 hacia el centro galáctico, l creciente en el sentido
    matemático estándar, IGUAL que la convención galáctica real que usan
    estos mapas HEALPix: no hace falta ningún espejo aquí, a diferencia
    del espejo puramente visual que aplica `faradaymr.plotting_sky` solo
    para que el dibujo quede con la orientación convencional en el papel,
    ver docstring del módulo).

    Se usa `healpy.get_interp_val` (interpolación bilineal entre los 4
    píxeles HEALPix vecinos, `lonlat=True`) en vez de un simple
    `ang2pix`+indexado directo: da un mapa suave en la grilla de destino,
    consistente con que la grilla de destino (180x91 por defecto, ver
    `config_fisica.N_L`/`N_B`) es más gruesa que el nside=128/512 de los
    mapas de origen -sin interpolar se vería un aliasing artificial que no
    es parte de los datos.

    Devuelve un arreglo (len(l_grid), len(b_grid)), con la MISMA
    convención de ejes [l, b] que `rm_map`/`i_map`/... de `sky_map`.
    """
    import healpy as hp

    l_deg = np.degrees(np.asarray(l_grid, dtype=float))
    b_deg = np.degrees(np.asarray(b_grid, dtype=float))
    ll, bb = np.meshgrid(l_deg, b_deg, indexing="ij")
    valores = hp.get_interp_val(
        healpix_map, ll.ravel(), bb.ravel(), lonlat=True, nest=nest
    )
    return valores.reshape(ll.shape)


def reproject_fits_to_grid(fits_path, l_grid, b_grid, hdu=0):
    """
    Reproyecta un mapa FITS con WCS celeste convencional (RA/Dec o
    Galactic con `CTYPE`/`CRVAL`/... estándar -NO HEALPix, que no lleva
    WCS de imagen) a la grilla (l, b) nativa del framework, usando
    `reproject` (astropy), tal como sugiere el issue #37 original.

    Por qué esta función existe ADEMÁS de `project_healpix_to_grid`: la
    sugerencia original del issue ("usar reproject de astropy") asume un
    mapa tipo imagen con WCS -el caso típico de un mapa de Planck
    reproyectado a un parche del cielo-, pero el dato real disponible y
    científicamente más directo para RM (Oppermann et al. 2012) viene en
    HEALPix, un formato que `reproject` no entiende nativamente (no tiene
    WCS de imagen, tiene un esquema de pixelización esférico distinto).
    Para HEALPix, la herramienta correcta es `healpy`
    (`project_healpix_to_grid`); esta función queda disponible para el
    caso general -un mapa FITS con WCS de verdad- que el issue original
    contemplaba, por si en el futuro se compara contra un producto de ese
    tipo (p.ej. un mosaico ya reproyectado de Planck/WMAP).

    Construye como destino una proyección CAR (plate carrée) simple sobre
    `l_grid`/`b_grid` (equiespaciados, como los genera `config_fisica`), y
    devuelve el arreglo reproyectado con forma (len(l_grid), len(b_grid)).
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from reproject import reproject_interp

    l_grid = np.asarray(l_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)
    n_l, n_b = len(l_grid), len(b_grid)
    dl_deg = float(np.degrees(l_grid[1] - l_grid[0]))
    db_deg = float(np.degrees(b_grid[1] - b_grid[0]))

    wcs_destino = WCS(naxis=2)
    wcs_destino.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]
    wcs_destino.wcs.crval = [np.degrees(l_grid[0]), np.degrees(b_grid[0])]
    wcs_destino.wcs.crpix = [1, 1]
    wcs_destino.wcs.cdelt = [dl_deg, db_deg]

    with fits.open(fits_path) as hdul:
        datos_origen = hdul[hdu].data
        wcs_origen = WCS(hdul[hdu].header)
        # reproject_interp devuelve (fila, columna) = (b, l); se transpone
        # al final para volver a la convención [l, b] del resto del
        # framework.
        reproyectado, _ = reproject_interp(
            (datos_origen, wcs_origen), wcs_destino, shape_out=(n_b, n_l)
        )
    return reproyectado.T


def subtract_foreground(observed_map, simulated_map):
    """
    Resta delgada RM_observado - RM_modelo (issue #37): una vez que ambos
    mapas están en la MISMA grilla (l, b) -por `project_healpix_to_grid` o
    `reproject_fits_to_grid`, según el formato de origen-, sustraer el
    foreground sintético de un mapa observado es, literalmente, esto: no
    hace falta más código que la resta en sí. Se deja como función
    nombrada (en vez de inline en cada script) porque es la pieza que el
    issue original pide como entregable explícito.
    """
    return np.asarray(observed_map) - np.asarray(simulated_map)


# --------------------------------------------------------------------------
# Estadística de comparación (insensible a la fase arbitraria en l, ver
# docstring del módulo)
# --------------------------------------------------------------------------


def perfil_estadistico_vs_latitud(b_grid, mapa_lb, bins_deg=None, estadistico="rms"):
    """
    Perfil de RM en función de |b|, resumiendo sobre TODAS las longitudes
    de golpe: exactamente la cantidad que sigue siendo comparable entre el
    modelo (fase de brazos arbitraria) y los datos reales (fase real),
    porque promediar en l borra la dependencia en la fase -ver docstring
    del módulo sobre por qué una resta punto a punto en l no es honesta,
    pero un perfil en |b| sí.

    `mapa_lb`: arreglo (n_l, n_b), misma convención que `rm_map` de
    `sky_map` (o lo que devuelve `project_healpix_to_grid`).
    `estadistico`: "rms" (raíz de <RM^2>, sensible a la amplitud típica
    incluyendo el signo) o "mean_abs" (<|RM|>, más robusto a valores
    atípicos aislados).

    Devuelve dict con `centros_deg` (centro de cada bin) y `valores`
    (mismo largo), ignorando bins sin píxeles.
    """
    mapa_lb = np.asarray(mapa_lb)
    b_deg_abs = np.abs(np.degrees(np.asarray(b_grid, dtype=float)))
    if bins_deg is None:
        bins_deg = np.linspace(0.0, float(np.max(b_deg_abs)), 18)

    centros, valores, n_pixeles = [], [], []
    for lo, hi in zip(bins_deg[:-1], bins_deg[1:]):
        mascara = (b_deg_abs >= lo) & (b_deg_abs < hi)
        if not np.any(mascara):
            continue
        datos = mapa_lb[:, mascara]
        datos = datos[np.isfinite(datos)]
        if datos.size == 0:
            continue
        if estadistico == "rms":
            valor = float(np.sqrt(np.mean(datos**2)))
        elif estadistico == "mean_abs":
            valor = float(np.mean(np.abs(datos)))
        else:
            raise ValueError(f"estadistico desconocido: {estadistico!r}")
        centros.append(0.5 * (lo + hi))
        valores.append(valor)
        n_pixeles.append(int(datos.size))

    return {
        "centros_deg": np.array(centros),
        "valores": np.array(valores),
        "n_pixeles": np.array(n_pixeles),
    }


def comparar_con_oppermann(l_grid, b_grid, rm_sim, path=RUTA_OPPERMANN2012):
    """
    Compara el mapa de RM sintético contra la reconstrucción observacional
    de Oppermann & Enßlin (2012), en la misma grilla (l, b).

    Devuelve un diccionario con:
    - `rm_obs_grid`, `err_obs_grid`: el mapa observado (y su
      incertidumbre) regrillado a `(l_grid, b_grid)`.
    - `perfil_modelo`, `perfil_obs`: perfiles RMS(|RM|) vs |b| (ver
      `perfil_estadistico_vs_latitud`) -la comparación estadísticamente
      honesta (ver docstring del módulo).
    - `rms_modelo`, `rms_obs`: RMS global (todas las l, b) de cada mapa.
    - `chi2_reducido_ingenuo`: `mean(((rm_obs - rm_sim) / err_obs)^2)`
      calculado punto a punto -se reporta con la palabra "ingenuo" a
      propósito: NO es un chi^2 válido en sentido estadístico estricto
      (mezcla el error de reconstrucción observacional con una diferencia
      de fase de brazos que no es "ruido", es una limitación estructural
      del modelo de juguete, ver docstring del módulo), pero su ORDEN DE
      MAGNITUD sigue siendo diagnóstico: un valor de unos pocos indica que
      el modelo está en el vecindario correcto punto a punto pese a la
      diferencia de fase; un valor de miles indica una discrepancia de
      amplitud real, no solo de fase.
    """
    import healpy as hp

    rm_obs_healpix = hp.read_map(path, hdu=3)
    err_obs_healpix = hp.read_map(path, hdu=4)

    rm_obs_grid = project_healpix_to_grid(rm_obs_healpix, l_grid, b_grid)
    err_obs_grid = project_healpix_to_grid(err_obs_healpix, l_grid, b_grid)

    rm_sim = np.asarray(rm_sim)
    residual = rm_obs_grid - rm_sim
    chi = residual / err_obs_grid

    return {
        "rm_obs_grid": rm_obs_grid,
        "err_obs_grid": err_obs_grid,
        "residual": residual,
        "perfil_modelo": perfil_estadistico_vs_latitud(b_grid, rm_sim),
        "perfil_obs": perfil_estadistico_vs_latitud(b_grid, rm_obs_grid),
        "rms_modelo": float(np.sqrt(np.nanmean(rm_sim**2))),
        "rms_obs": float(np.sqrt(np.nanmean(rm_obs_grid**2))),
        "chi2_reducido_ingenuo": float(np.nanmean(chi**2)),
        "referencia": "Oppermann & Enßlin (2012), A&A 542, A93",
    }


def comparar_con_catalogo_taylor(l_grid, b_grid, rm_sim, path=RUTA_TAYLOR2009):
    """
    Compara el modelo, punto a punto, contra las 37543 mediciones
    puntuales reales del catálogo NVSS de Taylor et al. (2009): interpola
    `rm_sim` en la posición exacta (l, b) de cada fuente y calcula el
    residuo `RM_catalogo - RM_modelo_interpolado`.

    Por qué esta comparación SÍ es punto a punto (a diferencia de
    `comparar_con_oppermann`, que se limita a perfiles en |b|, ver
    docstring del módulo): no se está afirmando que el modelo prediga la
    RM hacia una fuente en particular -sería exigirle acertar la fase
    exacta de los brazos-, sino usando la dispersión del residuo en
    conjunto como estadístico: si el modelo tiene la amplitud/escala
    correcta, el residuo debería tener una dispersión del orden de la RM
    observada misma (el modelo no explica nada punto a punto, pero
    tampoco debería estar sistemáticamente sesgado ni ser diez veces más
    grande en amplitud).

    La interpolación bilineal en l es PERIÓDICA (se extiende `rm_sim` una
    vuelta completa a cada lado en l antes de interpolar): sin este paso,
    `scipy.interpolate.RegularGridInterpolator` fallaría (devolvería NaN)
    para cualquier fuente cerca del borde l=+-pi, que de otro modo caería
    fuera del dominio de interpolación por un artefacto de índice, no por
    estar realmente fuera de la grilla (el cielo es continuo en l).

    Devuelve dict con `l_deg`, `b_deg`, `rm_obs`, `e_rm_obs`,
    `rm_modelo_interp`, `residual`, y estadísticos globales (`rms_obs`,
    `rms_modelo_en_fuentes`, `rms_residual`, `correlacion_pearson`).
    """
    from scipy.interpolate import RegularGridInterpolator

    catalogo = load_taylor2009_catalog(path)

    l_grid = np.asarray(l_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)
    rm_sim = np.asarray(rm_sim, dtype=float)

    l_ext = np.concatenate([l_grid - 2 * np.pi, l_grid, l_grid + 2 * np.pi])
    rm_ext = np.concatenate([rm_sim, rm_sim, rm_sim], axis=0)
    interpolador = RegularGridInterpolator(
        (l_ext, b_grid), rm_ext, bounds_error=False, fill_value=np.nan
    )

    dentro_de_cobertura = np.abs(catalogo["b_rad"]) <= np.max(np.abs(b_grid))
    l_fuente = catalogo["l_rad"][dentro_de_cobertura]
    b_fuente = catalogo["b_rad"][dentro_de_cobertura]
    rm_obs = catalogo["rm"][dentro_de_cobertura]
    e_rm_obs = catalogo["e_rm"][dentro_de_cobertura]

    rm_modelo_interp = interpolador(np.stack([l_fuente, b_fuente], axis=-1))
    residual = rm_obs - rm_modelo_interp

    valido = np.isfinite(residual)
    r_pearson = float(
        np.corrcoef(rm_obs[valido], rm_modelo_interp[valido])[0, 1]
    )

    return {
        "l_deg": np.degrees(l_fuente),
        "b_deg": np.degrees(b_fuente),
        "rm_obs": rm_obs,
        "e_rm_obs": e_rm_obs,
        "rm_modelo_interp": rm_modelo_interp,
        "residual": residual,
        "n_fuentes_comparadas": int(valido.sum()),
        "rms_obs": float(np.sqrt(np.mean(rm_obs[valido] ** 2))),
        "rms_modelo_en_fuentes": float(np.sqrt(np.mean(rm_modelo_interp[valido] ** 2))),
        "rms_residual": float(np.sqrt(np.mean(residual[valido] ** 2))),
        "correlacion_pearson": r_pearson,
        "referencia": "Taylor, Stil & Sunstrum (2009), ApJ 702, 1230",
    }


def comparar_morfologia_sincrotron(l_grid, b_grid, i_map, path=RUTA_HASLAM408):
    """
    Comparación PURAMENTE morfológica (forma espacial) entre la intensidad
    sincrotrón sintética y el mapa real de 408 MHz de Haslam et al.
    (1982): cada mapa se normaliza por su propio máximo antes de comparar,
    y nunca se reporta una diferencia en unidades físicas.

    Por qué no cuantitativa: `i_map` (de `los_raytrace.sky_map`) está en
    unidades arbitrarias -no hay, en este framework, una densidad de
    electrones relativistas ni una emisividad sincrotrón calibradas contra
    un valor físico real (ver `config_fisica.NE_REL_FRACCION`, una
    fracción ad hoc, no una normalización física)-, mientras que Haslam
    está en Kelvin de temperatura de brillo. Comparar esos números
    directamente sería una afirmación cuantitativa que los propios datos
    del modelo no respaldan; comparar la FORMA (dónde se concentra la
    emisión, qué tan rápido cae con |b|) sí es una validación honesta del
    modelo geométrico (disco + brazos), independiente de cualquier
    normalización de flujo.

    Devuelve dict con los mapas normalizados, el perfil en |b| de cada uno
    (normalizado a 1 en b=0) y la correlación de Pearson en log-log (más
    robusta que en escala lineal dado el enorme rango dinámico plano/polo
    de la emisión sincrotrón, ver `faradaymr.plotting_sky.mollweide_panel`,
    parámetro `escala="log"`).
    """
    import healpy as hp

    # `hp.read_map(path, hdu=1)` ya devuelve el mapa en RING (ver nota en
    # `load_haslam408_map`: con `nest=False` -el default-, healpy reordena
    # de NESTED a RING usando el `ORDERING` del header antes de devolver el
    # arreglo). Pasar `nest=True` acá le diría a
    # `project_healpix_to_grid`/`healpy.get_interp_val` que interprete ese
    # arreglo YA-RING como si todavía fuera NESTED -un doble manejo del
    # reordenamiento que muestrea los píxeles equivocados en cada punto de
    # la grilla de interpolación.
    haslam_healpix = hp.read_map(path, hdu=1)
    haslam_grid = project_healpix_to_grid(haslam_healpix, l_grid, b_grid, nest=False)

    i_map = np.asarray(i_map, dtype=float)
    i_norm = i_map / np.nanmax(i_map)
    haslam_norm = haslam_grid / np.nanmax(haslam_grid)

    log_i = np.log10(np.clip(i_norm, 1e-6, None))
    log_h = np.log10(np.clip(haslam_norm, 1e-6, None))
    r_pearson = float(np.corrcoef(log_i.ravel(), log_h.ravel())[0, 1])

    return {
        "i_norm": i_norm,
        "haslam_norm": haslam_norm,
        "haslam_grid_k": haslam_grid,
        "perfil_i_norm": perfil_estadistico_vs_latitud(
            b_grid, i_norm, estadistico="mean_abs"
        ),
        "perfil_haslam_norm": perfil_estadistico_vs_latitud(
            b_grid, haslam_norm, estadistico="mean_abs"
        ),
        "correlacion_log_pearson": r_pearson,
        "referencia": "Haslam et al. (1982), mapa 408 MHz (LAMBDA/NASA)",
    }


def restar_planck_030ghz(l_grid, b_grid, i_map, path=RUTA_PLANCK_030GHZ):
    """
    Resta RE-ESCALADA del foreground sincrotrón sintético contra Planck
    30 GHz (LFI, misión completa) -la pieza final que pide el issue #37
    ("generar una réplica... para sustraerla de datos de misiones como
    Planck"), con una salvedad que hay que declarar explícitamente: `i_map`
    (de `los_raytrace.sky_map`) está en unidades arbitrarias (ver docstring
    de `comparar_morfologia_sincrotron` para por qué: no hay, en este
    framework, una población de electrones relativistas calibrada contra
    un valor físico real), mientras que Planck está en K_CMB. Restar esos
    dos números directamente, sin reescalar, no tendría sentido físico
    -sería restar unidades incompatibles, no "eliminar la contaminación".

    Por eso esta función AJUSTA la amplitud del modelo antes de restar:
    encuentra, por mínimos cuadrados, el factor de escala `alpha` que
    minimiza sum((I_planck - alpha*I_modelo)^2) sobre toda la grilla
    -una regresión lineal sin término independiente, con solución cerrada
    `alpha = sum(I_planck * I_modelo) / sum(I_modelo^2)`- y resta
    `alpha * I_modelo` con `subtract_foreground`.

    Esto es, deliberadamente, una resta "recalibrada en amplitud", NO una
    resta independiente: no valida si la amplitud ABSOLUTA del modelo es
    correcta (esa validación, con unidades físicas reales, ya la hacen
    `comparar_con_oppermann`/`comparar_con_catalogo_taylor` sobre RM).
    Lo que SÍ valida es si, una vez fijada la amplitud, la FORMA espacial
    sintética explica una fracción razonable de la varianza espacial real
    de Planck a 30 GHz -si el modelo geométrico (disco+brazos+campo
    espiral) está bien orientado, el residuo debería quedar dominado por
    CMB+ruido+polvo, sin la estructura de disco/brazos que un mal ajuste
    de forma dejaría visible.

    Devuelve dict con `i_planck_grid` (K_CMB, regrillado a `l_grid`/
    `b_grid`), `alpha` (factor de reescalado ajustado), `i_modelo_escalado`,
    `residuo` (=`subtract_foreground(i_planck_grid, i_modelo_escalado)`),
    `rms_residuo`/`rms_planck` (K_CMB), y `fraccion_varianza_explicada`
    (1 - var(residuo)/var(planck): 1.0 sería un ajuste de forma perfecto,
    0.0 ningún poder explicativo -el número central de este chequeo).
    """
    import healpy as hp

    i_k, _q_k, _u_k = hp.read_map(path, hdu=1, field=(0, 1, 2))
    i_planck_grid = project_healpix_to_grid(i_k, l_grid, b_grid, nest=False)

    i_map = np.asarray(i_map, dtype=float)
    valido = np.isfinite(i_planck_grid) & np.isfinite(i_map)

    alpha = float(
        np.sum(i_planck_grid[valido] * i_map[valido]) / np.sum(i_map[valido] ** 2)
    )
    i_modelo_escalado = alpha * i_map
    residuo = subtract_foreground(i_planck_grid, i_modelo_escalado)

    varianza_planck = float(np.var(i_planck_grid[valido]))
    varianza_residuo = float(np.var(residuo[valido]))
    fraccion_varianza_explicada = (
        1.0 - varianza_residuo / varianza_planck if varianza_planck > 0 else float("nan")
    )

    return {
        "i_planck_grid": i_planck_grid,
        "alpha": alpha,
        "i_modelo_escalado": i_modelo_escalado,
        "residuo": residuo,
        "rms_residuo": float(np.sqrt(np.mean(residuo[valido] ** 2))),
        "rms_planck": float(np.sqrt(np.mean(i_planck_grid[valido] ** 2))),
        "fraccion_varianza_explicada": fraccion_varianza_explicada,
        "referencia": (
            "Planck Collaboration, mapa LFI 30GHz de misión completa "
            "(PR3/DX12), I_STOKES en K_CMB"
        ),
    }
