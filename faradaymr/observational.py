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


NU_PLANCK_030_HZ = 28.4e9  # frecuencia central efectiva (Planck 2018 II, tabla 4)
NU_HASLAM_HZ = 408e6
T_CMB_K = 2.7255  # Fixsen (2009)


def factor_kcmb_a_krj(nu_hz, t_cmb_k=T_CMB_K):
    """
    Factor que pasa una fluctuación de K_CMB (temperatura termodinámica) a
    K_RJ (temperatura de brillo de Rayleigh-Jeans): x^2 e^x / (e^x - 1)^2,
    con x = h nu / (k T_CMB). A 28.4 GHz vale ~0.98. Hace falta para
    comparar Planck con Haslam, que está en K_RJ: el índice espectral del
    sincrotrón se define sobre temperatura de brillo.
    """
    from scipy.constants import h, k

    x = h * nu_hz / (k * t_cmb_k)
    return x**2 * np.exp(x) / np.expm1(x) ** 2


def load_planck_030ghz_map(path: str = RUTA_PLANCK_030GHZ, nside_salida=None):
    """
    Carga el mapa Planck LFI 30 GHz, misión completa (PR3/DX12, "full"):
    el canal de frecuencia más baja de Planck y el más dominado por
    sincrotrón Galáctico (ver Planck 2018 results IV).

    Lo devuelve listo para comparar contra el modelo:

    - En K_RJ (el archivo está en K_CMB), ver `factor_kcmb_a_krj`.
    - Con U en convención IAU. El archivo usa la convención COSMO
      (`POLCCONV='COSMO'` en el header), que mide el ángulo al revés:
      U_IAU = -U_COSMO. Sin este cambio de signo, los ángulos de
      polarización de Planck quedan espejados respecto a los del modelo.
    - Opcionalmente degradado a `nside_salida` (promedio de píxeles). La
      grilla del modelo tiene píxeles de ~2°, así que comparar a nside=1024
      (~3.4') solo agrega ruido. Las desviaciones estándar del ruido de
      Q y U (`sigma_q_k_rj`, `sigma_u_k_rj`, de las columnas QQ_cov/UU_cov)
      se degradan como varianza de un promedio: var / n_subpíxeles.

    `healpy.read_map` ya reordena el archivo NESTED a RING (ver la nota en
    `load_haslam408_map`).
    """
    import healpy as hp
    from astropy.io import fits

    with fits.open(path) as hdul:
        cabecera = hdul[1].header
        columnas = hdul[1].columns.names
    convencion = str(cabecera.get("POLCCONV", "IAU")).strip().upper()

    i_k, q_k, u_k = hp.read_map(path, hdu=1, field=(0, 1, 2))
    tiene_ruido = "QQ_cov" in columnas and "UU_cov" in columnas
    if tiene_ruido:
        var_q, var_u = hp.read_map(
            path, hdu=1, field=(columnas.index("QQ_cov"), columnas.index("UU_cov"))
        )

    nside_origen = hp.get_nside(i_k)
    if nside_salida is not None and nside_salida != nside_origen:
        i_k, q_k, u_k = hp.ud_grade([i_k, q_k, u_k], nside_salida)
        if tiene_ruido:
            # power=2 divide por (nside_origen/nside_salida)^2 = número de
            # subpíxeles: la varianza de un promedio de N píxeles
            # independientes es var/N.
            var_q, var_u = hp.ud_grade([var_q, var_u], nside_salida, power=2)

    factor = factor_kcmb_a_krj(NU_PLANCK_030_HZ)

    def sin_unseen(mapa):
        return np.where(mapa == hp.UNSEEN, np.nan, mapa)

    signo_u = -1.0 if convencion == "COSMO" else 1.0
    datos = {
        "i_k_rj": sin_unseen(i_k) * factor,
        "q_k_rj": sin_unseen(q_k) * factor,
        "u_k_rj": signo_u * sin_unseen(u_k) * factor,
        "nside": hp.get_nside(i_k),
        "nside_origen": nside_origen,
        "nest": False,
        "convencion_polarizacion_origen": convencion,
        "referencia": "Planck Collaboration, mapa LFI 30GHz de misión completa (PR3/DX12)",
    }
    if tiene_ruido:
        datos["sigma_q_k_rj"] = np.sqrt(sin_unseen(var_q)) * factor
        datos["sigma_u_k_rj"] = np.sqrt(sin_unseen(var_u)) * factor
    return datos


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

    Los promedios se pesan por cos(b) (igual área en el cielo, ver nota en
    el cuerpo). Devuelve dict con `centros_deg` (centro de cada bin) y
    `valores` (mismo largo), ignorando bins sin píxeles.
    """
    mapa_lb = np.asarray(mapa_lb)
    b_rad = np.asarray(b_grid, dtype=float)
    b_deg_abs = np.abs(np.degrees(b_rad))
    if bins_deg is None:
        bins_deg = np.linspace(0.0, float(np.max(b_deg_abs)), 18)

    # La grilla (l, b) es rectangular, no de igual área: un píxel a latitud
    # b cubre un ángulo sólido proporcional a cos(b). Sin este peso, los
    # polos (donde la grilla se apila) pesarían más que el plano en
    # cualquier promedio, y el RMS/media de una banda quedaría sesgado.
    peso_fila = np.cos(b_rad)

    centros, valores, n_pixeles = [], [], []
    for lo, hi in zip(bins_deg[:-1], bins_deg[1:]):
        mascara = (b_deg_abs >= lo) & (b_deg_abs < hi)
        if not np.any(mascara):
            continue
        datos = mapa_lb[:, mascara]
        pesos = np.broadcast_to(peso_fila[mascara], datos.shape)
        bueno = np.isfinite(datos)
        datos, pesos = datos[bueno], pesos[bueno]
        if datos.size == 0:
            continue
        if estadistico == "rms":
            valor = float(np.sqrt(np.average(datos**2, weights=pesos)))
        elif estadistico == "mean_abs":
            valor = float(np.average(np.abs(datos), weights=pesos))
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


def perfil_rms_catalogo_vs_latitud(resultado_catalogo, bins_deg):
    """
    RMS(RM) del catálogo NVSS (Taylor+2009) en bandas de |b|, con el ruido
    de medida restado en cuadratura: sqrt(<RM²> - <σ²>). Sigue incluyendo
    la RM intrínseca de cada fuente (~6-10 rad/m², Schnitzeler 2010), así
    que es una cota superior de la RM galáctica; la reconstrucción de
    Oppermann+2012, que suaviza escalas pequeñas, es la cota inferior.
    `resultado_catalogo`: lo que devuelve `comparar_con_catalogo_taylor`.
    """
    b_abs = np.abs(np.asarray(resultado_catalogo["b_deg"], dtype=float))
    rm = np.asarray(resultado_catalogo["rm_obs"], dtype=float)
    ruido = np.asarray(resultado_catalogo["e_rm_obs"], dtype=float)
    centros, valores, n_fuentes = [], [], []
    for lo, hi in zip(bins_deg[:-1], bins_deg[1:]):
        en_banda = (b_abs >= lo) & (b_abs < hi)
        if np.sum(en_banda) < 10:
            continue
        centros.append(0.5 * (lo + hi))
        valores.append(np.sqrt(max(np.mean(rm[en_banda] ** 2) - np.mean(ruido[en_banda] ** 2), 0.0)))
        n_fuentes.append(int(np.sum(en_banda)))
    return {"centros_deg": np.array(centros), "valores": np.array(valores), "n_fuentes": np.array(n_fuentes)}


def rms_ponderado_por_area(b_grid, mapa_lb):
    """RMS global de un mapa (n_l, n_b) con peso cos(b) por píxel."""
    mapa_lb = np.asarray(mapa_lb, dtype=float)
    pesos = np.broadcast_to(np.cos(np.asarray(b_grid, dtype=float)), mapa_lb.shape)
    ok = np.isfinite(mapa_lb)
    return float(np.sqrt(np.average(mapa_lb[ok] ** 2, weights=pesos[ok])))


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
        "rms_modelo": rms_ponderado_por_area(b_grid, rm_sim),
        "rms_obs": rms_ponderado_por_area(b_grid, rm_obs_grid),
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


# --------------------------------------------------------------------------
# Planck LFI 30 GHz: resta del foreground y validación del modelo
# --------------------------------------------------------------------------

# nside=32 da píxeles de ~1.8°, del orden de la grilla (l, b) del modelo
# (~2° en el perfil "rapido"): degradar a esto antes de interpolar baja el
# ruido de Planck por píxel de ~40 µK a ~1 µK sin perder resolución útil.
NSIDE_VALIDACION_PLANCK = 32

# Criterios de la validación, fijados antes de mirar el resultado (ver
# `validar_contra_planck_030ghz`). Cambiarlos para que el modelo pase
# invalidaría la prueba.
COHERENCIA_ANGULAR_MINIMA = 0.5
SENAL_RUIDO_MINIMA_POLARIZACION = 5.0

# Máscaras de estructuras que ningún modelo de campo galáctico de gran
# escala reproduce (Planck Int. XLII 2016, sec. 3.4 y 3.4.3), con las
# definiciones de la literatura:
# - Loops y arcos visibles en polarización: centro (l, b) y radio en
#   grados, columnas "Polarisation" de la tabla 1 de Vidal et al. (2015,
#   MNRAS 452, 656). Se enmascara una banda de ±5° alrededor de cada
#   cresta (FWHM de las crestas ~5°, Planck 2015 XXV sec. 5.4).
LOOPS_POLARIZACION_VIDAL2015 = {
    "I (North Polar Spur)": (332.6, 20.7, 54.3),
    "III": (118.8, 13.2, 31.6),
    "IV": (315.8, 48.1, 19.3),
    "GCS": (344.0, 4.8, 18.5),
    "IIIS": (106.0, -22.0, 50.0),
    "VIIb": (0.7, -23.3, 45.9),
    "IX": (332.0, 16.0, 46.5),
    "X": (30.0, 35.0, 67.0),
    "XI": (227.0, 38.0, 81.0),
    "XII": (300.0, 0.7, 27.6),
}
SEMIANCHO_MASCARA_LOOPS_DEG = 5.0
# - Centro galáctico: |l| < 10° y |b| < 10°, como en los perfiles de
#   Planck Int. XLII (pie de la Fig. 10).
SEMIANCHO_MASCARA_CENTRO_DEG = 10.0
# - Región del Fan, estructura local muy polarizada: 100° < l < 170°
#   (Planck 2015 XXV sec. 5.4). Planck Int. XLII la excluye del perfil del
#   tercer cuadrante y muestra que deja residuos fuertes en todos los modelos.
REGION_FAN_L_DEG = (100.0, 170.0)

# Regiones de los perfiles en latitud (Planck Int. XLII, Fig. 4): Galaxia
# interior y tercer cuadrante, con l en (-180°, 180°].
REGIONES_PERFIL_LATITUD = {
    "galaxia_interior": (-90.0, 90.0),
    "tercer_cuadrante": (-180.0, -90.0),
}
ANCHO_BANDA_PERFIL_DEG = 5.0


def _pesos_area(b_grid, forma):
    return np.broadcast_to(np.cos(np.asarray(b_grid, dtype=float)), forma)


def _ajuste_lineal_con_fondo(x, y, w):
    """
    Mínimos cuadrados pesados de y ≈ a·x + c. El término constante c hace
    falta porque ningún mapa real tiene el cero del modelo: Haslam tiene
    un nivel cero y un fondo extragaláctico casi isótropo, y Planck el
    residuo de monopolo/dipolo del CMB. Sin c, ese fondo se absorbe en la
    pendiente y la sesga.

    Devuelve (a, c, r2), con r2 = 1 - var(residuo)/var(y), ambas
    varianzas pesadas por w.
    """
    # Se normaliza x antes de resolver: `i_map` a 28.4 GHz vale ~1e-15 en
    # unidades arbitrarias, y junto a una columna de unos `lstsq` descarta
    # ese valor singular por "numéricamente cero" (daba a=0 y r2=0).
    escala = float(np.max(np.abs(x))) or 1.0
    raiz_w = np.sqrt(w)
    diseno = np.column_stack([x / escala, np.ones_like(x)]) * raiz_w[:, None]
    (a_normalizado, c), *_ = np.linalg.lstsq(diseno, y * raiz_w, rcond=None)
    a = a_normalizado / escala

    def var(v):
        return np.average((v - np.average(v, weights=w)) ** 2, weights=w)

    r2 = 1.0 - var(y - (a * x + c)) / var(y)
    return float(a), float(c), float(r2)


def restar_planck_030ghz(
    l_grid, b_grid, i_map, path=RUTA_PLANCK_030GHZ, nside=NSIDE_VALIDACION_PLANCK
):
    """
    Resta del foreground sincrotrón sintético de Planck 30 GHz (el
    entregable del issue #37). `i_map` está en unidades arbitrarias y
    Planck en K, así que antes de restar se ajusta
    I_planck ≈ alpha·I_modelo + fondo (ver `_ajuste_lineal_con_fondo`) y
    se resta solo alpha·I_modelo: el fondo es parte del cielo (CMB, nivel
    cero), no del foreground.

    Esta resta por sí sola NO valida el modelo: con alpha libre, cualquier
    plantilla con un disco explica buena parte de la varianza de Planck.
    La validación está en `validar_contra_planck_030ghz`.

    Devuelve `i_planck_grid` y `residuo` en K_RJ, `alpha`, `fondo_k_rj` y
    `fraccion_varianza_explicada`.
    """
    planck = load_planck_030ghz_map(path, nside_salida=nside)
    i_planck_grid = project_healpix_to_grid(planck["i_k_rj"], l_grid, b_grid)

    i_map = np.asarray(i_map, dtype=float)
    valido = np.isfinite(i_planck_grid) & np.isfinite(i_map)
    w = _pesos_area(b_grid, i_map.shape)[valido]
    alpha, fondo, r2 = _ajuste_lineal_con_fondo(i_map[valido], i_planck_grid[valido], w)

    residuo = subtract_foreground(i_planck_grid, alpha * i_map)
    return {
        "i_planck_grid": i_planck_grid,
        "alpha": alpha,
        "fondo_k_rj": fondo,
        "i_modelo_escalado": alpha * i_map,
        "residuo": residuo,
        "rms_residuo": float(
            np.sqrt(np.average((residuo[valido] - fondo) ** 2, weights=w))
        ),
        "fraccion_varianza_explicada": r2,
        "referencia": planck["referencia"],
    }


def angulo_polarizacion(q, u):
    """Ángulo de polarización (IAU, desde el norte galáctico hacia el este) en rad."""
    return 0.5 * np.arctan2(u, q)


def coherencia_angular(psi_a, psi_b, pesos):
    """
    <cos 2(psi_a - psi_b)> pesado: 1 si los ángulos coinciden, 0 si no
    tienen relación, -1 si son perpendiculares. El factor 2 es porque la
    polarización lineal es simétrica ante psi -> psi + 180°.
    """
    return float(np.average(np.cos(2.0 * (psi_a - psi_b)), weights=pesos))


def mascara_estructuras_locales(l_grid, b_grid):
    """
    True donde el cielo queda FUERA de la comparación: crestas de los loops
    de Vidal et al. (2015), centro galáctico y región del Fan (ver las
    constantes de arriba). Forma (len(l_grid), len(b_grid)).
    """
    l_rad = np.asarray(l_grid, dtype=float)[:, None]
    b_rad = np.asarray(b_grid, dtype=float)[None, :]
    l_deg = np.degrees(l_rad) * np.ones_like(b_rad)
    b_deg = np.degrees(b_rad) * np.ones_like(l_rad)

    mascara = (np.abs(l_deg) < SEMIANCHO_MASCARA_CENTRO_DEG) & (
        np.abs(b_deg) < SEMIANCHO_MASCARA_CENTRO_DEG
    )
    l_360 = np.mod(l_deg, 360.0)
    mascara |= (l_360 > REGION_FAN_L_DEG[0]) & (l_360 < REGION_FAN_L_DEG[1])

    for l_c, b_c, radio in LOOPS_POLARIZACION_VIDAL2015.values():
        l_c, b_c = np.radians(l_c), np.radians(b_c)
        cos_sep = np.sin(b_rad) * np.sin(b_c) + np.cos(b_rad) * np.cos(b_c) * np.cos(l_rad - l_c)
        separacion = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        mascara |= np.abs(separacion - radio) < SEMIANCHO_MASCARA_LOOPS_DEG
    return mascara


def _perfil_latitud_con_signo(b_grid, mapa, valido, l_en_region, pesos):
    """Media pesada por cos b en bandas de b con signo, solo en píxeles válidos de la región."""
    b_deg = np.degrees(np.asarray(b_grid, dtype=float))
    bordes = np.arange(-85.0, 85.0 + 1e-9, ANCHO_BANDA_PERFIL_DEG)
    centros, valores = [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        filas = (b_deg >= lo) & (b_deg < hi)
        seleccion = valido & l_en_region[:, None] & filas[None, :]
        if not np.any(seleccion):
            continue
        centros.append(0.5 * (lo + hi))
        valores.append(np.average(mapa[seleccion], weights=pesos[seleccion]))
    return np.array(centros), np.array(valores)


def validar_contra_planck_030ghz(
    l_grid,
    b_grid,
    q_ensamble,
    u_ensamble,
    path_planck=RUTA_PLANCK_030GHZ,
    nside=NSIDE_VALIDACION_PLANCK,
):
    """
    Valida el campo magnético del modelo contra la polarización de Planck
    LFI 30 GHz siguiendo el procedimiento de Planck Collaboration Int. XLII
    (2016, A&A 596, A103), la comparación de referencia de modelos de campo
    galáctico (Sun10, JF12, Jaffe13) contra Planck:

    - Solo polarización. La intensidad total a 30 GHz está contaminada por
      emisión anómala de polvo y free-free (XLII sec. 3.4.1; Planck 2015
      XXV), y la solución Commander de sincrotrón es Haslam 408 MHz con un
      espectro fijo, así que un índice espectral medido con I sería
      contaminación o circularidad. A 30 GHz, P casi solo tiene sincrotrón
      y la rotación de Faraday es despreciable.
    - Varianza galáctica (XLII sec. 3.4.1): el cielo es una realización de
      la turbulencia. `q_ensamble`/`u_ensamble` (forma (N, n_l, n_b), a
      28.4 GHz) son N realizaciones del modelo; cada estadístico se da como
      media ± desviación estándar sobre ellas.
    - Máscaras (XLII sec. 3.4.3): crestas de loops y spurs (Vidal et al.
      2015), centro galáctico y región del Fan, que ningún modelo de gran
      escala incluye (ver `mascara_estructuras_locales`).

    Pruebas, con criterios fijados antes de ver el resultado:

    1. Ángulo de polarización: <cos 2Δψ> (pesado por P de Planck y cos b,
       en píxeles no enmascarados con P/σ_P >= 5) >= 0.5 y mayor que la
       plantilla trivial ψ = 0 (campo paralelo al plano).
    2. Forma de P: R² de P_planck ≈ a·P_modelo + c mayor que el de un disco
       plano-paralelo (P ∝ 1/sin|b|), en el cielo no enmascarado.
    3. Perfiles de P en latitud (XLII Fig. 4) en la Galaxia interior y el
       tercer cuadrante, con una sola normalización libre (degenerada con la
       de los electrones relativistas, como en XLII). Es informativa: se
       reportan los residuos en unidades de σ² = σ_varianza_galáctica² +
       σ_ruido², sin umbral, porque XLII encuentra residuos mayores que la
       varianza del modelo incluso para los modelos de la literatura.

    `valida` es True si pasan 1 y 2 con la media del ensamble.
    """
    planck = load_planck_030ghz_map(path_planck, nside_salida=nside)

    def a_grilla(mapa_healpix):
        return project_healpix_to_grid(mapa_healpix, l_grid, b_grid)

    q_planck = a_grilla(planck["q_k_rj"])
    u_planck = a_grilla(planck["u_k_rj"])
    sigma_p = np.sqrt(
        0.5 * (a_grilla(planck["sigma_q_k_rj"]) ** 2 + a_grilla(planck["sigma_u_k_rj"]) ** 2)
    )
    # Sesgo de ruido de P = sqrt(Q²+U²): se resta en cuadratura (Wardle &
    # Kronberg 1974).
    p_planck = np.sqrt(np.clip(q_planck**2 + u_planck**2 - sigma_p**2, 0.0, None))
    psi_planck = angulo_polarizacion(q_planck, u_planck)

    q_ensamble = np.asarray(q_ensamble, dtype=float)
    u_ensamble = np.asarray(u_ensamble, dtype=float)
    p_ensamble = np.hypot(q_ensamble, u_ensamble)
    forma = p_planck.shape
    pesos = np.array(_pesos_area(b_grid, forma))
    abs_b = np.abs(np.broadcast_to(np.asarray(b_grid, dtype=float), forma))
    enmascarado = mascara_estructuras_locales(l_grid, b_grid)
    usable = np.isfinite(p_planck) & ~enmascarado

    def media_y_dispersion(valores):
        valores = np.asarray(valores, dtype=float)
        return float(np.mean(valores)), float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0

    # --- 1. Ángulo de polarización ------------------------------------------
    detectado = usable & (p_planck >= SENAL_RUIDO_MINIMA_POLARIZACION * sigma_p)
    w_pol = (pesos * p_planck)[detectado]
    coherencias = [
        coherencia_angular(angulo_polarizacion(q, u)[detectado], psi_planck[detectado], w_pol)
        for q, u in zip(q_ensamble, u_ensamble)
    ]
    coherencia_media, coherencia_dispersion = media_y_dispersion(coherencias)
    coherencia_trivial = coherencia_angular(0.0, psi_planck[detectado], w_pol)
    prueba_angulo = {
        "coherencia_modelo_media": coherencia_media,
        "coherencia_modelo_dispersion_entre_realizaciones": coherencia_dispersion,
        "coherencia_plantilla_campo_paralelo_al_plano": coherencia_trivial,
        "fraccion_cielo_usada": float(np.sum(pesos[detectado]) / np.sum(pesos)),
        "criterio": (
            f"<cos 2Δψ> >= {COHERENCIA_ANGULAR_MINIMA} y mayor que la plantilla ψ=0, "
            f"en píxeles no enmascarados con P/σ >= {SENAL_RUIDO_MINIMA_POLARIZACION:.0f}"
        ),
        "pasa": bool(
            coherencia_media >= COHERENCIA_ANGULAR_MINIMA and coherencia_media > coherencia_trivial
        ),
    }

    # --- 2. Forma de la intensidad polarizada -------------------------------
    paso_b = float(np.min(np.abs(np.diff(np.asarray(b_grid, dtype=float)))))
    # El disco plano-paralelo diverge en b=0: se corta en el tamaño del píxel.
    p_disco = 1.0 / np.sin(np.maximum(abs_b, paso_b))
    r2_modelo = [
        _ajuste_lineal_con_fondo(p[usable], p_planck[usable], pesos[usable])[2] for p in p_ensamble
    ]
    r2_media, r2_dispersion = media_y_dispersion(r2_modelo)
    _, _, r2_disco = _ajuste_lineal_con_fondo(p_disco[usable], p_planck[usable], pesos[usable])
    prueba_forma_p = {
        "r2_modelo_media": r2_media,
        "r2_modelo_dispersion_entre_realizaciones": r2_dispersion,
        "r2_disco_plano_paralelo": r2_disco,
        "criterio": "R² del modelo > R² de un disco plano-paralelo (P ∝ 1/sin|b|), cielo no enmascarado",
        "pasa": bool(r2_media > r2_disco),
    }

    # --- 3. Perfiles de P en latitud (informativa) --------------------------
    l_deg = np.degrees(np.asarray(l_grid, dtype=float))
    perfiles = {}
    for nombre, (l_min, l_max) in REGIONES_PERFIL_LATITUD.items():
        en_region = (l_deg > l_min) & (l_deg <= l_max)
        centros, perfil_datos = _perfil_latitud_con_signo(b_grid, p_planck, usable, en_region, pesos)
        perfiles_modelo = np.array([
            _perfil_latitud_con_signo(b_grid, p, usable, en_region, pesos)[1] for p in p_ensamble
        ])
        # Ruido de la media pesada de cada banda: sqrt(Σ w² σ²) / Σ w.
        ruido = np.array([
            np.sqrt(np.sum((pesos * sigma_p)[s] ** 2)) / np.sum(pesos[s])
            for s in (
                usable & en_region[:, None]
                & ((np.degrees(np.asarray(b_grid))[None, :] >= c - ANCHO_BANDA_PERFIL_DEG / 2)
                   & (np.degrees(np.asarray(b_grid))[None, :] < c + ANCHO_BANDA_PERFIL_DEG / 2))
                for c in centros
            )
        ])
        perfiles[nombre] = {
            "b_deg": centros,
            "datos": perfil_datos,
            "modelo_realizaciones": perfiles_modelo,
            "ruido": ruido,
        }

    # Una normalización común para las dos regiones, por mínimos cuadrados
    # pesados por 1/σ², con σ de varianza galáctica y ruido.
    def componentes(perfil):
        media = perfil["modelo_realizaciones"].mean(axis=0)
        varianza_galactica = perfil["modelo_realizaciones"].std(axis=0, ddof=1)
        return media, varianza_galactica

    num = den = 0.0
    for perfil in perfiles.values():
        media, gv = componentes(perfil)
        # Primera pasada: σ con la varianza galáctica sin escalar da un
        # peso relativo razonable; la normalización se ajusta con él.
        peso = 1.0 / (perfil["ruido"] ** 2 + (gv * np.mean(perfil["datos"]) / np.mean(media)) ** 2)
        num += np.sum(peso * perfil["datos"] * media)
        den += np.sum(peso * media**2)
    normalizacion = num / den

    prueba_perfiles = {"normalizacion_k_rj_por_unidad_modelo": float(normalizacion)}
    for nombre, perfil in perfiles.items():
        media, gv = componentes(perfil)
        sigma_total = np.sqrt(perfil["ruido"] ** 2 + (normalizacion * gv) ** 2)
        residuo = (perfil["datos"] - normalizacion * media) / sigma_total
        perfil.update(
            modelo_media=normalizacion * media,
            sigma_total=sigma_total,
            residuo_en_sigmas=residuo,
        )
        prueba_perfiles[nombre] = {
            "chi2_reducido": float(np.mean(residuo**2)),
            "fraccion_bandas_dentro_de_3_sigma": float(np.mean(np.abs(residuo) <= 3.0)),
            "razon_datos_sobre_modelo_mediana": float(np.median(perfil["datos"] / (normalizacion * media))),
        }
    prueba_perfiles["criterio"] = (
        "informativa (Planck Int. XLII Fig. 4): residuos en σ = varianza galáctica ⊕ ruido"
    )

    p_modelo = p_ensamble[0]
    a_p, c_p, _ = _ajuste_lineal_con_fondo(p_modelo[usable], p_planck[usable], pesos[usable])
    psi_modelo = angulo_polarizacion(q_ensamble[0], u_ensamble[0])
    return {
        "angulo_polarizacion": prueba_angulo,
        "forma_intensidad_polarizada": prueba_forma_p,
        "perfiles_latitud_p": prueba_perfiles,
        "n_realizaciones": int(len(q_ensamble)),
        "valida": prueba_angulo["pasa"] and prueba_forma_p["pasa"],
        "mapas": {
            "p_planck_k_rj": p_planck,
            "p_modelo_escalado_k_rj": a_p * p_modelo + c_p,
            "delta_psi": np.where(
                detectado, 0.5 * np.angle(np.exp(2j * (psi_modelo - psi_planck))), np.nan
            ),
            "mascara": enmascarado,
        },
        "perfiles": perfiles,
        "referencia": (
            f"{planck['referencia']}; procedimiento de Planck Collaboration Int. XLII (2016); "
            "máscaras de Vidal et al. (2015) y Planck 2015 XXV"
        ),
    }


# Regiones para medir la remoción del foreground polarizado (en l de 0 a
# 360 y |b|). El plano interior concentra ~80 % de la potencia polarizada
# de Planck a 30 GHz y es donde la geometría real (tangencias de brazos,
# inversiones, campo en X) se aleja más de un modelo de juguete, así que se
# reporta por separado en vez de dejar que domine un número global. La
# franja |b| < 3° va aparte: ahí Q y U de Planck siguen a la intensidad
# total (r = -0.76 y -0.89 en |l| < 60°, |b| < 2°, con Q/I ~ U/I ~ -1.5 %)
# aunque a 30 GHz esa intensidad es sobre todo free-free y emisión anómala
# de polvo, que no polarizan: es la firma de la fuga de intensidad a
# polarización por desajuste de banda de LFI (Planck 2018 II), no de
# sincrotrón.
REGIONES_REMOCION = {
    "franja_interior": {"l_deg": ((0.0, 90.0), (270.0, 360.0)), "b_abs_deg": (0.0, 3.0)},
    "plano_interior": {"l_deg": ((0.0, 90.0), (270.0, 360.0)), "b_abs_deg": (3.0, 20.0)},
    "plano_exterior": {"l_deg": ((90.0, 270.0),), "b_abs_deg": (0.0, 20.0)},
    "alta_latitud": {"l_deg": ((0.0, 360.0),), "b_abs_deg": (20.0, 90.0)},
}


def _seleccion_region(l_grid, b_grid, region):
    l_deg = np.mod(np.degrees(np.asarray(l_grid, dtype=float)), 360.0)[:, None]
    b_abs = np.abs(np.degrees(np.asarray(b_grid, dtype=float)))[None, :]
    en_l = np.zeros_like(l_deg, dtype=bool)
    for lo, hi in region["l_deg"]:
        en_l |= (l_deg >= lo) & (l_deg < hi)
    lo_b, hi_b = region["b_abs_deg"]
    return en_l & (b_abs >= lo_b) & (b_abs < hi_b)


def remover_foreground_polarizado_planck(
    l_grid,
    b_grid,
    q_ensamble,
    u_ensamble,
    path_planck=RUTA_PLANCK_030GHZ,
    nside=NSIDE_VALIDACION_PLANCK,
):
    """
    Sustrae el foreground sincrotrón polarizado sintético de Planck LFI
    30 GHz por ajuste de plantilla (template fitting, la técnica más simple
    de separación de componentes que usa Planck; Planck 2015 X): en cada
    región se ajusta una amplitud libre `a` por mínimos cuadrados sobre Q y
    U a la vez,

        a = Σ w (Q_p Q_m + U_p U_m) / Σ w (Q_m² + U_m²),

    y se resta a·(Q_m, U_m). La amplitud es libre porque la emisividad del
    modelo no está en unidades físicas; lo que se pone a prueba es la
    geometría (dónde y con qué ángulo polariza la Galaxia), no la
    normalización.

    Plantilla: la media de las realizaciones de turbulencia a 28.4 GHz
    (`q_ensamble`, `u_ensamble`, forma (N, n_l, n_b)), que es la
    predicción del modelo para el cielo -el cielo real es una realización
    desconocida de la turbulencia. Se reporta también la dispersión al usar
    cada realización por separado.

    Métrica: fracción de la potencia polarizada removida, con el ruido de
    Planck restado (pesos cos b, sin las estructuras locales de
    `mascara_estructuras_locales`):

        f = 1 - Σ w (|P_residuo|² - σ²) / Σ w (|P_Planck|² - σ²)

    f = 1 es remoción perfecta, f = 0 no remueve nada (también si la
    plantilla está anticorrelacionada: la amplitud se restringe a a >= 0),
    f < 0 agrega potencia. Se compara contra una plantilla trivial (campo paralelo al
    plano con P ∝ 1/sin|b|) y contra una amplitud única para todo el cielo
    (que mide si el modelo reparte bien la emisión entre regiones).
    """
    planck = load_planck_030ghz_map(path_planck, nside_salida=nside)

    def a_grilla(mapa_healpix):
        return project_healpix_to_grid(mapa_healpix, l_grid, b_grid)

    q_p, u_p = a_grilla(planck["q_k_rj"]), a_grilla(planck["u_k_rj"])
    varianza_ruido = a_grilla(planck["sigma_q_k_rj"]) ** 2 + a_grilla(planck["sigma_u_k_rj"]) ** 2

    q_ensamble = np.asarray(q_ensamble, dtype=float)
    u_ensamble = np.asarray(u_ensamble, dtype=float)
    q_m, u_m = q_ensamble.mean(axis=0), u_ensamble.mean(axis=0)

    forma = q_p.shape
    pesos = np.array(_pesos_area(b_grid, forma))
    enmascarado = mascara_estructuras_locales(l_grid, b_grid)
    usable = np.isfinite(q_p) & np.isfinite(u_p) & ~enmascarado
    abs_b = np.abs(np.broadcast_to(np.asarray(b_grid, dtype=float), forma))
    paso_b = float(np.min(np.abs(np.diff(np.asarray(b_grid, dtype=float)))))
    q_trivial, u_trivial = 1.0 / np.sin(np.maximum(abs_b, paso_b)), np.zeros(forma)

    def amplitud(sel, q_t, u_t):
        # Mínimos cuadrados con a >= 0: una amplitud negativa sería una
        # emisividad negativa; si la plantilla está anticorrelacionada con
        # Planck, no remueve nada (a = 0) en vez de "remover" con el signo
        # cambiado.
        a = np.sum(pesos[sel] * (q_p[sel] * q_t[sel] + u_p[sel] * u_t[sel])) / np.sum(
            pesos[sel] * (q_t[sel] ** 2 + u_t[sel] ** 2)
        )
        return float(max(a, 0.0))

    def fraccion_removida(sel, a, q_t, u_t):
        r_q, r_u = q_p - a * q_t, u_p - a * u_t
        residuo = np.sum(pesos[sel] * (r_q[sel] ** 2 + r_u[sel] ** 2 - varianza_ruido[sel]))
        total = np.sum(pesos[sel] * (q_p[sel] ** 2 + u_p[sel] ** 2 - varianza_ruido[sel]))
        return float(1.0 - residuo / total)

    potencia_total = np.sum((pesos * (q_p**2 + u_p**2))[usable])
    # La amplitud única se ajusta sin la franja con posible fuga I→P (ver
    # REGIONES_REMOCION), que si no la fija ella sola.
    sin_franja = usable & ~_seleccion_region(l_grid, b_grid, REGIONES_REMOCION["franja_interior"])
    amplitud_global = amplitud(sin_franja, q_m, u_m)
    q_restado, u_restado = np.full(forma, np.nan), np.full(forma, np.nan)
    regiones = {}
    for nombre, region in REGIONES_REMOCION.items():
        sel = usable & _seleccion_region(l_grid, b_grid, region)
        a = amplitud(sel, q_m, u_m)
        por_realizacion = [
            fraccion_removida(sel, amplitud(sel, q, u), q, u) for q, u in zip(q_ensamble, u_ensamble)
        ]
        regiones[nombre] = {
            "fraccion_potencia_planck": float(np.sum((pesos * (q_p**2 + u_p**2))[sel]) / potencia_total),
            "amplitud": a,
            "fraccion_removida_modelo": fraccion_removida(sel, a, q_m, u_m),
            "fraccion_removida_una_realizacion_media": float(np.mean(por_realizacion)),
            "fraccion_removida_una_realizacion_dispersion": float(np.std(por_realizacion, ddof=1)),
            "fraccion_removida_plantilla_trivial": fraccion_removida(
                sel, amplitud(sel, q_trivial, u_trivial), q_trivial, u_trivial
            ),
            "fraccion_removida_amplitud_global": fraccion_removida(sel, amplitud_global, q_m, u_m),
        }
        q_restado[sel], u_restado[sel] = a * q_m[sel], a * u_m[sel]

    return {
        "regiones": regiones,
        "fraccion_removida_sin_franja_amplitud_global": fraccion_removida(
            sin_franja, amplitud_global, q_m, u_m
        ),
        "n_realizaciones": int(len(q_ensamble)),
        "mapas": {
            "p_planck_k_rj": np.where(usable, np.hypot(q_p, u_p), np.nan),
            "p_plantilla_k_rj": np.hypot(q_restado, u_restado),
            "p_residuo_k_rj": np.where(usable, np.hypot(q_p - q_restado, u_p - u_restado), np.nan),
        },
        "criterio": (
            "fracción de la potencia polarizada de Planck 30 GHz removida (ruido restado), "
            "amplitud libre (>= 0) por región; comparada con una plantilla trivial y con "
            "una amplitud única para todo el cielo sin la franja |b|<3° interior"
        ),
        "referencia": (
            f"{planck['referencia']}; ajuste de plantilla como en Planck 2015 X (A&A 594, A10)"
        ),
    }


# Regiones para el signo de la RM (l de 0 a 360). Son las que fijan la
# geometría del campo regular: el plano por cuadrante (inversiones del
# disco) y latitudes medias hacia la Galaxia interior (antisimetría
# norte-sur del halo).
REGIONES_SIGNO_RM = {
    "plano, l=20-90": ((20.0, 90.0), (-5.0, 5.0)),
    "plano, l=90-180": ((90.0, 180.0), (-5.0, 5.0)),
    "plano, l=180-270": ((180.0, 270.0), (-5.0, 5.0)),
    "plano, l=270-340": ((270.0, 340.0), (-5.0, 5.0)),
    "norte, l=0-90, b=10-45": ((0.0, 90.0), (10.0, 45.0)),
    "sur, l=0-90, b=-45-(-10)": ((0.0, 90.0), (-45.0, -10.0)),
    "norte, l=270-360, b=10-45": ((270.0, 360.0), (10.0, 45.0)),
    "sur, l=270-360, b=-45-(-10)": ((270.0, 360.0), (-45.0, -10.0)),
}


def signo_rm_por_region(l_grid, b_grid, rm_ensamble, rm_observado):
    """
    RM media (pesada por cos b) en cada región de `REGIONES_SIGNO_RM`, para
    el modelo -media y dispersión entre las realizaciones de turbulencia de
    `rm_ensamble` (N, n_l, n_b)- y para el mapa observado. El signo cuenta
    como reproducido si la media del ensamble tiene el signo observado; la
    dispersión dice si una sola realización podría tenerlo cambiado.
    """
    l_deg = np.mod(np.degrees(np.asarray(l_grid, dtype=float)), 360.0)[:, None]
    b_deg = np.degrees(np.asarray(b_grid, dtype=float))[None, :]
    rm_ensamble = np.asarray(rm_ensamble, dtype=float)
    pesos = np.array(_pesos_area(b_grid, rm_ensamble.shape[1:]))
    resultado = {}
    for nombre, ((l_lo, l_hi), (b_lo, b_hi)) in REGIONES_SIGNO_RM.items():
        sel = (l_deg >= l_lo) & (l_deg < l_hi) & (b_deg >= b_lo) & (b_deg < b_hi)
        sel = np.broadcast_to(sel, pesos.shape) & np.isfinite(rm_observado)
        medias = [float(np.average(rm[sel], weights=pesos[sel])) for rm in rm_ensamble]
        observado = float(np.average(np.asarray(rm_observado)[sel], weights=pesos[sel]))
        media = float(np.mean(medias))
        resultado[nombre] = {
            "rm_modelo_media": media,
            "rm_modelo_dispersion": float(np.std(medias, ddof=1)),
            "rm_observado": observado,
            "signo_correcto": bool(np.sign(media) == np.sign(observado)),
        }
    return resultado
