"""
Figuras de comparación entre el foreground galáctico sintético y los datos
observacionales reales cargados por `faradaymr.observational` (issue #37).

Separado de `faradaymr.plotting_sky` (que solo dibuja el mapa sintético en
sí) porque estas figuras necesitan dos fuentes de datos a la vez -modelo y
observación- y, en el caso del catálogo puntual, un tipo de gráfico
distinto (densidad, histograma) que no encaja en el molde de "un panel
Mollweide por observable" del otro módulo.

Convenciones de color en `faradaymr.estilo_figuras`: modelo en naranja,
observaciones de referencia en azul, NVSS en verde agua, referencias sin
física en gris.
"""

from __future__ import annotations

import os

import numpy as np

from . import estilo_figuras as estilo
from .plotting_sky import _preparar_grilla_mollweide, mollweide_panel


def _guardar(fig, ruta_destino, nombre_archivo):
    import matplotlib.pyplot as plt

    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa)
    plt.close(fig)
    return ruta_completa


def _barra_de_color(fig, malla, ax, etiqueta, extend="neither"):
    cbar = fig.colorbar(malla, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07,
                        aspect=30, extend=extend)
    cbar.set_label(etiqueta, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.grid(False)
    return cbar


def mapa_comparacion_mollweide(
    ruta_destino,
    l_grid,
    b_grid,
    rm_modelo,
    rm_obs_grid,
    nombre_archivo="comparacion_mapa_rm.png",
):
    """
    RM del modelo, RM observada (Oppermann & Enßlin 2012) y su residuo
    (observado - modelo), los tres con la MISMA escala de color: un mismo
    tono significa la misma RM en los tres paneles, así que el residuo se
    lee directamente contra la señal (un residuo pálido es un buen ajuste).
    """
    import matplotlib.pyplot as plt

    estilo.aplicar()
    rm_modelo = np.asarray(rm_modelo)
    rm_obs_grid = np.asarray(rm_obs_grid)
    amplitud = float(max(np.nanpercentile(np.abs(rm_modelo), 99),
                         np.nanpercentile(np.abs(rm_obs_grid), 99)))

    fig = plt.figure(figsize=(14, 4.4))
    for i, (mapa, titulo) in enumerate([
        (rm_modelo, estilo.ETIQUETA_MODELO),
        (rm_obs_grid, f"Observado ({estilo.ETIQUETA_OPPERMANN})"),
        (rm_obs_grid - rm_modelo, "Residuo (observado − modelo)"),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        malla = mollweide_panel(ax, l_grid, b_grid, mapa, cmap=estilo.MAPA_SIGNO,
                                simetrico=True, vmax=amplitud)
        ax.set_title(titulo, pad=10)
        _barra_de_color(fig, malla, ax, r"RM [rad m$^{-2}$]", extend="both")

    fig.tight_layout()
    fig.suptitle("Medida de rotación de Faraday: modelo contra el cielo real", fontsize=12, y=0.9)
    return _guardar(fig, ruta_destino, nombre_archivo)


def perfil_comparacion_latitud(
    ruta_destino,
    perfil_modelo,
    perfil_obs,
    nombre_archivo="comparacion_perfil_latitud.png",
    etiqueta_obs=estilo.ETIQUETA_OPPERMANN,
    perfil_catalogo=None,
):
    """
    RMS(RM) contra |b| para el modelo y los datos, escala log en y (el
    contraste plano-polo abarca más de un orden de magnitud). Es la
    comparación de amplitud central: no depende de la fase arbitraria de
    los brazos del modelo.

    Con `perfil_catalogo` (de `perfil_rms_catalogo_vs_latitud`) se dibuja
    también NVSS y se sombrea la franja entre las dos estimaciones
    observacionales: Oppermann suaviza las escalas pequeñas (cota
    inferior) y NVSS incluye la RM intrínseca de cada fuente (cota
    superior), así que la RM galáctica real está entre ambas.
    """
    import matplotlib.pyplot as plt

    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    if perfil_catalogo is not None:
        centros = np.asarray(perfil_catalogo["centros_deg"])
        cat = np.asarray(perfil_catalogo["valores"])
        opp = np.interp(centros, perfil_obs["centros_deg"], perfil_obs["valores"])
        ax.fill_between(centros, np.minimum(opp, cat), np.maximum(opp, cat),
                        color=estilo.REFERENCIA, alpha=0.18, lw=0,
                        label="rango observacional (entre las dos estimaciones)")
        ax.semilogy(centros, cat, "^-", color=estilo.NVSS, ms=5,
                    label=f"{estilo.ETIQUETA_NVSS}, ruido de medida restado")
    ax.semilogy(perfil_obs["centros_deg"], perfil_obs["valores"], "s-",
                color=estilo.OBSERVADO, ms=5, label=etiqueta_obs)
    ax.semilogy(perfil_modelo["centros_deg"], perfil_modelo["valores"], "o-",
                color=estilo.MODELO, ms=5, label=estilo.ETIQUETA_MODELO)
    ax.set_xlabel("|b| [grados]")
    ax.set_ylabel(r"RMS de la RM [rad m$^{-2}$]")
    ax.set_xlim(0, 85)
    ax.set_title("Amplitud de la RM contra latitud")
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


def dispersion_catalogo_vs_modelo(
    ruta_destino,
    resultado_comparacion_catalogo,
    nombre_archivo="comparacion_catalogo_nvss.png",
):
    """
    RM del modelo en la posición de cada fuente NVSS contra la RM medida.
    Con 37 000 fuentes una nube de puntos se satura, así que se dibuja la
    densidad (histograma 2D, escala log) y, encima, la mediana del modelo
    en intervalos de RM observada: si el modelo sigue a los datos, esa
    curva va sobre la diagonal 1:1. Dos paneles, plano y fuera del plano,
    porque la RM cambia un orden de magnitud entre ellos.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    estilo.aplicar()
    r = resultado_comparacion_catalogo
    obs = np.asarray(r["rm_obs"])
    mod = np.asarray(r["rm_modelo_interp"])
    abs_b = np.abs(np.asarray(r["b_deg"]))
    valido = np.isfinite(mod) & np.isfinite(obs)
    densidad = LinearSegmentedColormap.from_list("densidad", ["#eef3fb", estilo.OBSERVADO, "#0d366b"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (seleccion, titulo) in zip(axes, [
        (abs_b < 10.0, r"Plano, $|b| < 10^\circ$"),
        (abs_b >= 10.0, r"Fuera del plano, $|b| \geq 10^\circ$"),
    ]):
        s = valido & seleccion
        x, y = obs[s], mod[s]
        limite = float(np.percentile(np.abs(np.concatenate([x, y])), 98))
        bordes = np.linspace(-limite, limite, 61)
        h = ax.hist2d(x, y, bins=[bordes, bordes], cmap=densidad, norm=LogNorm(), cmin=1)
        ax.plot([-limite, limite], [-limite, limite], "--", color=estilo.REFERENCIA, lw=1.2,
                label="1:1")
        intervalos = np.linspace(-limite, limite, 13)
        centro, mediana = [], []
        for lo, hi in zip(intervalos[:-1], intervalos[1:]):
            en = (x >= lo) & (x < hi)
            if en.sum() >= 30:
                centro.append(0.5 * (lo + hi))
                mediana.append(np.median(y[en]))
        ax.plot(centro, mediana, "o-", color=estilo.MODELO, ms=5,
                label="mediana del modelo por intervalo")
        coef = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{titulo}:  r = {coef:+.2f},  N = {s.sum()}")
        ax.set_xlim(-limite, limite)
        ax.set_ylim(-limite, limite)
        ax.set_aspect("equal")
        ax.set_xlabel(r"RM medida, NVSS [rad m$^{-2}$]")
        ax.grid(False)
        cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("fuentes por celda", fontsize=8.5)
    axes[0].set_ylabel(r"RM del modelo en la fuente [rad m$^{-2}$]")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        f"Modelo contra {r['n_fuentes_comparadas']} fuentes extragalácticas del catálogo NVSS "
        f"(r total = {r['correlacion_pearson']:+.2f})",
        fontsize=11.5,
    )
    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


def histograma_comparacion_rm(
    ruta_destino,
    rm_modelo_flat,
    rm_obs_flat,
    rm_catalogo,
    nombre_archivo="comparacion_histograma_rm.png",
    pesos_mapa_flat=None,
    abs_b_mapa_flat=None,
    abs_b_catalogo=None,
):
    """
    Distribución de la RM: modelo (todo el mapa), Oppermann (todo el mapa) y
    catálogo NVSS (fuentes), como densidades normalizadas en escala log.
    Si se pasan las latitudes (`abs_b_mapa_flat`, `abs_b_catalogo`) se
    separa en plano (|b| < 10°) y fuera del plano, que tienen anchos muy
    distintos; mezclados, la distribución del plano domina las colas.

    `pesos_mapa_flat`: cos(b) de cada píxel (ángulo sólido); el catálogo son
    fuentes puntuales y no se pondera.
    """
    import matplotlib.pyplot as plt

    estilo.aplicar()
    series = [
        (np.asarray(rm_modelo_flat), estilo.ETIQUETA_MODELO, estilo.MODELO, pesos_mapa_flat, abs_b_mapa_flat),
        (np.asarray(rm_obs_flat), estilo.ETIQUETA_OPPERMANN, estilo.OBSERVADO, pesos_mapa_flat, abs_b_mapa_flat),
        (np.asarray(rm_catalogo), estilo.ETIQUETA_NVSS, estilo.NVSS, None, abs_b_catalogo),
    ]
    separar = abs_b_mapa_flat is not None and abs_b_catalogo is not None
    paneles = ([(lambda b: b < 10.0, r"Plano, $|b| < 10^\circ$"),
                (lambda b: b >= 10.0, r"Fuera del plano, $|b| \geq 10^\circ$")]
               if separar else [(None, "Todo el cielo")])

    fig, axes = plt.subplots(1, len(paneles), figsize=(5.6 * len(paneles), 4.4), squeeze=False)
    for ax, (criterio, titulo) in zip(axes[0], paneles):
        recortes = []
        for datos, etiqueta, color, pesos, abs_b in series:
            ok = np.isfinite(datos)
            if criterio is not None:
                ok &= criterio(np.asarray(abs_b))
            recortes.append((datos[ok], etiqueta, color, None if pesos is None else np.asarray(pesos)[ok]))
        limite = np.percentile(np.abs(np.concatenate([d for d, *_ in recortes])), 99)
        bins = np.linspace(-limite, limite, 61)
        for datos, etiqueta, color, pesos in recortes:
            ax.hist(datos, bins=bins, density=True, histtype="step", lw=1.8, color=color,
                    weights=pesos, label=etiqueta)
        ax.set_yscale("log")
        ax.set_xlabel(r"RM [rad m$^{-2}$]")
        ax.set_title(titulo)
    axes[0][0].set_ylabel("densidad de probabilidad")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Distribución de la RM", fontsize=12)
    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


def mapa_morfologia_sincrotron(
    ruta_destino,
    l_grid,
    b_grid,
    resultado_morfologia,
    nombre_archivo="comparacion_morfologia_sincrotron.png",
):
    """
    Intensidad sincrotrón del modelo contra Haslam 408 MHz, solo FORMA (la
    emisividad del modelo está en unidades arbitrarias): los dos mapas
    normalizados a su máximo, en escala log, y el perfil en |b| de cada uno
    normalizado al plano, que dice si el disco emisor del modelo es tan
    grueso como el real.
    """
    import matplotlib.pyplot as plt

    estilo.aplicar()
    r = resultado_morfologia
    fig = plt.figure(figsize=(15, 4.6))
    for i, (mapa, titulo) in enumerate([(r["i_norm"], estilo.ETIQUETA_MODELO + " (1.4 GHz)"),
                                        (r["haslam_norm"], estilo.ETIQUETA_HASLAM)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        malla = mollweide_panel(ax, l_grid, b_grid, mapa, cmap=estilo.MAPA_MAGNITUD,
                                escala="log", vmin=1e-3, vmax=1.0)
        ax.set_title(titulo, pad=10)
        _barra_de_color(fig, malla, ax, "intensidad / máximo del mapa")

    ax = fig.add_subplot(1, 3, 3)
    for perfil, etiqueta, color, marca in [
        (r["perfil_i_norm"], estilo.ETIQUETA_MODELO, estilo.MODELO, "o-"),
        (r["perfil_haslam_norm"], estilo.ETIQUETA_HASLAM, estilo.OBSERVADO, "s-"),
    ]:
        ax.semilogy(perfil["centros_deg"], perfil["valores"] / perfil["valores"][0], marca,
                    color=color, ms=5, label=etiqueta)
    ax.set_xlabel("|b| [grados]")
    ax.set_ylabel(r"$\langle I\rangle$ / valor en el plano")
    ax.set_title("Perfil en latitud")
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    # En el hueco entre las dos curvas (Haslam arriba, el modelo abajo).
    ax.text(0.97, 0.36,
            "Haslam incluye un fondo casi isótropo\n(extragaláctico + CMB + nivel cero):\n"
            "parte de su piso a alta |b| no es galáctico",
            transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color=estilo.TEXTO_SECUNDARIO)

    fig.suptitle(f"Forma de la emisión sincrotrón: correlación log-log r = "
                 f"{r['correlacion_log_pearson']:.2f}", fontsize=12, y=1.03)
    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


def perfil_rm_vs_longitud(
    ruta_destino,
    l_grid,
    b_grid,
    rm_modelo,
    rm_obs_grid,
    bandas_deg=((10.0, 30.0), (30.0, 85.0)),
    nombre_archivo="comparacion_rm_vs_longitud.png",
):
    """
    RM promedio (con signo) contra la longitud galáctica, modelo vs.
    Oppermann & Enßlin (2012). Conserva el signo, así que prueba el SENTIDO
    del campo regular y no solo su amplitud.

    Izquierda: el plano, |b| < 10° (el disco es simétrico respecto al plano,
    así que ahí sí se promedian los dos hemisferios). Derecha: cada banda de
    `bandas_deg` por separado al norte y al sur. Promediar en |b| mezclaría
    los dos hemisferios, y la RM del halo tiene signo opuesto en cada uno
    (antisimetría norte-sur): se cancelaría y quedaría solo ruido. En el
    título de cada panel va la correlación de Pearson entre las dos curvas.
    """
    import matplotlib.pyplot as plt

    estilo.aplicar()
    l_deg = np.degrees(np.asarray(l_grid, dtype=float))
    b = np.asarray(b_grid, dtype=float)
    b_deg = np.degrees(b)
    rm_modelo, rm_obs_grid = np.asarray(rm_modelo), np.asarray(rm_obs_grid)

    def dibujar(ax, filas, titulo):
        pesos = np.cos(b[filas])
        medias = []
        for mapa, etiqueta, color in [
            (rm_obs_grid, estilo.ETIQUETA_OPPERMANN, estilo.OBSERVADO),
            (rm_modelo, estilo.ETIQUETA_MODELO, estilo.MODELO),
        ]:
            media = np.average(mapa[:, filas], axis=1, weights=pesos)
            ax.plot(l_deg, media, color=color, label=etiqueta, lw=1.5)
            medias.append(media)
        r = np.corrcoef(*medias)[0, 1]
        ax.axhline(0, color=estilo.REFERENCIA, lw=0.8)
        ax.set_xlim(180, -180)  # l crece hacia la izquierda, como en los mapas
        ax.set_xticks(np.arange(180, -181, -60))
        ax.set_title(f"{titulo}   (r = {r:+.2f})")

    fig = plt.figure(figsize=(15, 6.6))
    rejilla = fig.add_gridspec(2, 1 + len(bandas_deg), width_ratios=[1.25] + [1.0] * len(bandas_deg))
    ax_plano = fig.add_subplot(rejilla[:, 0])
    dibujar(ax_plano, np.abs(b_deg) < 10.0, r"Plano, $|b| < 10^\circ$")
    ax_plano.set_xlabel("l [grados]")
    ax_plano.set_ylabel(r"$\langle$RM$\rangle$ [rad m$^{-2}$]")
    ax_plano.legend(loc="lower left")
    ax_plano.annotate("inversión del campo\n(brazo de Sagitario)", xy=(45, 250), xytext=(150, 230),
                      fontsize=8, color=estilo.TEXTO_SECUNDARIO,
                      arrowprops=dict(arrowstyle="->", color=estilo.TEXTO_SECUNDARIO, lw=0.8))

    for k, (lo, hi) in enumerate(bandas_deg):
        for fila, (signo, nombre) in enumerate([(1.0, "Norte"), (-1.0, "Sur")]):
            ax = fig.add_subplot(rejilla[fila, k + 1])
            filas = (signo * b_deg >= lo) & (signo * b_deg < hi)
            rango = (f"{lo:.0f}^\\circ < b < {hi:.0f}^\\circ" if signo > 0
                     else f"-{hi:.0f}^\\circ < b < -{lo:.0f}^\\circ")
            dibujar(ax, filas, rf"{nombre}, ${rango}$")
            if fila == 1:
                ax.set_xlabel("l [grados]")

    fig.suptitle(
        "RM con signo contra longitud: el plano y cada hemisferio por separado "
        "(el halo invierte el signo entre norte y sur)",
        fontsize=12,
    )
    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


_ETIQUETA_MASCARA = "contorno punteado: excluido de las estadísticas (loops/spurs locales, centro galáctico, Fan)"


def _sombrear_mascara(ax, l_grid, b_grid, mascara):
    """
    Contornea con una línea fina punteada las regiones enmascaradas: los
    datos se ven completos (un relleno opaco o semitransparente las hacía
    parecer borradas) y queda marcado qué no entra en las estadísticas.
    """
    lon, lat, m = _preparar_grilla_mollweide(l_grid, b_grid, np.asarray(mascara, dtype=float))
    ax.contour(lon, lat, m, levels=[0.5], colors="0.85", linewidths=0.6, linestyles=":", zorder=4)


def _limites_regiones_remocion(ax):
    """Bordes de las regiones de `REGIONES_REMOCION` (l dibujada con espejo: x = -l)."""
    estilo_linea = dict(color="cyan", lw=0.9, ls="--", zorder=5)
    x = np.linspace(-np.pi, np.pi, 361)
    for b in (20.0, -20.0):
        ax.plot(x, np.full_like(x, np.radians(b)), **estilo_linea)
    interior = np.linspace(-np.pi / 2, np.pi / 2, 181)
    for b in (3.0, -3.0):
        ax.plot(interior, np.full_like(interior, np.radians(b)), **estilo_linea)
    for x_borde in (-np.pi / 2, np.pi / 2):
        ax.plot([x_borde, x_borde], [np.radians(-20.0), np.radians(20.0)], **estilo_linea)


def mapa_validacion_planck(
    ruta_destino,
    l_grid,
    b_grid,
    resultado_validacion,
    nombre_archivo="validacion_planck_030ghz.png",
):
    """
    Las pruebas de `validar_contra_planck_030ghz` en una figura. Arriba:
    P de Planck, P del modelo (realización 0, reescalada con fondo libre,
    misma escala de color) y Δψ donde Planck detecta polarización; las
    regiones enmascaradas (loops, centro galáctico, Fan) solo se
    contornean: se ven, pero no entran en las estadísticas.
    Abajo: los perfiles de P en latitud de la Galaxia interior y del tercer
    cuadrante (Planck Int. XLII Fig. 4), con la banda de ±1σ de varianza
    galáctica ⊕ ruido alrededor de la media del ensamble.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    estilo.aplicar()
    r = resultado_validacion
    mapas = r["mapas"]
    mascara = np.asarray(mapas["mascara"], dtype=bool)

    tope_p = float(np.nanpercentile(np.where(mascara, np.nan, mapas["p_planck_k_rj"]), 99)) * 1e6
    paneles = [
        (mapas["p_planck_k_rj"] * 1e6, f"P {estilo.ETIQUETA_PLANCK}", "P [µK$_{RJ}$]",
         estilo.MAPA_MAGNITUD, False, 0.0, tope_p),
        (mapas["p_modelo_escalado_k_rj"] * 1e6, "P del modelo (amplitud ajustada)", "P [µK$_{RJ}$]",
         estilo.MAPA_MAGNITUD, False, 0.0, tope_p),
        (np.degrees(mapas["abs_delta_psi_dibujo"]),
         "Desalineación del ángulo de polarización |ψ modelo − ψ Planck|",
         "|Δψ| [grados]: 0° alineados, 90° perpendiculares\n(blanco: Planck no detecta polarización, P/σ < 3)",
         "angulo_abs", False, 0.0, 90.0),
    ]
    fig = plt.figure(figsize=(15, 8.8))
    rejilla = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.9])
    from matplotlib.colors import LinearSegmentedColormap

    # |Δψ|: secuencial de un tono, oscuro = alineados (bueno), claro =
    # perpendiculares; sobre fondo blanco, que es "sin detección".
    mapa_angulo = LinearSegmentedColormap.from_list(
        "angulo_abs", ["#0d366b", "#3987e5", "#e8a33a", "#f6e3c4"]
    )
    for k, (mapa, titulo, unidad, cmap, simetrico, vmin, vmax) in enumerate(paneles):
        ax = fig.add_subplot(rejilla[0, 2 * k : 2 * k + 2], projection="mollweide")
        es_angulo = cmap == "angulo_abs"
        ax.set_facecolor("white" if es_angulo else "0.88")
        malla = mollweide_panel(ax, l_grid, b_grid, mapa, cmap=mapa_angulo if es_angulo else cmap,
                                simetrico=simetrico, vmin=vmin, vmax=vmax)
        if es_angulo:
            lon, lat, m = _preparar_grilla_mollweide(l_grid, b_grid, mascara.astype(float))
            ax.contour(lon, lat, m, levels=[0.5], colors="0.3", linewidths=0.6, linestyles=":", zorder=4)
        else:
            _sombrear_mascara(ax, l_grid, b_grid, mascara)
        ax.set_title(titulo, pad=10, fontsize=9.5 if es_angulo else None)
        cbar = _barra_de_color(fig, malla, ax, unidad, extend="neither" if es_angulo else "max")
        if es_angulo:
            cbar.set_ticks([0, 15, 30, 45, 60, 75, 90])
            cbar.ax.xaxis.label.set_size(7.5)

    nombres = {
        "galaxia_interior": "Galaxia interior, |l| < 90°",
        "tercer_cuadrante": "Tercer cuadrante, 180° < l < 270°",
    }
    for k, (clave, perfil) in enumerate(r["perfiles"].items()):
        ax = fig.add_subplot(rejilla[1, 3 * k : 3 * k + 3])
        b_deg = np.asarray(perfil["b_deg"])
        media = np.asarray(perfil["modelo_media"]) * 1e6
        sigma = np.asarray(perfil["sigma_total"]) * 1e6
        ax.fill_between(b_deg, media - sigma, media + sigma, color=estilo.MODELO, alpha=0.22, lw=0,
                        label="modelo ±1σ (varianza entre realizaciones ⊕ ruido)")
        ax.plot(b_deg, media, color=estilo.MODELO, label="modelo (media de 8 realizaciones)")
        ax.plot(b_deg, np.asarray(perfil["datos"]) * 1e6, "o-", color=estilo.OBSERVADO, ms=3.5,
                label=estilo.ETIQUETA_PLANCK)
        prueba = r["perfiles_latitud_p"][clave]
        ax.set_title(f"{nombres.get(clave, clave)}:  χ²_red = {prueba['chi2_reducido']:.1f}, "
                     f"{100 * prueba['fraccion_bandas_dentro_de_3_sigma']:.0f} % de las bandas a ≤ 3σ")
        ax.set_xlabel("b [grados]")
        ax.set_ylabel("⟨P⟩ [µK$_{RJ}$]")
        if k == 0:
            ax.legend(loc="upper left")

    def veredicto(prueba):
        return "pasa" if prueba["pasa"] else "no pasa"

    ang = r["angulo_polarizacion"]
    forma = r["forma_intensidad_polarizada"]
    fig.legend(handles=[Line2D([], [], color="0.5", ls=":", lw=1.0, label=_ETIQUETA_MASCARA)],
               loc="upper center", bbox_to_anchor=(0.5, 0.96), fontsize=8)
    fig.suptitle(
        "Polarización a 30 GHz: modelo contra Planck — "
        f"ángulo ⟨cos 2Δψ⟩ = {ang['coherencia_modelo_media']:.2f} ± "
        f"{ang['coherencia_modelo_dispersion_entre_realizaciones']:.2f} (criterio ≥ 0.5: {veredicto(ang)}); "
        f"forma de P R² = {forma['r2_modelo_media']:.2f} contra {forma['r2_disco_plano_paralelo']:.2f} "
        f"de un disco plano-paralelo ({veredicto(forma)})",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(h_pad=2.5)
    return _guardar(fig, ruta_destino, nombre_archivo)


def mapa_remocion_planck(
    ruta_destino,
    l_grid,
    b_grid,
    resultado_remocion,
    nombre_archivo="remocion_foreground_planck.png",
):
    """
    Resultado de `remover_foreground_polarizado_planck`. Arriba, con la
    misma escala de color: P de Planck 30 GHz, la plantilla del modelo
    ajustada por región y el residuo |P_Planck - plantilla|. Las estructuras
    locales enmascaradas solo se contornean (se ven, pero no entran en las
    estadísticas); las líneas discontinuas son los bordes de las regiones
    (cada una con su amplitud, de ahí los saltos de la plantilla en ellos),
    y el rayado marca las regiones donde la plantilla no remueve nada
    (< 1 %). Abajo: fracción de la potencia polarizada removida en cada
    región, para la plantilla del modelo (media del ensamble y una sola
    realización), la plantilla trivial y la amplitud única para todo el
    cielo.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    estilo.aplicar()
    r = resultado_remocion
    mapas = r["mapas"]
    mascara = np.asarray(mapas["mascara"], dtype=bool)
    sin_remocion = np.asarray(mapas["sin_remocion"], dtype=bool)
    tope = float(np.nanpercentile(np.where(mascara, np.nan, mapas["p_planck_k_rj"]), 99)) * 1e6
    paneles = [
        (mapas["p_planck_k_rj"], f"P {estilo.ETIQUETA_PLANCK} (lo que hay que remover)", False),
        (np.where(sin_remocion, np.nan, mapas["p_plantilla_k_rj"]),
         "Plantilla del modelo (amplitud ajustada por región)", True),
        (mapas["p_residuo_k_rj"], "Residuo |P Planck − plantilla|", False),
    ]
    fig = plt.figure(figsize=(15, 9.4))
    rejilla = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95])
    for k, (mapa, titulo, rayar) in enumerate(paneles):
        ax = fig.add_subplot(rejilla[0, k], projection="mollweide")
        ax.set_facecolor("0.88")
        malla = mollweide_panel(ax, l_grid, b_grid, mapa * 1e6, cmap=estilo.MAPA_MAGNITUD,
                                vmin=0.0, vmax=tope)
        _sombrear_mascara(ax, l_grid, b_grid, mascara)
        if rayar and sin_remocion.any():
            lon, lat, z = _preparar_grilla_mollweide(l_grid, b_grid, sin_remocion.astype(float))
            ax.contourf(lon, lat, z, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=4)
        _limites_regiones_remocion(ax)
        ax.set_title(titulo, pad=10)
        _barra_de_color(fig, malla, ax, "P [µK$_{RJ}$]", extend="max")

    nombres = {
        "franja_interior": "Franja interior\n|l| < 90°, |b| < 3°\n(probable fuga I→P de Planck)",
        "plano_interior": "Plano interior\n|l| < 90°, 3° < |b| < 20°",
        "plano_exterior": "Plano exterior\n90° < l < 270°, |b| < 20°",
        "alta_latitud": "Alta latitud\n|b| > 20°\n(donde se observa el CMB)",
    }
    series = [
        ("fraccion_removida_modelo", "modelo (media de realizaciones)", estilo.MODELO, 1.0),
        ("fraccion_removida_una_realizacion_media", "modelo (una sola realización)", estilo.MODELO, 0.45),
        ("fraccion_removida_plantilla_trivial", "plantilla trivial: campo ∥ plano, P ∝ 1/sin|b|",
         estilo.REFERENCIA, 1.0),
        ("fraccion_removida_amplitud_global", "modelo con una sola amplitud para todo el cielo",
         estilo.SECUNDARIO, 1.0),
    ]
    ax = fig.add_subplot(rejilla[1, :])
    claves = list(r["regiones"])
    x = np.arange(len(claves))
    ancho = 0.8 / len(series)
    for j, (clave, etiqueta, color, alfa) in enumerate(series):
        valores = np.array([100 * r["regiones"][c][clave] for c in claves])
        errores = None
        if clave == "fraccion_removida_una_realizacion_media":
            errores = [100 * r["regiones"][c]["fraccion_removida_una_realizacion_dispersion"] for c in claves]
        barras = ax.bar(x + (j - (len(series) - 1) / 2) * ancho, valores, ancho * 0.92, yerr=errores,
                        color=color, alpha=alfa, label=etiqueta, capsize=3,
                        error_kw=dict(ecolor=estilo.TEXTO_SECUNDARIO, lw=1))
        if j == 0:
            for barra, valor in zip(barras, valores):
                ax.annotate(f"{valor:.0f} %", (barra.get_x() + barra.get_width() / 2, max(valor, 0)),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9,
                            fontweight="bold", color="#0b0b0b")
    ax.axhline(0.0, color=estilo.REFERENCIA, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{nombres.get(c, c)}\n{100 * r['regiones'][c]['fraccion_potencia_planck']:.0f} % de la potencia"
        for c in claves
    ], fontsize=9)
    ax.set_ylabel("potencia polarizada removida [%]")
    ax.set_ylim(min(-50.0, ax.get_ylim()[0]), 100.0)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(loc="upper left")
    fig.legend(
        handles=[
            Line2D([], [], color="0.5", ls=":", lw=1.0, label=_ETIQUETA_MASCARA),
            Line2D([], [], color="cyan", ls="--", lw=0.9, label="bordes de las regiones de abajo"),
            Patch(facecolor="0.88", hatch="////", label="la plantilla no remueve nada (< 1 %)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=8,
    )
    fig.suptitle("Remoción del foreground sincrotrón polarizado de Planck 30 GHz con la plantilla del modelo",
                 fontsize=12, y=1.0)
    fig.tight_layout(h_pad=2.0)
    return _guardar(fig, ruta_destino, nombre_archivo)
