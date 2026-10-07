"""
Figuras de comparación entre el foreground galáctico sintético y los datos
observacionales reales cargados por `faradaymr.observational` (issue #37).

Separado de `faradaymr.plotting_sky` (que solo dibuja el mapa sintético en
sí) porque estas figuras necesitan dos fuentes de datos a la vez -modelo y
observación- y, en el caso del catálogo puntual, un tipo de gráfico
distinto (dispersión, histograma) que no encaja en el molde de "un panel
Mollweide por observable" del otro módulo.
"""

from __future__ import annotations

import os

import numpy as np

from .plotting_sky import mollweide_panel


def mapa_comparacion_mollweide(
    ruta_destino,
    l_grid,
    b_grid,
    rm_modelo,
    rm_obs_grid,
    nombre_archivo="comparacion_mapa_rm.png",
):
    """
    Tres paneles Mollweide en una fila -modelo, observado (Oppermann et
    al. 2012), y su residuo (obs - modelo)- para comparar visualmente la
    estructura de gran escala.

    Modelo y observado comparten la MISMA escala de color (fijada por el
    mayor de los dos rangos): una comparación visual honesta exige que un
    mismo tono de rojo/azul signifique la misma RM en ambos paneles, no
    una escala renormalizada panel a panel que disimularía una diferencia
    real de amplitud (ver `faradaymr.observational`, el modelo sobreestima
    sistemáticamente la amplitud respecto a Oppermann, ver informe de la
    corrida). El residuo lleva su propia escala, siempre simétrica.
    """
    import matplotlib.pyplot as plt

    rm_modelo = np.asarray(rm_modelo)
    rm_obs_grid = np.asarray(rm_obs_grid)
    residual = rm_obs_grid - rm_modelo

    amplitud_compartida = float(
        max(
            np.nanpercentile(np.abs(rm_modelo), 99),
            np.nanpercentile(np.abs(rm_obs_grid), 99),
        )
    )

    fig = plt.figure(figsize=(13.5, 4.2))
    especificaciones = [
        (rm_modelo, "Modelo sintético", amplitud_compartida),
        (rm_obs_grid, "Observado (Oppermann & Enßlin 2012)", amplitud_compartida),
        (residual, "Residuo (observado - modelo)", None),
    ]
    for i, (mapa, titulo, amplitud) in enumerate(especificaciones):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        malla = mollweide_panel(
            ax, l_grid, b_grid, mapa, cmap="RdBu_r", simetrico=True,
            vmax=amplitud,
        )
        ax.set_title(titulo, fontsize=10, pad=10)
        cbar = fig.colorbar(
            malla, ax=ax, orientation="horizontal", fraction=0.055, pad=0.07, aspect=26,
            extend="both",
        )
        cbar.set_label(r"RM [rad m$^{-2}$]", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Faraday depth galáctica: modelo vs. reconstrucción observacional",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


def perfil_comparacion_latitud(
    ruta_destino,
    perfil_modelo,
    perfil_obs,
    nombre_archivo="comparacion_perfil_latitud.png",
    etiqueta_obs="Oppermann & Enßlin (2012)",
    perfil_catalogo=None,
):
    """
    RMS(RM) vs |b| -modelo y observado superpuestos, escala log en y (el
    contraste plano/polo abarca más de un orden de magnitud, ver
    `faradaymr.calibration`)-, la comparación estadística central del
    informe: insensible a la fase arbitraria de los brazos del modelo
    (ver docstring de `faradaymr.observational`), a diferencia de una
    resta punto a punto en el mapa completo.

    `perfil_catalogo` (opcional, de `perfil_rms_catalogo_vs_latitud`): el
    mismo perfil medido en las fuentes NVSS. Las dos estimaciones
    observacionales encierran a la RM galáctica real (ver esa función), y
    la figura las muestra juntas para que la comparación no dependa de
    elegir una.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 4.8))
    plt.semilogy(
        perfil_modelo["centros_deg"], perfil_modelo["valores"], "o-",
        color="firebrick", label="Modelo sintético",
    )
    plt.semilogy(
        perfil_obs["centros_deg"], perfil_obs["valores"], "s-",
        color="steelblue", label=etiqueta_obs,
    )
    if perfil_catalogo is not None:
        plt.semilogy(
            perfil_catalogo["centros_deg"], perfil_catalogo["valores"], "^-",
            color="darkorange", label="NVSS, Taylor+2009 (ruido de medida restado)",
        )
    plt.xlabel("|b| [grados]")
    plt.ylabel(r"RMS(RM) [rad m$^{-2}$]")
    plt.title("Contraste plano-polo: modelo vs. observado")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    plt.savefig(ruta_completa, dpi=200)
    plt.close()
    return ruta_completa


def dispersion_catalogo_vs_modelo(
    ruta_destino,
    resultado_comparacion_catalogo,
    nombre_archivo="comparacion_catalogo_nvss.png",
):
    """
    Dispersión RM_modelo_interpolado vs. RM_observado (catálogo NVSS,
    Taylor et al. 2009) en la posición real de cada fuente, coloreado por
    |b|, con la línea 1:1 de referencia.

    Qué buscar en esta figura (ver docstring de
    `faradaymr.observational.comparar_con_catalogo_taylor` sobre por qué
    NO se espera una correlación punto a punto fuerte): la nube de puntos
    debería estar centrada en torno a la línea 1:1 y con un ANCHO
    comparable en ambos ejes -no una nube mucho más ancha en el eje del
    modelo que en el del catálogo, que señalaría una sobreestimación
    sistemática de amplitud, no solo falta de correlación de fase.
    """
    import matplotlib.pyplot as plt

    r = resultado_comparacion_catalogo
    valido = np.isfinite(r["rm_modelo_interp"])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ax.scatter(
        r["rm_obs"][valido], r["rm_modelo_interp"][valido],
        c=np.abs(r["b_deg"][valido]), cmap="viridis", s=4, alpha=0.5,
    )
    limite = float(
        max(
            np.nanmax(np.abs(r["rm_obs"][valido])),
            np.nanmax(np.abs(r["rm_modelo_interp"][valido])),
        )
    )
    ax.plot([-limite, limite], [-limite, limite], "k--", lw=1, label="1:1")
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)
    ax.set_xlabel(r"RM observado, catálogo NVSS [rad m$^{-2}$]")
    ax.set_ylabel(r"RM modelo (interpolado en la fuente) [rad m$^{-2}$]")
    ax.set_title(
        f"r = {r['correlacion_pearson']:.3f}  "
        f"(N = {r['n_fuentes_comparadas']})"
    )
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    cbar = fig.colorbar(disp, ax=ax)
    cbar.set_label("|b| [grados]")
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200)
    plt.close(fig)
    return ruta_completa


def histograma_comparacion_rm(
    ruta_destino,
    rm_modelo_flat,
    rm_obs_flat,
    rm_catalogo,
    nombre_archivo="comparacion_histograma_rm.png",
    pesos_mapa_flat=None,
):
    """
    Distribución de valores de RM -modelo (todos los píxeles del mapa
    sintético), Oppermann (todos los píxeles regrillados) y catálogo NVSS
    (mediciones puntuales reales)- superpuestas como densidades
    normalizadas, escala log en y (la cola de valores extremos cerca del
    plano es justamente lo que separa al modelo de los datos reales, y en
    escala lineal esa cola queda invisible).

    `pesos_mapa_flat`: peso de cada píxel de los dos mapas (típicamente
    cos(b), el ángulo sólido relativo de un píxel de una grilla l-b
    rectangular); sin él, el histograma cuenta igual a un píxel del polo
    que a uno del plano y sobrerrepresenta las latitudes altas. El
    catálogo son fuentes puntuales (una fuente = un dato) y no se pondera.
    """
    import matplotlib.pyplot as plt

    limite = np.nanpercentile(
        np.abs(np.concatenate([rm_modelo_flat, rm_obs_flat, rm_catalogo])), 99
    )
    bins = np.linspace(-limite, limite, 80)

    plt.figure(figsize=(7, 4.8))
    for datos, etiqueta, color, pesos in [
        (rm_modelo_flat, "Modelo sintético (todo el mapa)", "firebrick", pesos_mapa_flat),
        (rm_obs_flat, "Oppermann & Enßlin 2012 (todo el mapa)", "steelblue", pesos_mapa_flat),
        (rm_catalogo, "Catálogo NVSS (fuentes puntuales)", "darkorange", None),
    ]:
        datos = np.asarray(datos)
        ok = np.isfinite(datos)
        pesos = None if pesos is None else np.asarray(pesos)[ok]
        datos = datos[ok]
        plt.hist(
            datos, bins=bins, density=True, histtype="step", lw=1.8,
            color=color, label=f"{etiqueta} (n={datos.size})", weights=pesos,
        )
    plt.yscale("log")
    plt.xlabel(r"RM [rad m$^{-2}$]")
    plt.ylabel("Densidad de probabilidad")
    plt.title("Distribución de RM: modelo vs. datos reales")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    plt.savefig(ruta_completa, dpi=200)
    plt.close()
    return ruta_completa


def mapa_morfologia_sincrotron(
    ruta_destino,
    l_grid,
    b_grid,
    resultado_morfologia,
    nombre_archivo="comparacion_morfologia_sincrotron.png",
):
    """
    Modelo vs. Haslam 408 MHz, solo FORMA (`i_map` está en unidades
    arbitrarias, así que nunca se compara la magnitud): dos mapas
    Mollweide normalizados a su máximo, en escala log, y el perfil de cada
    uno en |b| normalizado a su valor en el plano. El perfil es lo
    cuantitativo: dice si el disco modelado es tan delgado como el real.
    """
    import matplotlib.pyplot as plt

    r = resultado_morfologia
    fig = plt.figure(figsize=(15, 4.4))
    for i, (mapa, titulo) in enumerate(
        [(r["i_norm"], "Modelo sintético"), (r["haslam_norm"], "Haslam 408 MHz")]
    ):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        malla = mollweide_panel(
            ax, l_grid, b_grid, mapa, cmap="inferno", simetrico=False,
            escala="log", vmin=1e-4, vmax=1.0,
        )
        ax.set_title(titulo, fontsize=10, pad=10)
        cbar = fig.colorbar(
            malla, ax=ax, orientation="horizontal", fraction=0.055, pad=0.07, aspect=26
        )
        cbar.set_label("intensidad / máximo", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(1, 3, 3)
    for perfil, etiqueta, estilo in [
        (r["perfil_i_norm"], "Modelo sintético", "o-r"),
        (r["perfil_haslam_norm"], "Haslam 408 MHz", "s-b"),
    ]:
        ax.semilogy(perfil["centros_deg"], perfil["valores"] / perfil["valores"][0],
                    estilo, label=etiqueta)
    ax.set_xlabel("|b| [grados]")
    ax.set_ylabel(r"$\langle I\rangle$ / valor en el plano")
    ax.set_title("Perfil en latitud (media pesada por área)")
    ax.text(
        0.97, 0.55,
        "Haslam incluye un fondo casi isótropo\n(extragaláctico + CMB + nivel cero):\n"
        "el piso a alta |b| no es galáctico",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"),
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Morfología sincrotrón: r(log-log) = {r['correlacion_log_pearson']:.3f}",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


def perfil_rm_vs_longitud(
    ruta_destino,
    l_grid,
    b_grid,
    rm_modelo,
    rm_obs_grid,
    bandas_deg=((0.0, 10.0), (10.0, 30.0), (30.0, 85.0)),
    nombre_archivo="comparacion_rm_vs_longitud.png",
):
    """
    RM promedio (con signo) contra la longitud galáctica, en tres bandas de
    |b|, modelo vs. Oppermann & Enßlin (2012). Es la comparación clásica de
    la RM galáctica (el patrón de signo del campo regular): a diferencia del
    RMS, conserva el signo, así que prueba el SENTIDO del campo regular y
    no solo su amplitud. La fase de los brazos del modelo es arbitraria (ver
    `faradaymr.observational`), pero el signo de la RM de gran escala en
    función de l no depende de ella.
    """
    import matplotlib.pyplot as plt

    l_deg = np.degrees(np.asarray(l_grid, dtype=float))
    b = np.asarray(b_grid, dtype=float)
    b_abs = np.abs(np.degrees(b))

    fig, axes = plt.subplots(1, len(bandas_deg), figsize=(5.2 * len(bandas_deg), 4.2), sharex=True)
    for ax, (lo, hi) in zip(np.atleast_1d(axes), bandas_deg):
        filas = (b_abs >= lo) & (b_abs < hi)
        pesos = np.cos(b[filas])
        for mapa, etiqueta, color in [
            (rm_modelo, "Modelo sintético", "firebrick"),
            (rm_obs_grid, "Oppermann & Enßlin (2012)", "steelblue"),
        ]:
            media = np.average(np.asarray(mapa)[:, filas], axis=1, weights=pesos)
            ax.plot(l_deg, media, color=color, label=etiqueta)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlim(180, -180)  # l crece hacia la izquierda, como en los mapas
        ax.set_xticks(np.arange(180, -181, -60))
        ax.set_xlabel("l [grados]")
        ax.set_title(rf"${lo:.0f}^\circ \leq |b| < {hi:.0f}^\circ$")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(r"$\langle$RM$\rangle$ [rad m$^{-2}$]")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


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
    regiones enmascaradas (loops, centro galáctico, Fan) van en gris.
    Abajo: los perfiles de P en latitud de la Galaxia interior y del tercer
    cuadrante (Planck Int. XLII Fig. 4), con la banda de ±1σ de varianza
    galáctica ⊕ ruido alrededor de la media del ensamble.
    """
    import matplotlib.pyplot as plt

    r = resultado_validacion
    mapas = r["mapas"]
    mascara = np.asarray(mapas["mascara"], dtype=bool)

    def enmascarar(mapa):
        return np.where(mascara, np.nan, mapa)

    tope_p = float(np.nanpercentile(enmascarar(mapas["p_planck_k_rj"]), 99)) * 1e6
    paneles = [
        (enmascarar(mapas["p_planck_k_rj"]) * 1e6, "P Planck 28.4 GHz", "µK_RJ", "inferno", False, 0.0, tope_p),
        (enmascarar(mapas["p_modelo_escalado_k_rj"]) * 1e6, "P modelo (reescalado)", "µK_RJ", "inferno", False, 0.0, tope_p),
        (np.degrees(mapas["delta_psi"]), "ψ modelo − ψ Planck (P/σ ≥ 5)", "grados", "twilight", True, -90.0, 90.0),
    ]
    fig = plt.figure(figsize=(15, 8.5))
    rejilla = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.9])
    for k, (mapa, titulo, unidad, cmap, simetrico, vmin, vmax) in enumerate(paneles):
        ax = fig.add_subplot(rejilla[0, 2 * k : 2 * k + 2], projection="mollweide")
        ax.set_facecolor("0.85")
        malla = mollweide_panel(
            ax, l_grid, b_grid, mapa, cmap=cmap, simetrico=simetrico, vmin=vmin, vmax=vmax
        )
        ax.set_title(titulo, fontsize=10, pad=10)
        cbar = fig.colorbar(malla, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07, aspect=30)
        cbar.set_label(unidad, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    nombres = {
        "galaxia_interior": "Galaxia interior (|l| < 90°)",
        "tercer_cuadrante": "Tercer cuadrante (180° < l < 270°)",
    }
    for k, (clave, perfil) in enumerate(r["perfiles"].items()):
        ax = fig.add_subplot(rejilla[1, 3 * k : 3 * k + 3])
        b_deg = np.asarray(perfil["b_deg"])
        media = np.asarray(perfil["modelo_media"]) * 1e6
        sigma = np.asarray(perfil["sigma_total"]) * 1e6
        ax.fill_between(b_deg, media - sigma, media + sigma, color="firebrick", alpha=0.25,
                        label="modelo ±1σ (varianza galáctica ⊕ ruido)")
        ax.plot(b_deg, media, color="firebrick", label="modelo (media de realizaciones)")
        ax.plot(b_deg, np.asarray(perfil["datos"]) * 1e6, "o-", color="steelblue", ms=3,
                label="Planck 28.4 GHz")
        prueba = r["perfiles_latitud_p"][clave]
        ax.set_title(
            f"{nombres.get(clave, clave)}: χ²_red = {prueba['chi2_reducido']:.1f}, "
            f"{100 * prueba['fraccion_bandas_dentro_de_3_sigma']:.0f}% de bandas a ≤3σ",
            fontsize=9,
        )
        ax.set_xlabel("b [grados]")
        ax.set_ylabel("⟨P⟩ [µK_RJ]")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)

    def veredicto(prueba):
        return "pasa" if prueba["pasa"] else "NO pasa"

    ang = r["angulo_polarizacion"]
    forma = r["forma_intensidad_polarizada"]
    fig.suptitle(
        f"Validación contra Planck 30 GHz ({r['n_realizaciones']} realizaciones) — "
        f"⟨cos 2Δψ⟩ = {ang['coherencia_modelo_media']:.2f} ± {ang['coherencia_modelo_dispersion_entre_realizaciones']:.2f} "
        f"(ψ=0 trivial: {ang['coherencia_plantilla_campo_paralelo_al_plano']:.2f}; criterio ≥ 0.5): {veredicto(ang)}   |   "
        f"R²(P) = {forma['r2_modelo_media']:.2f} ± {forma['r2_modelo_dispersion_entre_realizaciones']:.2f} "
        f"(disco plano-paralelo {forma['r2_disco_plano_paralelo']:.2f}): {veredicto(forma)}",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout(h_pad=2.5)
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


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
    ajustada por región y el residuo |P_Planck - plantilla|; en gris, las
    estructuras locales enmascaradas. Abajo: fracción de la potencia
    polarizada removida en cada región, para la plantilla del modelo (media
    del ensamble y una sola realización), la plantilla trivial y la
    amplitud única para todo el cielo.
    """
    import matplotlib.pyplot as plt

    r = resultado_remocion
    mapas = r["mapas"]
    tope = float(np.nanpercentile(mapas["p_planck_k_rj"], 99)) * 1e6
    paneles = [
        (mapas["p_planck_k_rj"], "P Planck 28.4 GHz"),
        (mapas["p_plantilla_k_rj"], "Plantilla del modelo (amplitud por región)"),
        (mapas["p_residuo_k_rj"], "Residuo |P Planck − plantilla|"),
    ]
    fig = plt.figure(figsize=(15, 8.8))
    rejilla = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95])
    for k, (mapa, titulo) in enumerate(paneles):
        ax = fig.add_subplot(rejilla[0, k], projection="mollweide")
        ax.set_facecolor("0.85")
        malla = mollweide_panel(ax, l_grid, b_grid, mapa * 1e6, cmap="inferno", vmin=0.0, vmax=tope)
        ax.set_title(titulo, fontsize=10, pad=10)
        cbar = fig.colorbar(malla, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07,
                            aspect=30, extend="max")
        cbar.set_label("µK_RJ", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    nombres = {
        "franja_interior": "Franja interior\n|l|<90°, |b|<3°\n(posible fuga I→P)",
        "plano_interior": "Plano interior\n|l|<90°, 3°<|b|<20°",
        "plano_exterior": "Plano exterior\n90°<l<270°, |b|<20°",
        "alta_latitud": "Alta latitud\n|b|>20°",
    }
    series = [
        ("fraccion_removida_modelo", "modelo (media de realizaciones)", "firebrick"),
        ("fraccion_removida_una_realizacion_media", "modelo (una realización)", "salmon"),
        ("fraccion_removida_plantilla_trivial", "trivial: campo ∥ plano, P∝1/sin|b|", "0.6"),
        ("fraccion_removida_amplitud_global", "modelo, una amplitud para todo el cielo (sin la franja)", "steelblue"),
    ]
    ax = fig.add_subplot(rejilla[1, :])
    claves = list(r["regiones"])
    x = np.arange(len(claves))
    ancho = 0.8 / len(series)
    for j, (clave, etiqueta, color) in enumerate(series):
        valores = [r["regiones"][c][clave] for c in claves]
        errores = None
        if clave == "fraccion_removida_una_realizacion_media":
            errores = [r["regiones"][c]["fraccion_removida_una_realizacion_dispersion"] for c in claves]
        ax.bar(x + (j - (len(series) - 1) / 2) * ancho, valores, ancho, yerr=errores, color=color, label=etiqueta,
               capsize=3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{nombres.get(c, c)}\n({100 * r['regiones'][c]['fraccion_potencia_planck']:.0f}% de la potencia)"
        for c in claves
    ], fontsize=9)
    ax.set_ylabel("fracción de la potencia polarizada removida")
    ax.set_ylim(min(-0.5, ax.get_ylim()[0]), 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Remoción del foreground sincrotrón polarizado de Planck 30 GHz por ajuste de plantilla",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(h_pad=2.0)
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa
