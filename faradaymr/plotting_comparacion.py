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
    nombre_archivo="figura_comparacion_mapa_rm.png",
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
        max(np.nanmax(np.abs(rm_modelo)), np.nanmax(np.abs(rm_obs_grid)))
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
            malla, ax=ax, orientation="horizontal", fraction=0.055, pad=0.07, aspect=26
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
    nombre_archivo="figura_comparacion_perfil_latitud.png",
    etiqueta_obs="Oppermann & Enßlin (2012)",
):
    """
    RMS(RM) vs |b| -modelo y observado superpuestos, escala log en y (el
    contraste plano/polo abarca más de un orden de magnitud, ver
    `faradaymr.calibration`)-, la comparación estadística central del
    informe: insensible a la fase arbitraria de los brazos del modelo
    (ver docstring de `faradaymr.observational`), a diferencia de una
    resta punto a punto en el mapa completo.
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
    nombre_archivo="figura_dispersion_catalogo_nvss.png",
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
    nombre_archivo="figura_histograma_rm.png",
):
    """
    Distribución de valores de RM -modelo (todos los píxeles del mapa
    sintético), Oppermann (todos los píxeles regrillados) y catálogo NVSS
    (mediciones puntuales reales)- superpuestas como densidades
    normalizadas, escala log en y (la cola de valores extremos cerca del
    plano es justamente lo que separa al modelo de los datos reales, y en
    escala lineal esa cola queda invisible).
    """
    import matplotlib.pyplot as plt

    limite = np.nanpercentile(
        np.abs(np.concatenate([rm_modelo_flat, rm_obs_flat, rm_catalogo])), 99
    )
    bins = np.linspace(-limite, limite, 80)

    plt.figure(figsize=(7, 4.8))
    for datos, etiqueta, color in [
        (rm_modelo_flat, "Modelo sintético (todo el mapa)", "firebrick"),
        (rm_obs_flat, "Oppermann & Enßlin 2012 (todo el mapa)", "steelblue"),
        (rm_catalogo, "Catálogo NVSS (fuentes puntuales)", "darkorange"),
    ]:
        datos = np.asarray(datos)
        datos = datos[np.isfinite(datos)]
        plt.hist(
            datos, bins=bins, density=True, histtype="step", lw=1.8,
            color=color, label=f"{etiqueta} (n={datos.size})",
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
    nombre_archivo="figura_comparacion_morfologia_sincrotron.png",
):
    """
    Dos paneles Mollweide -intensidad sintética normalizada vs. Haslam 408
    MHz normalizado- en escala log, misma escala de color en ambos (ambos
    ya están normalizados a su propio máximo, ver
    `faradaymr.observational.comparar_morfologia_sincrotron`): compara
    FORMA, nunca magnitud absoluta (`i_map` está en unidades arbitrarias).
    """
    import matplotlib.pyplot as plt

    r = resultado_morfologia
    fig = plt.figure(figsize=(9.5, 4.2))
    for i, (mapa, titulo) in enumerate(
        [
            (r["i_norm"], "Modelo sintético (normalizado)"),
            (r["haslam_norm"], "Haslam 408 MHz (normalizado)"),
        ]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="mollweide")
        malla = mollweide_panel(
            ax, l_grid, b_grid, mapa, cmap="inferno", simetrico=False,
            escala="log", vmin=1e-4, vmax=1.0,
        )
        ax.set_title(titulo, fontsize=10, pad=10)
        cbar = fig.colorbar(
            malla, ax=ax, orientation="horizontal", fraction=0.055, pad=0.07, aspect=26
        )
        cbar.set_label("Intensidad normalizada (u.a.)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Morfología sincrotrón: r(log-log) = {r['correlacion_log_pearson']:.3f}",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


def mapa_resta_planck(
    ruta_destino,
    l_grid,
    b_grid,
    resultado_planck,
    nombre_archivo="figura_resta_planck_030ghz.png",
):
    """
    Tres paneles Mollweide -Planck 30 GHz real (K_CMB), modelo sintético ya
    reescalado a esa misma amplitud (`alpha * i_map`, ver
    `faradaymr.observational.restar_planck_030ghz`), y el residuo de la
    resta- el resultado final que pide el issue #37: una réplica sintética
    de la contaminación Galáctica, ajustada en amplitud, lista para restar
    de un mapa real de una misión como Planck.

    Planck y el modelo reescalado comparten la MISMA escala de color
    (fijada por el mayor de los dos, en log -el rango dinámico plano/polo
    de la emisión sincrotrón abarca órdenes de magnitud): un residuo
    "limpio" (sin la silueta del disco/brazos) es la señal visual de que el
    ajuste de FORMA (no solo de amplitud) es razonable, la fracción de
    varianza explicada reportada en el título es la versión numérica de esa
    misma lectura.
    """
    import matplotlib.pyplot as plt

    r = resultado_planck
    vmax_compartido = float(
        max(np.nanmax(r["i_planck_grid"]), np.nanmax(r["i_modelo_escalado"]))
    )

    fig = plt.figure(figsize=(13.5, 4.2))
    especificaciones = [
        (r["i_planck_grid"], "Planck 30 GHz real (K$_{CMB}$)", "log", vmax_compartido, False),
        (r["i_modelo_escalado"], r"Modelo sintético $\times\, \alpha$", "log", vmax_compartido, False),
        (r["residuo"], "Residuo (Planck − modelo escalado)", "lineal", None, True),
    ]
    for i, (mapa, titulo, escala, vmax, simetrico) in enumerate(especificaciones):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        malla = mollweide_panel(
            ax, l_grid, b_grid, mapa, cmap=("RdBu_r" if simetrico else "inferno"),
            simetrico=simetrico, escala=escala, vmax=vmax,
        )
        ax.set_title(titulo, fontsize=10, pad=10)
        cbar = fig.colorbar(
            malla, ax=ax, orientation="horizontal", fraction=0.055, pad=0.07, aspect=26
        )
        cbar.set_label("K$_{CMB}$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Resta contra Planck 30GHz: α={r['alpha']:.3g}, "
        f"varianza explicada={r['fraccion_varianza_explicada']:.2f}",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa
