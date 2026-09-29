"""
Figuras finales del Proyecto II (filamento WHIM).

Requiere haber corrido, desde la raíz del repo:
    python -m examples.filamento_whim.run                 # mapas (fig. 1)
    python -m examples.filamento_whim.run_barrido_theta   # barrido (figs. 2-6)
y luego:
    python -m examples.filamento_whim.plots

Convención: puntos = simulación Monte Carlo (perfil apilado de N semillas,
barras = bootstrap). El valor esperado del modelo
(`faradaymr.analysis.expected`) no se dibuja: es una validación del código,
no un resultado, y vive en tests/tests_filamento/test_expected.py y en las
llaves `esp_*` del .npz.
"""
import os

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from examples.filamento_whim import config_fisica
from examples.filamento_whim.umbral_deteccion import (
    ESCENARIOS,
    K_SIGMA,
    calcular as calcular_umbral,
)
from faradaymr.analysis.deteccion import (
    N_FUENTES_STUARDI,
    SIGMA_INTRINSECA_RAD_M2,
    campo_minimo_detectable,
)
from faradaymr.analysis.fitting import beta_dispersion_model, gaussian_model

BASE_DIR = os.path.dirname(__file__)
DIR_MAPAS = os.path.join(BASE_DIR, "results", "mapas_poster")
ARCHIVO_MC = os.path.join(BASE_DIR, "results", "barrido_theta", "barrido_theta_mc.npz")
DIR_PLOTS = os.path.join(BASE_DIR, "results", "plots")

# Paleta categórica validada (orden fijo) + tinta neutra para la predicción.
AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS = "#3d3d3a"
TINTA_SECUNDARIA = "#6b6a63"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _guardar(fig, nombre):
    os.makedirs(DIR_PLOTS, exist_ok=True)
    ruta = os.path.join(DIR_PLOTS, nombre)
    fig.savefig(ruta)
    plt.close(fig)
    print(f"Generada: {ruta}")


def _pie(fig, texto):
    fig.text(0.5, -0.02, texto, ha="center", va="top", fontsize=8.5,
             color=TINTA_SECUNDARIA, wrap=True)


def _puntos(ax, x, y, yerr, color, etiqueta, marcador="o"):
    ax.errorbar(x, y, yerr=yerr, fmt=marcador, ms=6.5, color=color,
                mfc="white", mew=1.6, capsize=3, elinewidth=1.2, lw=0,
                label=etiqueta, zorder=3)


def _cargar_mc():
    if not os.path.exists(ARCHIVO_MC):
        print("Falta el barrido Monte Carlo: ejecuta run_barrido_theta.py primero.")
        return None
    return np.load(ARCHIVO_MC)


def _indice_theta(datos, theta):
    return int(np.argmin(np.abs(datos["theta_grados"] - theta)))


def _cientifico(x):
    exponente = int(np.floor(np.log10(x)))
    mantisa = x / 10**exponente
    sup = str(exponente).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return (f"10{sup}" if np.isclose(mantisa, 1) else f"{mantisa:.1f}×10{sup}")


def _texto_parametros(datos):
    return (
        f"n₀ = {_cientifico(float(datos['n0_cm3']))} cm⁻³, β = {float(datos['beta']):.2f}, "
        f"r_c = {float(datos['rc_kpc']):.0f} kpc, L = {float(datos['longitud_kpc']) / 1e3:.0f} Mpc, "
        f"B = {float(datos['b0_ng']):.0f} nG; {int(datos['n_semillas'])} realizaciones."
    )


# ---------------------------------------------------------------------------
# Fig. 1 — mapas de RM
# ---------------------------------------------------------------------------
def fig1_mapas_rm():
    angulos = [0, 30, 60, 90]
    dx = config_fisica.DX_BASE.to_value(u.kpc)
    n = config_fisica.N_BASE
    mapas = {}
    for theta in angulos:
        ruta = os.path.join(DIR_MAPAS, f"theta_{theta:02d}deg", "rm_mapa.npy")
        if os.path.exists(ruta):
            mapas[theta] = np.load(ruta)
    if not mapas:
        print("Omitiendo fig. 1: faltan los mapas, ejecuta run.py primero.")
        return

    limite = n / 2 * dx
    recorte = 1600.0  # kpc: el filamento mide 2 Mpc; se deja margen a los lados
    pico = np.percentile(np.abs(np.concatenate([m.ravel() for m in mapas.values()])), 99.8)

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True,
                            gridspec_kw=dict(wspace=0.06))
    for ax, theta in zip(axs, angulos):
        ax.grid(False)
        if theta not in mapas:
            ax.set_title(f"θ = {theta}°: falta el mapa")
            continue
        im = ax.imshow(mapas[theta].T, cmap="RdBu_r", origin="lower",
                       extent=[-limite, limite, -limite, limite], vmin=-pico, vmax=pico)
        ax.set_xlim(-recorte, recorte)
        ax.set_ylim(-recorte, recorte)
        ax.set_title(rf"$\theta = {theta}^\circ$")
        ax.set_xlabel("x (kpc)")
    axs[0].set_ylabel("y (kpc)")
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.015)
    cb.set_label(r"RM (rad m$^{-2}$)")
    fig.suptitle("Mapas simulados de medida de rotación del filamento según el ángulo de visión",
                 y=1.02, fontsize=13)
    _pie(fig, "θ = ángulo entre el eje del filamento y la línea de visión (0° de frente, 90° de lado). "
              "Misma realización del campo turbulento en los cuatro paneles y escala de color común. "
              f"B = {config_fisica.B0.to_value(u.nG):.0f} nG.")
    _guardar(fig, "fig1_mapas_rm.png")


# ---------------------------------------------------------------------------
# Fig. 2 — perfiles transversales y forma funcional
# ---------------------------------------------------------------------------
def fig2_perfiles_transversales():
    datos = _cargar_mc()
    if datos is None:
        return
    d = datos["centros_kpc"]
    rc = float(datos["rc_kpc"])
    x = np.linspace(0, datos["bordes_kpc"][-1], 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))

    for theta, color, marcador in [(0, AZUL, "o"), (40, AGUA, "D"), (90, NARANJA, "s")]:
        i = _indice_theta(datos, theta)
        th = datos["theta_grados"][i]
        _puntos(ax1, d, datos["perfil_apilado"][i], datos["perfil_apilado_err"][i],
                color, rf"$\theta={th:.0f}^\circ$: $p={datos['mc_p'][i]:.2f}\pm{datos['mc_p_err'][i]:.2f}$",
                marcador)
        ax1.plot(x, beta_dispersion_model(x, datos["mc_sigma0"][i], rc, datos["mc_p"][i]),
                 color=color, lw=1.8)
    ax1.set(xlabel="Distancia transversal al eje proyectado d (kpc)",
            ylabel=r"$\sigma_{RM}(d)$ (rad m$^{-2}$)", ylim=(0, None),
            title=r"(a) Perfil de dispersión y ajuste $\sigma_0(1+d^2/r_c^2)^{-p}$")
    ax1.legend(fontsize=9)

    i = _indice_theta(datos, 90)
    perfil, err = datos["perfil_apilado"][i], datos["perfil_apilado_err"][i]
    _puntos(ax2, d, perfil, err, NARANJA, r"Simulación, $\theta=90^\circ$", "s")
    ax2.plot(x, beta_dispersion_model(x, datos["mc_sigma0"][i], rc, datos["mc_p"][i]),
             color=NARANJA, lw=1.8,
             label=rf"Ley beta ($r_c$ fijo): $R^2={datos['mc_r2_beta'][i]:.3f}$")
    ax2.plot(x, gaussian_model(x, datos["mc_sigma0_gauss"][i], datos["mc_w_gauss"][i]),
             "--", color=GRIS, lw=1.6,
             label=rf"Gaussiana: $R^2={datos['mc_r2_gauss'][i]:.3f}$")
    ax2.set(xlabel="Distancia transversal al eje proyectado d (kpc)",
            ylabel=r"$\sigma_{RM}(d)$ (rad m$^{-2}$)", ylim=(0, None),
            title=r"(b) Ley beta y gaussiana ajustadas al perfil a $\theta=90^\circ$")
    ax2.legend(fontsize=9)

    _pie(fig, "Puntos: perfil apilado de las realizaciones Monte Carlo (barras: bootstrap). "
              "Líneas de color: ajuste con r_c fijo al del perfil de densidad. "
              + _texto_parametros(datos))
    _guardar(fig, "fig2_perfiles_transversales.png")


# ---------------------------------------------------------------------------
# Fig. 3 — forma y ancho del perfil vs theta
# ---------------------------------------------------------------------------
def fig3_forma_vs_theta():
    datos = _cargar_mc()
    if datos is None:
        return
    th = datos["theta_grados"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))

    _puntos(ax1, th, datos["mc_p"], datos["mc_p_err"], AZUL, "Simulación Monte Carlo")
    p_f, p_l = float(datos["p_frontal"]), float(datos["p_lateral"])
    ax1.axhline(p_f, color=TINTA_SECUNDARIA, lw=1, ls="--")
    ax1.text(th.max(), p_f + 0.008, rf"límite de frente: $p=3\beta/2={p_f:.2f}$",
             ha="right", va="bottom", fontsize=9, color=TINTA_SECUNDARIA)
    ax1.axhline(p_l, color=TINTA_SECUNDARIA, lw=1, ls="--")
    ax1.text(th.min(), p_l - 0.008, rf"límite de lado: $p=(3\beta-\frac{{1}}{{2}})/2={p_l:.2f}$",
             ha="left", va="top", fontsize=9, color=TINTA_SECUNDARIA)
    ax1.set(xlabel=r"Ángulo de visión $\theta$ (grados)",
            ylabel=r"Exponente $p$ de $\sigma_{RM}\propto(1+d^2/r_c^2)^{-p}$",
            title=r"(a) Exponente $p$ del perfil vs. ángulo de visión",
            ylim=(0.6, 1.12))

    _puntos(ax2, th, datos["mc_hwhm"], datos["mc_hwhm_err"], AZUL, "Simulación Monte Carlo")
    ax2.set(xlabel=r"Ángulo de visión $\theta$ (grados)",
            ylabel=r"Semiancho a media altura $d_{1/2}$ (kpc)",
            title=r"(b) Semiancho a media altura vs. ángulo de visión",
            ylim=(0, None))
    cambio = 100 * (datos["mc_hwhm"][-1] / datos["mc_hwhm"][0] - 1)
    ax2.text(0.03, 0.22,
             rf"$d_{{1/2}}({th[-1]:.0f}^\circ)\,/\,d_{{1/2}}({th[0]:.0f}^\circ)$ = {1 + cambio / 100:.2f}",
             transform=ax2.transAxes, va="top", fontsize=9.5)

    _pie(fig, "Líneas punteadas en (a): límites de fórmula cerrada para θ = 0° (σ ∝ n_e(d)) y "
              "θ = 90° (σ² ∝ ∫n_e² dl) de un campo de paseo aleatorio sobre un perfil beta. "
              "Semiancho: d½ = r_c·√(2^(1/p) − 1). " + _texto_parametros(datos))
    _guardar(fig, "fig3_forma_vs_theta.png")


# ---------------------------------------------------------------------------
# Fig. 4 — amplitud en el eje vs theta
# ---------------------------------------------------------------------------
def fig4_sigma0_vs_theta():
    datos = _cargar_mc()
    if datos is None:
        return
    th = datos["theta_grados"]
    s0, s0_err = datos["mc_sigma0"], datos["mc_sigma0_err"]
    una = datos["una_realizacion_sigma0_std"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(th, s0 - una, s0 + una, color=AZUL, alpha=0.12, lw=0,
                    label="Dispersión de UN filamento (±1σ entre realizaciones)")
    _puntos(ax, th, s0, s0_err, AZUL, "Simulación Monte Carlo (media)")
    ax.set(xlabel=r"Ángulo de visión $\theta$ (grados)",
           ylabel=r"$\sigma_0$, dispersión de RM en el eje (rad m$^{-2}$)",
           title=r"Dispersión de RM en el eje vs. ángulo de visión", ylim=(0, None))
    razon = s0[0] / s0[-1]
    ax.text(0.97, 0.97, rf"$\sigma_0({th[0]:.0f}^\circ)\,/\,\sigma_0({th[-1]:.0f}^\circ)$ = {razon:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=9)
    _pie(fig, f"σ_RM es lineal en B: para otro campo multiplicar por B/{float(datos['b0_ng']):.0f} nG. "
              + _texto_parametros(datos))
    _guardar(fig, "fig4_sigma0_vs_theta.png")


# ---------------------------------------------------------------------------
# Fig. 5 — distribución de RM normalizada
# ---------------------------------------------------------------------------
def fig5_gaussianidad_rm():
    datos = _cargar_mc()
    if datos is None:
        return
    bordes = datos["hist_bordes_z"]
    centros = 0.5 * (bordes[1:] + bordes[:-1])
    ancho = np.diff(bordes)
    cuentas = datos["hist_z"].sum(axis=0)
    densidad = cuentas / (cuentas.sum() * ancho)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(centros, densidad, width=ancho * 0.9, color=AZUL, alpha=0.85,
           label="Simulación (todos los ángulos)")
    z = np.linspace(bordes[0], bordes[-1], 400)
    ax.plot(z, norm.pdf(z), color=GRIS, lw=2, label="Normal estándar N(0, 1)")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set(xlabel=r"RM / $\sigma_{RM}$ esperado en el mismo píxel",
           ylabel="Densidad de probabilidad",
           title="Distribución de RM normalizada por su dispersión esperada")
    n_total = cuentas.sum()
    asim = np.average(datos["z_asimetria"], weights=datos["hist_z"].sum(axis=1))
    curt = np.average(datos["z_exceso_curtosis"], weights=datos["hist_z"].sum(axis=1))
    ax.text(0.03, 0.97,
            f"media {np.average(datos['z_media'], weights=datos['hist_z'].sum(axis=1)):+.3f}\n"
            f"asimetría {asim:+.3f}\nexceso de curtosis {curt:+.3f}\n"
            f"{n_total:.1e} píxeles",
            transform=ax.transAxes, va="top", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=9)
    _pie(fig, "Cada valor de RM se divide entre la σ_RM esperada de su línea de visión "
              "(faradaymr.analysis.expected). Referencia: una variable gaussiana de media cero "
              "sigue N(0, 1), con media 0, asimetría 0 y exceso de curtosis 0.")
    _guardar(fig, "fig5_gaussianidad_rm.png")


# ---------------------------------------------------------------------------
# Fig. 6 — umbral teórico de detección
# ---------------------------------------------------------------------------
def fig6_umbral_deteccion():
    datos = _cargar_mc()
    if datos is None:
        return
    umbral = calcular_umbral(datos)
    th = datos["theta_grados"]
    colores = [NARANJA, AZUL]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    for (nombre, e), color in zip(umbral["escenarios"].items(), colores):
        ax1.plot(th, e["resumen"]["n_por_theta"], "o-", color=color, lw=1.8, ms=6,
                 mfc="white", mew=1.6, label=nombre)
    ax1.axhline(N_FUENTES_STUARDI, color=TINTA_SECUNDARIA, ls="--", lw=1)
    ax1.text(th.min(), N_FUENTES_STUARDI * 1.6,
             f"{N_FUENTES_STUARDI} fuentes: filamento A3667/3651 (Stuardi et al. 2026)",
             fontsize=8.5, color=TINTA_SECUNDARIA)
    ax1.set_yscale("log")
    ax1.set(xlabel=r"Ángulo de visión $\theta$ (grados)",
            ylabel=f"Fuentes de fondo necesarias ({K_SIGMA:.0f}σ)",
            title=f"(a) Fuentes para detectar B = {umbral['b0_ng']:.0f} nG")
    ax1.legend(fontsize=9, loc="center right")

    n = np.logspace(1, 9, 200)
    i0 = int(np.argmax(datos["mc_sigma0"]))
    for (nombre, e), color in zip(umbral["escenarios"].items(), colores):
        b_min = campo_minimo_detectable(
            n, datos["mc_sigma0"][i0], umbral["b0_ng"], k_sigma=K_SIGMA,
            sigma_fondo_rad_m2=SIGMA_INTRINSECA_RAD_M2, sigma_medicion_rad_m2=e["sigma_med"],
        )
        ax2.plot(n, b_min, color=color, lw=1.8, label=nombre)
    ax2.axhline(umbral["b0_ng"], color=GRIS, lw=1, ls="--")
    ax2.text(12, umbral["b0_ng"] * 1.25, f"campo del modelo ({umbral['b0_ng']:.0f} nG)",
             fontsize=8.5, color=GRIS)
    ax2.axvline(N_FUENTES_STUARDI, color=TINTA_SECUNDARIA, ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set(xlabel="Fuentes de fondo por muestra",
            ylabel=f"Campo mínimo detectable a {K_SIGMA:.0f}σ (nG)",
            title=rf"(b) Campo mínimo detectable vs. fuentes ($\theta={th[i0]:.0f}^\circ$)")
    ax2.legend(fontsize=9)
    _pie(fig, "Exceso de varianza de RM detrás del filamento frente a una región de control, "
              "con dispersión intrínseca de 7 rad m⁻² y, en el escenario POSSUM, 12 rad m⁻² de error "
              "de medición (Stuardi et al. 2026). Usa σ_RM en el eje, en el ángulo de σ₀ máximo.")
    _guardar(fig, "fig6_umbral_deteccion.png")


if __name__ == "__main__":
    fig1_mapas_rm()
    fig2_perfiles_transversales()
    fig3_forma_vs_theta()
    fig4_sigma0_vs_theta()
    fig5_gaussianidad_rm()
    fig6_umbral_deteccion()
