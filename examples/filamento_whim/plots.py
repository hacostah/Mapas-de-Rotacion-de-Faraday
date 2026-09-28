import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from faradaymr.analysis.spatial_stats import transverse_rm_dispersion
from faradaymr.analysis.fitting import (
    fit_transverse_dispersion,
    fit_beta_dispersion,
    p_random_walk_cilindro,
)
from faradaymr.simulation.geometry import filament_axis_from_viewing_angle
from examples.filamento_whim import config_fisica
from examples.filamento_whim.model import construir_escenario
from faradaymr.simulation.geometry import sky_footprint_mask, filament_body_mask

# Rutas de los datos generados por run.py y run_barrido_theta.py
BASE_DIR = os.path.dirname(__file__)
DIR_MAPAS = os.path.join(BASE_DIR, "results", "mapas_poster")
ARCHIVO_MC = os.path.join(BASE_DIR, "results", "barrido_theta", "barrido_theta_mc.npz")
DIR_PLOTS = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(DIR_PLOTS, exist_ok=True)

# Estilo unificado para figuras académicas
plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})


def cargar_mapa_rm(theta_grados):
    """Función auxiliar para leer el mapa RM de la ruta correcta."""
    ruta = os.path.join(DIR_MAPAS, f"theta_{theta_grados:02.0f}deg", "rm_mapa.npy")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Falta el mapa para {theta_grados}°. Ejecuta run.py primero.")
    return np.load(ruta)


def fig1_mapa_rm_ejemplo():
    # 0°=Frente, 90°=Lado real (antes [0,15,30] nunca mostraba el filamento
    # de lado; ver la nota en run.py.generar_mapas_poster).
    angulos = [0, 30, 60, 90]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    dx_kpc = config_fisica.DX_BASE.to_value(u.kpc)
    limite = config_fisica.N_BASE / 2 * dx_kpc

    mapas = {}
    for theta in angulos:
        try:
            mapas[theta] = cargar_mapa_rm(theta)
        except FileNotFoundError:
            pass

    # Escala de color COMPARTIDA entre los 4 paneles (antes cada uno tenía su
    # propio rango, lo que hace parecer visualmente similar la dispersión de
    # RM a distintos theta aunque su amplitud real sea muy distinta).
    if mapas:
        pico = max(np.max(np.abs(m)) for m in mapas.values())
    else:
        pico = 1.0

    for ax, theta in zip(axs, angulos):
        if theta not in mapas:
            ax.set_title(f"Falta el mapa para {theta}°. Ejecuta run.py primero.", fontsize=10)
            continue
        rm_map = mapas[theta]

        # `rm_map[i, j]`: el eje 0 es x y el eje 1 es y (ver
        # `faradaymr.simulation.geometry.projected_axis_distance`). imshow
        # dibuja el primer eje del arreglo en VERTICAL, así que hay que
        # transponer para que "x" quede horizontal como dice la etiqueta
        # (sin esto, el mapa salía con los ejes x/y intercambiados).
        im = ax.imshow(
            rm_map.T, cmap='RdBu_r', origin='lower',
            extent=[-limite, limite, -limite, limite],
            vmin=-pico, vmax=pico,
        )
        ax.set_title(f"Medida de Rotación - $\\theta = {theta}^\\circ$")
        ax.set_xlabel("x (kpc)")
        if theta == 0:
            ax.set_ylabel("y (kpc)")

    fig.colorbar(
        im, ax=axs, label=r"RM (rad/m$^2$)", fraction=0.023, pad=0.02,
    )
    ruta_salida = os.path.join(DIR_PLOTS, "fig1_mapas_rm.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generada: {ruta_salida}")


def fig2_perfil_transversal_con_ajuste():
    theta = 30
    try:
        rm_map = cargar_mapa_rm(theta)
    except FileNotFoundError:
        print(f"Omitiendo Fig 2: Falta mapa RM para {theta}°")
        return

    dx_kpc = config_fisica.DX_BASE.to_value(u.kpc)
    dist_max_kpc = config_fisica.DIST_MAX_AJUSTE.to_value(u.kpc)
    axis_dir = filament_axis_from_viewing_angle(np.deg2rad(theta))

    # Recalculamos ne (no se guarda en disco, solo los mapas 2D) con la
    # misma semilla que usó run.py para este mapa, únicamente para construir
    # la máscara de huella del filamento finito (ver sky_footprint_mask):
    # sin ella, los píxeles más allá de las puntas proyectadas diluyen la
    # dispersión medida en cada bin, igual que en el barrido Monte Carlo.
    _, _, _, ne, _, _ = construir_escenario(
        n_spec=config_fisica.N_SPEC,
        b0_microgauss=config_fisica.B0.to_value(u.microgauss),
        axis_direction=axis_dir,
        use_gpu=False,
        rng=np.random.RandomState(42),
        longitud_filamento_kpc=config_fisica.LONGITUD_FILAMENTO.to_value(u.kpc),
    )
    footprint = sky_footprint_mask(ne) & filament_body_mask(
        rm_map.shape, axis_dir, dx_kpc,
        config_fisica.LONGITUD_FILAMENTO.to_value(u.kpc), config_fisica.RC.to_value(u.kpc),
    )

    # Ventana de ajuste FIJA (igual que run_barrido_theta.py), no "hasta la
    # mitad de la caja": así esta figura es comparable con la Fig. 3. El
    # número de bordes también se comparte vía config (antes esta figura
    # usaba 15 bordes y el barrido Monte Carlo 12, así que no eran
    # estrictamente comparables bin a bin pese a compartir la ventana).
    limite_ventana = min(dist_max_kpc, config_fisica.N_BASE / 2 * dx_kpc)
    bordes = np.linspace(0, limite_ventana, config_fisica.N_BORDES_PERFIL)
    centros, dispersion = transverse_rm_dispersion(
        rm_map, axis_dir, dx_kpc, bordes, footprint_mask=footprint
    )

    if hasattr(centros, 'get'):
        centros = centros.get()
    if hasattr(dispersion, 'get'):
        dispersion = dispersion.get()

    validos = ~np.isnan(dispersion)
    centros_v, dispersion_v = centros[validos], dispersion[validos]
    ajuste_g = fit_transverse_dispersion(centros_v, dispersion_v)
    ajuste_b = fit_beta_dispersion(centros_v, dispersion_v)
    p_fijo = p_random_walk_cilindro(config_fisica.BETA)
    ajuste_b_fijo = fit_beta_dispersion(centros_v, dispersion_v, p_fijo=p_fijo)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(centros, dispersion, color='black', label='Datos (Simulación)', zorder=3)

    x_fit = np.linspace(0, max(centros), 200)
    y_gauss = ajuste_g.sigma0 * np.exp(-0.5 * (x_fit / ajuste_g.width) ** 2)
    y_beta = ajuste_b.sigma0 * (1.0 + (x_fit / ajuste_b.r_c) ** 2) ** (-ajuste_b.p)
    y_beta_fijo = ajuste_b_fijo.sigma0 * (1.0 + (x_fit / ajuste_b_fijo.r_c) ** 2) ** (-p_fijo)

    # Las etiquetas LaTeX necesitan ser "raw" (\sigma, \theta), pero el
    # salto de línea SÍ debe ser un newline real: mezclar ambas cosas en
    # una sola cadena raw hacía que "\n" saliera literal en la leyenda
    # (bug de la versión anterior de esta figura). Se arma con .join().
    etiqueta_gauss = "\n".join([
        "Ajuste Gaussiano",
        rf"$\sigma_0={ajuste_g.sigma0:.4f}$, $w={ajuste_g.width:.1f}$ kpc",
        rf"$R^2={ajuste_g.r_squared:.3f}$",
    ])
    etiqueta_beta = "\n".join([
        "Ajuste beta, p libre",
        rf"$\sigma_0={ajuste_b.sigma0:.4f}$, $r_c={ajuste_b.r_c:.1f}\pm{ajuste_b.r_c_err:.1f}$ kpc, $p={ajuste_b.p:.2f}$",
        rf"$R^2={ajuste_b.r_squared:.3f}$ (r_c y p casi degenerados, ver README)",
    ])
    etiqueta_beta_fijo = "\n".join([
        rf"Ajuste beta, $p={p_fijo:.2f}$ fijo (random-walk)",
        rf"$\sigma_0={ajuste_b_fijo.sigma0:.4f}$, $r_c={ajuste_b_fijo.r_c:.1f}\pm{ajuste_b_fijo.r_c_err:.1f}$ kpc",
        rf"$R^2={ajuste_b_fijo.r_squared:.3f}$",
    ])

    plt.plot(x_fit, y_gauss, 'r-', lw=2, label=etiqueta_gauss)
    plt.plot(x_fit, y_beta, 'g--', lw=2, label=etiqueta_beta)
    plt.plot(x_fit, y_beta_fijo, 'b-.', lw=2, label=etiqueta_beta_fijo)

    plt.xlabel("Distancia transversal al eje (kpc)")
    plt.ylabel(r"$\sigma_{RM}$ (rad/m$^2$)")
    plt.title(f"Perfil de Dispersión RM Transversal ($\\theta={theta}^\\circ$)")
    plt.legend(fontsize=8)

    ruta_salida = os.path.join(DIR_PLOTS, "fig2_perfil_transversal.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generada: {ruta_salida}")


def fig3_ancho_vs_theta():
    if not os.path.exists(ARCHIVO_MC):
        print("Omitiendo Fig 3: Falta el archivo del barrido Monte Carlo.")
        return

    datos = np.load(ARCHIVO_MC)
    theta = datos["theta_grados"]
    n_semillas = int(datos["n_semillas"]) if "n_semillas" in datos else 50

    width = datos["width_medio_kpc"]
    width_sem = datos["width_std_kpc"] / np.sqrt(n_semillas)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        theta, width, yerr=width_sem,
        fmt='o-', color='navy', ecolor='darkred', capsize=5,
        capthick=1.5, markerfacecolor='white', markeredgewidth=1.5,
        label="Ajuste gaussiano ($w$)",
    )

    # Se grafica el ajuste beta con p FIJO (rc_beta_fijo), no el de p libre:
    # con 3 parámetros libres, r_c y p quedan casi degenerados en el rango
    # de distancias de este proyecto (barras de error grandes, r_c no
    # identificado de forma confiable). El de p fijo (ver
    # `faradaymr.analysis.fitting.p_random_walk_cilindro`) es la
    # comprobación de validación real. El ajuste libre se sigue guardando en
    # el .npz (`rc_beta_medio_kpc`) para quien quiera inspeccionarlo.
    if "rc_beta_fijo_medio_kpc" in datos:
        rc_beta = datos["rc_beta_fijo_medio_kpc"]
        rc_beta_sem = datos["rc_beta_fijo_std_kpc"] / np.sqrt(n_semillas)
        p_fijo = float(datos["p_fijo"]) if "p_fijo" in datos else float("nan")
        ax.errorbar(
            theta, rc_beta, yerr=rc_beta_sem,
            fmt='s--', color='seagreen', ecolor='darkgreen', capsize=5,
            capthick=1.5, markerfacecolor='white', markeredgewidth=1.5,
            label=rf"Ajuste beta, $p={p_fijo:.2f}$ fijo ($r_c$)",
        )
    elif "rc_beta_medio_kpc" in datos:
        rc_beta = datos["rc_beta_medio_kpc"]
        rc_beta_sem = datos["rc_beta_std_kpc"] / np.sqrt(n_semillas)
        ax.errorbar(
            theta, rc_beta, yerr=rc_beta_sem,
            fmt='s--', color='seagreen', ecolor='darkgreen', capsize=5,
            capthick=1.5, markerfacecolor='white', markeredgewidth=1.5,
            label=r"Ajuste forma beta, p libre ($r_c$)",
        )

    # Eje y honesto: empezar en 0 en vez de recortar al rango de los datos,
    # que exageraba visualmente una variación de pocos por ciento como si
    # fuera una tendencia dramática (bug de la versión anterior de esta
    # figura). Se deja un margen del 15% sobre el máximo para legibilidad.
    techo = 1.15 * np.nanmax(np.concatenate([
        width + width_sem,
        rc_beta + rc_beta_sem,
    ]))
    ax.set_ylim(0, techo)

    variacion_pct = 100 * (np.nanmax(width) - np.nanmin(width)) / np.nanmean(width)
    ax.text(
        0.02, 0.97,
        f"Variación de $w$ en el barrido: {variacion_pct:.1f}%\n"
        f"Barras: error estándar de la media (SEM), N={n_semillas} semillas.",
        transform=ax.transAxes, fontsize=9, color="gray", va="top",
    )

    ax.set_xlabel(r"Ángulo de visión $\theta$ (grados)")
    ax.set_ylabel("Escala transversal característica (kpc)")
    ax.set_title("Ancho Aparente del Filamento vs. Ángulo de Visión")
    # Posición explícita (no "best"): con la anotación ahora arriba a la
    # izquierda, dejar que matplotlib elija podía volver a superponer la
    # leyenda con ella en otros rangos de datos; abajo a la derecha está
    # vacío en este gráfico (ambas curvas se achatan en la parte superior).
    ax.legend(loc="lower right")

    ruta_salida = os.path.join(DIR_PLOTS, "fig3_ancho_vs_theta.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generada: {ruta_salida}")


def fig4_sigma0_vs_theta():
    """
    sigma0(theta): la amplitud de RM en el propio eje del filamento.
    Con un filamento de longitud FINITA (ver `model.py`), esta es la
    cantidad con la dependencia angular más clara y físicamente más fácil
    de explicar: a theta≈0 (filamento "de frente") la línea de visión
    recorre casi toda la longitud del filamento a densidad casi constante;
    a theta≈90° (filamento "de lado") solo atraviesa el perfil radial en un
    tramo de unos pocos radios de núcleo. `w`/`r_c` (Fig. 3), en cambio,
    describen la FORMA del perfil transversal, que para un filamento con
    simetría cilíndrica no tiene por qué depender fuertemente de theta.
    """
    if not os.path.exists(ARCHIVO_MC):
        print("Omitiendo Fig 4: Falta el archivo del barrido Monte Carlo.")
        return

    datos = np.load(ARCHIVO_MC)
    theta = datos["theta_grados"]
    n_semillas = int(datos["n_semillas"]) if "n_semillas" in datos else 50
    sigma0 = datos["sigma0_medio"]
    sigma0_sem = datos["sigma0_std"] / np.sqrt(n_semillas)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        theta, sigma0, yerr=sigma0_sem,
        fmt='o-', color='darkorange', ecolor='saddlebrown', capsize=5,
        capthick=1.5, markerfacecolor='white', markeredgewidth=1.5,
    )
    ax.set_ylim(0, 1.15 * np.nanmax(sigma0 + sigma0_sem))
    ax.set_xlabel(r"Ángulo de visión $\theta$ (grados)")
    ax.set_ylabel(r"$\sigma_0$ en el eje del filamento (rad/m$^2$)")
    ax.set_title("Amplitud de RM en el Eje vs. Ángulo de Visión")
    ax.text(
        0.98, 0.95, f"Barras: SEM, N={n_semillas} semillas.",
        transform=ax.transAxes, fontsize=9, color="gray", ha="right", va="top",
    )

    ruta_salida = os.path.join(DIR_PLOTS, "fig4_sigma0_vs_theta.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generada: {ruta_salida}")


if __name__ == "__main__":
    fig1_mapa_rm_ejemplo()
    fig2_perfil_transversal_con_ajuste()
    fig3_ancho_vs_theta()
    fig4_sigma0_vs_theta()
