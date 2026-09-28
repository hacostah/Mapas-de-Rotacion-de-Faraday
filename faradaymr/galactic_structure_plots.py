"""
Vistas del escenario 3D (densidad de electrones térmicos, campo magnético
regular y turbulento) que arma `model.construir_escenario`, ANTES de
integrar ninguna línea de visión -a diferencia de `faradaymr.plotting_sky`
(mapas de cielo (l, b), el resultado final del ray tracing), estas figuras
muestran la estructura de la "Vía Láctea de juguete" en sí, vista desde
afuera, con distintos esquemas: cara-on (mirando el disco desde el polo),
de canto (mirando el disco desde el plano), y perfiles 1D radiales/
verticales.

Por qué hace falta esto además de los mapas de cielo: un mapa (l, b) es lo
que *vería* un observador en el Sol, pero no deja ver directamente la
geometría 3D que lo produce (¿dónde están los brazos exactamente? ¿qué tan
grueso es el disco? ¿el campo regular de verdad traza una espiral
coherente, o se pierde entre la turbulencia?) -preguntas que solo se
responden mirando la caja completa desde afuera.

Convención de coordenadas: igual que `model.construir_escenario` (ver su
docstring) -los arreglos vienen indexados en la esquina física (0,0,0) de
la caja, con el centro galáctico en el centro geométrico
(`centro_caja = box_size/2`); todas las funciones de este módulo reciben
`dx`/`box_size` y hacen ellas mismas el corrimiento a coordenadas
centradas en el centro galáctico (GC en (0,0,0), Sol en
`(-r0, 0, 0)`) para no tener que pasar arreglos de coordenadas 3D completos
(que ya pesan N^3, no hace falta duplicarlos solo para graficar).
"""

from __future__ import annotations

import os

import numpy as np


def _eje_centrado_en_gc(dx, box_size):
    """Eje 1D (kpc) de la caja, centrado en el centro galáctico -ver
    docstring del módulo para la convención."""
    n = int(round(box_size / dx))
    return np.arange(n) * dx - box_size / 2.0


def _indice_mas_cercano(eje, valor):
    return int(np.argmin(np.abs(eje - valor)))


def _curva_brazo_espiral(pitch_angle, r0, fase, r_min, r_max, n_puntos=400):
    """
    (x, y) de un brazo espiral logarítmico, usando la misma convención que
    `faradaymr.simulation.galactic_disk.spiral_arm_density_factor`:
    phi_brazo(r) = fase + ln(r/r0)/tan(pitch_angle) -ver la derivación en
    el docstring de ese módulo (es la inversa, en r<->phi, de la forma más
    común en la literatura r(phi) = r0*exp((phi-phi0)*tan(pitch))-.
    """
    r = np.linspace(max(r_min, 1e-3), r_max, n_puntos)
    phi = fase + np.log(r / r0) / np.tan(pitch_angle)
    return r * np.cos(phi), r * np.sin(phi)


def _marcar_gc_y_sol(ax, observer_pos_gc):
    ax.plot(0, 0, "*", color="gold", markersize=16, markeredgecolor="k",
             markeredgewidth=0.6, zorder=5, label="Centro galáctico")
    ax.plot(observer_pos_gc[0], observer_pos_gc[1], "o", color="cyan",
             markersize=7, markeredgecolor="k", markeredgewidth=0.6,
             zorder=5, label="Sol")


def corte_densidad_cara_on(
    ruta_destino, ne, dx, box_size, observer_pos_gc, r0, pitch_angle, n_arms,
    phase0=0.0, nombre_archivo="figura_estructura_densidad_cara_on.png",
):
    """
    Densidad de electrones térmicos en el plano galáctico (z=0), visto
    "cara-on" (desde el polo norte galáctico) -la vista estándar en la
    literatura de GMF (p.ej. Jansson & Farrar 2012, Fig. 5) para mostrar la
    estructura espiral de un vistazo, algo que ningún mapa de cielo (l, b)
    puede mostrar directamente porque el observador está DENTRO del disco.

    Se superponen las curvas analíticas de los `n_arms` brazos (misma
    fórmula que usa `spiral_arm_density_factor` para calcular el realce):
    confirma visualmente que el realce de densidad de verdad sigue esas
    curvas, no es un artefacto de la malla.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    eje = _eje_centrado_en_gc(dx, box_size)
    k_medio = _indice_mas_cercano(eje, 0.0)
    slice_ne = np.asarray(ne)[:, :, k_medio]

    fig, ax = plt.subplots(figsize=(7, 6.2))
    malla = ax.imshow(
        slice_ne.T, origin="lower",
        extent=[eje[0], eje[-1], eje[0], eje[-1]],
        cmap="inferno", norm=LogNorm(vmin=np.nanmax(slice_ne) / 1e3, vmax=np.nanmax(slice_ne)),
    )
    r_max = 0.95 * eje[-1]
    for i in range(n_arms):
        fase = 2.0 * np.pi * i / n_arms + phase0
        x_brazo, y_brazo = _curva_brazo_espiral(pitch_angle, r0, fase, 0.3, r_max)
        dentro = (np.abs(x_brazo) <= eje[-1]) & (np.abs(y_brazo) <= eje[-1])
        ax.plot(x_brazo[dentro], y_brazo[dentro], "-", color="cyan", lw=1.0, alpha=0.8)

    _marcar_gc_y_sol(ax, observer_pos_gc)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title("Densidad de electrones térmicos, plano galáctico (z=0)\nvisto cara-on desde el polo norte")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    cbar = fig.colorbar(malla, ax=ax)
    cbar.set_label(r"$n_e$ [cm$^{-3}$]")
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200)
    plt.close(fig)
    return ruta_completa


def corte_campo_cara_on(
    ruta_destino, bx, by, dx, box_size, observer_pos_gc,
    titulo="Campo magnético, plano galáctico (z=0)",
    nombre_archivo="figura_estructura_campo_cara_on.png",
    densidad_lineas=1.8,
):
    """
    Líneas de campo magnético (streamlines de bx, by) en el plano
    galáctico, coloreadas por |B| -la vista que muestra si el campo traza
    una espiral coherente o no (ver docstring de
    `faradaymr.fields.spiral_field`: sin componente regular, un campo
    puramente turbulento no sostendría ninguna dirección coherente aquí).
    """
    import matplotlib.pyplot as plt

    eje = _eje_centrado_en_gc(dx, box_size)
    k_medio = _indice_mas_cercano(eje, 0.0)
    bx_slice = np.asarray(bx)[:, :, k_medio]
    by_slice = np.asarray(by)[:, :, k_medio]
    b_mag = np.sqrt(bx_slice**2 + by_slice**2)

    fig, ax = plt.subplots(figsize=(7, 6.2))
    streams = ax.streamplot(
        eje, eje, bx_slice.T, by_slice.T, color=b_mag.T, cmap="viridis",
        density=densidad_lineas, linewidth=1.0, arrowsize=1.0,
    )
    _marcar_gc_y_sol(ax, observer_pos_gc)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(eje[0], eje[-1])
    ax.set_ylim(eje[0], eje[-1])
    cbar = fig.colorbar(streams.lines, ax=ax)
    cbar.set_label(r"$|B|$ [$\mu$G]")
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200)
    plt.close(fig)
    return ruta_completa


def comparacion_campo_regular_vs_total(
    ruta_destino, bx_reg, by_reg, bx_total, by_total, dx, box_size,
    observer_pos_gc, nombre_archivo="figura_estructura_campo_regular_vs_total.png",
):
    """
    Dos paneles lado a lado -campo SOLO regular vs. regular+turbulento-,
    misma escala de color: la comparación visual directa que motiva por
    qué hace falta una componente regular en primer lugar (ver docstring
    de `faradaymr.fields.spiral_field`): a la izquierda, líneas de campo
    perfectamente ordenadas en espiral; a la derecha, la misma espiral de
    fondo pero con las líneas enredadas por la turbulencia superpuesta -la
    turbulencia sola, sin la regular, despolarizaría casi toda la señal
    porque el ángulo de polarización promediaría a algo sin dirección
    preferida.
    """
    import matplotlib.pyplot as plt

    eje = _eje_centrado_en_gc(dx, box_size)
    k_medio = _indice_mas_cercano(eje, 0.0)

    bx_reg_s = np.asarray(bx_reg)[:, :, k_medio]
    by_reg_s = np.asarray(by_reg)[:, :, k_medio]
    bx_tot_s = np.asarray(bx_total)[:, :, k_medio]
    by_tot_s = np.asarray(by_total)[:, :, k_medio]

    b_mag_max = float(
        max(
            np.nanmax(np.sqrt(bx_reg_s**2 + by_reg_s**2)),
            np.nanmax(np.sqrt(bx_tot_s**2 + by_tot_s**2)),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, bx_s, by_s, titulo in [
        (axes[0], bx_reg_s, by_reg_s, "Solo componente regular"),
        (axes[1], bx_tot_s, by_tot_s, "Regular + turbulenta"),
    ]:
        b_mag = np.sqrt(bx_s**2 + by_s**2)
        streams = ax.streamplot(
            eje, eje, bx_s.T, by_s.T, color=b_mag.T, cmap="viridis",
            density=1.8, linewidth=1.0, arrowsize=1.0,
            norm=plt.Normalize(vmin=0, vmax=b_mag_max),
        )
        _marcar_gc_y_sol(ax, observer_pos_gc)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_title(titulo)
        ax.set_aspect("equal")
        ax.set_xlim(eje[0], eje[-1])
        ax.set_ylim(eje[0], eje[-1])

    axes[0].legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(streams.lines, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(r"$|B|$ [$\mu$G]")
    fig.suptitle("¿Por qué hace falta un campo regular? Coherencia vs. desorden", fontsize=12)
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta_completa


def corte_borde_on(
    ruta_destino, ne, dx, box_size, observer_pos_gc,
    nombre_archivo="figura_estructura_densidad_borde_on.png",
):
    """
    Densidad de electrones en el corte vertical (x, z) que pasa por el eje
    Sol-centro galáctico (y = y_Sol = 0), visto "de canto": la vista que
    muestra la escala de altura del disco (`SCALE_HEIGHT_NE`), invisible
    en la vista cara-on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    eje = _eje_centrado_en_gc(dx, box_size)
    j_sol = _indice_mas_cercano(eje, observer_pos_gc[1])
    slice_ne = np.asarray(ne)[:, j_sol, :]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    malla = ax.imshow(
        slice_ne.T, origin="lower",
        extent=[eje[0], eje[-1], eje[0], eje[-1]],
        cmap="inferno", norm=LogNorm(vmin=np.nanmax(slice_ne) / 1e3, vmax=np.nanmax(slice_ne)),
        aspect="auto",
    )
    ax.axvline(0, color="gold", lw=1.2, ls="--", label="Centro galáctico")
    ax.axvline(observer_pos_gc[0], color="cyan", lw=1.2, ls="--", label="Sol")
    ax.set_xlabel("x [kpc] (a lo largo del eje Sol-centro galáctico)")
    ax.set_ylabel("z [kpc] (altura sobre el plano)")
    ax.set_title("Densidad de electrones térmicos, corte de canto (y = y_Sol)")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(malla, ax=ax)
    cbar.set_label(r"$n_e$ [cm$^{-3}$]")
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200)
    plt.close(fig)
    return ruta_completa


def perfiles_radial_y_vertical(
    ruta_destino, ne, bx, by, bz, dx, box_size, observer_pos_gc, r0,
    nombre_archivo="figura_estructura_perfiles_radial_vertical.png",
):
    """
    Dos perfiles 1D -n_e y |B| en función del radio galactocéntrico R (a
    z=0, a lo largo del eje Sol-centro galáctico) y en función de la
    altura |z| (en R=r0, la posición del Sol)- con escala log en y: la
    forma estándar de mostrar los parámetros de un modelo de disco tipo
    GMF en la literatura (comparar contra, p.ej., Fig. 3 de Jansson &
    Farrar 2012), y el chequeo más directo de que `SCALE_RADIAL_NE`/
    `SCALE_HEIGHT_NE`/`SCALE_RADIAL_B`/`SCALE_HEIGHT_B` (config_fisica.py)
    de verdad producen el decaimiento exponencial esperado en los arreglos
    3D ya generados (no solo en la fórmula).
    """
    import matplotlib.pyplot as plt

    eje = _eje_centrado_en_gc(dx, box_size)
    j_medio = _indice_mas_cercano(eje, 0.0)
    k_medio = _indice_mas_cercano(eje, 0.0)

    ne = np.asarray(ne)
    b_mag = np.sqrt(np.asarray(bx) ** 2 + np.asarray(by) ** 2 + np.asarray(bz) ** 2)

    ne_radial = ne[:, j_medio, k_medio]
    b_radial = b_mag[:, j_medio, k_medio]
    R_radial = np.abs(eje)

    i_sol = _indice_mas_cercano(eje, observer_pos_gc[0])
    ne_vertical = ne[i_sol, j_medio, :]
    b_vertical = b_mag[i_sol, j_medio, :]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    orden = np.argsort(R_radial)
    axes[0].semilogy(R_radial[orden], np.clip(ne_radial[orden], 1e-6, None), "-",
                       color="firebrick", label=r"$n_e$")
    ax0b = axes[0].twinx()
    ax0b.semilogy(R_radial[orden], np.clip(b_radial[orden], 1e-6, None), "--",
                   color="steelblue", label=r"$|B|$")
    axes[0].axvline(r0, color="gold", lw=1, ls=":", label="R$_{solar}$")
    axes[0].set_xlabel("R [kpc] (a lo largo del eje Sol-GC, z=0)")
    axes[0].set_ylabel(r"$n_e$ [cm$^{-3}$]", color="firebrick")
    ax0b.set_ylabel(r"$|B|$ [$\mu$G]", color="steelblue")
    axes[0].set_title("Perfil radial")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].semilogy(eje, np.clip(ne_vertical, 1e-6, None), "-",
                       color="firebrick", label=r"$n_e$")
    ax1b = axes[1].twinx()
    ax1b.semilogy(eje, np.clip(b_vertical, 1e-6, None), "--",
                   color="steelblue", label=r"$|B|$")
    axes[1].axvline(0, color="gray", lw=1, ls=":")
    axes[1].set_xlabel("z [kpc] (en R = R$_{solar}$)")
    axes[1].set_ylabel(r"$n_e$ [cm$^{-3}$]", color="firebrick")
    ax1b.set_ylabel(r"$|B|$ [$\mu$G]", color="steelblue")
    axes[1].set_title("Perfil vertical (en el radio solar)")

    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200)
    plt.close(fig)
    return ruta_completa


def perfil_rotacion_galactica(
    ruta_destino, r0, v0=220.0, r_core=2.0, r_max=20.0,
    nombre_archivo="figura_estructura_rotacion_galactica.png",
):
    """
    Estructura MECÁNICA (dinámica orbital) del disco, en vez de la
    electromagnética (n_e, B) que cubre el resto de este módulo: velocidad
    circular V_c(R), velocidad angular Omega(R)=V_c/R y período orbital
    T(R)=2*pi*R/V_c, las tres cantidades estándar para describir cómo
    ROTA la Vía Láctea como cuerpo, independientemente de su contenido de
    plasma/campo magnético.

    V_c(R) = v0 * R / sqrt(R^2 + r_core^2): la forma funcional más simple
    que (a) sube casi linealmente cerca del centro (rotación de cuerpo
    rígido, evita la divergencia no física de un punto masivo en R=0) y
    (b) se aplana a v0 para R >> r_core -la curva de rotación de la Vía
    Láctea es observacionalmente CASI PLANA más allá de un par de kpc
    (Sofue 2013, "Rotation Curve of the Milky Way and the Andromeda
    Galaxy", PASJ; Reid et al. 2014; Eilers et al. 2019 miden ~229 km/s en
    R_solar). No es un ajuste a un modelo de masa (bulbo+disco+halo)
    -este framework no tiene ninguno-, es la aproximación observacional
    más simple que reproduce ese hecho, con `r_core` fijado solo para que
    la curva no diverja en R=0 (no una escala física medida, análogo al
    mismo tipo de nota que ya usa `config_fisica.ARM_WIDTH`).

    v0 : velocidad circular asintótica (km/s); default 220 km/s, el valor
        "de libro de texto" citado en Sofue (2013) y consistente en orden
        de magnitud con las medidas más recientes (~229 km/s, Eilers+2019).
    r_core : radio de transición cuerpo-rígido -> plano (kpc); default 2.0,
        del orden del bulbo galáctico, sin pretensión de precisión.
    r_max : hasta qué radio graficar (kpc).

    Marca R_solar y su período orbital -el "año galáctico" del Sol,
    conocido observacionalmente en ~225-250 Myr (Sofue 2013)- como chequeo
    de orden de magnitud visible de un vistazo, el mismo espíritu que
    `faradaymr.calibration`.
    """
    import matplotlib.pyplot as plt

    KPC_KM = 3.0856775814913673e16  # 1 kpc en km
    GYR_S = 3.15576e16  # 1 Gyr en s (año juliano)

    radio = np.linspace(1e-3, r_max, 400)
    v_c = v0 * radio / np.sqrt(radio**2 + r_core**2)
    omega = v_c / radio  # km/s/kpc
    periodo_gyr = (2.0 * np.pi * radio * KPC_KM) / (v_c * GYR_S)  # Gyr

    v_c_r0 = v0 * r0 / np.sqrt(r0**2 + r_core**2)
    periodo_r0_myr = (2.0 * np.pi * r0 * KPC_KM) / (v_c_r0 * GYR_S) * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(radio, v_c, "-", color="darkorange")
    axes[0].axvline(r0, color="gray", lw=1, ls=":")
    axes[0].axhline(v0, color="gray", lw=0.7, ls="--", alpha=0.6)
    axes[0].set_xlabel("R [kpc]")
    axes[0].set_ylabel(r"$V_c$ [km/s]")
    axes[0].set_title("Velocidad circular (curva de rotación)")
    axes[0].set_ylim(0, v0 * 1.15)

    axes[1].plot(radio, omega, "-", color="teal")
    axes[1].axvline(r0, color="gray", lw=1, ls=":")
    axes[1].set_xlabel("R [kpc]")
    axes[1].set_ylabel(r"$\Omega$ [km s$^{-1}$ kpc$^{-1}$]")
    axes[1].set_title("Velocidad angular")

    axes[2].plot(radio, periodo_gyr * 1e3, "-", color="indigo")
    axes[2].axvline(r0, color="gray", lw=1, ls=":",
                     label=f"R$_\\odot$={r0:.1f} kpc\nT={periodo_r0_myr:.0f} Myr")
    axes[2].set_xlabel("R [kpc]")
    axes[2].set_ylabel("Período orbital [Myr]")
    axes[2].set_title("Período orbital")
    axes[2].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Estructura mecánica del disco: rotación (V$_0$={v0:.0f} km/s, "
        f"Sofue 2013)",
        fontsize=11,
    )
    fig.tight_layout()
    ruta_completa = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta_completa, dpi=200)
    plt.close(fig)
    return ruta_completa


def generar_graficos_estructura(
    ruta_destino, ne, bx, by, bz, dx, box_size, observer_pos,
    r0, pitch_angle, n_arms, phase0=0.0, componentes_regular=None,
):
    """
    Orquesta todas las figuras de estructura 3D de este módulo para una
    corrida. `observer_pos` en coordenadas de CAJA (no centradas, ver
    docstring del módulo); se centra en el GC una sola vez acá.

    `componentes_regular`: tupla opcional `(bx_reg, by_reg, bz_reg)` -si
    se pasa (ver `model.construir_escenario(..., return_components=True)`),
    agrega la figura de comparación regular-vs-total; si es `None`, se
    omite esa figura sin fallar (por si alguien llama esto con una corrida
    vieja que no guardó las componentes por separado).

    Devuelve la lista de rutas de archivo generadas.
    """
    # `run.py` llama a esta función ANTES de `faradaymr.io.save_maps` (que
    # es lo que normalmente crea `ruta_destino`) -en una copia nueva del
    # repo (p.ej. recién clonada en Colab, ver `Faraday_MR_Colab.ipynb`),
    # `results/foreground_galactico/` todavía no existe en ese punto, y
    # `fig.savefig` no crea directorios por sí solo: sin este
    # `os.makedirs`, la primera figura de este módulo fallaba con
    # `FileNotFoundError` (visto en una corrida real en Colab). Mismo
    # patrón defensivo que ya usan `faradaymr.io.save_maps`/`save_fits`.
    os.makedirs(ruta_destino, exist_ok=True)

    observer_pos_gc = np.asarray(observer_pos) - box_size / 2.0

    rutas = [
        corte_densidad_cara_on(ruta_destino, ne, dx, box_size, observer_pos_gc, r0, pitch_angle, n_arms, phase0=phase0),
        corte_campo_cara_on(ruta_destino, bx, by, dx, box_size, observer_pos_gc),
        corte_borde_on(ruta_destino, ne, dx, box_size, observer_pos_gc),
        perfiles_radial_y_vertical(ruta_destino, ne, bx, by, bz, dx, box_size, observer_pos_gc, r0),
        perfil_rotacion_galactica(ruta_destino, r0),
    ]
    if componentes_regular is not None:
        bx_reg, by_reg, _ = componentes_regular
        rutas.append(
            comparacion_campo_regular_vs_total(
                ruta_destino, bx_reg, by_reg, bx, by, dx, box_size, observer_pos_gc
            )
        )
    return rutas
