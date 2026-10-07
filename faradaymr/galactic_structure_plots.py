"""
Figuras de la estructura 3D del modelo (disco, brazos, campo regular y
turbulento), vistas desde afuera y ANTES de integrar ninguna línea de
visión. Un mapa (l, b) muestra lo que ve el Sol, no la geometría que lo
produce; estas figuras responden a las preguntas de fondo: ¿los brazos
están donde deben?, ¿el disco decae con las escalas pedidas?, ¿la
turbulencia realmente sigue Kolmogorov y es gaussiana?

Una figura por pregunta física, sin duplicados:
  - estructura_disco:       n_e en z=0 y contraste de brazos n_e / n_e(sin brazos)
  - estructura_perfiles:    n_e, B regular y B turbulento vs R y vs |z|
  - estructura_campo:       líneas de campo, regular vs regular+turbulento
  - estructura_turbulencia: corte, espectro E(k), PDF de componentes y de |B|

Convención de coordenadas: la de `model.construir_escenario`. Los arreglos
se indexan desde la esquina de la caja; el centro galáctico (GC) está en el
centro geométrico. Aquí todo se reexpresa con el GC en (0, 0, 0) y el Sol
en (-r0, 0, 0). Longitudes en kpc, campos en microgauss, n_e en cm^-3.
"""

from __future__ import annotations

import os

import numpy as np

from .analysis.spatial_stats import isotropic_energy_spectrum

# El modelo de densidad extiende los brazos hasta R -> 0, donde una
# espiral logarítmica da vueltas sin fin. Las curvas superpuestas se
# dibujan desde este radio para que no se apilen en una mancha ilegible.
R_MIN_CURVA_BRAZO_KPC = 2.0


def _eje_centrado_en_gc(dx, box_size):
    n = int(round(box_size / dx))
    return np.arange(n) * dx - box_size / 2.0


def _indice_mas_cercano(eje, valor):
    return int(np.argmin(np.abs(eje - valor)))


def _radio_cilindrico(eje):
    xx, yy = np.meshgrid(eje, eje, indexing="ij")
    return np.hypot(xx, yy)


def _curva_brazo_espiral(pitch_angle, r0, fase, r_min, r_max, n_puntos=400):
    """(x, y) de un brazo logarítmico: phi(r) = fase + ln(r/r0)/tan(pitch),
    la misma expresión que usa `spiral_arm_density_factor`."""
    r = np.linspace(max(r_min, 1e-3), r_max, n_puntos)
    phi = fase + np.log(r / r0) / np.tan(pitch_angle)
    return r * np.cos(phi), r * np.sin(phi)


def _marcar_gc_y_sol(ax, observer_pos_gc):
    ax.plot(0, 0, "*", color="gold", markersize=14, markeredgecolor="k",
            markeredgewidth=0.6, zorder=5, label="Centro galáctico")
    ax.plot(observer_pos_gc[0], observer_pos_gc[1], "o", color="cyan",
            markersize=7, markeredgecolor="k", markeredgewidth=0.6,
            zorder=5, label="Sol")


def _guardar(fig, ruta_destino, nombre_archivo):
    import matplotlib.pyplot as plt

    ruta = os.path.join(ruta_destino, nombre_archivo)
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------
# Disco y brazos
# --------------------------------------------------------------------------


def figura_disco_y_brazos(
    ruta_destino, ne, dx, box_size, observer_pos_gc, r0, pitch_angle, n_arms,
    phase0=0.0, ne_axisimetrico=None, arm_r_min=None, nombre_archivo="estructura_disco.png",
):
    """
    Izquierda: n_e en el plano z=0 visto desde el polo norte, con las curvas
    analíticas de los brazos encima.

    Derecha (solo si se pasa `ne_axisimetrico`, la misma densidad sin
    brazos): el cociente n_e / n_e(sin brazos), que quita la caída radial y
    deja solo el realce de los brazos. Si el realce no cae sobre las curvas,
    la fase o el pitch de la densidad y los de la curva no coinciden.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    eje = _eje_centrado_en_gc(dx, box_size)
    k0 = _indice_mas_cercano(eje, 0.0)
    corte = np.asarray(ne)[:, :, k0]
    extent = [eje[0], eje[-1], eje[0], eje[-1]]

    n_paneles = 1 if ne_axisimetrico is None else 2
    fig, axes = plt.subplots(1, n_paneles, figsize=(6.4 * n_paneles, 5.6), squeeze=False)
    axes = axes[0]

    img = axes[0].imshow(
        corte.T, origin="lower", extent=extent, cmap="inferno",
        norm=LogNorm(vmin=corte.max() / 50.0, vmax=corte.max()),
    )
    fig.colorbar(img, ax=axes[0], fraction=0.046, pad=0.03).set_label(r"$n_e$ [cm$^{-3}$]")
    axes[0].set_title(r"Densidad de electrones, plano $z=0$")

    if ne_axisimetrico is not None:
        # Fuera del corte radial de n_e (R > NE_RADIAL_CUTOFF) las dos
        # densidades son 0: se deja en blanco en vez de dividir 0/0.
        referencia = np.asarray(ne_axisimetrico)[:, :, k0]
        contraste = np.divide(
            corte, referencia, out=np.full_like(corte, np.nan, dtype=float), where=referencia > 0
        )
        img2 = axes[1].imshow(
            contraste.T, origin="lower", extent=extent, cmap="magma",
            vmin=1.0, vmax=max(2.0, float(np.nanmax(contraste))),
        )
        fig.colorbar(img2, ax=axes[1], fraction=0.046, pad=0.03).set_label(
            r"$n_e\,/\,n_e^{\rm sin\ brazos}$"
        )
        axes[1].set_title("Contraste de brazos (caída radial removida)")

    r_max = 0.95 * eje[-1]
    for ax in axes:
        for i in range(n_arms):
            fase = 2.0 * np.pi * i / n_arms + phase0
            r_inicio = R_MIN_CURVA_BRAZO_KPC if arm_r_min is None else arm_r_min
            x, y = _curva_brazo_espiral(pitch_angle, r0, fase, r_inicio, r_max)
            dentro = (np.abs(x) <= eje[-1]) & (np.abs(y) <= eje[-1])
            ax.plot(x[dentro], y[dentro], "-", color="cyan", lw=0.9, alpha=0.8)
        _marcar_gc_y_sol(ax, observer_pos_gc)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Disco de {n_arms} brazos, visto desde el polo norte galáctico", y=1.0)
    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


# --------------------------------------------------------------------------
# Perfiles radial / vertical
# --------------------------------------------------------------------------


def _media_y_rms_en_anillos(plano, radio, bordes):
    media, rms = [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        v = plano[(radio >= lo) & (radio < hi)]
        media.append(v.mean() if v.size else np.nan)
        rms.append(np.sqrt((v**2).mean()) if v.size else np.nan)
    return np.array(media), np.array(rms)


def figura_perfiles(
    ruta_destino, ne, bx_reg, by_reg, bz_reg, dx, box_size, r0,
    turbulento=None, ne_axisimetrico=None, escalas_b=None,
    nombre_archivo="estructura_perfiles.png",
):
    """
    n_e, campo regular y campo turbulento contra R (anillos en z=0) y contra
    |z| (anillo de radio r0).

    Se promedia en azimut para que el perfil no dependa de en qué dirección
    se corta respecto a los brazos. Referencias punteadas: `ne_axisimetrico`
    (la densidad sin brazos: lo que separa la curva medida de ella es el
    aporte de los brazos) y `escalas_b` = dict(b0, escala_radial_b,
    escala_altura_b, ancho_vertical_b) para el disco del campo regular. Con
    `turbulento=(bx, by, bz)` se dibuja su RMS. Es la amplitud EFECTIVA de
    la malla (B de literatura × sqrt(L_coh ISM / L_integral malla), ver
    config_fisica.py), no el campo turbulento real, que es mayor que el
    regular; los rótulos lo dicen para que la figura no se lea al revés.
    """
    import matplotlib.pyplot as plt

    eje = _eje_centrado_en_gc(dx, box_size)
    radio = _radio_cilindrico(eje)
    k0 = _indice_mas_cercano(eje, 0.0)

    ne = np.asarray(ne)
    b_reg = np.sqrt(sum(np.asarray(c) ** 2 for c in (bx_reg, by_reg, bz_reg)))
    b_turb = None if turbulento is None else np.sqrt(sum(np.asarray(c) ** 2 for c in turbulento))

    bordes = np.arange(0.0, 0.95 * eje[-1] + dx, dx)
    r_centro = 0.5 * (bordes[:-1] + bordes[1:])
    anillo_solar = np.abs(radio - r0) < dx
    z_pos = eje >= 0
    z = eje[z_pos]

    def vertical(campo):
        perfil = np.array([campo[:, :, k][anillo_solar].mean() for k in range(len(eje))])
        return perfil[z_pos] / perfil[k0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    media, _ = _media_y_rms_en_anillos(ne[:, :, k0], radio, bordes)
    ax.semilogy(r_centro, media, "o-", color="firebrick", ms=3, label="con brazos (media en azimut)")
    if ne_axisimetrico is not None:
        ref, _ = _media_y_rms_en_anillos(np.asarray(ne_axisimetrico)[:, :, k0], radio, bordes)
        ax.semilogy(r_centro, ref, "k--", label="sin brazos")
    ax.axvline(r0, color="gray", ls=":", lw=1)
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel(r"$n_e$ [cm$^{-3}$]")
    ax.set_title(r"$n_e(R)$ en $z=0$")
    ax.legend(fontsize=8)

    ax = axes[1]
    media, _ = _media_y_rms_en_anillos(b_reg[:, :, k0], radio, bordes)
    ax.semilogy(r_centro, media, "o-", color="steelblue", ms=3, label=r"$|B|$ regular (disco + halo)")
    if b_turb is not None:
        _, rms = _media_y_rms_en_anillos(b_turb[:, :, k0], radio, bordes)
        ax.semilogy(r_centro, rms, "s-", color="darkorange", ms=3, label=r"$B$ turbulento efectivo (RMS)")
    if escalas_b:
        # Misma forma que `LogarithmicSpiralField`: exponencial, constante
        # dentro de `radio_nucleo` y nula dentro de `radio_sin_campo`.
        r_env = r_centro
        if escalas_b.get("radio_nucleo") is not None:
            r_env = np.maximum(r_centro, escalas_b["radio_nucleo"])
        envolvente = escalas_b["b0"] * np.exp(-(r_env - r0) / escalas_b["escala_radial_b"])
        if escalas_b.get("radio_sin_campo") is not None:
            envolvente = np.where(r_centro < escalas_b["radio_sin_campo"], np.nan, envolvente)
        ax.semilogy(r_centro, envolvente, "k--", label="envolvente del disco")
    ax.axvline(r0, color="gray", ls=":", lw=1)
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel(r"$B$ [$\mu$G]")
    ax.set_title(r"Campo magnético vs R ($z=0$)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.semilogy(z, vertical(ne), "o-", color="firebrick", ms=3, label=r"$n_e$")
    ax.semilogy(z, vertical(b_reg), "o-", color="steelblue", ms=3, label=r"$|B|$ regular (disco + halo)")
    if b_turb is not None:
        ax.semilogy(z, vertical(b_turb**2) ** 0.5, "s-", color="darkorange", ms=3, label=r"$B$ turbulento efectivo (RMS)")
    if ne_axisimetrico is not None:
        ax.semilogy(z, vertical(np.asarray(ne_axisimetrico)), "k--", lw=1, label=r"$n_e$ sin brazos")
    if escalas_b:
        h_b, w_b = escalas_b["escala_altura_b"], escalas_b.get("ancho_vertical_b")
        z_fino = np.linspace(0.0, z.max(), 400)
        if w_b is None:
            corte, etiqueta = np.exp(-z_fino / h_b), r"disco: $e^{-|z|/h_B}$"
        else:
            def corte_disco(zz):
                return 1.0 - 1.0 / (1.0 + np.exp(-2.0 * (zz - h_b) / w_b))
            corte = corte_disco(z_fino) / corte_disco(0.0)
            etiqueta = r"disco: $1-L(z;h_B,w_B)$ (JF12)"
        ax.semilogy(z_fino, corte, "k:", lw=1, label=etiqueta)
        if escalas_b.get("nota_vertical"):
            ax.annotate(
                escalas_b["nota_vertical"], xy=(0.5, 0.2), xytext=(1.6, 0.05),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8),
            )
    ax.set_xlim(0, 8)
    ax.set_ylim(1e-4, 3)
    ax.set_xlabel(r"$|z|$ [kpc]")
    ax.set_ylabel(r"normalizado a $z=0$")
    ax.set_title(r"Perfil vertical (anillo $R=R_\odot\pm\Delta x$)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


# --------------------------------------------------------------------------
# Campo regular vs total
# --------------------------------------------------------------------------


def figura_campo_regular_vs_total(
    ruta_destino, bx_reg, by_reg, bx_total, by_total, dx, box_size,
    observer_pos_gc, nombre_archivo="estructura_campo.png",
):
    """
    Líneas de campo en z=0 con solo la componente regular (izquierda) y con
    regular + turbulenta (derecha), misma escala de color. Muestra por qué
    hace falta la componente regular: sin ella no queda una dirección
    coherente a lo largo de la línea de visión y la polarización se cancela.
    """
    import matplotlib.pyplot as plt

    eje = _eje_centrado_en_gc(dx, box_size)
    k0 = _indice_mas_cercano(eje, 0.0)
    pares = [
        ("Solo regular", np.asarray(bx_reg)[:, :, k0], np.asarray(by_reg)[:, :, k0]),
        ("Regular + turbulento", np.asarray(bx_total)[:, :, k0], np.asarray(by_total)[:, :, k0]),
    ]
    b_max = max(np.hypot(bx, by).max() for _, bx, by in pares)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, (titulo, bx, by) in zip(axes, pares):
        lineas = ax.streamplot(
            eje, eje, bx.T, by.T, color=np.hypot(bx, by).T, cmap="viridis",
            density=1.8, linewidth=1.0, arrowsize=1.0,
            norm=plt.Normalize(vmin=0, vmax=b_max),
        )
        _marcar_gc_y_sol(ax, observer_pos_gc)
        ax.set_title(titulo)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal")
        ax.set_xlim(eje[0], eje[-1])
        ax.set_ylim(eje[0], eje[-1])
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(lineas.lines, ax=axes, fraction=0.025, pad=0.02).set_label(r"$|B_{xy}|$ [$\mu$G]")
    return _guardar(fig, ruta_destino, nombre_archivo)


# --------------------------------------------------------------------------
# Turbulencia
# --------------------------------------------------------------------------


def figura_turbulencia(
    ruta_destino, bx_t, by_t, bz_t, dx, box_size, spectral_index=5.0 / 3.0,
    k_banda=None, turbulento_homogeneo=None, nombre_archivo="estructura_turbulencia.png",
):
    """
    Diagnóstico del campo turbulento, cuatro paneles:
      1. corte de |B_turb| en z=0 (textura y tamaño típico de los remolinos);
      2. espectro de energía E(k) contra la ley impuesta k^-n (Kolmogorov:
         n=5/3): la banda de turbulencia (k_banda) queda sin sombrear y
         fuera de ella el espectro es cero por construcción; se ajusta la
         pendiente dentro de la banda;
      3. PDF de las componentes contra una gaussiana (el campo es gaussiano
         por construcción; una curtosis en exceso lejos de 0 delataría un
         problema de muestreo);
      4. PDF de |B| contra la Maxwelliana de un vector gaussiano isótropo
         con el mismo B_rms.

    `k_banda`: (k_min, k_max) [1/kpc] donde hay turbulencia. Por defecto,
    todo el rango donde E(k) > 0.

    `turbulento_homogeneo`: (bx, by, bz) del mismo generador SIN envolvente
    espacial. El campo del modelo está modulado por una envolvente en R y z
    (B = ∇×(f A)), así que su histograma mezcla celdas de amplitudes muy
    distintas y no es gaussiano aunque el generador sí lo sea (curtosis
    grande, pico en |B|=0 por las celdas lejanas al disco). Si se pasa,
    los paneles 2-4 (espectro y estadística de un punto, que prueban el
    generador) usan este campo; el panel 1 sigue mostrando el del modelo.
    """
    import matplotlib.pyplot as plt

    bx_t, by_t, bz_t = (np.asarray(c) for c in (bx_t, by_t, bz_t))
    eje = _eje_centrado_en_gc(dx, box_size)
    k0 = _indice_mas_cercano(eje, 0.0)
    b_mag_modelo = np.sqrt(bx_t**2 + by_t**2 + bz_t**2)
    sufijo = ""
    if turbulento_homogeneo is not None:
        bx_t, by_t, bz_t = (np.asarray(c) for c in turbulento_homogeneo)
        sufijo = " (generador, sin envolvente)"
    b_mag = np.sqrt(bx_t**2 + by_t**2 + bz_t**2)
    b_rms = float(np.sqrt((b_mag**2).mean()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    img = ax.imshow(b_mag_modelo[:, :, k0].T, origin="lower", cmap="viridis",
                    extent=[eje[0], eje[-1], eje[0], eje[-1]])
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03).set_label(
        r"$|B_{\rm turb}|$ efectivo [$\mu$G]"
    )
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(r"Corte $z=0$ (modelo, con envolvente)")

    ax = axes[0, 1]
    k, e_k = isotropic_energy_spectrum(bx_t, by_t, bz_t, dx)
    e_max = e_k.max()
    activo = e_k > 1e-8 * e_max  # descarta cascarones vacíos (ruido de máquina)
    ax.loglog(k[activo], e_k[activo], "o-", color="k", ms=3, label="medido")
    if k_banda is None:
        k_banda = (k[activo].min(), k[activo].max())
    en_banda = activo & (k >= k_banda[0]) & (k <= k_banda[1])
    if en_banda.sum() >= 3:
        pendiente, ordenada = np.polyfit(np.log(k[en_banda]), np.log(e_k[en_banda]), 1)
        kk = np.array(k_banda)
        ax.loglog(kk, np.exp(ordenada) * kk**pendiente, "r-", lw=2,
                  label=f"ajuste en la banda: pendiente {pendiente:.2f}")
        impuesta = np.exp(ordenada) * kk[0] ** (pendiente + spectral_index) * kk ** (-spectral_index)
        ax.loglog(kk, impuesta, "b--", lw=1.5, label=rf"impuesto: $k^{{-{spectral_index:.2f}}}$")
    ax.axvspan(k.min(), k_banda[0], color="gray", alpha=0.15, label="fuera de la banda")
    ax.axvspan(k_banda[1], k.max(), color="gray", alpha=0.15)
    ax.set_xticks([0.5, 1, 2, 3])
    ax.set_xticklabels(["0.5", "1", "2", "3"])
    ax.minorticks_off()
    ax.set_xlabel(r"$k$ [kpc$^{-1}$]")
    ax.set_ylabel(r"$E(k)$ [$\mu{\rm G}^2$ kpc]")
    ax.set_title("Espectro de energía magnética" + sufijo, fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    comp = np.concatenate([bx_t.ravel(), by_t.ravel(), bz_t.ravel()])
    z = comp / comp.std()
    ax.hist(z, bins=80, density=True, color="steelblue", alpha=0.7, label=r"$B_x, B_y, B_z$")
    xs = np.linspace(-5, 5, 300)
    ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), "k-", label=r"$\mathcal{N}(0,1)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(r"$B_i/\sigma$")
    ax.set_ylabel("densidad de probabilidad")
    ax.set_title(f"Gaussianidad{sufijo}: curtosis en exceso = {(z**4).mean() - 3.0:+.2f}", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    s = b_rms / np.sqrt(3.0)
    bs = np.linspace(0, b_mag.max(), 300)
    ax.hist(b_mag.ravel(), bins=80, density=True, color="darkorange", alpha=0.7, label="medido")
    ax.plot(bs, np.sqrt(2 / np.pi) * bs**2 / s**3 * np.exp(-0.5 * (bs / s) ** 2), "k-",
            label="Maxwelliana (vector gaussiano isótropo)")
    ax.axvline(b_rms, color="gray", ls=":", label=rf"$B_{{\rm rms}}={b_rms:.2f}\,\mu$G")
    ax.set_xlabel(r"$|B_{\rm turb}|$ [$\mu$G]")
    ax.set_ylabel("densidad de probabilidad")
    ax.set_title(r"Distribución de $|B|$" + sufijo, fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Campo magnético turbulento", y=1.0)
    fig.tight_layout()
    return _guardar(fig, ruta_destino, nombre_archivo)


# --------------------------------------------------------------------------
# Orquestador
# --------------------------------------------------------------------------


def generar_graficos_estructura(
    ruta_destino, ne, bx, by, bz, dx, box_size, observer_pos,
    r0, pitch_angle, n_arms, phase0=0.0, arm_r_min=None, componentes_regular=None,
    componentes_turbulento=None, ne_axisimetrico=None, escalas_b=None,
    spectral_index=5.0 / 3.0, k_banda=None, turbulento_homogeneo=None,
):
    """
    Genera las figuras de estructura de una corrida y devuelve sus rutas.

    `observer_pos` viene en coordenadas de caja; aquí se centra en el GC.
    `componentes_regular` / `componentes_turbulento`: tuplas (bx, by, bz) de
    `model.construir_escenario(..., return_components=True)`; las figuras
    que las necesitan se omiten si faltan. El resto de opcionales solo
    agrega curvas de referencia (ver cada figura).
    """
    os.makedirs(ruta_destino, exist_ok=True)  # `run.py` grafica antes de guardar los mapas

    observer_pos_gc = np.asarray(observer_pos) - box_size / 2.0
    bx_reg, by_reg, bz_reg = componentes_regular if componentes_regular else (bx, by, bz)

    rutas = [
        figura_disco_y_brazos(
            ruta_destino, ne, dx, box_size, observer_pos_gc, r0, pitch_angle,
            n_arms, phase0=phase0, ne_axisimetrico=ne_axisimetrico, arm_r_min=arm_r_min,
        ),
        figura_perfiles(
            ruta_destino, ne, bx_reg, by_reg, bz_reg, dx, box_size, r0,
            turbulento=componentes_turbulento, ne_axisimetrico=ne_axisimetrico,
            escalas_b=escalas_b,
        ),
    ]
    if componentes_regular is not None:
        rutas.append(
            figura_campo_regular_vs_total(
                ruta_destino, bx_reg, by_reg, bx, by, dx, box_size, observer_pos_gc
            )
        )
    if componentes_turbulento is not None:
        rutas.append(
            figura_turbulencia(
                ruta_destino, *componentes_turbulento, dx, box_size,
                spectral_index=spectral_index, k_banda=k_banda,
                turbulento_homogeneo=turbulento_homogeneo,
            )
        )
    return rutas
