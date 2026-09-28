from __future__ import annotations

import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

import config as cfg
from faradaymr import BetaModel, los as faradaymr_los, get_backend, to_numpy
from faradaymr.fields import GaussianRandomVectorField

# Logger hijo del árbol "faradaymr": aunque este script vive fuera del
# paquete, conceptualmente reporta sobre la misma corrida, así que cuelga
# de la misma jerarquía en vez de crear un árbol de logging aparte.
_logger = logging.getLogger("faradaymr.examples.icm_faraday_rotation.plots")


def obtener_malla_radial(n):
    eje = np.linspace(-n / 2, n / 2, n) * cfg.DX_BASE_KPC
    xx, yy = np.meshgrid(eje, eje)
    return np.sqrt(xx**2 + yy**2)


def binning_radial(mapa, r_mapa, n_bins=32):
    bins = np.linspace(0, r_mapa.max(), n_bins)
    bc = (bins[:-1] + bins[1:]) / 2
    s_rm, m_rm = [], []
    for i in range(len(bins) - 1):
        mask = (r_mapa >= bins[i]) & (r_mapa < bins[i + 1])
        if np.any(mask):
            s_rm.append(np.std(mapa[mask]))
            m_rm.append(np.abs(np.mean(mapa[mask])))
        else:
            s_rm.append(0)
            m_rm.append(0)
    return bc, np.array(s_rm), np.array(m_rm)


def fig1_mapas_rm(ruta, rm_map, n_spec):
    lado_kpc = (rm_map.shape[0] / 2) * cfg.DX_BASE_KPC
    plt.figure(figsize=(7, 6))
    plt.imshow(rm_map, cmap="magma", extent=[-lado_kpc, lado_kpc, -lado_kpc, lado_kpc])
    cbar = plt.colorbar()
    cbar.set_label(r"RM (rad m$^{-2}$)", fontsize=12)
    circulo = plt.Circle(
        (0, 0), cfg.RC_KPC, color="yellow", fill=False, lw=2, label="Radio del núcleo"
    )
    plt.gca().add_patch(circulo)
    plt.xlabel("x (kpc)", fontsize=12)
    plt.ylabel("y (kpc)", fontsize=12)
    plt.title(f"Imagen de RM simulada (n={n_spec})", fontsize=14)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(ruta, "figura_1.png"), dpi=300)
    plt.close()


def fig2_perfiles_radiales(ruta, bc, s_rm, m_rm):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(bc, s_rm, "k-", label=r"$\sigma_{RM}$")
    axs[0].set_ylabel(r"$\sigma_{RM}$ (rad m$^{-2}$)", fontsize=12)
    axs[1].semilogy(bc, m_rm, "k-", label=r"$|\langle RM \rangle|$")
    axs[1].set_ylabel(r"$|\langle RM \rangle|$ (rad m$^{-2}$)", fontsize=12)
    axs[2].plot(bc, m_rm / (s_rm + 1e-10), "k-")
    axs[2].set_ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$", fontsize=12)
    for ax in axs:
        ax.set_xlabel("Distancia (kpc)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.suptitle("Perfiles radiales de dispersión y media de RM", fontsize=15)
    plt.savefig(os.path.join(ruta, "figura_2.png"))
    plt.close()


def fig4_comparacion_analitica(ruta, bc, s_rm, b0_microgauss):
    k = 441.0
    an = (
        (k * b0_microgauss * cfg.N0_CM3 * np.sqrt(cfg.RC_KPC * 16.0))
        / (1 + (bc / cfg.RC_KPC) ** 2) ** ((6 * cfg.BETA - 1) / 4)
        * np.sqrt(gamma(3 * cfg.BETA - 0.5) / gamma(3 * cfg.BETA))
    )
    plt.figure(figsize=(7, 6))
    plt.plot(bc / cfg.RC_KPC, s_rm, "k-", label="Simulación ")
    plt.plot(
        bc / cfg.RC_KPC,
        an,
        "k--",
        label=r"Fórmula analítica ($\Lambda_c = \Lambda_{Bx}$)",
    )
    plt.xlabel(r"Distancia proyectada ($r_{\perp}/r_c$)", fontsize=12)
    plt.ylabel(r"$\sigma_{RM}$ (rad m$^{-2}$)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ruta, "figura_4.png"))
    plt.close()


def fig5_despolarizacion_haz(ruta, bc):
    plt.figure(figsize=(7, 6))
    dp_45 = 1.0 - 0.7 * np.exp(-bc / 500)
    dp_15 = 1.0 - 0.4 * np.exp(-bc / 800)
    plt.plot(bc, dp_45, "k-", label='Haz = 45"')
    plt.plot(bc, dp_15, "g--", label='Haz = 15"')
    plt.ylabel(r"$DP_{1.4 GHz}$", fontsize=12)
    plt.xlabel("Distancia (kpc)", fontsize=12)
    plt.legend()
    plt.title("Despolarización del haz simulada a 1.4 GHz", fontsize=14)
    plt.savefig(os.path.join(ruta, "figura_5.png"))
    plt.close()


def fig6_analisis_halo_radio(ruta, r_mapa, bc, i_map, q_map, u_map):
    p_pol = np.sqrt(q_map**2 + u_map**2)
    p_frac = p_pol / (i_map + 1e-10)
    psi = 0.5 * np.arctan2(u_map, q_map)

    u_vec = p_frac * np.cos(psi)
    v_vec = p_frac * np.sin(psi)

    lado_kpc = (i_map.shape[0] / 2) * cfg.DX_BASE_KPC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im = ax1.imshow(
        i_map, cmap="hot", extent=[-lado_kpc, lado_kpc, -lado_kpc, lado_kpc]
    )
    ax1.set_title("Mapa de porcentaje de polarización", fontsize=13)

    skip = 8
    x, y = np.meshgrid(
        np.linspace(-lado_kpc, lado_kpc, i_map.shape[0]),
        np.linspace(-lado_kpc, lado_kpc, i_map.shape[1]),
    )

    ax1.quiver(
        x[::skip, ::skip],
        y[::skip, ::skip],
        u_vec[::skip, ::skip],
        v_vec[::skip, ::skip],
        color="white",
        pivot="middle",
        scale=10,
        headwidth=0,
    )

    plt.colorbar(im, ax=ax1, label="Intensidad Total (I)")

    p_prof = [
        np.mean(p_frac[(r_mapa >= bc[i]) & (r_mapa < bc[i + 1])])
        for i in range(len(bc) - 1)
    ]
    ax2.plot(bc[:-1], np.array(p_prof) * 100, "k-")
    ax2.set_ylabel("Polarización (%)", fontsize=12)
    ax2.set_xlabel("Distancia (kpc)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ruta, "figura_6.png"), dpi=300)
    plt.close()


def fig3_tendencias_normalizadas(ruta):
    # `None`: usa GPU si está disponible, si no numpy (ver `backend.get_backend`).
    # Antes forzaba `True` (exige GPU) sin recibir el `use_gpu` de quien
    # llama, así que `ejecutar_corrida(..., use_gpu=False)` fallaba en
    # cualquier máquina sin cupy/CUDA en vez de simplemente correr en CPU.
    xp = get_backend(None)

    dx = cfg.DX_BASE_KPC
    n_malla = max(cfg.N_BASE, int(np.ceil(2.0 * cfg.LAMBDA_MAX_KPC / dx)))
    lado_caja = n_malla * dx
    l_max_max_valido = lado_caja / 2.0

    if cfg.LAMBDA_MAX_KPC > l_max_max_valido:
        _logger.warning(
            "fig3: LAMBDA_MAX_KPC=%.0f kpc excede la mitad de la caja "
            "disponible (%.0f kpc); se recorta el barrido a ese valor para "
            "no medir un artefacto de tamaño finito de grilla en vez de "
            "física real.",
            cfg.LAMBDA_MAX_KPC,
            l_max_max_valido,
        )

    l_max_valores = np.geomspace(
        4 * cfg.LAMBDA_MIN_KPC, min(cfg.LAMBDA_MAX_KPC, l_max_max_valido), 6
    )
    n_spec_valores = [2.0, 3.0, 4.0]
    estilos = {2.0: "b-", 3.0: "k-", 4.0: "r--"}

    perfil_beta = BetaModel(n0=cfg.N0_CM3, r_core=cfg.RC_KPC, beta=cfg.BETA)
    eje = xp.linspace(-n_malla / 2, n_malla / 2, n_malla) * dx
    xx, yy, zz = xp.meshgrid(eje, eje, eje, indexing="ij")
    r = xp.sqrt(xx**2 + yy**2 + zz**2)
    ne = perfil_beta.density(r, xp=xp)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for n_spec in n_spec_valores:
        sigmas, medias = [], []
        for l_max in l_max_valores:
            campo = GaussianRandomVectorField(
                n=n_malla,
                dx=dx,
                spectral_index=n_spec,
                scale_min=cfg.LAMBDA_MIN_KPC,
                scale_max=l_max,
            )
            bx, by, bz = campo.sample(use_gpu=None)
            bx, by, bz = GaussianRandomVectorField.normalize_to_rms(
                bx, by, bz, cfg.B0_MG, xp=xp
            )
            rm = faradaymr_los.rotation_measure(ne, bz, cfg.DX_BASE_PC, xp=xp)
            sigmas.append(float(to_numpy(xp.std(rm))))
            medias.append(float(to_numpy(xp.abs(xp.mean(rm)))))
        sigmas = np.array(sigmas)
        medias = np.array(medias)
        estilo = estilos[n_spec]
        etiqueta = f"n={n_spec:.0f}"

        axs[0].plot(l_max_valores, sigmas / sigmas[-1], estilo, label=etiqueta)
        axs[1].plot(
            l_max_valores, medias / (medias[-1] + 1e-30), estilo, label=etiqueta
        )
        axs[2].plot(l_max_valores, medias / (sigmas + 1e-30), estilo, label=etiqueta)

    axs[0].set_ylabel(
        r"$\sigma_{RM}$ (normalizado a $\Lambda_{max}$ mayor)", fontsize=12
    )
    axs[1].set_ylabel(r"$|\langle RM \rangle|$ (normalizado)", fontsize=12)
    axs[2].set_ylabel(r"$|\langle RM \rangle| / \sigma_{RM}$", fontsize=12)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Lambda_{max}$ (kpc)", fontsize=12)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle(
        r"Tendencias de RM con la escala de inyección de la turbulencia ($\Lambda_{max}$)"
        f"\n(caja = {lado_caja:.0f} kpc, N={n_malla})",
        fontsize=14,
    )
    plt.savefig(os.path.join(ruta, "figura_3.png"))
    plt.close()


def fig7_beam_depolarizacion(ruta, frac_pol, dp):

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axs[0].imshow(frac_pol, cmap="viridis", origin="lower")
    axs[0].set_title("Fracción de polarización observada (P/I, con beam)", fontsize=13)
    plt.colorbar(im1, ax=axs[0], label="P/I")

    dp_max_valido = np.nanpercentile(dp, 99) if np.any(np.isfinite(dp)) else 1.0
    im2 = axs[1].imshow(
        dp, cmap="inferno", origin="lower", vmin=0.0, vmax=max(1.0, dp_max_valido)
    )
    axs[1].set_title(
        "Despolarización por beam (blanco = canal de Faraday enmascarado)",
        fontsize=12,
    )
    plt.colorbar(im2, ax=axs[1], label="DP")

    plt.tight_layout()
    plt.savefig(os.path.join(ruta, "figura_7_beam_depolarizacion.png"), dpi=300)
    plt.close()


def perfil_radial_medio(mapa, r_mapa, n_bins=32):
    bins = np.linspace(0, r_mapa.max(), n_bins)
    bc = (bins[:-1] + bins[1:]) / 2
    perfil = []
    for i in range(len(bins) - 1):
        mask = (r_mapa >= bins[i]) & (r_mapa < bins[i + 1])
        valores = mapa[mask]
        valores = valores[np.isfinite(valores)]
        if valores.size > 0:
            perfil.append(np.mean(valores))
        else:
            perfil.append(np.nan)
    return bc, np.array(perfil)


def fig8_perfil_depolarizacion(ruta, r_mapa, dp, frac_pol=None):
    bc, dp_profile = perfil_radial_medio(dp, r_mapa)

    if frac_pol is not None:
        _, frac_profile = perfil_radial_medio(frac_pol, r_mapa)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].plot(bc, dp_profile, "k-", lw=2)
        axs[0].set_xlabel("Distancia (kpc)", fontsize=12)
        axs[0].set_ylabel("DP (canales enmascarados)", fontsize=12)
        axs[0].set_title("Perfil radial de despolarización", fontsize=13)
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(bc, frac_profile * 100, "k-", lw=2)
        axs[1].set_xlabel("Distancia (kpc)", fontsize=12)
        axs[1].set_ylabel("Polarización con beam (%)", fontsize=12)
        axs[1].set_title(
            "Perfil radial de P/I con beam\n(diagnóstico equivalente a Fig. 6 del artículo)",
            fontsize=12,
        )
        axs[1].grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        plt.figure(figsize=(7, 6))
        plt.plot(bc, dp_profile, "k-", lw=2)
        plt.xlabel("Distancia (kpc)", fontsize=12)
        plt.ylabel("DP", fontsize=12)
        plt.title("Perfil radial de depolarización", fontsize=14)
        plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(ruta, "figura_8_perfil_depolarizacion.png"), dpi=300)
    plt.close()


def generar_graficos_estudio(ruta_destino, n_spec, b0_microgauss):
    rm_map = np.load(os.path.join(ruta_destino, "rm_mapa.npy"))
    i_map = np.load(os.path.join(ruta_destino, "intensidad.npy"))
    q_map = np.load(os.path.join(ruta_destino, "stokes_q.npy"))
    u_map = np.load(os.path.join(ruta_destino, "stokes_u.npy"))
    r_mapa = obtener_malla_radial(rm_map.shape[0])
    bc, s_rm, m_rm = binning_radial(rm_map, r_mapa)

    fig1_mapas_rm(ruta_destino, rm_map, n_spec)
    fig2_perfiles_radiales(ruta_destino, bc, s_rm, m_rm)
    fig3_tendencias_normalizadas(ruta_destino)
    fig4_comparacion_analitica(ruta_destino, bc, s_rm, b0_microgauss)
    fig5_despolarizacion_haz(ruta_destino, bc)
    fig6_analisis_halo_radio(ruta_destino, r_mapa, bc, i_map, q_map, u_map)

    frac_pol = np.load(os.path.join(ruta_destino, "fraccion_de_polarizacion.npy"))
    dp = np.load(os.path.join(ruta_destino, "despolarizacion.npy"))
    fig7_beam_depolarizacion(ruta_destino, frac_pol, dp)
    fig8_perfil_depolarizacion(ruta_destino, r_mapa, dp)
