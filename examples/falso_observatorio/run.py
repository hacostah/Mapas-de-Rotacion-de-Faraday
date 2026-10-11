"""Mapa analítico: observación Q/U, desenredado n-pi y diagnóstico del ruido."""
import numpy as np
import matplotlib.pyplot as plt
from faradaymr.instrument.noise import add_gaussian_noise
from faradaymr.analysis.rm_fit import resolve_n_pi_ambiguity_batch, unwrap_polarization_angle
import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.model import construir_mapa_rm_analitico, observar_stokes_qu
from examples.falso_observatorio._presentation import RESULTS, setup_style, save_figure, write_json


def resolve_rm_vectorized(wavelengths_m, psi_obs_rad, rm_candidatos):
    """Compatibilidad: los extremos de candidatos definen la ventana continua LS.

    Ya no discretiza RM; comparte el mismo ajuste exacto por ramas de la biblioteca.
    """
    grid = np.asarray(rm_candidatos, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("rm_candidatos debe tener al menos dos valores finitos.")
    return resolve_n_pi_ambiguity_batch(wavelengths_m, psi_obs_rad,
                                      float(grid.min()), float(grid.max()))


def naive_rm(psi, wavelengths=cfg.LONGITUDES_ONDA_M):
    x = np.asarray(wavelengths)**2
    xc = x - x.mean()
    y = unwrap_polarization_angle(psi, axis=0)
    return np.sum(xc[:, None] * y, axis=0) / np.sum(xc**2)


def main(output_dir=None):
    dest = RESULTS if output_dir is None else output_dir
    from pathlib import Path
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    setup_style()
    truth = np.asarray(construir_mapa_rm_analitico(use_gpu=False))
    q, u, _ = observar_stokes_qu(truth, use_gpu=False)
    rng = np.random.default_rng(cfg.SEMILLA)
    sigmas = [0.0, 0.005, 0.03, 0.08]
    fig, axes = plt.subplots(4, 4, figsize=(15, 13), layout="constrained")
    lo, hi = float(truth.min()), float(truth.max())
    summaries = []
    arrays = {"rm_true": truth, "frequencies_ghz": cfg.FRECUENCIAS_GHZ}
    for row, sigma in enumerate(sigmas):
        qn = add_gaussian_noise(q, sigma, rng=rng)
        un = add_gaussian_noise(u, sigma, rng=rng)
        psi = 0.5 * np.arctan2(un, qn)
        fit = resolve_n_pi_ambiguity_batch(cfg.LONGITUDES_ONDA_M,
                 psi.reshape(3, -1), cfg.RM_ESTUDIO_MIN, cfg.RM_ESTUDIO_MAX,
                 return_diagnostics=True)
        rm = fit["rm"].reshape(truth.shape)
        error = rm - truth
        failures = np.abs(error) > cfg.UMBRAL_ERROR_RM
        rate = float(failures.mean())
        summaries.append({"sigma_qu_over_p": sigma, "n_pixels": truth.size,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))), "max_error": float(np.max(np.abs(error))),
            "failure_fraction": rate, "failure_count": int(failures.sum()),
            "boundary_fraction": float(fit["at_boundary"].mean())})
        arrays[f"rm_sigma_{sigma:g}"] = rm
        arrays[f"q_sigma_{sigma:g}"] = qn
        arrays[f"u_sigma_{sigma:g}"] = un
        arrays[f"n_pi_sigma_{sigma:g}"] = fit["n_pi"].reshape(q.shape)
        # Mapa reconstruido en escala completa: no ocultar aliases por saturación.
        im0 = axes[row, 0].imshow(truth, origin="lower", cmap="inferno", vmin=lo, vmax=hi)
        im1 = axes[row, 1].imshow(rm, origin="lower", cmap="inferno",
                                vmin=cfg.RM_ESTUDIO_MIN, vmax=cfg.RM_ESTUDIO_MAX)
        im2 = axes[row, 2].imshow(error, origin="lower", cmap="inferno", vmin=-2800, vmax=2800)
        im3 = axes[row, 3].imshow(failures, origin="lower", cmap="Greys", vmin=0, vmax=1)
        for col, im in enumerate([im0, im1, im2]):
            fig.colorbar(im, ax=axes[row, col], fraction=0.045, pad=0.02,
                         label=r"rad m$^{-2}$")
        axes[row, 0].set_ylabel(rf"$\sigma/P={sigma:g}$" + "\npíxel y")
        axes[row, 3].set_title(f"Errores > 35: {100*rate:.2f}%")
        for ax in axes[row]:
            ax.set_xlabel("píxel x")
            ax.set_xticks([0, 64, 127])
            ax.set_yticks([0, 64, 127])
        if row == 0:
            for ax, title in zip(axes[row, :3], ["RM verdadera", "RM reconstruida", "Error firmado"]):
                ax.set_title(title)
    fig.suptitle("Reconstrucción píxel a píxel en tres frecuencias (1.4, 2.0, 5.0 GHz)\n"
                 r"Ventana de búsqueda: $[-1500,1500]$ rad m$^{-2}$; intensidad $P=1$", fontsize=15)
    save_figure(fig, "mapas_reconstruccion_rm_ruido", dest)

    # Figura de precisión para ruido bajo con escalas compartidas y error completo.
    rm_low = arrays["rm_sigma_0.005"]
    err_low = rm_low - truth
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for a, m, title in zip(ax[:2], [truth, rm_low], ["RM verdadera", "RM recuperada"]):
        im = a.imshow(m, origin="lower", cmap="inferno", vmin=lo, vmax=hi)
        a.set_title(title)
        fig.colorbar(im, ax=a, label=r"rad m$^{-2}$", fraction=0.045)
    lim = float(np.max(np.abs(err_low)))
    im = ax[2].imshow(err_low, origin="lower", cmap="inferno", vmin=-lim, vmax=lim)
    ax[2].set_title("Error firmado (escala completa)")
    fig.colorbar(im, ax=ax[2], label=r"rad m$^{-2}$", fraction=0.045)
    for a in ax:
        a.set_xlabel("píxel x")
        a.set_ylabel("píxel y")
    fig.suptitle(r"Precisión con $\sigma/P=0.005$ y ventana amplia de RM")
    save_figure(fig, "mapa_precision_ruido_bajo", dest)
    np.savez_compressed(dest / "mapas_datos.npz", **arrays)
    write_json(dest / "mapas_metricas.json", {"search_window": [cfg.RM_ESTUDIO_MIN, cfg.RM_ESTUDIO_MAX],
          "seed": cfg.SEMILLA, "threshold_rm": cfg.UMBRAL_ERROR_RM,
          "rm_true_min": lo, "rm_true_max": hi, "results": summaries})
    print("Mapas:", summaries, flush=True)
    return summaries


if __name__ == "__main__":
    main()
