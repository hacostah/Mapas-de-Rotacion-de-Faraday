"""Paisaje de costo obtenido de las mismas ramas LS que explora el solver."""
import numpy as np
import matplotlib.pyplot as plt
import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.model import construir_mapa_rm_analitico, observar_stokes_qu
from faradaymr.analysis.rm_fit import iter_rm_branch_fits
from examples.falso_observatorio._presentation import setup_style, save_figure


def main(output_dir=None):
    setup_style()
    truth = construir_mapa_rm_analitico(use_gpu=False)
    _, _, psi_map = observar_stokes_qu(truth, use_gpu=False)
    i = cfg.N_PIXELES//2
    psi = psi_map[:, i, i, None]
    rows = [(float(rm[0]), float(cost[0])) for rm, _, cost, _ in iter_rm_branch_fits(
                cfg.LONGITUDES_ONDA_M, psi, cfg.RM_ESTUDIO_MIN, cfg.RM_ESTUDIO_MAX)]
    rows.sort()
    rows = np.asarray(rows)
    best = rows[np.argmin(rows[:, 1])]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.scatter(rows[:, 0], np.maximum(rows[:, 1], 1e-25), s=16, color="#64748b", label="Ramas LS evaluadas")
    ax.scatter(best[0], max(best[1], 1e-25), s=90, color="#2563eb", label=f"Mínimo: RM={best[0]:.2f}")
    ax.axvline(truth[i,i], color="#0f766e", ls="--", lw=1.3, label="RM verdadera (solo referencia)")
    ax.set_yscale("log")
    ax.set(xlabel=r"RM ajustada por rama [rad m$^{-2}$]", ylabel=r"Suma de residuos angulares$^2$ [rad$^2$]",
           title="Mínimos cuadrados por ramas nπ, sin ruido")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    save_figure(fig, "paisaje_minimos_cuadrados", output_dir)


if __name__ == "__main__":
    main()
