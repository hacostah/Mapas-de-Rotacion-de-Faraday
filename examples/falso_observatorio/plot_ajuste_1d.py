"""Datos observados levantados por enteros*pi, seguidos de ajuste LS."""
import numpy as np
import matplotlib.pyplot as plt
import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.model import construir_mapa_rm_analitico, observar_stokes_qu
from examples.falso_observatorio.run import resolve_rm_vectorized
from faradaymr.analysis.rm_fit import resolve_n_pi_ambiguity_batch
from examples.falso_observatorio._presentation import setup_style, save_figure


def main(output_dir=None):
    setup_style()
    truth = construir_mapa_rm_analitico(use_gpu=False)
    q, u, _ = observar_stokes_qu(truth, use_gpu=False)
    i = cfg.N_PIXELES // 2
    rng = np.random.default_rng(142)
    sigma = 0.005
    psi = 0.5*np.arctan2(u[:, i, i] + rng.normal(0, sigma, 3),
                        q[:, i, i] + rng.normal(0, sigma, 3))
    fit = resolve_n_pi_ambiguity_batch(cfg.LONGITUDES_ONDA_M, psi[:, None],
                      cfg.RM_ESTUDIO_MIN, cfg.RM_ESTUDIO_MAX, return_diagnostics=True)
    rm, psi0 = float(fit["rm"][0]), float(fit["psi0"][0])
    lifted = fit["psi_unwrapped"][:, 0]
    x = cfg.LAMBDA_CUADRADO
    xx = np.linspace(0, x.max()*1.08, 150)
    raw_rm, raw_psi0 = np.polyfit(x, psi, 1)
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.axhspan(-np.pi/2, np.pi/2, color="#94a3b8", alpha=0.13, label="Intervalo observado módulo π")
    ax.plot(xx, psi0+rm*xx, color="#2563eb", label=rf"LS con ramas: RM={rm:.2f} rad m$^{{-2}}$")
    ax.plot(xx, raw_psi0+raw_rm*xx, "--", color="#dc2626", label="LS directo sin desenredar")
    ax.scatter(x, psi, color="#dc2626", s=65, label="Ángulos observados", zorder=4)
    ax.scatter(x, lifted, color="#2563eb", marker="*", s=140,
               label=r"Observados + $n_i\pi$ (datos)", zorder=5)
    for j in range(3):
        ax.plot([x[j], x[j]], [psi[j], lifted[j]], ":", color="#334155")
        n = int(fit["n_pi"][j, 0])
        if n:
            ax.text(x[j]+0.0005, (psi[j]+lifted[j])/2, rf"${n:+d}\pi$", fontsize=11)
    ax.set(xlabel=r"$\lambda^2$ [m$^2$]", ylabel="Ángulo de polarización [rad]",
           title=f"Desenredado del píxel central; RM verdadera = {truth[i,i]:.2f}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=10)
    save_figure(fig, "ajuste_1d_salida_solver", output_dir)


if __name__ == "__main__":
    main()
