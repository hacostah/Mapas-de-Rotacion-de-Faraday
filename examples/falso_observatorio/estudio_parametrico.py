"""Monte Carlo reproducible de ruido, frecuencias y ventana de RM.

Los IC de Wilson son puntuales, binomiales, para tasas en una población
sintética especificada. El cruce del 10% es una estimación sobre una malla;
no existe un límite de ruido exacto o universal de un radiotelescopio.
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.run import resolve_rm_vectorized, naive_rm
from faradaymr.analysis.rm_fit import resolve_n_pi_ambiguity_batch
from examples.falso_observatorio._presentation import RESULTS, setup_style, save_figure, write_json


def wilson_interval(k, n, z=1.959963984540054):
    """IC binomial de Wilson al 95%; no degenera en cero para k=0."""
    k = np.asarray(k, dtype=float)
    if n <= 0 or np.any((k < 0) | (k > n)):
        raise ValueError("Conteos binomiales inválidos.")
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return np.maximum(0, center-half), np.minimum(1, center+half)


def threshold_crossing(sigmas, rates, target=0.1):
    """Primer cruce y su intervalo de malla, sin inventar extrapolaciones."""
    s, p = np.asarray(sigmas), np.asarray(rates)
    above = np.flatnonzero(p >= target)
    if not len(above):
        return {"status": "not_reached", "estimate": None,
                "grid_bracket": None, "tested_max": float(s[-1])}
    j = int(above[0])
    if j == 0:
        return {"status": "already_exceeded", "estimate": None,
                "grid_bracket": [float(s[0]), float(s[0])]}
    estimate = s[j-1] + (target-p[j-1])*(s[j]-s[j-1])/(p[j]-p[j-1])
    return {"status": "crossed", "estimate": float(estimate),
            "grid_bracket": [float(s[j-1]), float(s[j])]}


def estres_estadistico(sigmas, configuraciones, n_muestra=10000, n_repeticiones=5,
                      seed=42, rm_min=cfg.RM_ESTUDIO_MIN, rm_max=cfg.RM_ESTUDIO_MAX,
                      truth_min=cfg.RM_VERDAD_MIN, truth_max=cfg.RM_VERDAD_MAX,
                      threshold_rm=cfg.UMBRAL_ERROR_RM, frecuencias_ghz=None,
                      progress=False):
    """Compara subconjuntos de frecuencias con la misma población y ruido.

    `fallos`: error > threshold_rm O solución no identificable (empate LS).
    `error_fraction`: error > threshold_rm para el estimador de menor |RM|.
    Dos canales pueden ajustar muchas ramas exactamente: no identificable
    NO significa que el 100% de sus estimaciones puntuales estén equivocadas.
    Guarda conteos, IC de Wilson, RMSE/MAE y variabilidad entre repeticiones.
    """
    s = np.asarray(sigmas, dtype=float)
    if s.ndim != 1 or not len(s) or np.any(~np.isfinite(s)) or np.any(s < 0) or np.any(np.diff(s) <= 0):
        raise ValueError("sigmas debe ser finito, no negativo y estrictamente creciente.")
    if n_muestra < 1 or n_repeticiones < 1 or int(n_muestra) != n_muestra or int(n_repeticiones) != n_repeticiones:
        raise ValueError("El tamaño y repeticiones deben ser enteros positivos.")
    if not (rm_min <= truth_min < truth_max <= rm_max) or threshold_rm <= 0:
        raise ValueError("La distribución verdadera debe estar contenida en la ventana y el umbral ser positivo.")
    freq = np.asarray(cfg.FRECUENCIAS_GHZ if frecuencias_ghz is None else frecuencias_ghz, dtype=float)
    if freq.ndim != 1 or np.any(freq <= 0) or np.any(~np.isfinite(freq)) or len(np.unique(freq)) != len(freq):
        raise ValueError("Frecuencias positivas, finitas y distintas son obligatorias.")
    indices = {name: np.asarray(idx, dtype=int) for name, idx in configuraciones.items()}
    for idx in indices.values():
        if idx.ndim != 1 or len(idx) < 2 or len(np.unique(idx)) != len(idx) or np.any((idx < 0) | (idx >= len(freq))):
            raise ValueError("Cada configuración necesita al menos dos índices distintos válidos.")
    waves = cfg.VELOCIDAD_LUZ/(freq*1e9)
    n = int(n_muestra*n_repeticiones)
    accum = {name: {"fail": np.zeros(len(s), dtype=int), "err": np.zeros(len(s), dtype=int),
             "amb": np.zeros(len(s), dtype=int), "boundary": np.zeros(len(s), dtype=int),
             "sq": np.zeros(len(s)), "abs": np.zeros(len(s)), "sum": np.zeros(len(s)),
             "rates_rep": [], "naive": np.zeros(len(s), dtype=int)} for name in indices}
    for rep, seed_seq in enumerate(np.random.SeedSequence(seed).spawn(n_repeticiones)):
        rng = np.random.default_rng(seed_seq)
        truth = rng.uniform(truth_min, truth_max, n_muestra)
        psi0 = rng.uniform(-np.pi/2, np.pi/2, n_muestra)
        phase = psi0[None, :] + waves[:, None]**2 * truth[None, :]
        qi, ui = np.cos(2*phase), np.sin(2*phase)
        rep_counts = {name: [] for name in indices}
        for j, sigma in enumerate(s):
            # Ruido nuevo en cada sigma y repetición; común entre configuraciones.
            psi = .5*np.arctan2(ui+rng.normal(0, sigma, ui.shape),
                                qi+rng.normal(0, sigma, qi.shape))
            for name, idx in indices.items():
                fit = resolve_n_pi_ambiguity_batch(waves[idx], psi[idx], rm_min, rm_max,
                                                   return_diagnostics=True)
                error = fit["rm"]-truth
                wrong = np.abs(error) > threshold_rm
                ambiguous = ~fit["unique"]
                fail = wrong | ambiguous
                a = accum[name]
                a["fail"][j] += int(fail.sum())
                a["err"][j] += int(wrong.sum())
                a["amb"][j] += int(ambiguous.sum())
                a["boundary"][j] += int(fit["at_boundary"].sum())
                a["sq"][j] += np.sum(error**2)
                a["abs"][j] += np.sum(np.abs(error))
                a["sum"][j] += np.sum(error)
                # El unwrap secuencial requiere canales ordenados en lambda².
                order = idx[np.argsort(waves[idx]**2)]
                naive = naive_rm(psi[order], waves[order])
                a["naive"][j] += int((np.abs(naive-truth) > threshold_rm).sum())
                rep_counts[name].append(float(fail.mean()))
        for name in indices:
            accum[name]["rates_rep"].append(rep_counts[name])
        if progress:
            print(f"Repetición {rep+1}/{n_repeticiones}: {n_muestra} líneas de visión independientes", flush=True)
    result = {}
    for name, idx in indices.items():
        a = accum[name]
        lower, upper = wilson_interval(a["fail"], n)
        err_lo, err_hi = wilson_interval(a["err"], n)
        rates = a["fail"]/n
        result[name] = {"frequencies_ghz": freq[idx].tolist(), "sigmas": s.tolist(),
            "n_trials_per_sigma": n, "failure_counts": a["fail"].tolist(), "fallos": rates.tolist(),
            "ci95_low": lower.tolist(), "ci95_high": upper.tolist(),
            "error_counts": a["err"].tolist(), "error_fraction": (a["err"]/n).tolist(),
            "error_ci95_low": err_lo.tolist(), "error_ci95_high": err_hi.tolist(),
            "nonidentifiable_fraction": (a["amb"]/n).tolist(),
            "boundary_fraction": (a["boundary"]/n).tolist(),
            "rmse": np.sqrt(a["sq"]/n).tolist(), "mae": (a["abs"]/n).tolist(),
            "bias": (a["sum"]/n).tolist(), "rates_per_repetition": a["rates_rep"],
            "naive_error_fraction": (a["naive"]/n).tolist(),
            "threshold_10pct": threshold_crossing(s, rates)}
    result["_meta"] = {"seed": seed, "n_muestra": n_muestra, "n_repeticiones": n_repeticiones,
        "rm_min": rm_min, "rm_max": rm_max, "truth_min": truth_min, "truth_max": truth_max,
        "threshold_rm": threshold_rm, "polarized_intensity": 1.0,
        "noise_model": "independent Gaussian Q,U, same sigma all channels; sigma/P",
        "failure_definition": "abs(RM_est-RM_true)>threshold_rm OR tied LS branches",
        "confidence": "pointwise Wilson 95% for synthetic independent lines of sight",
        "solver": "complete integer-branch angular least squares, continuous bounded RM",
        "frequencies_all_ghz": freq.tolist()}
    return result


def save_results(result, dest, stem="estudio_parametrico"):
    write_json(dest / f"{stem}.json", result)
    fields = ["configuration", "sigma_over_p", "n_trials", "failures", "failure_fraction", "ci95_low", "ci95_high",
              "error_fraction", "nonidentifiable_fraction", "rmse", "mae", "bias", "boundary_fraction", "naive_error_fraction"]
    with (dest / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, data in result.items():
            if name.startswith("_"):
                continue
            for j, sigma in enumerate(data["sigmas"]):
                row = {"configuration": name, "sigma_over_p": sigma, "n_trials": data["n_trials_per_sigma"],
                       "failures": data["failure_counts"][j], "failure_fraction": data["fallos"][j]}
                row.update({key: data[key][j] for key in fields[5:]})
                writer.writerow(row)


def refinar_umbrales(coarse, dest=RESULTS):
    """Confirma el cruce con población y ruido nuevos en una malla 10x más fina.

    La selección del intervalo usa el barrido anterior; las tasas e IC se
    recalculan con nuevas semillas. El intervalo de malla no es un IC del umbral.
    """
    from pathlib import Path
    dest = Path(dest)
    meta = coarse["_meta"]
    fine = {}
    seeds = {}
    for i, (name, data) in enumerate(coarse.items()):
        if name.startswith("_") or data["threshold_10pct"]["status"] != "crossed":
            continue
        lower, upper = data["threshold_10pct"]["grid_bracket"]
        seed = int(meta["seed"])+1000+i
        frequencies = data["frequencies_ghz"]
        trial = estres_estadistico(np.linspace(lower, upper, 11),
                    {name: list(range(len(frequencies)))},
                    meta["n_muestra"], meta["n_repeticiones"], seed=seed,
                    rm_min=meta["rm_min"], rm_max=meta["rm_max"],
                    truth_min=meta["truth_min"], truth_max=meta["truth_max"],
                    threshold_rm=meta["threshold_rm"], frecuencias_ghz=frequencies)
        fine[name] = trial[name]
        fine[name]["coarse_grid_bracket"] = [lower, upper]
        seeds[name] = seed
        print("Umbral refinado", name, fine[name]["threshold_10pct"], flush=True)
    fine["_meta"] = dict(meta, seeds_by_configuration=seeds,
                         sampling="new independent population and noise for refinement")
    save_results(fine, dest, "umbrales_refinados")
    return fine


def main(n_muestra=10000, n_repeticiones=5, output_dir=None):
    from pathlib import Path
    dest = RESULTS if output_dir is None else Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    setup_style()
    freq = np.array([5.0, 2.0, 1.4, 3.0])
    configurations = {"3F original": [0, 1, 2], "3F alternativa": [0, 3, 2],
                      "4F con 3 GHz": [0, 3, 1, 2],
                      "2F 5 y 2 GHz": [0, 1], "2F 5 y 1.4 GHz": [0, 2],
                      "2F 2 y 1.4 GHz": [1, 2]}
    sigmas = np.r_[np.arange(0, .1501, .005), [.2, .25, .3, .4]]
    result = estres_estadistico(sigmas, configurations, n_muestra, n_repeticiones,
                              frecuencias_ghz=freq, progress=True)
    save_results(result, dest)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    colors = ["#2563eb", "#ea580c", "#0f766e"]
    for (name, data), color in zip(list(result.items())[:3], colors):
        s = np.asarray(data["sigmas"])
        p = np.asarray(data["fallos"])
        for ax in axes:
            ax.plot(s, p, "o-", ms=3, lw=1.7, color=color, label=name+": "+", ".join(f"{v:g}" for v in data["frequencies_ghz"])+" GHz")
            ax.fill_between(s, data["ci95_low"], data["ci95_high"], color=color, alpha=.18)
    for ax in axes:
        ax.axhline(.1, color="#dc2626", ls=":", lw=1.4, label="Tolerancia del 10%")
        ax.set_xlabel(r"Ruido en Q/U: $\sigma/P$")
        ax.set_ylabel("Fracción de errores > 35 rad m$^{-2}$ o ramas no identificables")
        ax.grid(alpha=.2)
    axes[0].set(xlim=(0,.4), ylim=(-.02,1.06), title="Barrido de ruido y frecuencias")
    axes[0].axhline(1, color="#64748b", ls="--", label="2F: ramas no identificables (tres pares)")
    axes[0].legend(fontsize=8, loc="center right")
    axes[1].set(xlim=(0,.06), ylim=(-.005,.4), title="Detalle del comienzo de los errores (IC 95%)")
    fig.suptitle(r"RM verdadera uniforme en $[-1000,1000]$; búsqueda en $[-1500,1500]$ rad m$^{-2}$"+f"\n{n_muestra*n_repeticiones:,} líneas de visión por punto; P=1")
    save_figure(fig, "limite_ruido_prior_astrofisico", dest)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for (name, data), color in zip(list(result.items())[:3], colors):
        axes[0].plot(sigmas, data["rmse"], "o-", color=color, ms=3, label=name)
    base = result["3F original"]
    axes[1].plot(sigmas, base["naive_error_fraction"], "--", color="#dc2626", label="Unwrap secuencial + LS")
    axes[1].plot(sigmas, base["error_fraction"], "o-", color="#2563eb", ms=3, label="Ramas nπ + LS")
    axes[0].set(xlabel=r"$\sigma/P$", ylabel=r"RMSE [rad m$^{-2}$]", title="Magnitud del error, incluidos aliases")
    axes[1].set(xlabel=r"$\sigma/P$", ylabel="Fracción con |error RM| > 35", title="Comparación de métodos: 1.4, 2.0 y 5.0 GHz", ylim=(-.03,1.03))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    save_figure(fig, "errores_y_comparacion_metodos", dest)
    refinar_umbrales(result, dest)
    print("Umbrales:", {name: d["threshold_10pct"] for name,d in result.items() if not name.startswith("_")}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-muestra", type=int, default=10000)
    parser.add_argument("--n-repeticiones", type=int, default=5)
    args = parser.parse_args()
    main(args.n_muestra, args.n_repeticiones)
