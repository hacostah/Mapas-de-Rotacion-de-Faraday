"""
Calcula el umbral teórico de detección que promete la propuesta del
proyecto: dado el sigma0(theta) que predice el modelo (barrido Monte Carlo,
ver `run_barrido_theta.py`), ¿cuántas fuentes de fondo hacen falta en un
experimento de stacking para distinguir ese exceso del ruido de RM
extragaláctico?

Uso: desde la raíz del repo, después de correr `run_barrido_theta.py`,

    python -m examples.filamento_whim.umbral_deteccion

Genera un reporte de texto en
examples/filamento_whim/results/umbral_deteccion.txt y una figura de apoyo
en examples/filamento_whim/results/plots/fig5_umbral_deteccion.png.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from faradaymr.analysis.deteccion import resumen_umbral_deteccion

BASE_DIR = os.path.dirname(__file__)
ARCHIVO_MC = os.path.join(BASE_DIR, "results", "barrido_theta", "barrido_theta_mc.npz")
DIR_PLOTS = os.path.join(BASE_DIR, "results", "plots")
ARCHIVO_REPORTE = os.path.join(BASE_DIR, "results", "umbral_deteccion.txt")

# Dispersión de RM extragaláctica de fondo a 1.4 GHz. 7 rad/m² es el valor
# típico citado por Stuardi et al. 2026 (POSSUM, filamento A3667/3651),
# consistente con Oppermann et al. 2015 y Schnitzeler et al. 2019.
SIGMA_FONDO_RAD_M2 = 7.0


def calcular_y_reportar():
    if not os.path.exists(ARCHIVO_MC):
        print(
            "Falta el archivo del barrido Monte Carlo "
            f"({ARCHIVO_MC}). Ejecuta run_barrido_theta.py primero."
        )
        return None

    datos = np.load(ARCHIVO_MC)
    theta = datos["theta_grados"]
    sigma0 = datos["sigma0_medio"]

    resumen_1s = resumen_umbral_deteccion(theta, sigma0, SIGMA_FONDO_RAD_M2, k_sigma=1.0)
    resumen_3s = resumen_umbral_deteccion(theta, sigma0, SIGMA_FONDO_RAD_M2, k_sigma=3.0)

    lineas = [
        "Umbral teorico de deteccion (stacking) -- filamento WHIM idealizado",
        "=" * 70,
        f"Fondo de dispersion de RM extragalactica asumido: {SIGMA_FONDO_RAD_M2:.1f} rad/m^2"
        " (Stuardi et al. 2026, POSSUM A3667/3651).",
        "",
        f"sigma0(theta) del modelo: {sigma0.min():.4f} - {sigma0.max():.4f} rad/m^2 "
        f"(theta = {theta.min():.0f} - {theta.max():.0f} grados).",
        "",
        "Deteccion marginal (1 sigma, N minimo para que el exceso de varianza",
        "supere el error de la varianza de control):",
        f"  N minimo = {resumen_1s['n_minimo']:.2e}  (en theta={resumen_1s['theta_n_minimo']:.0f} grados,"
        f" donde sigma0 es maximo)",
        f"  N maximo = {resumen_1s['n_maximo']:.2e}  (en theta={resumen_1s['theta_n_maximo']:.0f} grados,"
        f" donde sigma0 es minimo)",
        "",
        "Deteccion convincente (3 sigma):",
        f"  N minimo = {resumen_3s['n_minimo']:.2e}",
        f"  N maximo = {resumen_3s['n_maximo']:.2e}",
        "",
        "Mensaje: con los parametros actuales del modelo (config_fisica.py),",
        "sigma0 es 2-3 ordenes de magnitud menor que la dispersion de fondo",
        "tipica a 1.4 GHz. Un filamento INDIVIDUAL no es detectable por este",
        "metodo: hace falta stacking de un numero de fuentes del orden de",
        f"{resumen_1s['n_minimo']:.0e}-{resumen_1s['n_maximo']:.0e} incluso para una deteccion marginal,",
        "muy por encima de lo que un solo campo de POSSUM provee (decenas de",
        "fuentes). Esto es consistente con la propia conclusion de Stuardi et",
        'al. 2026: "extremely challenging... requiere una combinacion de',
        'RM-grids ultra-densos y observaciones a frecuencias mas bajas".',
        "Esta es la version cuantitativa del \"umbral teorico\" que promete",
        "la propuesta del proyecto.",
        "",
        "(Estimacion de orden de magnitud: supone ruido gaussiano y solo",
        "propaga el error estadistico de la varianza muestral; ignora",
        "sistematicos como la complejidad del foreground galactico, que el",
        "propio POSSUM identifica como su limitacion principal.)",
    ]
    reporte = "\n".join(lineas)

    os.makedirs(os.path.dirname(ARCHIVO_REPORTE), exist_ok=True)
    with open(ARCHIVO_REPORTE, "w", encoding="utf-8") as f:
        f.write(reporte + "\n")
    print(reporte)
    print(f"\nReporte guardado en: {ARCHIVO_REPORTE}")

    _figura_umbral(theta, resumen_1s["n_por_theta"], resumen_3s["n_por_theta"])

    return resumen_1s, resumen_3s


def _figura_umbral(theta, n_1sigma, n_3sigma):
    os.makedirs(DIR_PLOTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(theta, n_1sigma, "o-", color="teal", label=r"$1\sigma$ (marginal)")
    ax.plot(theta, n_3sigma, "s--", color="firebrick", label=r"$3\sigma$ (convincente)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Ángulo de visión $\theta$ (grados)")
    ax.set_ylabel("N de fuentes necesarias (stacking)")
    ax.set_title("Umbral de detección vs. ángulo de visión")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.text(
        theta.min(), 60, "~50 fuentes: tamaño típico de un campo de POSSUM",
        fontsize=8, color="gray",
    )
    ax.legend()
    ruta_salida = os.path.join(DIR_PLOTS, "fig5_umbral_deteccion.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generada: {ruta_salida}")


if __name__ == "__main__":
    calcular_y_reportar()
