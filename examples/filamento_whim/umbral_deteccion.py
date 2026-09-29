"""
Umbral teórico de detección del filamento WHIM: a partir del sigma0(theta)
del barrido Monte Carlo, calcula

1. cuántas fuentes de fondo hacen falta para detectar el filamento a 3 sigma,
   en dos escenarios de ruido por fuente:
   - "POSSUM": dispersión intrínseca 7 rad/m² + error de medición 12 rad/m²
     (Stuardi et al. 2026);
   - "intrínseco": solo los 7 rad/m² intrínsecos (error de medición cero);
2. el campo magnético mínimo detectable en función del número de fuentes.

Uso, desde la raíz del repo y después de `run_barrido_theta.py`:

    python -m examples.filamento_whim.umbral_deteccion

Escribe results/umbral_deteccion.txt.
"""
from __future__ import annotations

import os

import numpy as np

from faradaymr.analysis.deteccion import (
    N_FUENTES_STUARDI,
    SIGMA_INTRINSECA_RAD_M2,
    SIGMA_MEDICION_POSSUM_RAD_M2,
    campo_minimo_detectable,
    resumen_umbral_deteccion,
)

BASE_DIR = os.path.dirname(__file__)
ARCHIVO_MC = os.path.join(BASE_DIR, "results", "barrido_theta", "barrido_theta_mc.npz")
ARCHIVO_REPORTE = os.path.join(BASE_DIR, "results", "umbral_deteccion.txt")

K_SIGMA = 3.0
ESCENARIOS = {
    "POSSUM (7 intr. + 12 medición)": SIGMA_MEDICION_POSSUM_RAD_M2,
    "solo intrínseco (7 rad/m², RM-grid ideal)": 0.0,
}


def calcular(datos):
    """Devuelve un diccionario con todos los números del umbral."""
    theta = datos["theta_grados"]
    sigma0 = datos["mc_sigma0"]
    b0_ng = float(datos["b0_ng"])
    salida = {"theta": theta, "sigma0": sigma0, "b0_ng": b0_ng, "escenarios": {}}
    for nombre, sigma_med in ESCENARIOS.items():
        res = resumen_umbral_deteccion(
            theta, sigma0, SIGMA_INTRINSECA_RAD_M2, k_sigma=K_SIGMA,
            sigma_medicion_rad_m2=sigma_med,
        )
        b_min_stuardi = campo_minimo_detectable(
            N_FUENTES_STUARDI, sigma0.max(), b0_ng, k_sigma=K_SIGMA,
            sigma_fondo_rad_m2=SIGMA_INTRINSECA_RAD_M2,
            sigma_medicion_rad_m2=sigma_med,
        )
        salida["escenarios"][nombre] = dict(resumen=res, sigma_med=sigma_med,
                                            b_min_ng_54=float(b_min_stuardi))
    return salida


def reporte(u, datos):
    """
    Reporte de texto con los números calculados. No contiene conclusiones
    escritas a mano: todo lo que dice sale de `u` (umbral) y `datos` (.npz del
    barrido), así que cambia si cambian los parámetros o los resultados.
    La interpretación queda a cargo de quien lee el reporte.
    """
    theta, sigma0, b0 = u["theta"], u["sigma0"], u["b0_ng"]
    i_max, i_min = int(np.argmax(sigma0)), int(np.argmin(sigma0))
    pesos = datos["hist_z"].sum(axis=1)
    lineas = [
        "Umbral teórico de detección — filamento WHIM idealizado",
        "=" * 64,
        "",
        "Entradas",
        f"  Campo del modelo (RMS):                 B = {b0:.1f} nG",
        f"  Dispersión intrínseca de RM por fuente: {SIGMA_INTRINSECA_RAD_M2:.1f} rad/m²",
        f"  Nivel de detección:                     {K_SIGMA:.0f} sigma",
        f"  Tamaño de muestra de referencia:        {N_FUENTES_STUARDI} fuentes (Stuardi et al. 2026)",
        "",
        "Señal del modelo (sigma0 = dispersión de RM en el eje)",
        f"  máximo: {sigma0[i_max]:.4f} rad/m² en theta = {theta[i_max]:.0f}°",
        f"  mínimo: {sigma0[i_min]:.4f} rad/m² en theta = {theta[i_min]:.0f}°",
        f"  sigma_intrínseca / sigma0: {SIGMA_INTRINSECA_RAD_M2 / sigma0[i_max]:.0f}"
        f" a {SIGMA_INTRINSECA_RAD_M2 / sigma0[i_min]:.0f}",
        "",
        "Gaussianidad de RM/sigma_esperada (todos los ángulos, "
        f"{pesos.sum():.2e} píxeles)",
        f"  media {np.average(datos['z_media'], weights=pesos):+.4f}, "
        f"desviación {np.average(datos['z_std'], weights=pesos):.4f}, "
        f"asimetría {np.average(datos['z_asimetria'], weights=pesos):+.4f}, "
        f"exceso de curtosis {np.average(datos['z_exceso_curtosis'], weights=pesos):+.4f}",
        "  (para una normal: 0, 1, 0, 0)",
        "",
        f"Resultados por escenario de ruido (detección a {K_SIGMA:.0f} sigma)",
    ]
    for nombre, e in u["escenarios"].items():
        r = e["resumen"]
        b_min = e["b_min_ng_54"]
        lineas += [
            f"  {nombre}:",
            f"    fuentes necesarias por muestra: {r['n_minimo']:.2e} (theta={r['theta_n_minimo']:.0f}°)"
            f" a {r['n_maximo']:.2e} (theta={r['theta_n_maximo']:.0f}°)",
            f"    campo mínimo detectable con {N_FUENTES_STUARDI} fuentes (theta={theta[i_max]:.0f}°):"
            f" {b_min:.0f} nG = {b_min / b0:.0f} x B del modelo",
            f"    ¿B del modelo detectable con {N_FUENTES_STUARDI} fuentes?: "
            f"{'sí' if r['n_minimo'] <= N_FUENTES_STUARDI else 'no'}",
        ]
    lineas += [
        "",
        "Supuestos del cálculo (no son resultados):",
        "  - RM gaussiana por fuente; se usa sigma0 del eje del filamento.",
        "  - Comparación de varianzas entre la región del filamento y una",
        "    región de control con el mismo número de fuentes.",
        "  - Sin foreground galáctico ni otros sistemáticos.",
    ]
    return "\n".join(lineas)


def calcular_y_reportar():
    if not os.path.exists(ARCHIVO_MC):
        print(f"Falta {ARCHIVO_MC}. Ejecuta run_barrido_theta.py primero.")
        return None
    datos = np.load(ARCHIVO_MC)
    u = calcular(datos)
    texto = reporte(u, datos)
    os.makedirs(os.path.dirname(ARCHIVO_REPORTE), exist_ok=True)
    with open(ARCHIVO_REPORTE, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto)
    print(f"\nReporte guardado en: {ARCHIVO_REPORTE}")
    return u


if __name__ == "__main__":
    calcular_y_reportar()
