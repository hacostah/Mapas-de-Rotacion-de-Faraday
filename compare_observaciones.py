"""
Comparación/sustracción del foreground galáctico sintético contra datos
reales (issue #37): carga los mapas de la última corrida de `run.py` desde
`results/foreground_galactico`, los compara contra tres conjuntos de datos
observacionales reales (ver `faradaymr.observational` para las referencias
y las limitaciones de cada comparación), genera las figuras de comparación
y deja un resumen numérico en JSON.

No vuelve a correr la simulación (usa lo que `run.py` ya dejó guardado en
disco con `faradaymr.io.save_maps`): comparar contra datos reales es un
paso posterior y más caro (descarga/lectura de catálogos externos), que no
tiene sentido repetir cada vez que se corre la simulación.

Uso: `python compare_observaciones.py` después de correr `run.py` al menos
una vez.
"""

from __future__ import annotations

import json
import os

import numpy as np

from faradaymr import observational as obs
from faradaymr.io import load_map, save_maps
from faradaymr.plotting_comparacion import (
    dispersion_catalogo_vs_modelo,
    histograma_comparacion_rm,
    mapa_comparacion_mollweide,
    mapa_morfologia_sincrotron,
    mapa_resta_planck,
    perfil_comparacion_latitud,
)

RUTA_RESULTADOS = os.path.join(os.path.dirname(__file__), "results", "foreground_galactico")


def ejecutar_comparacion(ruta_resultados: str = RUTA_RESULTADOS):
    print(f"Cargando la última corrida desde {ruta_resultados} ...")
    l_grid = load_map(ruta_resultados, "l_grid")
    b_grid = load_map(ruta_resultados, "b_grid")
    rm_map = load_map(ruta_resultados, "rm_mapa")
    i_map = load_map(ruta_resultados, "intensidad")

    print("Comparando contra Oppermann & Enßlin (2012, reconstrucción de cielo completo)...")
    resultado_opp = obs.comparar_con_oppermann(l_grid, b_grid, rm_map)

    print("Comparando contra el catálogo NVSS de Taylor, Stil & Sunstrum (2009)...")
    resultado_cat = obs.comparar_con_catalogo_taylor(l_grid, b_grid, rm_map)

    print("Comparando morfología sincrotrón contra Haslam et al. (1982, 408 MHz)...")
    resultado_morf = obs.comparar_morfologia_sincrotron(l_grid, b_grid, i_map)

    print("Restando el foreground sintético (reescalado) contra Planck 30GHz...")
    resultado_planck = obs.restar_planck_030ghz(l_grid, b_grid, i_map)

    print("Generando figuras de comparación...")
    mapa_comparacion_mollweide(ruta_resultados, l_grid, b_grid, rm_map, resultado_opp["rm_obs_grid"])
    perfil_comparacion_latitud(ruta_resultados, resultado_opp["perfil_modelo"], resultado_opp["perfil_obs"])
    dispersion_catalogo_vs_modelo(ruta_resultados, resultado_cat)
    histograma_comparacion_rm(
        ruta_resultados,
        np.asarray(rm_map).ravel(),
        resultado_opp["rm_obs_grid"].ravel(),
        resultado_cat["rm_obs"],
    )
    mapa_morfologia_sincrotron(ruta_resultados, l_grid, b_grid, resultado_morf)
    mapa_resta_planck(ruta_resultados, l_grid, b_grid, resultado_planck)

    # El producto de datos en sí (no solo la figura): el mapa ya limpio de
    # foreground sintético, listo para usarse aguas abajo (p.ej. un
    # análisis de B-modos de CMB) sin tener que volver a correr esta
    # comparación completa cada vez.
    ruta_planck = os.path.join(ruta_resultados, "resta_planck_030ghz")
    save_maps(
        ruta_planck,
        {
            "i_planck_030ghz_k_cmb": resultado_planck["i_planck_grid"],
            "i_modelo_escalado_k_cmb": resultado_planck["i_modelo_escalado"],
            "residuo_k_cmb": resultado_planck["residuo"],
            "l_grid": l_grid,
            "b_grid": b_grid,
        },
    )

    perfil_b = [
        {
            "b_deg": float(b),
            "rms_modelo": float(vm),
            "rms_oppermann": float(vo),
            "razon_modelo_sobre_obs": float(vm / vo) if vo != 0 else None,
        }
        for b, vm, vo in zip(
            resultado_opp["perfil_modelo"]["centros_deg"],
            resultado_opp["perfil_modelo"]["valores"],
            resultado_opp["perfil_obs"]["valores"],
        )
    ]

    resumen = {
        "oppermann_2012": {
            "referencia": resultado_opp["referencia"],
            "rms_modelo_global_rad_m2": resultado_opp["rms_modelo"],
            "rms_observado_global_rad_m2": resultado_opp["rms_obs"],
            "chi2_reducido_ingenuo": resultado_opp["chi2_reducido_ingenuo"],
            "perfil_rms_vs_b": perfil_b,
        },
        "catalogo_taylor_2009": {
            "referencia": resultado_cat["referencia"],
            "n_fuentes_comparadas": resultado_cat["n_fuentes_comparadas"],
            "rms_observado_rad_m2": resultado_cat["rms_obs"],
            "rms_modelo_en_fuentes_rad_m2": resultado_cat["rms_modelo_en_fuentes"],
            "rms_residual_rad_m2": resultado_cat["rms_residual"],
            "correlacion_pearson": resultado_cat["correlacion_pearson"],
        },
        "morfologia_haslam_408mhz": {
            "referencia": resultado_morf["referencia"],
            "correlacion_log_pearson": resultado_morf["correlacion_log_pearson"],
        },
        "resta_planck_030ghz": {
            "referencia": resultado_planck["referencia"],
            "alpha_reescalado": resultado_planck["alpha"],
            "rms_planck_k_cmb": resultado_planck["rms_planck"],
            "rms_residuo_k_cmb": resultado_planck["rms_residuo"],
            "fraccion_varianza_explicada": resultado_planck["fraccion_varianza_explicada"],
        },
    }

    ruta_resumen = os.path.join(ruta_resultados, "resumen_comparacion_observacional.json")
    with open(ruta_resumen, "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nResumen guardado en {ruta_resumen}")
    return resumen


if __name__ == "__main__":
    ejecutar_comparacion()
