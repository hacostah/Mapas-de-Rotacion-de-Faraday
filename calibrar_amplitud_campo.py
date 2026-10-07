"""
Calibración estadística de `FACTOR_CAMPO_REGULAR` (config_fisica.py): el
factor único que lleva las amplitudes de literatura del campo regular
(disco + halo de JF12) a valores efectivos para este modelo de juguete.

Solo se calibra el campo regular. La amplitud efectiva de la turbulencia
no se ajusta: se deriva de su longitud de coherencia (`FACTOR_TURBULENCIA`
en `config.py`, ver config_fisica.py). Ajustar las dos a la vez contra el
mismo perfil de RM no las separa: ambas suben la RMS en todas las
latitudes.

Método. La RM es lineal en B, así que con la semilla de turbulencia fija

    RM(f) = f · RM_regular + RM_turbulenta,

con RM_regular calculada con factor 1. Basta un ray tracing por
componente para evaluar cualquier f sin volver a simular (ver
`tests/test_calibracion_amplitud.py`). Se eligen dos estimaciones de f
independientes:

1. Oppermann & Enßlin (2012): el f que minimiza la suma de cuadrados de
   log(RMS_modelo / RMS_obs) sobre las bandas de |b|, ponderada por el
   número de píxeles de cada banda.
2. Catálogo NVSS (Taylor, Stil & Sunstrum 2009): el f con el que la RMS
   del modelo en la posición de las fuentes iguala a la observada.

Los dos datos tienen sistemáticas distintas (el catálogo suma la RM
intrínseca de cada fuente y el ruido de medida; la reconstrucción de
Oppermann suprime potencia en escalas pequeñas), así que se propone su
media geométrica, y el script imprime las dos para ver cuánto difieren.

Uso: `python calibrar_amplitud_campo.py`. Simula con la configuración
actual (semilla 0, la misma que `run.py`) e imprime el valor a poner en
`FACTOR_CAMPO_REGULAR`; no modifica config_fisica.py.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import config as cfg
from faradaymr import get_backend, los_raytrace, to_numpy
from faradaymr import observational as obs
from model import construir_escenario

RANGO_FACTOR = (0.05, 5.0)


def mapas_rm_por_componente(seed: int = 0, use_gpu=None):
    """
    (l_grid, b_grid, rm_regular, rm_turbulenta): la RM del campo regular
    con FACTOR_CAMPO_REGULAR = 1 y la de la turbulencia, integradas por
    separado sobre la misma grilla de cielo que `run.py`. Con GPU se
    integra en cupy y se devuelve en numpy (las comparaciones contra datos
    reales son de CPU).
    """
    xp = get_backend(use_gpu)
    factor_original = cfg.FACTOR_CAMPO_REGULAR
    cfg.FACTOR_CAMPO_REGULAR = 1.0
    try:
        (_, _, _, ne, ne_rel, observer_pos, box_size, dx, regular, turbulento) = (
            construir_escenario(
                use_gpu=use_gpu, rng=np.random.RandomState(seed), return_components=True
            )
        )
    finally:
        cfg.FACTOR_CAMPO_REGULAR = factor_original

    l_grid = np.linspace(-np.pi, np.pi, cfg.N_L, endpoint=False)
    b_max = np.radians(cfg.B_MAX_DEG)
    b_grid = np.linspace(-b_max, b_max, cfg.N_B)

    def rm_de(campo):
        rm_map, _i, _q, _u = los_raytrace.sky_map(
            *campo, ne, ne_rel, observer_pos, dx, box_size, xp.asarray(l_grid), xp.asarray(b_grid),
            dl=cfg.DL_KPC, frequency=cfg.NU_HZ, wavelength=cfg.LAMBDA_ONDA_M,
            p_index=cfg.P_SPEC, xp=xp, pixel_chunk_size=cfg.PIXEL_CHUNK_SIZE,
            length_unit_pc=cfg.KPC_A_PC,
        )
        return to_numpy(rm_map)

    return l_grid, b_grid, rm_de(regular), rm_de(turbulento)


def factor_oppermann(l_grid, b_grid, rm_regular, rm_turbulenta) -> float:
    """f que minimiza el residuo en log del perfil RMS(RM) vs |b| contra Oppermann+2012."""
    perfil_obs = obs.comparar_con_oppermann(l_grid, b_grid, rm_regular)["perfil_obs"]
    pesos = perfil_obs["n_pixeles"].astype(float)

    def costo(factor):
        rm = factor * rm_regular + rm_turbulenta
        perfil_modelo = obs.perfil_estadistico_vs_latitud(b_grid, rm)
        residuo = np.log(perfil_modelo["valores"] / perfil_obs["valores"])
        return float(np.sum(pesos * residuo**2))

    return float(minimize_scalar(costo, bounds=RANGO_FACTOR, method="bounded").x)


def factor_catalogo(l_grid, b_grid, rm_regular, rm_turbulenta) -> float:
    """f con el que la RMS del modelo en las fuentes NVSS iguala la observada."""
    def exceso(factor):
        resultado = obs.comparar_con_catalogo_taylor(
            l_grid, b_grid, factor * rm_regular + rm_turbulenta
        )
        return resultado["rms_modelo_en_fuentes"] - resultado["rms_obs"]

    return float(brentq(exceso, *RANGO_FACTOR))


def calibrar(seed: int = 0, use_gpu=None) -> dict:
    l_grid, b_grid, rm_regular, rm_turbulenta = mapas_rm_por_componente(seed, use_gpu=use_gpu)

    f_oppermann = factor_oppermann(l_grid, b_grid, rm_regular, rm_turbulenta)
    f_catalogo = factor_catalogo(l_grid, b_grid, rm_regular, rm_turbulenta)
    f_combinado = float(np.sqrt(f_oppermann * f_catalogo))

    print(f"Turbulencia efectiva (derivada, no ajustada): {cfg.B0_TURBULENTO_MG:.3f} uG "
          f"(= {cfg.B0_TURBULENTO_LITERATURA_MG:.1f} uG x {cfg.FACTOR_TURBULENCIA:.3f})")
    print(f"factor regular (perfil vs |b|, Oppermann+2012):  {f_oppermann:.3f}")
    print(f"factor regular (RMS en fuentes NVSS, Taylor+09): {f_catalogo:.3f}")
    print(f"factor regular combinado (media geométrica):     {f_combinado:.3f}")
    print(f"\nFACTOR_CAMPO_REGULAR actual: {cfg.FACTOR_CAMPO_REGULAR:.3f} -> propuesto: {f_combinado:.2f}")

    return {
        "factor_oppermann": f_oppermann,
        "factor_catalogo": f_catalogo,
        "factor_combinado": f_combinado,
    }


if __name__ == "__main__":
    calibrar()
