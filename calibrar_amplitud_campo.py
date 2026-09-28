"""
Calibración estadística de la amplitud del campo magnético (`B0_REGULAR`,
`B0_TURBULENTO`) contra los dos conjuntos de datos reales de RM ya
cargados por `faradaymr.observational`: Oppermann & Enßlin (2012, mapa de
cielo completo) y Taylor, Stil & Sunstrum (2009, catálogo NVSS de fuentes
puntuales).

Por qué hace falta (ver `results/foreground_galactico/resumen_comparacion_observacional.json`
antes de esta calibración): `B0_REGULAR`/`B0_TURBULENTO` en `config_fisica.py`
eran valores de orden de magnitud tomados de la literatura (Jansson &
Farrar 2012, Cordes & Lazio 2002), nunca ajustados contra un dato real -el
primer chequeo cuantitativo (esta sesión) mostró que el modelo sobreestima
la RM real por un factor de varias veces en TODO el rango de latitud.

Por qué un solo factor de escala (no un ajuste independiente de cada
parámetro): RM es una integral LINEAL en el campo magnético a lo largo de
la línea de visión (`faradaymr.los.rotation_measure`); con una semilla de
campo turbulento fija, escalar `B0_REGULAR` y `B0_TURBULENTO` por el MISMO
factor `alpha` escala el mapa de RM completo por exactamente ese mismo
`alpha` (no hace falta volver a correr el ray tracing para explorar
distintos `alpha`: `RM(alpha) = alpha * RM(alpha=1)`, verificado en
`tests/test_calibracion_amplitud.py`). Se preserva la razón
`B0_TURBULENTO/B0_REGULAR` (2003/2004 = 3/2 en el config original) porque
es la relación físicamente motivada -orden de magnitud comparable, la
razón estándar reportada para el ISM (Beck 2001)- que no hay datos en este
proyecto para reajustar independientemente (no hay una comparación real de
grado de polarización/despolarización todavía, que es lo que
distinguiría la contribución de cada componente por separado).

Método: para cada banda de |b| del perfil `RMS(RM) vs |b|`
(`faradaymr.observational.perfil_estadistico_vs_latitud`, ver
`comparar_con_oppermann`), la razón observado/modelo en escala log es
`log(alpha) = log(rms_obs_i) - log(rms_modelo_i)` -el modelo actual es
`alpha=1`-; el `alpha` que minimiza la suma de cuadrados de ese residuo
sobre todas las bandas (ponderada por cuántos píxeles informan cada banda)
tiene solución cerrada: la media geométrica ponderada de las razones
por-banda. Se calcula la misma razón, de forma independiente, contra el
catálogo puntual NVSS (`rms_obs/rms_modelo_en_fuentes`, un solo número
global en vez de un perfil) como validación cruzada: si ambas fuentes de
datos -una reconstrucción Bayesiana de cielo completo, un catálogo de
mediciones puntuales crudas, con sistemáticas y errores completamente
distintos- dan un factor de escala consistente, el ajuste no es un
artefacto de una particularidad de un solo conjunto de datos.

Uso: `python calibrar_amplitud_campo.py` después de correr `run.py` al
menos una vez (usa el `rm_mapa.npy` guardado, no vuelve a simular).
"""

from __future__ import annotations

import os

import numpy as np

from faradaymr import observational as obs
from faradaymr.io import load_map

RUTA_RESULTADOS = os.path.join(os.path.dirname(__file__), "results", "foreground_galactico")


def calcular_alpha_oppermann(l_grid, b_grid, rm_map) -> float:
    """
    Media geométrica, ponderada por número de píxeles por banda, de la
    razón (RMS observado / RMS modelo) en cada banda de |b| -la solución
    cerrada del ajuste por mínimos cuadrados en log-espacio descrito en el
    docstring del módulo.
    """
    resultado = obs.comparar_con_oppermann(l_grid, b_grid, rm_map)
    perfil_modelo = resultado["perfil_modelo"]
    perfil_obs = resultado["perfil_obs"]

    razones = perfil_obs["valores"] / perfil_modelo["valores"]
    pesos = perfil_modelo["n_pixeles"].astype(float)
    return float(np.exp(np.sum(pesos * np.log(razones)) / np.sum(pesos)))


def calcular_alpha_catalogo(l_grid, b_grid, rm_map) -> float:
    """Razón (RMS observado / RMS modelo en las fuentes) contra el catálogo
    NVSS -un único número global, ver docstring del módulo."""
    resultado = obs.comparar_con_catalogo_taylor(l_grid, b_grid, rm_map)
    return float(resultado["rms_obs"] / resultado["rms_modelo_en_fuentes"])


def calibrar(ruta_resultados: str = RUTA_RESULTADOS) -> dict:
    l_grid = load_map(ruta_resultados, "l_grid")
    b_grid = load_map(ruta_resultados, "b_grid")
    rm_map = load_map(ruta_resultados, "rm_mapa")

    alpha_oppermann = calcular_alpha_oppermann(l_grid, b_grid, rm_map)
    alpha_catalogo = calcular_alpha_catalogo(l_grid, b_grid, rm_map)
    alpha_combinado = float(np.sqrt(alpha_oppermann * alpha_catalogo))

    print(f"alpha (perfil vs |b|, Oppermann+2012):        {alpha_oppermann:.4f}")
    print(f"alpha (RMS global, catálogo NVSS Taylor+2009): {alpha_catalogo:.4f}")
    print(f"alpha combinado (media geométrica de ambos):   {alpha_combinado:.4f}")
    print()
    print("Valores actuales -> calibrados (config_fisica.py):")
    print(f"  B0_REGULAR:    2.0 uG -> {2.0 * alpha_combinado:.3f} uG")
    print(f"  B0_TURBULENTO: 3.0 uG -> {3.0 * alpha_combinado:.3f} uG")

    return {
        "alpha_oppermann": alpha_oppermann,
        "alpha_catalogo": alpha_catalogo,
        "alpha_combinado": alpha_combinado,
    }


if __name__ == "__main__":
    calibrar()
