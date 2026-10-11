import numpy as np

from examples.falso_observatorio.estudio_parametrico import estres_estadistico


def test_estres_estadistico_estructura():
    configuraciones = {
        "Configuracion 3": [0, 1, 2],
        "Configuracion 2": [0, 1],
    }
    sigmas = np.array([0.0, 0.2])

    resultados = estres_estadistico(
        sigmas, configuraciones, n_muestra=20, n_repeticiones=2
    )

    assert isinstance(resultados, dict)
    assert list(k for k in resultados if not k.startswith("_")) == [
        "Configuracion 3", "Configuracion 2"
    ]
    assert len(resultados["Configuracion 3"]["fallos"]) == len(sigmas)
    assert len(resultados["Configuracion 2"]["fallos"]) == len(sigmas)
    assert resultados["_meta"]["rm_min"] < resultados["_meta"]["rm_max"]


def test_tres_frecuencias_sin_ruido_recuperan_la_rama():
    configuraciones = {"Completas": [0, 1, 2]}
    resultados = estres_estadistico(
        np.array([0.0]), configuraciones, n_muestra=100, n_repeticiones=2
    )
    assert resultados["Completas"]["fallos"][0] == 0.0


def test_dos_frecuencias_son_intrinsecamente_ambiguas():
    configuraciones = {"Ambiguas": [0, 1]}
    resultados = estres_estadistico(
        np.array([0.0, 0.1]), configuraciones, n_muestra=20, n_repeticiones=2
    )
    assert resultados["Ambiguas"]["fallos"] == [1.0, 1.0]
