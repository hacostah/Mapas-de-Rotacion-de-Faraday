import numpy as np
import pytest

from examples.falso_observatorio.model import construir_mapa_rm_analitico
from examples.falso_observatorio.estudio_parametrico import estres_estadistico

def test_estres_estadistico_estructura():
    """Verifica que el diccionario de salida tenga las llaves y dimensiones correctas."""
    rm_verdad = construir_mapa_rm_analitico()
    configuraciones = {
        "Configuracion 3": [0, 1, 2],
        "Configuracion 2": [0, 1]
    }
    sigmas = np.array([0.0, 0.2, 0.4])
    
    # Corremos con repeticiones al mínimo (1) para no ralentizar pytest
    resultados = estres_estadistico(
        rm_verdad, 
        configuraciones, 
        sigmas, 
        n_repeticiones=1
    )
    
    # Verificaciones
    assert isinstance(resultados, dict)
    assert list(resultados.keys()) == ["Configuracion 3", "Configuracion 2"]
    assert len(resultados["Configuracion 3"]) == len(sigmas)
    assert len(resultados["Configuracion 2"]) == len(sigmas)

def test_estres_estadistico_fisica_cero_ruido():
    """
    Verifica el fenómeno de aliasing: 
    A cero ruido, 3 frecuencias resuelven el RM, mientras que 2 frecuencias colapsan.
    """
    rm_verdad = construir_mapa_rm_analitico()
    configuraciones = {
        "Completas": [0, 1, 2],
        "Ambiguas": [0, 1]
    }
    sigmas = np.array([0.0]) # Sin ruido instrumental inyectado
    
    resultados = estres_estadistico(
        rm_verdad,
        configuraciones,
        sigmas,
        n_repeticiones=1
    )
    
    # 1. Con 3 puntos (frecuencias), la recta ideal se recupera perfectamente sin errores
    assert resultados["Completas"][0] == 0.0
    
    # 2. Con 2 puntos, el sistema está degenerado (aliasing) y falla estrepitosamente
    # Validamos que la tasa de falla sea mayor al 80% incluso sin ruido térmico
    assert resultados["Ambiguas"][0] > 0.8