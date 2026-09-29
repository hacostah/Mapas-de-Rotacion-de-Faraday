import numpy as np
import pytest

import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.model import construir_mapa_rm_analitico, observar_stokes_qu

def _to_cpu(array):
    """
    Función auxiliar para mover arreglos de VRAM (CuPy) a RAM (NumPy).
    Si el arreglo ya es de NumPy, lo devuelve intacto.
    """
    return array.get() if hasattr(array, 'get') else array

def test_configuracion_frecuencias():
    """Verifica la correcta conversión de frecuencias a longitudes de onda al cuadrado."""
    assert len(cfg.FRECUENCIAS_GHZ) == 3
    assert len(cfg.LAMBDA_CUADRADO) == 3
    
    # Las longitudes de onda al cuadrado deben ser estrictamente positivas
    np.testing.assert_array_less(0, cfg.LAMBDA_CUADRADO)
    
    # Verificamos un cálculo conocido: 5.0 GHz -> lambda ~ 0.0599 m -> lambda^2 ~ 0.00359 m^2
    c = 299792458.0
    lambda_5ghz = c / (5.0 * 1e9)
    assert np.isclose(cfg.LAMBDA_CUADRADO[0], lambda_5ghz**2)

def test_construir_mapa_rm_analitico():
    """Verifica la dimensión y los límites físicos del mapa de RM analítico."""
    # Extraemos a CPU por si se generó en GPU
    rm_mapa = _to_cpu(construir_mapa_rm_analitico())
    
    # 1. Dimensión esperada de la malla
    assert rm_mapa.shape == (cfg.N_PIXELES, cfg.N_PIXELES)
    
    # 2. Límites físicos: El RM base es 812 * 1e-3 * 3.0 * 100 = 243.6 rad/m^2. 
    # Con la perturbación gaussiana máxima de 50.0, el valor pico debe ser ~293.6
    rm_base_teorico = 812.0 * cfg.N_E_CM3 * cfg.B_Z_BASE_UG * cfg.PROFUNDIDAD_KPC
    
    assert np.min(rm_mapa) >= rm_base_teorico
    assert np.max(rm_mapa) > rm_base_teorico + 49.0  # Cerca del pico central

def test_observar_stokes_qu():
    """Verifica las propiedades fundamentales de las observaciones simuladas."""
    rm_mapa = construir_mapa_rm_analitico()
    mapas_Q, mapas_U, mapas_psi_obs = observar_stokes_qu(rm_mapa)
    
    # Extraemos explícitamente los resultados a la RAM (CPU) para validarlos con NumPy
    mapas_Q = _to_cpu(mapas_Q)
    mapas_U = _to_cpu(mapas_U)
    mapas_psi_obs = _to_cpu(mapas_psi_obs)
    
    # 1. Verificación de dimensiones (3 frecuencias x NxN píxeles)
    shape_esperado = (3, cfg.N_PIXELES, cfg.N_PIXELES)
    assert mapas_Q.shape == shape_esperado
    assert mapas_U.shape == shape_esperado
    assert mapas_psi_obs.shape == shape_esperado
    
    # 2. Conservación de la intensidad polarizada: Q^2 + U^2 = P^2
    polarizacion_total = mapas_Q**2 + mapas_U**2
    np.testing.assert_allclose(polarizacion_total, 1.0, rtol=1e-5)
    
    # 3. Límite del radiotelescopio (Ambigüedad n-pi): 
    assert np.max(mapas_psi_obs) <= np.pi / 2.0
    assert np.min(mapas_psi_obs) >= -np.pi / 2.0