import numpy as np
import pytest
from faradaymr.analysis.fitting import gaussian_model, fit_transverse_dispersion

def test_fit_recupera_parametros_exactos():
    """Criterio 1: Recupera sigma0 y width con perfil sintético y ruido mínimo."""
    centros = np.linspace(-5, 5, 20)
    sigma0_true = 8.5
    width_true = 1.2
    
    # Crear perfil ideal y añadir ruido flotante mínimo
    sigma_rm_ideal = gaussian_model(centros, sigma0_true, width_true)
    ruido = np.random.normal(0, 1e-5, size=centros.size)
    sigma_rm_sintetico = sigma_rm_ideal + ruido
    
    resultado = fit_transverse_dispersion(centros, sigma_rm_sintetico)
    
    # Verificar recuperación dentro de tolerancia numérica
    assert np.isclose(resultado.sigma0, sigma0_true, atol=1e-3)
    assert np.isclose(resultado.width, width_true, atol=1e-3)
    assert resultado.r_squared > 0.999

def test_fit_maneja_nans():
    """Criterio 2: Maneja perfiles con bins NaN sin romperse."""
    centros = np.linspace(0, 10, 15)
    sigma0_true = 10.0
    width_true = 2.0
    
    sigma_rm = gaussian_model(centros, sigma0_true, width_true)
    
    # Inyectar NaNs simulando bins vacíos en el perfil radial
    sigma_rm[2] = np.nan
    sigma_rm[5] = np.nan
    sigma_rm[-1] = np.nan
    
    resultado = fit_transverse_dispersion(centros, sigma_rm)
    
    # El ajuste debe realizarse correctamente saltando los NaNs
    assert np.isclose(resultado.sigma0, sigma0_true, rtol=1e-5)
    assert np.isfinite(resultado.r_squared)

def test_fit_error_pocos_puntos():
    """Criterio 3: Lanza un error claro si no hay suficientes puntos válidos."""
    centros = np.array([0.0, 1.0, 2.0, 3.0])
    
    # Solo 2 puntos válidos, el resto NaNs
    sigma_rm = np.array([5.0, 3.0, np.nan, np.nan])
    
    with pytest.raises(ValueError, match="No hay suficientes bins válidos"):
        fit_transverse_dispersion(centros, sigma_rm)