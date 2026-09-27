import os
import tempfile
import numpy as np
from unittest import mock
import astropy.units as u

from examples.filamento_whim.run_barrido_theta import barrer_angulos

@mock.patch('examples.filamento_whim.config_fisica.N_BASE', 32)
@mock.patch('examples.filamento_whim.config_fisica.LONGITUD_FILAMENTO', 300.0 * u.kpc)
def test_barrido_theta_smoke():
    """
    Test de humo para el barrido.
    Utiliza una malla pequeña (N_BASE=32) y un filamento muy corto para
    que la validación geométrica pase y el test corra rápido en CPU.
    """
    thetas_prueba = [0, 45]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        resultados = barrer_angulos(
            thetas_grados=thetas_prueba,
            ruta_resultados=temp_dir,
            use_gpu=False,
            n_bins=5,  # Menos bins porque la malla es más chica
            seed=42
        )
        
        # Validar salidas
        assert "theta_grados" in resultados
        assert "width_kpc" in resultados
        assert "width_err_kpc" in resultados
        assert "sigma0" in resultados
        
        assert len(resultados["theta_grados"]) == 2
        assert len(resultados["width_kpc"]) == 2
        
        # Verificar que el npz se puede guardar
        archivo_salida = os.path.join(temp_dir, "test_salida.npz")
        np.savez(archivo_salida, **resultados)
        assert os.path.exists(archivo_salida)

@mock.patch('examples.filamento_whim.run_barrido_theta.barrer_angulos')
def test_barrer_angulos_monte_carlo_agregacion(mock_barrer):
    """
    Verifica que la capa Monte Carlo orqueste el número correcto de semillas
    y agregue correctamente los arrays (mean y std), sin ejecutar la física real.
    """
    from examples.filamento_whim.run_barrido_theta import barrer_angulos_monte_carlo

    # 1. Configuramos el mock para que devuelva datos falsos predecibles según la semilla.
    # Semilla 0 -> anchos: [8.0, 18.0], sigma0: [4.0, 4.0]
    # Semilla 1 -> anchos: [10.0, 20.0], sigma0: [5.0, 5.0]
    # Semilla 2 -> anchos: [12.0, 22.0], sigma0: [6.0, 6.0]
    # Esperamos que el promedio de anchos sea [10.0, 20.0] y sigma0 [5.0, 5.0]
    def mock_side_effect(thetas_grados, ruta_resultados, use_gpu, n_bins, seed):
        desplazamiento = (seed - 1) * 2.0  # -2, 0, +2
        return {
            "theta_grados": thetas_grados,
            "width_kpc": np.array([10.0, 20.0]) + desplazamiento,
            "width_err_kpc": np.array([1.0, 1.0]),  # No se usa en la agregación
            "sigma0": np.array([5.0, 5.0]) + (seed - 1) * 1.0,
        }
    
    mock_barrer.side_effect = mock_side_effect

    # 2. Ejecutamos la función Monte Carlo
    thetas_prueba = [0, 45]
    n_semillas = 3
    
    resultados = barrer_angulos_monte_carlo(
        thetas_grados=thetas_prueba,
        ruta_resultados="ruta_falsa",
        n_semillas=n_semillas,
        use_gpu=False
    )

    # 3. Verificamos el comportamiento del orquestador
    assert mock_barrer.call_count == n_semillas, "No se llamó a la función el número correcto de veces"
    
    # 4. Verificamos la agregación matemática (mean y std)
    np.testing.assert_array_equal(resultados["width_medio_kpc"], [10.0, 20.0])
    np.testing.assert_array_equal(resultados["sigma0_medio"], [5.0, 5.0])
    
    # La desviación estándar poblacional de [-2, 0, 2] es sqrt(8/3) ≈ 1.63299
    # Para los anchos la std debería ser > 0
    assert np.all(resultados["width_std_kpc"] > 0)
    assert np.all(resultados["sigma0_std"] > 0)
    assert resultados["width_medio_kpc"].shape == (2,)