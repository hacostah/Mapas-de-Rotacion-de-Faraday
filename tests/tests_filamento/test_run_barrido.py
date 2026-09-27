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