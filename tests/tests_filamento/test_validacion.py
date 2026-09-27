import pytest
import warnings
from examples.filamento_whim.validacion import verificar_caja_suficiente

def test_verificar_caja_generosa_0_grados():
    """
    Filamento de 500 kpc en caja de 1000 kpc. 
    theta=0° -> factor 1. No debe emitir warning y debe retornar True.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Falla si se emite cualquier warning
        resultado = verificar_caja_suficiente(
            n_base=100, 
            dx_base_kpc=10.0, 
            longitud_filamento_kpc=500.0, 
            thetas_grados=[0]
        )
        assert resultado is True

def test_verificar_caja_insuficiente_89_grados():
    """
    Filamento de 500 kpc en caja de 1000 kpc. 
    theta=89° -> factor ~57. Necesita ~28500 kpc. Debe retornar False y emitir warning.
    """
    with pytest.warns(UserWarning, match="no alcanza para theta=89°"):
        resultado = verificar_caja_suficiente(
            n_base=100, 
            dx_base_kpc=10.0, 
            longitud_filamento_kpc=500.0, 
            thetas_grados=[0, 45, 89]
        )
        assert resultado is False

def test_verificar_caja_exacta():
    """
    Filamento de 1000 kpc en caja de 1000 kpc con theta=0. 
    Debe caber exactamente y retornar True.
    """
    resultado = verificar_caja_suficiente(
        n_base=100, 
        dx_base_kpc=10.0, 
        longitud_filamento_kpc=1000.0, 
        thetas_grados=[0]
    )
    assert resultado is True