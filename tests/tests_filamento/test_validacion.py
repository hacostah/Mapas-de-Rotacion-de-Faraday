import numpy as np
import pytest
import warnings
from examples.filamento_whim.validacion import verificar_caja_suficiente


def test_verificar_caja_generosa_90_grados():
    """
    Filamento de 5000 kpc (mucho mas largo que la caja) visto de lado
    (theta=90 grados): cos(90)=0, la profundidad necesaria a lo largo de
    la linea de vision es ~0, así que una caja de solo 1000 kpc alcanza
    sin problema. No debe emitir warning y debe retornar True.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Falla si se emite cualquier warning
        resultado = verificar_caja_suficiente(
            n_base=100,
            dx_base_kpc=10.0,
            longitud_filamento_kpc=5000.0,
            thetas_grados=[90],
        )
        assert resultado is True


def test_verificar_caja_insuficiente_cerca_de_0_grados():
    """
    Filamento de 5000 kpc en caja de 1000 kpc. El peor caso del barrido
    es el theta mas cercano a 0 grados (aqui, 5 grados): la profundidad
    requerida es longitud*cos(5 grados) ~ 4981 kpc, muy por encima de la
    caja. Debe retornar False y advertir sobre theta=5, no sobre theta=85
    (que es el angulo mas oblicuo, pero el que menos profundidad exige).
    """
    with pytest.warns(UserWarning, match=r"theta=5\.0"):
        resultado = verificar_caja_suficiente(
            n_base=100,
            dx_base_kpc=10.0,
            longitud_filamento_kpc=5000.0,
            thetas_grados=[5, 45, 85],
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
        thetas_grados=[0],
    )
    assert resultado is True


def test_verificar_caja_peor_caso_es_el_theta_mas_cercano_a_cero():
    """
    Regresión directa del bug de la version anterior: para un mismo
    filamento y caja, el angulo que mas profundidad exige es el mas
    cercano a 0 grados, no el mas cercano a 90 grados.
    """
    # Con longitud=1500 y caja=1000: a theta=10 grados hace falta
    # ~1477 kpc (> 1000, falla); a theta=80 grados solo ~260 kpc (<1000, pasa).
    with pytest.warns(UserWarning):
        solo_theta_chico = verificar_caja_suficiente(
            n_base=100, dx_base_kpc=10.0, longitud_filamento_kpc=1500.0,
            thetas_grados=[10],
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solo_theta_grande = verificar_caja_suficiente(
            n_base=100, dx_base_kpc=10.0, longitud_filamento_kpc=1500.0,
            thetas_grados=[80],
        )
    assert solo_theta_chico is False
    assert solo_theta_grande is True


def test_verificar_caja_acepta_array_numpy_de_thetas():
    # Bug corregido: `if not thetas_grados` lanza ValueError con un array de
    # numpy de más de un elemento ("truth value... is ambiguous"). Esto no
    # se disparaba en run_barrido_theta.py (convierte a lista antes de
    # llamar aquí), pero sí en cualquier llamada directa con un array -por
    # ejemplo, pasando theta_grados=np.linspace(...) sin convertir.
    resultado = verificar_caja_suficiente(
        n_base=100,
        dx_base_kpc=10.0,
        longitud_filamento_kpc=1000.0,
        thetas_grados=np.array([0.0, 45.0, 90.0]),
    )
    assert resultado is True


def test_verificar_caja_array_vacio_retorna_true():
    resultado = verificar_caja_suficiente(
        n_base=100,
        dx_base_kpc=10.0,
        longitud_filamento_kpc=1000.0,
        thetas_grados=np.array([]),
    )
    assert resultado is True


def test_profundidad_radial_detecta_la_caja_vieja_y_acepta_la_nueva():
    import warnings
    from examples.filamento_whim.validacion import verificar_profundidad_radial
    # Caja anterior del proyecto: +/-1280 kpc, beta=0.5, ventana de 4 r_c.
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        assert not verificar_profundidad_radial(128, 20.0, 300.0, 0.5, 1200.0)
        assert len(avisos) == 1
    # Configuración actual: +/-2000 kpc, beta=2/3, ventana de 3 r_c.
    assert verificar_profundidad_radial(160, 25.0, 300.0, 2 / 3, 900.0)


def test_fraccion_retenida_tiende_a_uno_con_caja_infinita():
    from examples.filamento_whim.validacion import fraccion_varianza_retenida
    assert fraccion_varianza_retenida(600.0, 1e6, 300.0, 2 / 3) > 0.999
