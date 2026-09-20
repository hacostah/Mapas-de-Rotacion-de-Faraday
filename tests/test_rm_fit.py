import numpy as np

from faradaymr.analysis import estimate_rm_lsq
from faradaymr.analysis.rm_fit import unwrap_polarization_angle, resolve_n_pi_ambiguity


def test_recupera_rm_sin_ruido():
    wavelengths_m = np.array([0.10, 0.16, 0.22, 0.31])
    rm_true = 83.5
    psi_0_true = -0.27

    psi_obs_rad = psi_0_true + rm_true * wavelengths_m**2

    rm, psi_0 = estimate_rm_lsq(wavelengths_m, psi_obs_rad)

    assert np.isclose(rm, rm_true, atol=1e-12)
    assert np.isclose(psi_0, psi_0_true, atol=1e-12)


def test_dos_frecuencias_determinan_la_recta():
    wavelengths_m = np.array([0.12, 0.28])
    rm_true = -41.0
    psi_0_true = 0.9

    psi_obs_rad = psi_0_true + rm_true * wavelengths_m**2

    rm, psi_0 = estimate_rm_lsq(wavelengths_m, psi_obs_rad)

    assert np.isclose(rm, rm_true, atol=1e-12)
    assert np.isclose(psi_0, psi_0_true, atol=1e-12)


def test_multiples_frecuencias_recuperan_rm_con_ruido():
    rng = np.random.default_rng(25)
    wavelengths_m = np.linspace(0.08, 0.32, 20)
    rm_true = 120.0
    psi_0_true = 0.15

    psi_obs_rad = psi_0_true + rm_true * wavelengths_m**2
    psi_obs_rad += rng.normal(scale=0.01, size=wavelengths_m.size)

    rm, psi_0 = estimate_rm_lsq(wavelengths_m, psi_obs_rad)

    assert np.isclose(rm, rm_true, atol=0.5)
    assert np.isclose(psi_0, psi_0_true, atol=0.02)


def test_unwrap_polarization_angle():
    """Verifica que los saltos artificiales de pi se desenreden correctamente."""
    # 1. Creamos ángulos verdaderos que crecen linealmente (sin saltos)
    angulos_verdaderos = np.array([0.1, 1.5, 2.9]) # Radianes
    
    # 2. Simulamos la limitación del telescopio: forzamos los ángulos 
    # al intervalo [-pi/2, pi/2] (esto introduce un salto artificial)
    angulos_observados = (angulos_verdaderos + np.pi/2) % np.pi - np.pi/2
    
    # 3. Aplicamos nuestra función de corrección
    angulos_corregidos = unwrap_polarization_angle(angulos_observados)
    
    # 4. Verificación: La diferencia (pendiente) entre los ángulos corregidos 
    # debe ser idéntica a la diferencia entre los ángulos verdaderos
    np.testing.assert_allclose(np.diff(angulos_corregidos), np.diff(angulos_verdaderos))


def test_resolve_n_pi_ambiguity():
    """Verifica que el ajuste por fuerza bruta recupera el RM en frecuencias discretas."""
    # 1. Configuración de nuestro Falso Observatorio
    frecuencias_ghz = np.array([5.0, 2.0, 1.4])
    c = 299792458.0 # Velocidad de la luz en m/s
    longitudes_onda_m = c / (frecuencias_ghz * 1e9)
    lambda2 = longitudes_onda_m ** 2
    
    # 2. Física verdadera (RM alto para forzar que la rotación supere pi/2)
    rm_verdadero = 150.0  # rad/m^2
    psi0_verdadero = 0.5  # radianes
    angulos_verdaderos = psi0_verdadero + rm_verdadero * lambda2
    
    # 3. Medición del telescopio (aparece la ambigüedad n*pi)
    angulos_observados = (angulos_verdaderos + np.pi/2) % np.pi - np.pi/2
    
    # 4. Recuperación
    rm_recuperado = resolve_n_pi_ambiguity(longitudes_onda_m, angulos_observados)
    
    # 5. Verificación: El RM calculado debe ser igual al RM verdadero que inyectamos
    np.testing.assert_allclose(rm_recuperado, rm_verdadero, rtol=1e-3)