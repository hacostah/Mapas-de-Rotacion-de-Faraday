import numpy as np
from faradaymr.analysis.deteccion import n_fuentes_necesarias, resumen_umbral_deteccion


def test_n_fuentes_necesarias_orden_de_magnitud_consistente_con_el_proyecto():
    # sigma0~0.05 rad/m2 frente a un fondo de 7 rad/m2 (Stuardi et al. 2026):
    # el orden de magnitud esperado es ~10^8-10^9 fuentes para una detección
    # marginal (1 sigma). Es la cifra que motiva agregar este módulo.
    n = n_fuentes_necesarias(sigma0_rad_m2=0.05, sigma_fondo_rad_m2=7.0, k_sigma=1.0)
    assert 1e7 < n < 1e10


def test_n_fuentes_necesarias_escala_como_cuarta_potencia_inversa_de_sigma0():
    # N ~ (sigma_fondo^2/sigma0^2)^2, así que duplicar sigma0 debe dividir N
    # entre 2^4 = 16.
    n1 = n_fuentes_necesarias(sigma0_rad_m2=0.05)
    n2 = n_fuentes_necesarias(sigma0_rad_m2=0.10)
    assert np.isclose(n1 / n2, 16.0, rtol=1e-6)


def test_n_fuentes_necesarias_escala_con_k_sigma_al_cuadrado():
    n_1s = n_fuentes_necesarias(sigma0_rad_m2=0.05, k_sigma=1.0)
    n_3s = n_fuentes_necesarias(sigma0_rad_m2=0.05, k_sigma=3.0)
    assert np.isclose(n_3s / n_1s, 9.0, rtol=1e-6)


def test_n_fuentes_necesarias_sigma0_cero_da_infinito():
    assert n_fuentes_necesarias(sigma0_rad_m2=0.0) == float("inf")


def test_resumen_umbral_deteccion_identifica_extremos():
    theta = np.array([0.0, 45.0, 90.0])
    sigma0 = np.array([0.08, 0.06, 0.05])  # decreciente con theta

    resumen = resumen_umbral_deteccion(theta, sigma0, sigma_fondo_rad_m2=7.0)

    # sigma0 máximo (theta=0) -> N mínimo; sigma0 mínimo (theta=90) -> N máximo.
    assert resumen["theta_n_minimo"] == 0.0
    assert resumen["theta_n_maximo"] == 90.0
    assert resumen["n_minimo"] < resumen["n_maximo"]
