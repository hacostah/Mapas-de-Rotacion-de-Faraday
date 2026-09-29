import numpy as np
import pytest
from faradaymr.analysis.fitting import (
    gaussian_model,
    fit_transverse_dispersion,
    beta_dispersion_model,
    fit_beta_dispersion,
    p_random_walk_cilindro,
)

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


def test_p_random_walk_cilindro_beta_0_5_da_p_0_5():
    # Valor de este proyecto (config_fisica.BETA = 0.5): p = (3*0.5-0.5)/2 = 0.5.
    assert np.isclose(p_random_walk_cilindro(0.5), 0.5)


def test_fit_beta_libre_recupera_parametros_con_ruido_minimo():
    centros = np.linspace(0, 20, 25)
    sigma0_true, rc_true, p_true = 5.0, 4.0, 0.7
    sigma_rm = beta_dispersion_model(centros, sigma0_true, rc_true, p_true)
    sigma_rm += np.random.normal(0, 1e-6, size=centros.size)

    resultado = fit_beta_dispersion(centros, sigma_rm)

    assert np.isclose(resultado.sigma0, sigma0_true, atol=1e-2)
    assert np.isclose(resultado.r_c, rc_true, atol=1e-2)
    assert np.isclose(resultado.p, p_true, atol=1e-2)
    assert resultado.r_squared > 0.999


def test_fit_beta_p_fijo_ajusta_solo_dos_parametros():
    centros = np.linspace(0, 20, 25)
    sigma0_true, rc_true, p_true = 5.0, 4.0, 0.5
    sigma_rm = beta_dispersion_model(centros, sigma0_true, rc_true, p_true)
    sigma_rm += np.random.normal(0, 1e-6, size=centros.size)

    resultado = fit_beta_dispersion(centros, sigma_rm, p_fijo=p_true)

    assert np.isclose(resultado.sigma0, sigma0_true, atol=1e-2)
    assert np.isclose(resultado.r_c, rc_true, atol=1e-2)
    assert resultado.p == p_true
    assert resultado.p_err == 0.0


def test_fit_beta_p_fijo_recupera_rc_donde_el_libre_degenera():
    # Caso realista de este proyecto: pocos bins y ruido no despreciable
    # hacen que el ajuste de 3 parámetros libres (sigma0, r_c, p) quede mal
    # condicionado. Con p fijo al valor físico esperado, el ajuste de 2
    # parámetros sigue recuperando r_c de forma mucho más ajustada.
    rng = np.random.default_rng(11)
    centros = np.linspace(0, 15, 8)
    sigma0_true, rc_true, p_true = 0.05, 3.0, 0.5
    sigma_rm = beta_dispersion_model(centros, sigma0_true, rc_true, p_true)
    sigma_rm += rng.normal(0, sigma0_true * 0.05, size=centros.size)

    resultado_fijo = fit_beta_dispersion(centros, sigma_rm, p_fijo=p_true)

    assert np.isclose(resultado_fijo.r_c, rc_true, rtol=0.2)
    assert resultado_fijo.r_c_err < rc_true  # error razonable, no explota


def test_fit_beta_error_pocos_puntos_con_p_fijo():
    centros = np.array([0.0, 1.0, 2.0])
    sigma_rm = np.array([5.0, np.nan, np.nan])

    with pytest.raises(ValueError, match="No hay suficientes bins válidos"):
        fit_beta_dispersion(centros, sigma_rm, p_fijo=0.5)

def test_p_vista_frontal_y_lateral_para_beta_dos_tercios():
    from faradaymr.analysis.fitting import p_vista_frontal, p_vista_lateral
    # beta = 2/3 (Tanimura et al. 2020): p = 1 de frente y 0.75 de lado.
    assert np.isclose(p_vista_frontal(2 / 3), 1.0)
    assert np.isclose(p_vista_lateral(2 / 3), 0.75)


def test_fit_beta_rc_fijo_recupera_p():
    from faradaymr.analysis.fitting import beta_dispersion_model, fit_beta_dispersion
    d = np.linspace(40, 860, 11)
    perfil = beta_dispersion_model(d, 0.03, 300.0, 0.8)
    res = fit_beta_dispersion(d, perfil, rc_fijo=300.0)
    assert np.isclose(res.p, 0.8, rtol=1e-5)
    assert np.isclose(res.sigma0, 0.03, rtol=1e-5)
    assert res.r_c == 300.0 and res.r_c_err == 0.0


def test_fit_beta_no_permite_fijar_p_y_rc_a_la_vez():
    from faradaymr.analysis.fitting import fit_beta_dispersion
    d = np.linspace(40, 860, 11)
    with pytest.raises(ValueError):
        fit_beta_dispersion(d, np.exp(-d / 300), p_fijo=0.5, rc_fijo=300.0)


def test_semiancho_media_altura_es_donde_el_perfil_cae_a_la_mitad():
    from faradaymr.analysis.fitting import beta_dispersion_model, semiancho_media_altura
    for p in [0.5, 0.75, 1.0]:
        d12 = semiancho_media_altura(300.0, p)
        assert np.isclose(beta_dispersion_model(d12, 1.0, 300.0, p), 0.5)


def test_fit_ponderado_acepta_errores_por_bin():
    from faradaymr.analysis.fitting import beta_dispersion_model, fit_beta_dispersion
    d = np.linspace(40, 860, 11)
    perfil = beta_dispersion_model(d, 0.03, 300.0, 0.8)
    res = fit_beta_dispersion(d, perfil, rc_fijo=300.0, errores=0.01 * perfil)
    assert np.isclose(res.p, 0.8, rtol=1e-5)
    assert res.p_err > 0
