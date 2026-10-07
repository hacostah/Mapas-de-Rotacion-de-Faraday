"""
Tests de `faradaymr.observational`. La mayoría usan datos sintéticos
pequeños (no dependen de descargar nada); los que sí leen los archivos
reales de `data/external/` se saltan automáticamente si no están
presentes (`pytest.importorskip`/existencia de archivo), para que el
suite completo siga corriendo en una copia del repo sin esos datos.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from faradaymr import observational as obs

_TIENE_OPPERMANN = os.path.isfile(obs.RUTA_OPPERMANN2012)
_TIENE_TAYLOR = os.path.isfile(obs.RUTA_TAYLOR2009)
_TIENE_HASLAM = os.path.isfile(obs.RUTA_HASLAM408)
_TIENE_PLANCK = os.path.isfile(obs.RUTA_PLANCK_030GHZ)


def test_subtract_foreground_es_una_resta_simple():
    observado = np.array([[1.0, 2.0], [3.0, 4.0]])
    modelo = np.array([[0.5, 0.5], [0.5, 0.5]])
    residuo = obs.subtract_foreground(observado, modelo)
    np.testing.assert_allclose(residuo, [[0.5, 1.5], [2.5, 3.5]])


def test_parse_sexagesimal_dec_signo_positivo_y_negativo():
    assert obs._parse_sexagesimal_dec("+09 57 06.6") == pytest.approx(9.951833, abs=1e-5)
    assert obs._parse_sexagesimal_dec("-17 04 00.7") == pytest.approx(-17.066861, abs=1e-5)


def test_parse_sexagesimal_dec_signo_entre_menos_uno_y_cero():
    # float("-00") == -0.0, y -0.0 < 0 es False: el caso que un chequeo de
    # signo ingenuo (`d < 0`) perdería silenciosamente.
    assert obs._parse_sexagesimal_dec("-00 43 01.3") == pytest.approx(-0.717028, abs=1e-5)
    assert obs._parse_sexagesimal_dec("+00 43 01.3") == pytest.approx(0.717028, abs=1e-5)


def test_load_taylor2009_catalog_parsea_formato_vizier(tmp_path):
    contenido = (
        "#\n"
        "#   VizieR Astronomical Server\n"
        "RAJ2000\tDEJ2000\tRM\te_RM\tPk\tSi\tm\n"
        '"h:m:s"\t"d:m:s"\trad/m2\trad/m2\tmJy\tmJy\t%\n'
        "-----------\t-----------\t------\t----\t-------\t-------\t------\n"
        "00 00 00.00\t+00 00 00.0\t  10.0\t 2.0\t   5.00\t  100.0\t  5.00\n"
        "12 00 00.00\t-30 00 00.0\t -20.0\t 3.0\t   6.00\t  200.0\t  3.00\n"
        "\n"
    )
    archivo = tmp_path / "catalogo_min.tsv"
    archivo.write_text(contenido)

    catalogo = obs.load_taylor2009_catalog(str(archivo))

    assert catalogo["n_fuentes"] == 2
    assert catalogo["rm"].tolist() == [10.0, -20.0]
    assert catalogo["e_rm"].tolist() == [2.0, 3.0]
    assert np.all(np.abs(catalogo["b_deg"]) <= 90.0)
    assert np.all(np.abs(catalogo["l_deg"]) <= 180.0)
    np.testing.assert_allclose(catalogo["l_rad"], np.radians(catalogo["l_deg"]))


def test_perfil_estadistico_vs_latitud_separa_bandas_correctamente():
    n_l = 4
    b_grid = np.radians(np.array([1.0, 2.0, 40.0, 41.0, 80.0]))
    # mapa (n_l, n_b): cada columna (banda de b) tiene el mismo valor en
    # todas las l, así el RMS de la banda es trivialmente ese valor.
    columnas = np.array([10.0, 10.0, 20.0, 20.0, 5.0])
    mapa = np.tile(columnas, (n_l, 1))
    bins = np.array([0.0, 10.0, 50.0, 90.0])

    perfil = obs.perfil_estadistico_vs_latitud(b_grid, mapa, bins_deg=bins)

    np.testing.assert_allclose(perfil["valores"], [10.0, 20.0, 5.0])
    assert perfil["n_pixeles"].tolist() == [2 * n_l, 2 * n_l, 1 * n_l]


def test_perfil_estadistico_vs_latitud_mean_abs_vs_rms():
    b_grid = np.radians(np.array([1.0, 1.0]))
    mapa = np.array([[3.0, -3.0]])  # una sola l, una sola banda de b
    bins = np.array([0.0, 5.0])

    perfil_rms = obs.perfil_estadistico_vs_latitud(b_grid, mapa, bins_deg=bins, estadistico="rms")
    perfil_abs = obs.perfil_estadistico_vs_latitud(b_grid, mapa, bins_deg=bins, estadistico="mean_abs")

    np.testing.assert_allclose(perfil_rms["valores"], [3.0])
    np.testing.assert_allclose(perfil_abs["valores"], [3.0])


@pytest.mark.skipif(not _TIENE_OPPERMANN, reason="requiere data/external/ (no versionado)")
def test_project_healpix_to_grid_con_mapa_constante():
    healpy = pytest.importorskip("healpy")
    nside = 8
    npix = healpy.nside2npix(nside)
    mapa_constante = np.full(npix, 7.5)

    l_grid = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    b_grid = np.linspace(-np.radians(80), np.radians(80), 7)
    grilla = obs.project_healpix_to_grid(mapa_constante, l_grid, b_grid)

    assert grilla.shape == (12, 7)
    np.testing.assert_allclose(grilla, 7.5, atol=1e-6)


@pytest.mark.skipif(not _TIENE_OPPERMANN, reason="requiere data/external/oppermann2012_galactic_faraday_depth.fits")
def test_load_oppermann2012_map_da_rm_razonable_en_el_polo():
    """
    Chequeo de sanidad contra el mismo rango que ya usa
    `faradaymr.calibration.validar_rm_polo`: la RM galáctica real hacia el
    polo norte debe caer, en orden de magnitud, dentro de unos pocos
    rad/m^2 hasta ~10 rad/m^2 (no cientos, como cerca del plano).
    """
    import healpy as hp

    datos = obs.load_oppermann2012_map()
    assert datos["nside"] == 128
    rm_polo = hp.get_interp_val(datos["rm_healpix"], 0.0, 90.0, lonlat=True)
    assert -20.0 < rm_polo < 20.0


@pytest.mark.skipif(not _TIENE_TAYLOR, reason="requiere data/external/taylor2009_nvss_rm_catalog.tsv")
def test_load_taylor2009_catalog_real_tiene_el_tamano_publicado():
    catalogo = obs.load_taylor2009_catalog()
    assert catalogo["n_fuentes"] == 37543


@pytest.mark.skipif(
    not (_TIENE_OPPERMANN and _TIENE_TAYLOR),
    reason="requiere data/external/ (no versionado)",
)
def test_comparar_con_oppermann_y_catalogo_extremo_a_extremo():
    """
    Extremo a extremo con un mapa sintético trivial (RM=0 en todas
    partes): el residuo frente a cada dato real debe ser, entonces,
    exactamente el dato real -el chequeo de que la composición
    carga+regrillado+resta no introduce un offset o una distorsión
    espuria, sin depender de ningún valor de RM sintético en particular.
    """
    n_l, n_b = 36, 19
    l_grid = np.linspace(-np.pi, np.pi, n_l, endpoint=False)
    b_grid = np.linspace(-np.radians(80), np.radians(80), n_b)
    rm_sim = np.zeros((n_l, n_b))

    resultado_opp = obs.comparar_con_oppermann(l_grid, b_grid, rm_sim)
    np.testing.assert_allclose(resultado_opp["residual"], resultado_opp["rm_obs_grid"])
    assert resultado_opp["rms_modelo"] == 0.0

    resultado_cat = obs.comparar_con_catalogo_taylor(l_grid, b_grid, rm_sim)
    np.testing.assert_allclose(resultado_cat["rm_modelo_interp"], 0.0, atol=1e-9)
    np.testing.assert_allclose(resultado_cat["residual"], resultado_cat["rm_obs"])


def test_restar_planck_recupera_alpha_conocido_en_un_caso_sintetico(tmp_path, monkeypatch):
    """
    Sin descargar nada: construye un HEALPix sintético diminuto (nside=4)
    donde el "Planck real" es, por construcción, EXACTAMENTE
    `alpha_verdadero * mapa_modelo` -así el alpha ajustado por mínimos
    cuadrados debe recuperar `alpha_verdadero` de forma exacta, y el
    residuo debe salir cero en todas partes (el chequeo de que el ajuste
    lineal en sí, no solo la resta, está bien compuesto).
    """
    import healpy as hp
    from astropy.io import fits

    nside = 4
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(0)
    mapa_base = rng.uniform(0.1, 1.0, size=npix)  # siempre positivo, como i_map

    alpha_verdadero = 3.7
    mapa_planck_sintetico = alpha_verdadero * mapa_base

    col = fits.Column(name="I_STOKES", format="E", array=mapa_planck_sintetico.astype(np.float32))
    col_q = fits.Column(name="Q_STOKES", format="E", array=np.zeros(npix, dtype=np.float32))
    col_u = fits.Column(name="U_STOKES", format="E", array=np.zeros(npix, dtype=np.float32))
    hdu = fits.BinTableHDU.from_columns([col, col_q, col_u])
    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = "RING"
    hdu.header["NSIDE"] = nside
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(tmp_path / "planck_sintetico.fits", overwrite=True)

    l_grid = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    b_grid = np.linspace(-np.radians(80), np.radians(80), 13)

    # `i_map` sintético (en la grilla nativa del modelo): se construye
    # regrillando el mismo `mapa_base` HEALPix, así ambos lados de la
    # comparación son, por construcción, la misma forma espacial salvo el
    # factor de escala -el escenario que hace que alpha_ajustado sea
    # exactamente alpha_verdadero, sin ruido de regrillado de por medio.
    i_map_modelo = obs.project_healpix_to_grid(mapa_base, l_grid, b_grid)

    resultado = obs.restar_planck_030ghz(
        l_grid, b_grid, i_map_modelo, path=str(tmp_path / "planck_sintetico.fits"), nside=nside
    )

    # La resta trabaja en K_RJ: el archivo sintético está en K_CMB, como Planck.
    alpha_krj = alpha_verdadero * obs.factor_kcmb_a_krj(obs.NU_PLANCK_030_HZ)
    assert resultado["alpha"] == pytest.approx(alpha_krj, rel=1e-5)
    assert resultado["fondo_k_rj"] == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(resultado["residuo"], 0.0, atol=1e-4)
    assert resultado["fraccion_varianza_explicada"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not _TIENE_PLANCK, reason="requiere data/external/planck_030ghz_iqu.fits")
def test_load_planck_030ghz_map_tiene_la_resolucion_publicada():
    datos = obs.load_planck_030ghz_map()
    assert datos["nside"] == 1024
    assert datos["i_k_rj"].shape == datos["q_k_rj"].shape == datos["u_k_rj"].shape
    assert datos["convencion_polarizacion_origen"] == "COSMO"


def test_factor_kcmb_a_krj_a_28_4_ghz_y_limite_de_baja_frecuencia():
    # x = h nu / k T_CMB = 0.500 a 28.4 GHz -> x^2 e^x / (e^x - 1)^2 = 0.979
    assert obs.factor_kcmb_a_krj(28.4e9) == pytest.approx(0.9794, abs=1e-3)
    # Muy por debajo del pico del CMB las dos temperaturas coinciden.
    assert obs.factor_kcmb_a_krj(1e6) == pytest.approx(1.0, abs=1e-6)


def test_load_planck_pasa_u_de_cosmo_a_iau(tmp_path):
    import healpy as hp
    from astropy.io import fits

    npix = hp.nside2npix(2)
    columnas = [
        fits.Column(name=nombre, format="E", array=np.full(npix, valor, dtype=np.float32))
        for nombre, valor in [("I_STOKES", 1.0), ("Q_STOKES", 0.5), ("U_STOKES", 0.25)]
    ]
    hdu = fits.BinTableHDU.from_columns(columnas)
    hdu.header.update(PIXTYPE="HEALPIX", ORDERING="RING", NSIDE=2, POLCCONV="COSMO")
    ruta = tmp_path / "planck_cosmo.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(ruta)

    datos = obs.load_planck_030ghz_map(str(ruta))
    factor = obs.factor_kcmb_a_krj(obs.NU_PLANCK_030_HZ)
    np.testing.assert_allclose(datos["q_k_rj"], 0.5 * factor, rtol=1e-6)
    np.testing.assert_allclose(datos["u_k_rj"], -0.25 * factor, rtol=1e-6)


def test_ajuste_con_fondo_no_pierde_plantillas_de_amplitud_minuscula():
    # El I del modelo a 28.4 GHz vale ~1e-15 (u.a.): sin normalizar, lstsq
    # tomaba esa columna por cero y devolvía pendiente 0.
    rng = np.random.default_rng(1)
    x = rng.uniform(1e-15, 3e-15, size=500)
    y = 2e10 * x + 3.0
    a, c, r2 = obs._ajuste_lineal_con_fondo(x, y, np.ones_like(x))
    assert a == pytest.approx(2e10, rel=1e-8)
    assert c == pytest.approx(3.0, rel=1e-8)
    assert r2 == pytest.approx(1.0, abs=1e-10)


def test_coherencia_angular_respeta_la_simetria_de_180_grados():
    psi = np.radians(np.array([10.0, 50.0, -70.0]))
    pesos = np.ones(3)
    assert obs.coherencia_angular(psi, psi, pesos) == pytest.approx(1.0)
    assert obs.coherencia_angular(psi, psi + np.pi, pesos) == pytest.approx(1.0)
    assert obs.coherencia_angular(psi, psi + np.pi / 2, pesos) == pytest.approx(-1.0)


def test_rms_rm_alta_latitud_pondera_por_area_y_recorta_en_b():
    import numpy as np
    from faradaymr.calibration import rms_rm_alta_latitud

    b = np.radians(np.array([-80.0, -30.0, 0.0, 30.0, 80.0]))
    rm = np.zeros((4, 5))
    rm[:, [0, 4]] = 10.0  # solo las bandas |b| >= 60
    rm[:, [1, 2, 3]] = 1e6  # deben ignorarse
    assert np.isclose(rms_rm_alta_latitud(rm, b), 10.0)
