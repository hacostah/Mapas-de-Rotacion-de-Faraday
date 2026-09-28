import numpy as np
import matplotlib

matplotlib.use("Agg")

from faradaymr.plotting_sky import (
    angulo_polarizacion_enmascarado,
    mapa_sintetico_estilo_hammurabi,
)


def test_angulo_enmascarado_oculta_ruido_y_conserva_senal_real():
    # El test de humo de abajo (correr la función completa y verificar que
    # el PNG existe) NO puede fallar por una máscara rota: un umbral
    # invertido, o directamente sin aplicar, igual produce un archivo
    # válido -solo con contenido incorrecto, invisible sin abrir la imagen.
    # Este test verifica el arreglo NUMÉRICO en sí: de 3 celdas, solo una
    # tiene señal polarizada real (P grande); las otras dos son ruido
    # numérico de amplitud mucho menor. Con el umbral por defecto
    # (1e-3 del máximo), las dos celdas de ruido deben quedar en NaN y la
    # celda con señal real debe conservar su ángulo exacto.
    q_map = np.array([1e-9, 1e-9, 1.0])
    u_map = np.array([-2e-9, 3e-9, 0.0])

    p_map, psi_obs = angulo_polarizacion_enmascarado(q_map, u_map)

    assert np.isnan(psi_obs[0])
    assert np.isnan(psi_obs[1])
    assert not np.isnan(psi_obs[2])
    assert np.isclose(psi_obs[2], 0.5 * np.arctan2(0.0, 1.0))
    np.testing.assert_allclose(p_map, np.hypot(q_map, u_map))


def test_angulo_enmascarado_no_enmascara_si_todo_el_mapa_esta_por_encima_del_umbral():
    # Caso opuesto: si TODA la señal es fuerte (uniforme), nada debe
    # quedar enmascarado -el umbral es relativo al máximo del mapa, así
    # que una máscara con el sentido invertido (p.ej. `<=` en vez de `>=`)
    # taparía justo este caso, el más simple de todos.
    q_map = np.full(5, 2.0)
    u_map = np.full(5, 0.0)

    _, psi_obs = angulo_polarizacion_enmascarado(q_map, u_map)

    assert not np.any(np.isnan(psi_obs))


def test_mapa_sintetico_corre_con_escala_log_y_angulo_enmascarado(tmp_path):
    l = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    b = np.linspace(-1.4, 1.4, 19)
    rng = np.random.RandomState(0)
    i_map = 10.0 ** rng.uniform(-12, -9, (36, 19))
    q_map = np.zeros((36, 19))
    u_map = np.zeros((36, 19))
    q_map[18, 9] = 1e-9  # una sola celda con señal polarizada
    rm_map = rng.normal(size=(36, 19))
    ruta = mapa_sintetico_estilo_hammurabi(str(tmp_path), l, b, rm_map, i_map, q_map, u_map)
    assert (tmp_path / "figura_1_mapa_de_cielo_mollweide.png").exists()
    assert ruta.endswith(".png")
