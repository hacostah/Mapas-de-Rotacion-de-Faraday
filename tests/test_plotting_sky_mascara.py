import numpy as np
import matplotlib

matplotlib.use("Agg")

from faradaymr.plotting_sky import mapa_sintetico_estilo_hammurabi


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
