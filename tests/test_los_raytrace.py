import numpy as np

from faradaymr import los_raytrace as lr


def test_sample_line_of_sight_interpola_campo_constante():
    # Un campo constante debe interpolar al mismo valor sin importar desde
    # dónde ni hacia dónde se muestree: es el caso más simple para separar
    # un error de escala/indexado de un error real de interpolación.
    valor = 3.5
    field = np.full((20, 20, 20), valor)
    dx = 1.0
    observer_pos = np.array([10.0, 10.0, 10.0])
    direction = np.array([1.0, 0.0, 0.0])

    perfil = lr.sample_line_of_sight(
        field, observer_pos, direction, n_samples=8, dl=0.5, dx=dx, xp=np
    )

    assert np.allclose(perfil, valor)


def test_sample_line_of_sight_interpola_gradiente_lineal_en_x():
    # Campo que vale exactamente la coordenada física x en cada celda
    # (field[i, j, k] = i * dx): un rayo a lo largo de x, arrancando en
    # x0, debe reproducir x0 + n*dl en el punto n-ésimo, sin más error que
    # el de redondeo de la interpolación lineal (que es exacta para un
    # gradiente lineal, no solo aproximada).
    n_celdas = 40
    dx = 2.0
    x_fisico = np.arange(n_celdas) * dx
    field = np.broadcast_to(x_fisico[:, None, None], (n_celdas, n_celdas, n_celdas)).copy()

    observer_pos = np.array([5.0, 3.0, 3.0])
    direction = np.array([1.0, 0.0, 0.0])
    dl = 1.5
    n_samples = 10

    perfil = lr.sample_line_of_sight(
        field, observer_pos, direction, n_samples, dl, dx, xp=np
    )
    esperado = observer_pos[0] + np.arange(n_samples) * dl

    assert np.allclose(perfil, esperado)


def test_ray_box_exit_distance_observador_en_el_centro_ejes_alineados():
    # Desde el centro exacto de una caja cúbica, mirando a lo largo de un
    # eje, la salida tiene que estar a media caja de distancia -el caso
    # más simple posible de intersección rayo-caja.
    box_size = 100.0
    observer_pos = np.array([50.0, 50.0, 50.0])
    direction = np.array([1.0, 0.0, 0.0])

    t_salida = lr.ray_box_exit_distance(observer_pos, direction, box_size, xp=np)

    assert np.isclose(t_salida, 50.0)


def test_ray_box_exit_distance_toma_el_eje_que_limita_primero():
    # Caja no cúbica: el rayo diagonal debe salir por la pared del eje que
    # esté más cerca, no por un promedio de las tres paredes.
    box_size = np.array([10.0, 100.0, 100.0])
    observer_pos = np.array([5.0, 5.0, 5.0])
    direction = np.array([1.0, 0.0, 0.0])  # solo el eje x es restrictivo

    t_salida = lr.ray_box_exit_distance(observer_pos, direction, box_size, xp=np)

    assert np.isclose(t_salida, 5.0)  # (10 - 5) / 1


def test_direction_from_galactic_es_unitaria_y_respeta_convencion():
    # l=0, b=0 debe apuntar exactamente al eje x (convención documentada
    # en el módulo); toda dirección, sea cual sea (l, b), debe ser unitaria.
    d_frente = lr.direction_from_galactic(0.0, 0.0, xp=np)
    assert np.allclose(d_frente, [1.0, 0.0, 0.0])

    rng = np.random.RandomState(0)
    l = rng.uniform(-np.pi, np.pi, size=20)
    b = rng.uniform(-np.pi / 2, np.pi / 2, size=20)
    d = lr.direction_from_galactic(l, b, xp=np)

    assert np.allclose(np.sum(d**2, axis=0), 1.0)


def test_sky_map_rm_analitica_para_campo_uniforme():
    # Con n_e y B_parallel constantes en toda la caja, la RM que ve el
    # observador interior debe coincidir con la misma fórmula analítica
    # que ya valida `los.rotation_measure` para el caso de eje fijo
    # (RM = 0.812 * n_e * B_parallel * n_samples * dl): lo único que
    # cambia es *cómo* se junta el perfil 1D (muestreo con observador
    # interior en vez de un corte a lo largo de un eje), la suma en sí es
    # la misma función sin modificar.
    n_celdas = 60
    dx = 1.0
    box_size = n_celdas * dx
    ne_val = 1e-3
    bz_val = 2.0

    ne = np.full((n_celdas, n_celdas, n_celdas), ne_val)
    bz = np.full((n_celdas, n_celdas, n_celdas), bz_val)
    bx = np.zeros_like(bz)
    by = np.zeros_like(bz)
    ne_rel = np.zeros_like(ne)  # no interesa I/Q/U en esta prueba

    observer_pos = np.array([box_size / 2, box_size / 2, box_size / 2])
    dl = 0.5

    # Mirando exactamente a lo largo del eje z (l=0, b=pi/2): el campo
    # relevante para RM es B_parallel = bz (ver los.rotation_measure, que
    # usa bz como componente paralela cuando el LoS es z).
    l_grid = np.array([0.0])
    b_grid = np.array([np.pi / 2])

    rm_map, _, _, _ = lr.sky_map(
        bx,
        by,
        bz,
        ne,
        ne_rel,
        observer_pos,
        dx,
        box_size,
        l_grid,
        b_grid,
        dl,
        frequency=1.0,
        wavelength=0.0,
        p_index=3.0,
        xp=np,
    )

    t_salida_esperado = box_size - observer_pos[2]  # dirección +z, desde el centro
    n_samples_esperado = int(np.floor((t_salida_esperado * (1 - 1e-6)) / dl))
    rm_esperada = 0.812 * ne_val * bz_val * n_samples_esperado * dl

    assert np.isclose(rm_map[0, 0], rm_esperada)


def test_sky_map_devuelve_forma_correcta():
    n_celdas = 12
    dx = 1.0
    box_size = n_celdas * dx
    rng = np.random.RandomState(2)
    bx, by, bz, ne, ne_rel = rng.uniform(0.1, 1.0, size=(5, n_celdas, n_celdas, n_celdas))

    observer_pos = np.array([box_size / 2, box_size / 2, box_size / 2])
    l_grid = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    b_grid = np.linspace(-np.pi / 3, np.pi / 3, 3)

    rm_map, i_map, q_map, u_map = lr.sky_map(
        bx,
        by,
        bz,
        ne,
        ne_rel,
        observer_pos,
        dx,
        box_size,
        l_grid,
        b_grid,
        dl=1.0,
        frequency=1.4,
        wavelength=0.21,
        p_index=3.0,
        xp=np,
    )

    forma_esperada = (len(l_grid), len(b_grid))
    assert rm_map.shape == forma_esperada
    assert i_map.shape == forma_esperada
    assert q_map.shape == forma_esperada
    assert u_map.shape == forma_esperada
    assert np.all(np.isfinite(rm_map))
    assert np.all(np.isfinite(i_map))
