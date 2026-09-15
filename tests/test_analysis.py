import numpy as np
import pytest

from faradaymr.analysis import radial_profile, transverse_rm_dispersion
from faradaymr.simulation.geometry import projected_axis_distance

def test_mapa_uniforme_no_tiene_dispersion():
    # Un mapa de RM perfectamente uniforme no tiene ninguna fluctuacion
    # que medir: sea cual sea el binning en distancia, la desviacion
    # estandar dentro de cada bin tiene que dar exactamente cero. Este es
    # el caso mas simple posible y sirve para descartar un bug de escala
    # o de indexado en el binning antes de probar casos mas realistas.
    mapa = np.full((20, 20), 5.0)
    xx, yy = np.meshgrid(np.arange(20) - 10, np.arange(20) - 10)
    distancia = np.sqrt(xx**2 + yy**2)

    _, valores = radial_profile(mapa, distancia, bins=5, statistic="std")

    assert np.allclose(valores, 0.0)


def test_perfil_recupera_una_relacion_lineal_exacta():
    # Si el observable depende de la distancia de forma exactamente
    # lineal (map2d = 3*d + 1, sin ruido), el promedio dentro de cada bin
    # tiene que reproducir esa recta evaluada en el centro del bin, sin
    # ningun error de por medio. Se eligen bordes de bin que coinciden
    # exactamente con los valores de distancia usados, para que el
    # promedio de cada bin sea un numero exacto y no una aproximacion.
    distancia = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    mapa = 3.0 * distancia + 1.0
    bordes = [-0.5, 0.5, 1.5, 2.5, 3.5]

    centros, valores = radial_profile(mapa, distancia, bins=bordes, statistic="mean")

    assert np.allclose(centros, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(valores, 3.0 * centros + 1.0)


def test_centros_de_bin_son_el_punto_medio_geometrico_de_los_bordes():
    # El "centro" que devuelve la funcion es una convencion geometrica
    # (punto medio de los bordes del bin), no un centroide pesado por los
    # datos que caen adentro. Vale la pena dejarlo explicito en un test
    # porque es justo lo que despues se usa como eje x al graficar el
    # perfil transversal de RM del Proyecto II.
    distancia = np.linspace(0, 10, 1000)
    mapa = np.zeros_like(distancia)
    bordes = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    centros, _ = radial_profile(mapa, distancia, bins=bordes)

    assert np.allclose(centros, [1.0, 3.0, 5.0, 7.0, 9.0])


def test_bin_sin_puntos_da_nan_no_cero():
    # Un bin de distancia que no contiene ningun pixel del mapa no es lo
    # mismo que un bin con dispersion cero: es, fisicamente, "no hay
    # medicion ahi". Confundir ambos casos (por ejemplo si el codigo
    # devolviera 0.0 en vez de NaN) haria pensar que el campo es
    # perfectamente uniforme en una zona que en realidad no fue muestreada.
    distancia = np.array([0.0, 0.5, 1.0])
    mapa = np.array([1.0, 2.0, 3.0])

    _, valores = radial_profile(mapa, distancia, bins=[0.0, 1.0, 2.0, 3.0])

    assert np.isnan(valores[-1])


def test_transverse_rm_dispersion_alineado_eje_x():
    pixel_size = 1.0
    ny, nx = 5, 5
    rm_map = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            # filamento a lo largo de x (eje 0): la distancia perpendicular
            # varía con j (eje 1 = y), no con i.
            rm_map[i, j] = abs(j - 2) * pixel_size

    filament_axis_3d = [1.0, 0.0, 0.0]
    bins = np.array([-0.5, 0.5, 1.5, 2.5])

    bin_centers, rm_dispersion = transverse_rm_dispersion(
        rm_map, filament_axis_3d, pixel_size, bins, xp=np
    )

    np.testing.assert_allclose(bin_centers, [0.0, 1.0, 2.0], atol=1e-7)
    # todos los píxeles de una misma banda perpendicular tienen el mismo
    # RM exacto: la dispersión transversal debe ser cero.
    np.testing.assert_allclose(rm_dispersion, [0.0, 0.0, 0.0], atol=1e-7)


def test_transverse_rm_dispersion_proyeccion_z_degenerada():
    rm_map = np.ones((5, 5))
    filament_axis_3d = [0.0, 0.0, 1.0]  # proyección (x,y) nula
    resultado = transverse_rm_dispersion(rm_map, filament_axis_3d, 1.0, bins=3, xp=np)
    assert resultado is not None
    assert not np.any(np.isnan(resultado[1]))


def test_transverse_rm_dispersion_no_duplica_logica():
    # transverse_rm_dispersion no debe tener lógica propia: es, por
    # definición, projected_axis_distance + radial_profile(statistic="std").
    rng = np.random.default_rng(3)
    rm_map = rng.normal(size=(30, 30))
    filament_axis_3d = [np.sin(0.4), 0.0, np.cos(0.4)]
    pixel_size = 2.0
    bins = 6

    distance_map = projected_axis_distance(rm_map.shape, filament_axis_3d, pixel_size, xp=np)
    centros_esperados, valores_esperados = radial_profile(
        rm_map, distance_map, bins, statistic="std"
    )
    centros, valores = transverse_rm_dispersion(rm_map, filament_axis_3d, pixel_size, bins)

    assert np.allclose(centros, centros_esperados)
    assert np.allclose(valores, valores_esperados, equal_nan=True)


def test_dispersion_transversal_decrece_al_alejarse_del_eje_del_filamento():
    # Mismo observable físico de antes (Proyecto II), ahora sobre un mapa
    # 2D real y pasando por la API definitiva (eje 3D + pixel_size), en vez
    # de fabricar a mano un arreglo de "distancias": cubre también la
    # proyección geométrica, no solo el binning.
    rng = np.random.default_rng(42)
    pixel_size = 1.0
    n = 400
    filament_axis_3d = [0.0, 1.0, 0.0]  # filamento a lo largo de "y"

    distance_map = projected_axis_distance((n, n), filament_axis_3d, pixel_size, xp=np)
    amplitud_0, escala = 50.0, 60.0
    amplitud = amplitud_0 * np.exp(-distance_map / escala)
    rm_map = amplitud * rng.normal(size=(n, n))

    bordes = np.linspace(0.0, distance_map.max(), 6)
    centros, dispersion = transverse_rm_dispersion(
        rm_map, filament_axis_3d, pixel_size, bordes
    )

    dispersion_teorica = amplitud_0 * np.exp(-centros / escala)
    assert np.allclose(dispersion, dispersion_teorica, rtol=0.1)
    assert np.all(np.diff(dispersion) < 0)