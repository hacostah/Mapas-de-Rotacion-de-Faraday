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


def test_radial_profile_con_mascara_excluye_pixeles():
    # `mascara` debe descartar por completo los píxeles marcados con False,
    # cambiando el estadístico calculado en el bin (no solo ponerlos a NaN
    # dentro del cálculo, que seguiría afectando media/std del bin).
    distancia = np.array([0.0, 0.0, 1.0, 1.0])
    mapa = np.array([10.0, -10.0, 100.0, 5.0])  # bin 1 con outlier 100.0
    # Al excluir el outlier, el bin 1 queda con un solo punto (std=0, no NaN:
    # NaN es para un bin que queda TOTALMENTE vacío, ver el siguiente test).
    mascara = np.array([True, True, False, True])

    _, valores_sin_mascara = radial_profile(mapa, distancia, bins=[-0.5, 0.5, 1.5], statistic="std")
    _, valores_con_mascara = radial_profile(
        mapa, distancia, bins=[-0.5, 0.5, 1.5], statistic="std", mascara=mascara
    )

    assert valores_sin_mascara[1] > 0.0  # el outlier infla el std sin máscara
    assert np.isclose(valores_con_mascara[1], 0.0)  # un solo punto: std=0
    assert np.isclose(valores_con_mascara[0], valores_sin_mascara[0])


def test_radial_profile_con_mascara_bin_vacio_da_nan():
    distancia = np.array([0.0, 0.0, 1.0, 1.0])
    mapa = np.array([10.0, -10.0, 5.0, 5.0])
    mascara = np.array([True, True, False, False])  # bin 1 queda totalmente vacío

    _, valores = radial_profile(
        mapa, distancia, bins=[-0.5, 0.5, 1.5], statistic="std", mascara=mascara
    )

    assert np.isnan(valores[1])


def test_transverse_rm_dispersion_footprint_mask_evita_dilucion():
    # Reproduce el bug de dilución: un filamento finito deja RM=0 en los
    # píxeles más allá de sus puntas proyectadas. Sin excluirlos
    # (footprint_mask), esos ceros se mezclan con el ruido real del
    # filamento en el mismo bin de distancia y bajan el std medido.
    #
    # Con axis=[1,0,0] (eje 0 = "a lo largo del filamento"), la distancia
    # transversal que calcula `projected_axis_distance` depende solo del eje
    # 1 ("y"); un mismo bin de distancia agrupa TODAS las filas del eje 0
    # ("x", posición a lo largo del filamento) que caigan en esa banda de y.
    # Simulamos la longitud finita restringiendo el ruido real a una franja
    # de filas en x (|x-centro|<=5) y dejando el resto en 0 -exactamente lo
    # que hace la máscara axial de `model.construir_escenario` para un
    # filamento visto de lado.
    pixel_size = 1.0
    n = 41
    filament_axis_3d = [1.0, 0.0, 0.0]

    rng = np.random.default_rng(7)
    rm_map = np.zeros((n, n))
    footprint = np.zeros((n, n), dtype=bool)
    centro = n // 2
    for i in range(centro - 5, centro + 6):
        rm_map[i, :] = rng.normal(scale=3.0, size=n)
        footprint[i, :] = True

    bordes = np.linspace(0, centro, 6)
    _, dispersion_diluida = transverse_rm_dispersion(
        rm_map, filament_axis_3d, pixel_size, bordes
    )
    _, dispersion_corregida = transverse_rm_dispersion(
        rm_map, filament_axis_3d, pixel_size, bordes, footprint_mask=footprint
    )

    # La dispersión sin máscara está diluida por las filas en 0 fuera del
    # tramo finito (30 de 41 filas, en este mapa), así que sistemáticamente
    # da un valor menor que excluyéndolas.
    validos = ~np.isnan(dispersion_diluida) & ~np.isnan(dispersion_corregida)
    assert np.any(validos)
    assert np.all(dispersion_diluida[validos] < dispersion_corregida[validos])


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