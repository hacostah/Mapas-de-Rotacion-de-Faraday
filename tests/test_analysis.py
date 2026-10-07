import numpy as np

from faradaymr.analysis import radial_profile, transverse_rm_dispersion


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


def test_transverse_rm_dispersion_es_radial_profile_con_std():
    # transverse_rm_dispersion no deberia tener ninguna logica propia:
    # es, por definicion, radial_profile fijando statistic="std". Si en
    # el futuro alguien "optimiza" una de las dos por separado y quedan
    # desincronizadas, este test tiene que fallar.
    rng = np.random.default_rng(0)
    mapa_rm = rng.normal(size=(50, 50))
    xx, yy = np.meshgrid(np.arange(50) - 25, np.arange(50) - 25)
    distancia = np.abs(xx.astype(float))  # distancia a un eje, no a un punto

    centros_a, valores_a = radial_profile(mapa_rm, distancia, bins=8, statistic="std")
    centros_b, valores_b = transverse_rm_dispersion(mapa_rm, distancia, bins=8)

    assert np.allclose(centros_a, centros_b)
    assert np.allclose(valores_a, valores_b, equal_nan=True)


def test_dispersion_transversal_decrece_al_alejarse_del_eje_del_filamento():
    # Este es el observable fisico real del Proyecto II: un filamento
    # turbulento donde la amplitud del campo (y por lo tanto de RM) cae
    # con la distancia al eje. Se arma un RM sintetico como
    # amplitud(d) * ruido_gaussiano_de_media_cero, con amplitud
    # exponencialmente decreciente -- el promedio de RM da ~0 en todos
    # lados (el campo turbulento no tiene direccion privilegiada, tal
    # como dice el abstract), pero su *dispersion* si debe caer con la
    # distancia al eje. Se usa una muestra grande para que el ruido
    # estadistico del propio estimador de std no tape la tendencia.
    rng = np.random.default_rng(42)
    distancia = rng.uniform(0.0, 10.0, size=200_000)
    amplitud_0, escala = 50.0, 3.0
    amplitud = amplitud_0 * np.exp(-distancia / escala)
    rm_sintetico = amplitud * rng.normal(size=distancia.size)

    bordes = np.linspace(0.0, 10.0, 6)
    centros, dispersion = transverse_rm_dispersion(rm_sintetico, distancia, bins=bordes)

    # la dispersion medida en cada bin debe acercarse a la amplitud
    # teorica evaluada en el centro del bin (dentro de un 10%, que es
    # margen mas que suficiente con 40000 puntos por bin en promedio)
    dispersion_teorica = amplitud_0 * np.exp(-centros / escala)
    assert np.allclose(dispersion, dispersion_teorica, rtol=0.1)

    # y sobre todo: la tendencia tiene que ser monotona decreciente,
    # que es la firma observacional que el proyecto busca medir
    assert np.all(np.diff(dispersion) < 0)


def test_radial_profile_acepta_bins_como_entero_o_como_bordes_explicitos():
    # scipy.stats.binned_statistic admite pasar solo el numero de bins
    # (bordes automaticos equiespaciados) o los bordes exactos. El
    # framework no le agrega restricciones propias a esa flexibilidad,
    # asi que ambas formas de uso tienen que seguir funcionando.
    rng = np.random.default_rng(1)
    distancia = rng.uniform(0, 5, size=500)
    mapa = rng.normal(size=500)

    centros_auto, valores_auto = radial_profile(mapa, distancia, bins=4)
    centros_manual, valores_manual = radial_profile(
        mapa, distancia, bins=np.linspace(distancia.min(), distancia.max(), 5)
    )

    assert centros_auto.shape == (4,)
    assert valores_auto.shape == (4,)
    assert np.allclose(centros_auto, centros_manual)
    assert np.allclose(valores_auto, valores_manual, equal_nan=True)


def test_espectro_isotropico_recupera_parseval_y_pendiente_de_kolmogorov():
    # El espectro de energía debe (a) integrar a <|B|^2> y (b) tener la
    # pendiente -n impuesta dentro de la banda con turbulencia.
    from faradaymr import GaussianRandomVectorField
    from faradaymr.analysis.spatial_stats import isotropic_energy_spectrum

    n, dx = 48, 0.5
    campo = GaussianRandomVectorField(
        n=n, dx=dx, spectral_index=5.0 / 3.0, scale_min=1.0, scale_max=6.0
    )
    bx, by, bz = campo.sample(use_gpu=False, rng=np.random.RandomState(1))

    k, e_k = isotropic_energy_spectrum(bx, by, bz, dx)

    dk = k[1] - k[0]
    varianza = float(np.mean(bx**2 + by**2 + bz**2))
    assert np.isclose(np.sum(e_k * dk), varianza, rtol=1e-6)

    banda = (k > np.pi / 6.0) & (k < np.pi / 1.0)
    pendiente = np.polyfit(np.log(k[banda]), np.log(e_k[banda]), 1)[0]
    assert abs(pendiente + 5.0 / 3.0) < 0.35
