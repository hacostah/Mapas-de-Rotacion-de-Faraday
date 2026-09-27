import numpy as np

from faradaymr.fields import GaussianRandomVectorField, LogarithmicSpiralField


def _malla_cartesiana(n, dx):
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    return np.meshgrid(coords, coords, coords, indexing="ij")


def test_magnitud_vale_b0_en_r0_y_z_cero():
    # Por construcción, en R=r0 y z=0 la envolvente exponencial vale 1 en
    # ambos factores, así que |B| debe reducirse exactamente a b0 ahí (no
    # aproximadamente: es el punto de anclaje de la normalización).
    campo = LogarithmicSpiralField(
        b0=2.0, r0=8.0, pitch_angle=np.radians(-12.0), scale_radial=5.0, scale_height=1.0
    )
    x = np.array([8.0])
    y = np.array([0.0])
    z = np.array([0.0])
    bx, by, bz = campo.sample(x, y, z, xp=np)

    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    assert np.isclose(b_mag[0], 2.0, rtol=1e-10)


def test_orientacion_coincide_con_el_pitch_angle():
    # La propiedad que define a una espiral logarítmica: en cualquier
    # punto, tan(pitch_angle) = B_R / B_phi. Se verifica proyectando
    # (bx, by) de vuelta a la base polar local (R_hat, phi_hat) en varios
    # puntos del disco, no solo en el radio de referencia.
    pitch = np.radians(-11.5)
    campo = LogarithmicSpiralField(
        b0=1.0, r0=8.0, pitch_angle=pitch, scale_radial=5.0, scale_height=1.0
    )
    x = np.array([8.0, 5.0, -3.0, 0.0])
    y = np.array([0.0, 3.0, 4.0, -6.0])
    z = np.zeros(4)
    bx, by, bz = campo.sample(x, y, z, xp=np)

    phi = np.arctan2(y, x)
    b_radial = bx * np.cos(phi) + by * np.sin(phi)
    b_azimutal = -bx * np.sin(phi) + by * np.cos(phi)

    assert np.allclose(np.arctan2(b_radial, b_azimutal), pitch, atol=1e-10)


def test_magnitud_decae_con_radio_y_altura():
    campo = LogarithmicSpiralField(
        b0=1.0, r0=8.0, pitch_angle=np.radians(-12.0), scale_radial=4.0, scale_height=0.5
    )

    def b_en(x, y, z):
        bx, by, bz = campo.sample(np.array([x]), np.array([y]), np.array([z]), xp=np)
        return np.sqrt(bx**2 + by**2 + bz**2)[0]

    # La envolvente radial es una exponencial decreciente en R (más
    # intenso hacia el centro, como el disco de gas que lo sostiene), no
    # un pico en r0: crece hacia adentro y decae hacia afuera de forma
    # monótona.
    assert b_en(2.0, 0.0, 0.0) > b_en(8.0, 0.0, 0.0) > b_en(14.0, 0.0, 0.0)
    # Alejarse del plano galáctico también.
    assert b_en(8.0, 0.0, 0.0) > b_en(8.0, 0.0, 2.0)


def test_handedness_invierte_el_signo_sin_cambiar_la_magnitud():
    campo_directo = LogarithmicSpiralField(
        b0=1.0, r0=8.0, pitch_angle=np.radians(-12.0), scale_radial=4.0, scale_height=1.0,
        handedness=1,
    )
    campo_invertido = LogarithmicSpiralField(
        b0=1.0, r0=8.0, pitch_angle=np.radians(-12.0), scale_radial=4.0, scale_height=1.0,
        handedness=-1,
    )
    x, y, z = np.array([6.0]), np.array([3.0]), np.array([0.5])

    bx1, by1, bz1 = campo_directo.sample(x, y, z, xp=np)
    bx2, by2, bz2 = campo_invertido.sample(x, y, z, xp=np)

    assert np.allclose((bx1, by1, bz1), (-bx2, -by2, -bz2))


def test_se_suma_directamente_al_campo_turbulento_sin_cambiar_su_interfaz():
    # Criterio de aceptación central: la salida de LogarithmicSpiralField
    # debe poder sumarse celda a celda con GaussianRandomVectorField.sample()
    # tal cual, sin adaptar ninguna de las dos interfaces.
    n, dx = 16, 1.0
    x, y, z = _malla_cartesiana(n, dx)

    turbulento = GaussianRandomVectorField(
        n=n, dx=dx, spectral_index=3.0, scale_min=2.0, scale_max=8.0
    )
    bx_t, by_t, bz_t = turbulento.sample(use_gpu=False, rng=np.random.RandomState(0))

    regular = LogarithmicSpiralField(
        b0=3.0, r0=4.0, pitch_angle=np.radians(-12.0), scale_radial=5.0, scale_height=2.0
    )
    bx_r, by_r, bz_r = regular.sample(x, y, z, xp=np)

    assert bx_t.shape == bx_r.shape == (n, n, n)

    bx, by, bz = bx_t + bx_r, by_t + by_r, bz_t + bz_r

    assert np.all(np.isfinite(bx)) and np.all(np.isfinite(by)) and np.all(np.isfinite(bz))
    # El campo regular es una componente coherente (no promedia a cero como
    # la turbulenta): la suma debe tener una amplitud RMS mayor que la del
    # campo turbulento solo, o el "regular" no estaría aportando nada.
    rms_turbulento = np.sqrt(np.mean(bx_t**2 + by_t**2 + bz_t**2))
    rms_total = np.sqrt(np.mean(bx**2 + by**2 + bz**2))
    assert rms_total > rms_turbulento
