import numpy as np

from faradaymr.simulation import GalacticDiskProfile, spiral_arm_density_factor


def test_sin_brazos_en_r0_y_z_cero_vale_n_e0():
    # En R=r0, z=0 ambas exponenciales de la envolvente valen 1, así que
    # sin modulación de brazos (arm_contrast=False) la densidad debe ser
    # exactamente n_e0: es el punto de anclaje de la normalización, igual
    # que en BetaModel.density(0)==n0.
    perfil = GalacticDiskProfile(
        n_e0=0.03, r0=8.0, scale_radial=3.5, scale_height=1.0,
        pitch_angle=np.radians(-12.0), arm_contrast=False,
    )
    densidad = perfil.density(np.array([8.0]), np.array([0.0]), np.array([0.0]), xp=np)
    assert np.isclose(densidad[0], 0.03)


def test_envolvente_decae_con_radio_y_altura():
    perfil = GalacticDiskProfile(
        n_e0=0.03, r0=8.0, scale_radial=3.5, scale_height=1.0,
        pitch_angle=np.radians(-12.0), arm_contrast=False,
    )

    def n_en(x, y, z):
        return perfil.density(np.array([x]), np.array([y]), np.array([z]), xp=np)[0]

    # Disco más denso hacia el centro que en el radio solar, y ese a su
    # vez más denso que más afuera -caída monótona en R, como cualquier
    # disco exponencial real.
    assert n_en(2.0, 0.0, 0.0) > n_en(8.0, 0.0, 0.0) > n_en(14.0, 0.0, 0.0)
    # Y cae al alejarse del plano galáctico (|z| creciente).
    assert n_en(8.0, 0.0, 0.0) > n_en(8.0, 0.0, 2.0)


def test_factor_espiral_vale_maximo_sobre_la_curva_del_brazo():
    # Construcción central de esta función: phi_brazo(r) = ln(r/r0)/tan(p)
    # (ver derivación en el docstring del módulo, consistente con la
    # convención tan(pitch)=B_R/B_phi de LogarithmicSpiralField). Si se
    # evalúa el factor exactamente sobre esa curva, para un solo brazo
    # (n_arms=1), la distancia angular al brazo debe ser cero y el factor
    # debe alcanzar su máximo posible (1 + 1 = 2), para cualquier radio,
    # no solo en r0.
    pitch = np.radians(-12.0)
    r0 = 8.0
    radios = np.array([3.0, 8.0, 15.0])
    fases = np.log(radios / r0) / np.tan(pitch)  # phi_brazo(r), fase=0

    xx = radios * np.cos(fases)
    yy = radios * np.sin(fases)

    factor = spiral_arm_density_factor(xx, yy, pitch, r0, n_arms=1, arm_width=0.5, xp=np)

    assert np.allclose(factor, 2.0, atol=1e-6)


def test_factor_espiral_decae_lejos_del_brazo():
    # A mitad de camino en fase entre dos brazos (n_arms=2 -> brazos
    # separados pi radianes) el punto está lo más lejos posible de ambos:
    # el factor ahí debe ser sensiblemente menor que sobre cualquiera de
    # los dos brazos, para el mismo radio.
    pitch = np.radians(-12.0)
    r0 = 8.0
    r = 8.0
    fase_brazo_0 = np.log(r / r0) / np.tan(pitch)

    xx_sobre_brazo = np.array([r * np.cos(fase_brazo_0)])
    yy_sobre_brazo = np.array([r * np.sin(fase_brazo_0)])
    xx_entre_brazos = np.array([r * np.cos(fase_brazo_0 + np.pi / 2)])
    yy_entre_brazos = np.array([r * np.sin(fase_brazo_0 + np.pi / 2)])

    factor_sobre_brazo = spiral_arm_density_factor(
        xx_sobre_brazo, yy_sobre_brazo, pitch, r0, n_arms=2, arm_width=0.5, xp=np
    )
    factor_entre_brazos = spiral_arm_density_factor(
        xx_entre_brazos, yy_entre_brazos, pitch, r0, n_arms=2, arm_width=0.5, xp=np
    )

    assert factor_sobre_brazo[0] > factor_entre_brazos[0]


def test_arm_contrast_false_ignora_brazos():
    perfil_con_brazos = GalacticDiskProfile(
        n_e0=0.03, r0=8.0, scale_radial=3.5, scale_height=1.0,
        pitch_angle=np.radians(-12.0), n_arms=4, arm_width=0.5, arm_contrast=True,
    )
    perfil_sin_brazos = GalacticDiskProfile(
        n_e0=0.03, r0=8.0, scale_radial=3.5, scale_height=1.0,
        pitch_angle=np.radians(-12.0), arm_contrast=False,
    )

    xx = np.array([5.0, 8.0, 11.0])
    yy = np.array([1.0, -2.0, 3.0])
    zz = np.zeros(3)

    con_brazos = perfil_con_brazos.density(xx, yy, zz, xp=np)
    sin_brazos = perfil_sin_brazos.density(xx, yy, zz, xp=np)

    # El factor espiral es siempre >=1 por construcción (una suma de 1 más
    # términos gaussianos no negativos), así que la versión con brazos
    # nunca puede ser menor que la envolvente sola.
    assert np.all(con_brazos >= sin_brazos - 1e-12)
    assert np.any(con_brazos > sin_brazos)


def test_phase0_rota_los_brazos_sin_cambiar_su_espaciado():
    # phase0 debe desplazar los n_arms brazos como un bloque rígido: el
    # factor con phase0=delta en la posición phi es idéntico al factor con
    # phase0=0 evaluado en phi-delta (una rotación pura, sin cambiar forma
    # ni espaciado entre brazos).
    pitch = np.radians(-12.0)
    r0 = 8.0
    r = 8.0
    delta = np.pi / 4

    fase_prueba = 0.7
    xx_con_phase0 = np.array([r * np.cos(fase_prueba)])
    yy_con_phase0 = np.array([r * np.sin(fase_prueba)])
    xx_sin_phase0 = np.array([r * np.cos(fase_prueba - delta)])
    yy_sin_phase0 = np.array([r * np.sin(fase_prueba - delta)])

    factor_con = spiral_arm_density_factor(
        xx_con_phase0, yy_con_phase0, pitch, r0, n_arms=4, arm_width=0.5,
        phase0=delta, xp=np,
    )
    factor_sin = spiral_arm_density_factor(
        xx_sin_phase0, yy_sin_phase0, pitch, r0, n_arms=4, arm_width=0.5,
        phase0=0.0, xp=np,
    )
    assert np.allclose(factor_con, factor_sin)


def test_arm_phase0_config_pone_al_observador_lo_mas_lejos_posible_de_un_brazo():
    # Regresión del bug reportado: con phase0=0 (el default de la función)
    # y N_ARMS=4, un brazo cae exactamente en phi=180°, que es DONDE
    # `model.construir_escenario` coloca al observador (desplazado -R_SOLAR
    # a lo largo de x desde el centro galáctico) -el Sol termina sobre la
    # cresta de un brazo, n_e local sale 2x el valor de referencia
    # "interbrazo" que declara NE0 en config_fisica.py. `ARM_PHASE0_DEG`
    # (180/N_ARMS) corrige esto poniendo al observador exactamente a mitad
    # de camino entre dos brazos -el mínimo local del factor espiral.
    pitch = np.radians(-12.0)
    r0 = 8.0
    n_arms = 4
    phase0_corregido = np.radians(180.0 / n_arms)

    # Observador: phi=180° (x=-r0, y=0), mismo r0 que el radio de referencia.
    xx_obs = np.array([-r0])
    yy_obs = np.array([0.0])

    factor_sin_correccion = spiral_arm_density_factor(
        xx_obs, yy_obs, pitch, r0, n_arms=n_arms, arm_width=0.5, phase0=0.0, xp=np
    )
    factor_corregido = spiral_arm_density_factor(
        xx_obs, yy_obs, pitch, r0, n_arms=n_arms, arm_width=0.5,
        phase0=phase0_corregido, xp=np,
    )

    # Antes de corregir: el observador cae justo sobre un brazo (factor
    # máximo posible, 2.0, ver test_factor_espiral_vale_maximo_sobre_la_curva_del_brazo).
    assert np.isclose(factor_sin_correccion[0], 2.0, atol=1e-6)
    # Con la corrección: el observador queda en el mínimo local posible
    # del factor espiral (estrictamente menor que sobre cualquier brazo).
    assert factor_corregido[0] < factor_sin_correccion[0]
    assert factor_corregido[0] < 1.1


def test_callable_es_azucar_sintactica_de_density():
    perfil = GalacticDiskProfile(
        n_e0=0.03, r0=8.0, scale_radial=3.5, scale_height=1.0,
        pitch_angle=np.radians(-12.0),
    )
    xx, yy, zz = np.array([8.0]), np.array([0.0]), np.array([0.0])
    assert np.allclose(perfil(xx, yy, zz, xp=np), perfil.density(xx, yy, zz, xp=np))
