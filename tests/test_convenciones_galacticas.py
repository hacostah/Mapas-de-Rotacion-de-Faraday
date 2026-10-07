"""
Convenciones físicas de la geometría galáctica en las coordenadas del
código (l=0 hacia +x, l=90 hacia +y, phi=arctan2(y, x), Sol en (-R0, 0)).
"""

import numpy as np

import config as cfg
from faradaymr.fields import LogarithmicSpiralField
from faradaymr.simulation import GalacticDiskProfile


def _campo_config():
    return LogarithmicSpiralField(
        b0=cfg.B0_REGULAR_MG, r0=cfg.R_SOLAR_KPC, pitch_angle=cfg.PITCH_ANGLE_RAD,
        scale_radial=cfg.SCALE_RADIAL_B_KPC, scale_height=cfg.SCALE_HEIGHT_B_KPC,
        handedness=cfg.HANDEDNESS,
    )


def test_campo_local_apunta_hacia_l_cercano_a_90_y_espira_hacia_adentro():
    # Manchester 1974, Han et al. 2006: el campo regular local es horario
    # visto desde el polo norte, apunta hacia l ~ 90 y algo hacia el centro.
    bx, by, bz = _campo_config().sample(np.array(-cfg.R_SOLAR_KPC), np.array(0.0), np.array(0.0))
    l_campo = np.degrees(np.arctan2(by, bx))
    assert 70.0 < l_campo < 90.0
    assert bx > 0  # componente hacia el centro galáctico (+x desde el Sol)


def test_brazos_son_trailing_r_crece_con_phi_antihorario():
    # Brazo: phi = ln(r/r0)/tan(p); trailing con rotación horaria => dr/dphi > 0.
    assert np.tan(cfg.PITCH_ANGLE_RAD) > 0
    # Sentido de rotación: el Sol, en (-R0, 0), se mueve hacia +y (l=90),
    # lo que es momento angular L_z < 0 (horario). El campo regular local
    # debe circular en ese mismo sentido.
    _, by, _ = _campo_config().sample(np.array(-cfg.R_SOLAR_KPC), np.array(0.0), np.array(0.0))
    assert by > 0


def test_envolvente_ne2001_vale_uno_en_r0_y_no_diverge_al_centro():
    perfil = GalacticDiskProfile(
        n_e0=0.03, r0=8.0, scale_radial=3.5, scale_height=1.0,
        pitch_angle=np.radians(12.0), arm_contrast=False,
        radial_profile="ne2001", radial_cutoff=17.5,
    )
    assert np.isclose(perfil.density(np.array(8.0), np.array(0.0), np.array(0.0)), 0.03)
    centro = perfil.density(np.array(0.0), np.array(0.0), np.array(0.0))
    assert centro < 0.06  # la exponencial daba ~0.29 cm^-3
    assert perfil.density(np.array(18.0), np.array(0.0), np.array(0.0)) == 0.0
