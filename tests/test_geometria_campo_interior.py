"""
Geometría del campo regular y de los brazos en la Galaxia interior y en la
transición disco-halo (ver config_fisica.py): inversión del campo del
disco dentro de R = 7 kpc, núcleo de Sun et al. (2008) / JF12, corte
vertical de JF12 y radio interior de los brazos.
"""

from __future__ import annotations

import numpy as np

from faradaymr.fields import LogarithmicSpiralField
from faradaymr.simulation.galactic_disk import spiral_arm_density_factor


def _campo(**kwargs):
    parametros = dict(
        b0=2.0, r0=8.0, pitch_angle=np.radians(12.0), scale_radial=5.0,
        scale_height=0.4, handedness=-1,
    )
    parametros.update(kwargs)
    return LogarithmicSpiralField(**parametros)


def _b_phi(campo, radio, z=0.0):
    # Sobre el eje -x (phi = 180°), e_phi = (0, -1): B_phi = -B_y.
    x, y, zz = np.array([-radio]), np.array([0.0]), np.array([z])
    return float(-campo.sample(x, y, zz)[1][0])


def test_anillo_invertido_cambia_el_sentido_solo_adentro():
    campo = _campo(anillos_invertidos=((0.0, 7.0),))
    referencia = _campo()
    assert _b_phi(campo, 8.0) == _b_phi(referencia, 8.0) < 0  # horario en el Sol
    assert _b_phi(campo, 6.0) == -_b_phi(referencia, 6.0) > 0  # antihorario adentro


def test_nucleo_plano_y_sin_campo_en_la_barra():
    campo = _campo(radio_nucleo=5.0, radio_sin_campo=3.0)
    assert np.isclose(_b_phi(campo, 4.0), _b_phi(campo, 5.0))
    assert _b_phi(campo, 2.0) == 0.0
    assert np.isclose(_b_phi(campo, 8.0), _b_phi(_campo(), 8.0))


def test_corte_vertical_jf12_conserva_la_columna():
    exponencial = _campo()
    jf12 = _campo(ancho_vertical=0.27)
    z = np.linspace(0.0, 5.0, 5001)
    columna = [
        np.trapezoid([abs(_b_phi(c, 8.0, zz)) for zz in z], z) for c in (exponencial, jf12)
    ]
    assert np.isclose(columna[1], columna[0], rtol=0.1)


def test_brazos_apagados_dentro_de_r_min():
    x = np.array([0.5, 1.0, 6.0, 8.0])
    y = np.zeros_like(x)
    sin_corte = spiral_arm_density_factor(x, y, np.radians(12.0), 8.0, arm_width=0.5)
    con_corte = spiral_arm_density_factor(x, y, np.radians(12.0), 8.0, arm_width=0.5, r_min=3.0)
    assert np.all(con_corte[:2] < 1.01)
    assert np.allclose(con_corte[2:], sin_corte[2:], rtol=1e-3)
