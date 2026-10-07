from __future__ import annotations

import numpy as np
import astropy.units as u

import config_fisica as cfg_units
from faradaymr.fields import GaussianRandomVectorField

N_BASE = cfg_units.N_BASE
DX_BASE_KPC = cfg_units.DX_BASE.to(u.kpc).value

R_SOLAR_KPC = cfg_units.R_SOLAR.to(u.kpc).value

NE0_CM3 = cfg_units.NE0.to(u.cm**-3).value
SCALE_RADIAL_NE_KPC = cfg_units.SCALE_RADIAL_NE.to(u.kpc).value
SCALE_HEIGHT_NE_KPC = cfg_units.SCALE_HEIGHT_NE.to(u.kpc).value
NE_RADIAL_PROFILE = cfg_units.NE_RADIAL_PROFILE
NE_RADIAL_CUTOFF_KPC = cfg_units.NE_RADIAL_CUTOFF.to(u.kpc).value
N_ARMS = cfg_units.N_ARMS
ARM_R_MIN_KPC = cfg_units.ARM_R_MIN.to(u.kpc).value
ARM_WIDTH_KPC = cfg_units.ARM_WIDTH.to(u.kpc).value
ARM_PHASE0_RAD = np.radians(cfg_units.ARM_PHASE0_DEG)
PITCH_ANGLE_DEG = cfg_units.PITCH_ANGLE_DEG
PITCH_ANGLE_RAD = np.radians(PITCH_ANGLE_DEG)

B0_REGULAR_MG = cfg_units.B0_REGULAR.to(u.microgauss).value
FACTOR_CAMPO_REGULAR = cfg_units.FACTOR_CAMPO_REGULAR
SCALE_RADIAL_B_KPC = cfg_units.SCALE_RADIAL_B.to(u.kpc).value
SCALE_HEIGHT_B_KPC = cfg_units.SCALE_HEIGHT_B.to(u.kpc).value
ANCHO_VERTICAL_B_KPC = cfg_units.ANCHO_VERTICAL_B.to(u.kpc).value
RADIO_NUCLEO_B_KPC = (
    None if cfg_units.RADIO_NUCLEO_B is None else cfg_units.RADIO_NUCLEO_B.to(u.kpc).value
)
RADIO_SIN_CAMPO_B_KPC = (
    None if cfg_units.RADIO_SIN_CAMPO_B is None else cfg_units.RADIO_SIN_CAMPO_B.to(u.kpc).value
)
HANDEDNESS = cfg_units.HANDEDNESS
ANILLOS_INVERSION_CAMPO_KPC = tuple(
    (r_min.to(u.kpc).value, r_max.to(u.kpc).value)
    for r_min, r_max in cfg_units.ANILLOS_INVERSION_CAMPO
)

USAR_CAMPO_HALO = cfg_units.USAR_CAMPO_HALO
B_HALO_NORTE_MG = cfg_units.B_HALO_NORTE.to(u.microgauss).value
B_HALO_SUR_MG = cfg_units.B_HALO_SUR.to(u.microgauss).value
R_HALO_NORTE_KPC = cfg_units.R_HALO_NORTE.to(u.kpc).value
R_HALO_SUR_KPC = cfg_units.R_HALO_SUR.to(u.kpc).value
ANCHO_HALO_KPC = cfg_units.ANCHO_HALO.to(u.kpc).value
ESCALA_ALTURA_HALO_KPC = cfg_units.ESCALA_ALTURA_HALO.to(u.kpc).value
ALTURA_DISCO_HALO_KPC = cfg_units.ALTURA_DISCO_HALO.to(u.kpc).value
ANCHO_DISCO_HALO_KPC = cfg_units.ANCHO_DISCO_HALO.to(u.kpc).value
SENTIDO_HALO_TOROIDAL = cfg_units.SENTIDO_HALO_TOROIDAL
B_CAMPO_X_MG = cfg_units.B_CAMPO_X.to(u.microgauss).value
ELEVACION_CAMPO_X_RAD = np.radians(cfg_units.ELEVACION_CAMPO_X_DEG)
R_CRITICO_CAMPO_X_KPC = cfg_units.R_CRITICO_CAMPO_X.to(u.kpc).value
R_ESCALA_CAMPO_X_KPC = cfg_units.R_ESCALA_CAMPO_X.to(u.kpc).value

SPECTRAL_INDEX = cfg_units.SPECTRAL_INDEX
USAR_ENVOLVENTE_TURBULENCIA = cfg_units.USAR_ENVOLVENTE_TURBULENCIA
ESCALA_RADIAL_TURBULENCIA_KPC = cfg_units.ESCALA_RADIAL_TURBULENCIA.to(u.kpc).value
ESCALA_ALTURA_TURBULENCIA_KPC = cfg_units.ESCALA_ALTURA_TURBULENCIA.to(u.kpc).value
LAMBDA_MIN_KPC = cfg_units.LAMBDA_MIN.to(u.kpc).value
LAMBDA_MAX_KPC = cfg_units.LAMBDA_MAX.to(u.kpc).value

# Amplitud efectiva de la turbulencia (ver la sección de campo magnético de
# config_fisica.py): la RM aleatoria crece como sqrt(L_coh), así que para
# que la turbulencia de kpc de la malla dé la RM de una de ~0.1 kpc se
# escala por sqrt(L_ISM / L_integral_modelo).
L_INTEGRAL_TURBULENCIA_KPC = GaussianRandomVectorField(
    n=N_BASE,
    dx=DX_BASE_KPC,
    spectral_index=SPECTRAL_INDEX,
    scale_min=LAMBDA_MIN_KPC,
    scale_max=LAMBDA_MAX_KPC,
).integral_scale()
FACTOR_TURBULENCIA = float(
    np.sqrt(cfg_units.L_COHERENCIA_ISM.to(u.kpc).value / L_INTEGRAL_TURBULENCIA_KPC)
)
B0_TURBULENTO_LITERATURA_MG = cfg_units.B0_TURBULENTO.to(u.microgauss).value
B0_TURBULENTO_MG = B0_TURBULENTO_LITERATURA_MG * FACTOR_TURBULENCIA

P_SPEC = cfg_units.P_SPEC
NE_REL_FRACCION = cfg_units.NE_REL_FRACCION

NU_HZ = cfg_units.NU.to(u.Hz).value
LAMBDA_ONDA_M = cfg_units.LAMBDA_ONDA.to(u.m).value
NU_PLANCK_HZ = cfg_units.NU_PLANCK.to(u.Hz).value
LAMBDA_PLANCK_M = cfg_units.LAMBDA_PLANCK.to(u.m).value
T_CMB_K = cfg_units.T_CMB.to(u.K).value
N_REALIZACIONES_VARIANZA_GALACTICA = cfg_units.N_REALIZACIONES_VARIANZA_GALACTICA

N_L = cfg_units.N_L
N_B = cfg_units.N_B
B_MAX_DEG = cfg_units.B_MAX_DEG
DL_KPC = cfg_units.DL.to(u.kpc).value
PIXEL_CHUNK_SIZE = cfg_units.PIXEL_CHUNK_SIZE

# pc por kpc: la caja está en kpc pero RM (0.812 * n_e[cm^-3] * B[uG] * dl[pc])
# necesita dl en pc; se pasa a `los_raytrace.sky_map(length_unit_pc=...)`.
KPC_A_PC = 1000.0
