from __future__ import annotations

import numpy as np
import astropy.units as u

import config_fisica as cfg_units

N_BASE = cfg_units.N_BASE
DX_BASE_KPC = cfg_units.DX_BASE.to(u.kpc).value

R_SOLAR_KPC = cfg_units.R_SOLAR.to(u.kpc).value

NE0_CM3 = cfg_units.NE0.to(u.cm**-3).value
SCALE_RADIAL_NE_KPC = cfg_units.SCALE_RADIAL_NE.to(u.kpc).value
SCALE_HEIGHT_NE_KPC = cfg_units.SCALE_HEIGHT_NE.to(u.kpc).value
N_ARMS = cfg_units.N_ARMS
ARM_WIDTH_KPC = cfg_units.ARM_WIDTH.to(u.kpc).value
PITCH_ANGLE_DEG = cfg_units.PITCH_ANGLE_DEG
PITCH_ANGLE_RAD = np.radians(PITCH_ANGLE_DEG)

B0_REGULAR_MG = cfg_units.B0_REGULAR.to(u.microgauss).value
SCALE_RADIAL_B_KPC = cfg_units.SCALE_RADIAL_B.to(u.kpc).value
SCALE_HEIGHT_B_KPC = cfg_units.SCALE_HEIGHT_B.to(u.kpc).value
HANDEDNESS = cfg_units.HANDEDNESS

B0_TURBULENTO_MG = cfg_units.B0_TURBULENTO.to(u.microgauss).value
SPECTRAL_INDEX = cfg_units.SPECTRAL_INDEX
LAMBDA_MIN_KPC = cfg_units.LAMBDA_MIN.to(u.kpc).value
LAMBDA_MAX_KPC = cfg_units.LAMBDA_MAX.to(u.kpc).value

P_SPEC = cfg_units.P_SPEC
NE_REL_FRACCION = cfg_units.NE_REL_FRACCION

NU_HZ = cfg_units.NU.to(u.Hz).value
LAMBDA_ONDA_M = cfg_units.LAMBDA_ONDA.to(u.m).value

N_L = cfg_units.N_L
N_B = cfg_units.N_B
B_MAX_DEG = cfg_units.B_MAX_DEG
DL_KPC = cfg_units.DL.to(u.kpc).value
PIXEL_CHUNK_SIZE = cfg_units.PIXEL_CHUNK_SIZE
