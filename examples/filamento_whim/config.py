from __future__ import annotations
import astropy.units as u

# Se usa importación explícita para evitar colisiones
import examples.filamento_whim.config_fisica as cfg_units

N_BASE = cfg_units.N_BASE
BETA = cfg_units.BETA
MU = cfg_units.MU
N_SPEC = cfg_units.N_SPEC
P_SPEC = cfg_units.P_SPEC

DX_BASE_KPC = cfg_units.DX_BASE.to(u.kpc).value
N0_CM3 = cfg_units.N0.to(u.cm**-3).value
RC_KPC = cfg_units.RC.to(u.kpc).value
B0_MG = cfg_units.B0.to(u.microgauss).value
LAMBDA_MIN_KPC = cfg_units.LAMBDA_MIN.to(u.kpc).value
LAMBDA_MAX_KPC = cfg_units.LAMBDA_MAX.to(u.kpc).value
NU_HZ = cfg_units.NU.to(u.Hz).value
LAMBDA_ONDA_M = cfg_units.LAMBDA_ONDA.to(u.m).value
DX_BASE_PC = cfg_units.DX_BASE.to(u.pc).value