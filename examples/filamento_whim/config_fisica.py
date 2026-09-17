import astropy.units as u
import astropy.constants as const

# --- Parámetros físicos del filamento WHIM ---
# Densidad electrónica de ~10^-5 cm^-3 observada en filamentos inter-cúmulos[cite: 2]
N0 = 1e-5 * (u.cm**-3) 
# Radio característico (core) transversal del filamento
RC = 300.0 * u.kpc
BETA = 0.5
# Campo magnético de 10 nG en filamentos locales[cite: 2]
B0 = 0.01 * u.microgauss 
MU = 0.6 # Peso molecular medio aproximado para plasma ionizado

# --- Parámetros de la malla y refinamiento ---
N_BASE = 128
N_REFINADO = 256
DX_BASE = 20.0 * u.kpc
DX_REFINADO = 10.0 * u.kpc
RADIO_REFINAMIENTO = 500.0 * u.kpc

# --- Parámetros de turbulencia magnética ---
LAMBDA_MIN = 10.0 * u.kpc
LAMBDA_MAX = 500.0 * u.kpc
N_SPEC = 3.0
P_SPEC = 3.0

# --- Parámetros observacionales y de ruido ---
NU = 1.4e9 * u.Hz
LAMBDA_ONDA = const.c / NU
BEAM_FWHM = 50.0 * u.kpc # Haz más amplio debido a la escala extendida del filamento
AGREGAR_RUIDO = True
DESV_EST_RUIDO = 1.0 # Ajustable según el nivel de ruido sintético requerido