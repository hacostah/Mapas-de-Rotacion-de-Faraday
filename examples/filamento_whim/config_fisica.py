import astropy.units as u
import astropy.constants as const

# --- Parámetros físicos del filamento WHIM ---
# Densidad electrónica de ~1e-5 cm^-3, congruente con el rango 10^-6 - 10^-4 cm^-3
# para el WHIM filamentario genérico (ej. Tanimura et al. 2020)[cite: 3].
N0 = 1e-5 * (u.cm**-3) 

# Radio del núcleo (core radius). Se usan 300 kpc para permitir que la densidad 
# decaiga correctamente dentro del dominio simulado. Valores empíricos para 
# el core de filamentos suelen ser de decenas a pocos cientos de kpc, 
# a diferencia de la extensión total que sí ronda 1-2 Mpc (Tudorache et al. 2025).
RC = 300.0 * u.kpc  

# Índice de caída radial. Se define en 0.5 para diferenciarlo del perfil más 
# concentrado (2/3) utilizado canónicamente en los cúmulos de galaxias.
BETA = 0.5

# Campo magnético. 40 nG (0.04 µG) representa mediciones observacionales 
# recientes basadas en RM extragalácticas a baja frecuencia (Carretti et al. 2022)[cite: 2].
# (Un valor de 10 nG rozaría el límite inferior teórico; Akahori & Ryu 2010)[cite: 4].
B0 = 0.04 * u.microgauss 

MU = 0.6 # Peso molecular medio aproximado para plasma ionizado

# --- Parámetros de la malla ---
N_BASE = 128
DX_BASE = 20.0 * u.kpc

# --- Parámetros de turbulencia magnética ---
LAMBDA_MIN = 10.0 * u.kpc
LAMBDA_MAX = 500.0 * u.kpc
N_SPEC = 3.0
P_SPEC = 3.0

# --- Parámetros observacionales ---
NU = 1.4e9 * u.Hz
LAMBDA_ONDA = const.c / NU