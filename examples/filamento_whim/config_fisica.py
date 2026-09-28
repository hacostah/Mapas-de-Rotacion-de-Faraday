import astropy.units as u
import astropy.constants as const

# --- Parámetros físicos del filamento WHIM ---
# Densidad electrónica de ~1e-5 cm^-3, congruente con el rango 10^-6 - 10^-4 cm^-3
# para el WHIM filamentario genérico (ej. Tanimura et al. 2020).
N0 = 1e-5 * (u.cm**-3) 

# Radio del núcleo (core radius). Se usan 300 kpc para permitir que la densidad 
# decaiga correctamente dentro del dominio simulado. Valores empíricos para 
# el core de filamentos suelen ser de decenas a pocos cientos de kpc, 
# a diferencia de la extensión total que sí ronda 1-2 Mpc (Tudorache et al. 2025).
RC = 300.0 * u.kpc  

# Longitud fisica del filamento (extensión total)
LONGITUD_FILAMENTO = 2000.0 * u.kpc

# Índice de caída radial. Se define en 0.5 para diferenciarlo del perfil más 
# concentrado (2/3) utilizado canónicamente en los cúmulos de galaxias.
BETA = 0.5

# Campo magnético. 40 nG (0.04 µG) representa mediciones observacionales 
# recientes basadas en RM extragalácticas a baja frecuencia (Carretti et al. 2022).
# (Un valor de 10 nG rozaría el límite inferior teórico; Akahori & Ryu 2010).
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

# --- Ventana de ajuste del perfil transversal ---
# Distancia máxima (desde el eje del filamento) usada para binear y ajustar
# sigma_RM(d). Se fija a un valor FÍSICO independiente de N_BASE/DX_BASE a
# propósito: si en cambio se usa "hasta la mitad de la caja" (como hacía
# antes `run_barrido_theta.py`), el ancho/r_c ajustado queda contaminado
# por el tamaño de la malla en vez de solo por la física del filamento
# (ver discusión de Issue 2). 4*RC es un margen amplio sobre el radio de
# núcleo que sigue cabiendo con margen dentro de la caja por defecto
# (N_BASE*DX_BASE = 128*20 = 2560 kpc; 4*RC = 1200 kpc < 2560/2 = 1280 kpc).
DIST_MAX_AJUSTE = 4.0 * RC