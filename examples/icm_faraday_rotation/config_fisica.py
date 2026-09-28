import astropy.units as u

# --- Observatorio: respuesta instrumental ---
BEAM_FWHM = 50.0 * u.kpc  # resolución angular del beam, en distancia física
DESV_EST_RUIDO = 0.01  # desviación estándar del ruido instrumental (misma escala que los mapas, adimensional)
AGREGAR_RUIDO = True  # activa/desactiva el ruido instrumental

# --- Malla ---
N_BASE = 128  # celdas por lado de la caja cúbica base (adimensional)
N_REFINADO = 256  # celdas por lado de una malla refinada (adimensional)
DX_BASE = 10.0 * u.kpc  # tamaño de celda de la malla base
DX_REFINADO = 3.0 * u.kpc  # tamaño de celda de una malla refinada
RADIO_REFINAMIENTO = 200.0 * u.kpc  # radio dentro del cual aplicaría el refinamiento

# --- Medio intracúmulo (ICM): perfil beta ---
N0 = 1e-3 * (u.cm**-3)  # densidad central de electrones térmicos
RC = 400.0 * u.kpc  # radio de núcleo del perfil beta
BETA = 0.6  # índice del perfil beta (adimensional)
B0 = 1.0 * u.microgauss  # intensidad típica (RMS) del campo magnético
MU = 0.5  # fracción molecular (adimensional; no usada aún por `model.py`)

# --- Turbulencia magnética ---
LAMBDA_MIN = 6.0 * u.kpc  # escala de disipación (remolinos más chicos)
LAMBDA_MAX = 768.0 * u.kpc  # escala de inyección (remolinos más grandes)
N_SPEC = 3.0  # índice del espectro de potencia magnético, P(k) ∝ k^-n
P_SPEC = 3.0  # índice de energía de los electrones relativistas, dN/dE ∝ E^-p

# --- Observación ---
NU = 1.4e9 * u.Hz  # frecuencia de observación (banda L, típica en RM)
C = 3e8 * (u.m / u.s)  # velocidad de la luz
LAMBDA_ONDA = C / NU  # longitud de onda de observación, derivada de NU
