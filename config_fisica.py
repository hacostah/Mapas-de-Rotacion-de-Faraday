"""
Parámetros físicos del escenario "Vía Láctea vista desde dentro" (Proyecto
III: foreground galáctico).

Cada número acá debe poder señalar una referencia (ver `plan_faradaymr.md`,
criterio 0.1 de "qué hace que un toy model sea de tesis"): un parámetro sin
anclaje es el primero que tumba un comité. Los que todavía no tienen una
calibración cuantitativa contra un dato real (más allá del orden de
magnitud de la referencia citada) quedan marcados como tal -esa
calibración es, precisamente, el siguiente issue sugerido tras este
`run.py` (ver notas de la corrida).
"""

import astropy.units as u

# --- Malla ---
# Una caja de 32 kpc de lado (64 celdas de 0.5 kpc) alcanza para contener
# el radio solar (8 kpc) con margen hasta más allá del disco óptico
# (~15 kpc), sin gastar resolución en un volumen que no es el objeto de
# este proyecto -a diferencia del ICM, que sí necesita cientos de kpc de
# lado. Mismo tamaño de caja/celda que ya se validó en la prueba de
# inspección visual del ray tracer (`tests/test_los_raytrace_visual.py`).
N_BASE = 64
DX_BASE = 0.5 * u.kpc

# --- Observador ---
# Radio galactocéntrico solar estándar en la literatura de campo
# magnético galáctico (Jansson & Farrar 2012; Sun et al. 2008 usan el
# mismo valor redondo).
R_SOLAR = 8.0 * u.kpc

# --- Densidad de electrones térmicos: disco + brazos espirales ---
# Densidad local (en R=R_solar, z=0, fuera de cualquier brazo): valor de
# referencia de NE2001 en la posición del Sol (Cordes & Lazio 2002).
NE0 = 0.03 * (u.cm**-3)
# Escala radial del disco ionizado delgado: orden de magnitud de la
# componente de disco fino de NE2001 (Cordes & Lazio 2002).
SCALE_RADIAL_NE = 3.5 * u.kpc
# Escala vertical del disco ionizado (Gaensler et al. 2008).
SCALE_HEIGHT_NE = 1.0 * u.kpc
# Número de brazos espirales (modelo de 4 brazos, Vallée 2016).
N_ARMS = 4
# Ancho gaussiano (1 sigma) del realce de densidad cerca de cada brazo.
# No hay, todavía, una referencia cuantitativa específica para este
# parámetro -queda como uno de los números "sin anclaje fino" que señala
# el criterio 0.1 del plan; se usa el mismo orden de magnitud propuesto en
# `carencias_framework_faradaymr.md` (issue "Modelo de estructura espiral").
ARM_WIDTH = 0.5 * u.kpc
# Pitch angle observacional (medido desde la dirección azimutal, no
# radial -ver docstring de `faradaymr.simulation.galactic_disk`), espiral
# trailing (signo negativo): Vallée (2015, 2016).
PITCH_ANGLE_DEG = -12.0

# --- Campo magnético regular (espiral logarítmica) ---
# Campo local, mismo orden que la componente de disco de JF12 (Jansson &
# Farrar 2012) y consistente con el valor citado en Cordes & Lazio (2002).
B0_REGULAR = 2.0 * u.microgauss
# Escala radial de decaimiento del campo regular (mismo orden que el
# disco de JF12).
SCALE_RADIAL_B = 5.0 * u.kpc
# Espesor del disco magnetizado (JF12).
SCALE_HEIGHT_B = 1.0 * u.kpc
HANDEDNESS = 1  # sentido de enrollamiento; no fijado por ninguna
                 # referencia particular en este toy model (ver nota en
                 # `run.py` sobre qué falta calibrar).

# --- Turbulencia (componente aleatoria del campo magnético) ---
# Amplitud RMS comparable a la del campo regular: es la relación estándar
# reportada para el ISM (Beck 2001; Sun et al. 2008 adoptan 3 uG en su
# modelo de referencia, ver Waelkens et al. 2008 sec. 5.2).
B0_TURBULENTO = 3.0 * u.microgauss
# Espectro de Kolmogorov, el estándar adoptado para la turbulencia del
# ISM galáctico (Han et al. 2004; Waelkens et al. 2008 sec. 5.2).
SPECTRAL_INDEX = 5.0 / 3.0
# Escala de disipación: limitada por la resolución de la malla (2 celdas),
# no una escala física real medida -mismo criterio que ya usa
# `icm_faraday_rotation/config_fisica.py` para LAMBDA_MIN.
LAMBDA_MIN = 2.0 * DX_BASE
# Escala de inyección: equivalente al corte k0~1 kpc^-1 adoptado por
# Waelkens et al. (2008)/Han et al. (2004) para la turbulencia del ISM.
LAMBDA_MAX = 6.0 * u.kpc

# --- Población de electrones relativistas (emisión sincrotrón) ---
# Índice espectral: mismo valor que Page et al. (2007)/Sun et al. (2008)
# adoptan para nu>408 MHz.
P_SPEC = 3.0
# n_rel = fracción * n_e: la misma convención ad hoc que ya usa
# `examples/icm_faraday_rotation` (no hay, todavía, un modelo separado de
# electrones relativistas para este proyecto).
NE_REL_FRACCION = 0.01

# --- Observación ---
NU = 1.4e9 * u.Hz  # banda L, igual que el ejemplo del ICM
C = 3e8 * (u.m / u.s)
LAMBDA_ONDA = C / NU

# --- Mapa de cielo (l, b) ---
N_L = 180
N_B = 91
# Se recorta cerca de los polos exactos por la singularidad de
# coordenadas de (l, b) en esa latitud (ver
# `los_raytrace.los_frame_from_galactic`), no por ninguna razón física.
B_MAX_DEG = 85.0
DL = 0.1 * u.kpc  # paso de integración a lo largo de cada rayo

# Cuántos píxeles (l, b) se resuelven juntos en un solo lote vectorizado
# dentro de `faradaymr.los_raytrace.sky_map` (ver docstring de esa
# función). No es un parámetro físico -no cambia ningún resultado, solo
# el pico de memoria de GPU/CPU durante el cómputo-, así que vive acá por
# conveniencia de tener un solo lugar de configuración por corrida, no
# porque tenga significado físico. 8192 es conservador para una GPU de
# Colab de gama media (T4, 16GB): con N_L=180, N_B=91 (16380 píxeles en
# total) ya entra en dos lotes; subir este número en una GPU con más
# memoria (A100) acelera la corrida al reducir el número de lanzamientos
# de kernel, bajarlo es la manera de correr en una GPU con menos memoria
# o en CPU con RAM limitada.
PIXEL_CHUNK_SIZE = 8192
