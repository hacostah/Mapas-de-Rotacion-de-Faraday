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

import os

import astropy.units as u

# --- Perfil de resolución: "rapido" (CPU, por defecto) o "exhaustivo" (GPU) ---
# Los parámetros de malla/mapa de cielo de más abajo (N_BASE, N_L, N_B, DL,
# PIXEL_CHUNK_SIZE) son los únicos que dependen de este perfil -no son
# físicos, solo determinan cuánto detalle numérico se resuelve y cuánto
# cuesta correr la simulación (ver notas puntuales en cada uno). Los
# parámetros FÍSICOS (NE0, B0_REGULAR, pitch_angle, etc., más abajo) NO
# dependen del perfil: son los mismos disco/campo/observador sin importar
# a qué resolución se muestreen.
#
# "rapido" (default): la resolución ya validada en esta sesión contra datos
# reales (ver `resumen_comparacion_observacional.json`) -corre en segundos
# en CPU pura (numpy), sin necesitar GPU. `ARM_WIDTH` (ver más abajo) queda
# sub-resuelta a esta resolución (literalmente 1 celda) -un hueco real,
# conocido, que NO se corrige acá para no invalidar la calibración
# estadística de `B0_REGULAR`/`B0_TURBULENTO` ya hecha contra esta misma
# resolución (ver esa nota más abajo) sin volver a correrla.
#
# "exhaustivo": 64x más celdas en la malla 3D (4x por eje) y 4x más
# píxeles en el mapa de cielo -resuelve `ARM_WIDTH` con 4 celdas en vez de
# 1, y el mapa de cielo se acerca más a la resolución angular de los datos
# reales (ver `faradaymr.observational`). Pensado para correr con
# `use_gpu=True` (ver `faradaymr.get_backend`); en CPU puede tardar minutos
# en vez de segundos. Activar con la variable de entorno
# `FARADAYMR_PERFIL_RESOLUCION=exhaustivo` ANTES de importar este módulo
# (ver `Faraday_MR_Colab.ipynb`, que la fija antes de correr `run.py`) -por
# defecto ("rapido") no cambia nada del comportamiento ya validado.
PERFIL_RESOLUCION = os.environ.get("FARADAYMR_PERFIL_RESOLUCION", "rapido")
if PERFIL_RESOLUCION not in ("rapido", "exhaustivo"):
    raise ValueError(
        f"FARADAYMR_PERFIL_RESOLUCION={PERFIL_RESOLUCION!r} no reconocido "
        "(valores válidos: 'rapido', 'exhaustivo')."
    )

# --- Malla ---
# Una caja de 32 kpc de lado (64 celdas de 0.5 kpc en el perfil "rapido")
# alcanza para contener el radio solar (8 kpc) con margen hasta más allá
# del disco óptico (~15 kpc), sin gastar resolución en un volumen que no es
# el objeto de este proyecto -a diferencia del ICM, que sí necesita
# cientos de kpc de lado. Mismo tamaño de caja/celda que ya se validó en la
# prueba de inspección visual del ray tracer
# (`tests/test_los_raytrace_visual.py`).
if PERFIL_RESOLUCION == "exhaustivo":
    N_BASE = 256
    DX_BASE = 0.125 * u.kpc
else:
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
# Desplazamiento de fase RÍGIDO de los `N_ARMS` brazos (ver
# `faradaymr.simulation.galactic_disk.spiral_arm_density_factor`,
# parámetro `phase0`). Con `phase0=0.0` (el default de esa función) los
# brazos caen en 0/90/180/270°; el observador de `model.construir_escenario`
# está fijo en phi=180° (desplazado -R_SOLAR a lo largo de x desde el
# centro galáctico) -con N_ARMS par, eso pone al Sol EXACTAMENTE sobre la
# cresta de un brazo (verificado numéricamente: n_e ahí sale 2x el valor
# de referencia "fuera de cualquier brazo" que declara `NE0` más arriba),
# un artefacto de construcción de la caja, no una elección física: el Sol
# real está en una región interbrazo (entre Sagitario-Carina y Perseo, ver
# Vallée 2016), no encima de un brazo. Se fija en la MITAD del espaciado
# entre brazos (360°/N_ARMS/2 = 45° para N_ARMS=4): la posición que pone
# al observador lo más lejos posible (en fase) de cualquier brazo, la
# aproximación más simple y simétrica a "interbrazo" sin necesidad de fijar
# la orientación real y asimétrica de los 4 brazos de la Vía Láctea (fuera
# del alcance de este modelo de juguete, ver docstring de
# `faradaymr.los_raytrace.direction_from_galactic`).
ARM_PHASE0_DEG = 180.0 / N_ARMS
# Pitch angle observacional (medido desde la dirección azimutal, no
# radial -ver docstring de `faradaymr.simulation.galactic_disk`), espiral
# trailing (signo negativo): Vallée (2015, 2016).
PITCH_ANGLE_DEG = -12.0

# --- Campo magnético regular (espiral logarítmica) ---
# CALIBRADO estadísticamente (no solo orden de magnitud de literatura)
# contra los dos conjuntos de datos reales de RM ya cargados por
# `faradaymr.observational` -ver `calibrar_amplitud_campo.py` para el
# método completo (mínimos cuadrados en log-espacio sobre el perfil
# RMS(RM) vs |b| de Oppermann & Enßlin 2012, con validación cruzada
# independiente contra el catálogo puntual NVSS de Taylor+2009). El valor
# original (2.0 uG, "mismo orden que JF12/Cordes & Lazio") sobreestimaba
# la RM real por un factor de ~3.7x (alpha=0.269 calibrado = 1/3.7); con
# `B0_TURBULENTO` (ver más abajo) escalado por el MISMO factor para
# preservar la razón regular/turbulento (ver docstring de
# `calibrar_amplitud_campo.py` sobre por qué un factor común, no
# independiente).
#
# TENSIÓN CON LA LITERATURA, declarada explícitamente (no escondida): 0.54
# uG es notablemente más bajo que el ~2 uG de JF12 para el disco Galáctico
# real. La lectura más honesta no es "JF12 está mal" sino que este modelo
# de juguete (disco+brazos idealizados, sin reversión de campo, sin burbuja
# local, con `ARM_WIDTH` sub-resuelta -ver nota en la sección de malla más
# arriba-) integra RM de forma sistemáticamente más eficiente que la Vía
# Láctea real a lo largo de cada línea de visión completa hasta el borde
# de la caja; la calibración compensa esa diferencia estructural, no
# necesariamente mide "el B real". Corregir `ARM_WIDTH` (aumentar la
# resolución de malla, ver nota arriba) es el candidato más probable para
# cerrar esta tensión sin necesitar un B tan bajo -tarea pendiente,
# deliberadamente no resuelta en esta calibración (ver
# `calibrar_amplitud_campo.py`, docstring del módulo).
B0_REGULAR = 2.0 * 0.269 * u.microgauss
# Escala radial de decaimiento del campo regular (mismo orden que el
# disco de JF12).
SCALE_RADIAL_B = 5.0 * u.kpc
# Espesor del disco magnetizado (JF12).
SCALE_HEIGHT_B = 1.0 * u.kpc
HANDEDNESS = 1  # sentido de enrollamiento; no fijado por ninguna
                 # referencia particular en este toy model (ver nota en
                 # `run.py` sobre qué falta calibrar).

# --- Turbulencia (componente aleatoria del campo magnético) ---
# Amplitud RMS comparable a la del campo regular (razón preservada en
# 3:2, la relación estándar reportada para el ISM: Beck 2001; Sun et al.
# 2008 adoptan 3 uG en su modelo de referencia, ver Waelkens et al. 2008
# sec. 5.2) -escalado por el MISMO factor de calibración que
# `B0_REGULAR` (0.269, ver esa nota para el método y la tensión declarada
# con la literatura).
B0_TURBULENTO = 3.0 * 0.269 * u.microgauss
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
# Se recorta cerca de los polos exactos por la singularidad de
# coordenadas de (l, b) en esa latitud (ver
# `los_raytrace.los_frame_from_galactic`), no por ninguna razón física.
B_MAX_DEG = 85.0
if PERFIL_RESOLUCION == "exhaustivo":
    # El doble de resolución angular por eje que "rapido" (píxeles de
    # ~1°x~0.9°): todavía más grueso que los datos reales usados en
    # `faradaymr.observational` (nside=128 de Oppermann+2012 ~0.5°,
    # nside=1024 de Planck ~0.06°), pero ya no tan grueso como para que un
    # cruce de brazo espiral quede resuelto por menos de un puñado de
    # píxeles en l. 65160 píxeles en total (4x los 16380 de "rapido").
    N_L = 360
    N_B = 181
    # A DX_BASE=0.125 kpc (ver arriba), 0.025 kpc caben 5 muestras por
    # celda -la misma proporción muestras/celda que 0.1 kpc a DX_BASE=0.5
    # kpc ("rapido")-, conservador para que la rotación de Faraday
    # acumulada entre dos celdas no salte de golpe (ver
    # `faradaymr.los_raytrace.sample_line_of_sight`, parámetro `offset`).
    DL = 0.025 * u.kpc
else:
    N_L = 180
    N_B = 91
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
# o en CPU con RAM limitada. En el perfil "exhaustivo" (más píxeles, más
# celdas por campo) se sube a 16384 -pensado para una GPU con más memoria
# que un T4 (A100, L4); bajarlo si la GPU asignada por Colab tiene menos
# memoria y el cómputo falla por falta de memoria.
PIXEL_CHUNK_SIZE = 16384 if PERFIL_RESOLUCION == "exhaustivo" else 8192
