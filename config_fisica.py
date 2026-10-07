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
# Escala vertical del disco ionizado: la del disco grueso de NE2001
# (H1 = 0.95 kpc, n1 = 0.033 cm^-3; Cordes & Lazio 2002), coherente con
# NE0. Gaensler et al. (2008) obtienen una escala mayor (1.8 kpc) pero con
# n0 = 0.014 cm^-3: lo que fija la RM y la DM hacia el polo es la columna
# n0 * h, ~25-30 pc cm^-3 en los dos casos (la DM polar de este modelo,
# 31 pc cm^-3, se valida en `run.py`).
SCALE_HEIGHT_NE = 1.0 * u.kpc
# Forma radial del disco de n_e. Una exponencial normalizada en R_solar
# (la opción "exponencial") sube a ~0.3 cm^-3 en el centro y, con el realce
# de los brazos, pasa de 1 cm^-3: ~10x lo que da NE2001 en el disco
# interior, y eso inflaba la RM de la línea de visión al centro galáctico
# hasta ~2000 rad/m^2. Se usa la forma del disco grueso de NE2001 (Cordes &
# Lazio 2002): cos(pi R / 2A) con A = 17.5 kpc, casi plana adentro.
NE_RADIAL_PROFILE = "ne2001"
NE_RADIAL_CUTOFF = 17.5 * u.kpc
# Número de brazos espirales (modelo de 4 brazos, Vallée 2016).
N_ARMS = 4
# Radio donde empiezan los brazos: el extremo de la barra, ~3 kpc (Vallée
# 2016). Sin él, las espirales logarítmicas se amontonan hacia el centro y
# n_e sube a 0.2 cm^-3 en R < 1 kpc por solapamiento de brazos vecinos.
ARM_R_MIN = 3.0 * u.kpc
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
# Pitch angle (medido desde la dirección azimutal, tan(p) = B_R/B_phi, con
# phi creciendo en sentido antihorario visto desde el polo norte, que es
# lo que da arctan2(y, x) con l=90 hacia +y). La Galaxia rota en sentido
# HORARIO visto desde el polo norte (el Sol, en (-R0, 0), se mueve hacia
# +y = l=90) y sus brazos son "trailing": al alejarse del centro la
# espiral se retrasa, o sea r CRECE con phi antihorario => tan(p) > 0.
# Vallée escribe -12 a -13 grados porque mide el azimut en sentido horario;
# en las coordenadas de este código el mismo brazo tiene p = +12 grados.
# Comprobado con las tangentes reales de Scutum-Crux (l=+31 y l=310): los
# puntos tangentes quedan en (R=4.1 kpc, phi=121) y (R=6.1 kpc, phi=220),
# es decir dr/(r dphi) ~ +0.23 = tan(13 grados). Con -12 (como estaba) los
# brazos se enrollaban al revés (leading).
PITCH_ANGLE_DEG = 12.0

# --- Campo magnético: valores de literatura y factores efectivos ---
# Todas las amplitudes de campo de esta sección son las publicadas (disco:
# Beck 2001 / JF12; halo: JF12; turbulencia: Beck 2001, Sun et al. 2008).
# Dos factores, declarados aparte, las convierten en valores efectivos
# para esta malla (`model.construir_escenario` los aplica):
#
# - FACTOR_TURBULENCIA (derivado en `config.py`, no ajustado): la RM de un
#   campo aleatorio crece como B_rms * sqrt(L_coh * L). La turbulencia de
#   este modelo tiene longitud integral ~1.6 kpc (espectro de Kolmogorov
#   entre LAMBDA_MIN y LAMBDA_MAX, limitado por la malla), contra ~0.1 kpc
#   en el ISM real (L_COHERENCIA_ISM, Haverkorn et al. 2008). Para que su
#   RM sea la de una turbulencia real de 3 µG, la amplitud se multiplica por
#   sqrt(L_COHERENCIA_ISM / L_integral_modelo) ≈ 0.25.
#
# - FACTOR_CAMPO_REGULAR (calibrado con `calibrar_amplitud_campo.py`): un
#   único factor para todo el campo regular (disco + halo), ajustado por
#   mínimos cuadrados en log sobre el perfil RMS(RM) vs |b| de Oppermann &
#   Enßlin (2012), con el catálogo NVSS de Taylor+2009 como validación
#   cruzada. Absorbe lo que la geometría de juguete no tiene (reversiones
#   del campo entre brazos, que cancelan RM en el plano) y el n_e del
#   modelo, que no es el de NE2001 con el que JF12 ajustó sus amplitudes.
#
# Antes había un solo factor 0.454 para todo; con eso la turbulencia de
# kpc dominaba la RM de alta latitud (2.5x la observada en el polo) y
# el campo regular quedaba castigado por un exceso que no era suyo.
# Última calibración (con la inversión interior, ANILLOS_INVERSION_CAMPO,
# y el núcleo de RADIO_NUCLEO_B / RADIO_SIN_CAMPO_B): 0.80 (Oppermann) y
# 1.25 (NVSS), media geométrica 1.00. Es decir, con la geometría completa
# las amplitudes publicadas del campo regular no necesitan reescalarse.
FACTOR_CAMPO_REGULAR = 1.0

# Disco (espiral logarítmica).
B0_REGULAR = 2.0 * u.microgauss
# Escala radial de decaimiento del campo regular (mismo orden que el
# disco de JF12).
SCALE_RADIAL_B = 5.0 * u.kpc
# Espesor del disco magnetizado: h_disk = 0.40 kpc de JF12. Antes era
# 1 kpc (atribuido a JF12 por error); con el halo de JF12 encima, el disco
# tiene que cortarse donde empieza el halo, o su RM simétrica en b tapa la
# antisimetría norte-sur a latitudes medias. A dx = 0.5 kpc queda resuelto
# por ~1 celda: la integral a lo largo del rayo se conserva, el detalle no.
SCALE_HEIGHT_B = 0.4 * u.kpc
# Forma del corte vertical del disco: la de JF12, 1 - L(z, h_disk, w_disk)
# con w_disk = 0.27 kpc, el complemento exacto de cómo se enciende el halo
# toroidal (ALTURA_DISCO_HALO/ANCHO_DISCO_HALO, más abajo). Con una
# exponencial exp(-|z|/h) el disco se apagaba antes de que el halo
# encendiera y |B| regular caía a 0.2 de su valor en z=0 a |z|~0.5 kpc
# para volver a subir. La columna integrada (~h por lado) es la misma.
ANCHO_VERTICAL_B = 0.27 * u.kpc
# Galaxia interior. La exponencial en R hacía crecer el campo del disco a
# ~9 µG en el centro, y con la inversión interior daba RM de ±300-400
# rad/m^2 hacia |l| < 20° (observado: 0 ± 100). Los modelos de referencia
# no extrapolan así: Sun et al. (2008) dejan |B| constante dentro de
# R_c = 5 kpc, y JF12 no tienen campo de disco dentro de 3 kpc (la región
# de la barra). Con las dos cosas la RMS en |l| < 30°, |b| < 5° baja de
# 616 a 325 rad/m^2 (observado: 210) y FACTOR_CAMPO_REGULAR queda en 1.0.
RADIO_NUCLEO_B = 5.0 * u.kpc
RADIO_SIN_CAMPO_B = 3.0 * u.kpc
# Sentido del campo regular: -1 = horario visto desde el polo norte
# (B_phi < 0), que es el del campo local: apunta hacia l ~ 90 grados
# (Manchester 1974; Han et al. 2006), con B_R < 0 (espirando hacia
# adentro, l algo menor que 90). Con B_phi > 0 y p > 0 el campo local
# apuntaría hacia l ~ 270, invirtiendo el signo global de la RM. Junto con
# p > 0 da B_R = -sin(p)|B| < 0, como se espera para brazos trailing.
HANDEDNESS = -1
# Inversión del campo del disco en la Galaxia interior. Sin ella el campo
# es horario en todo el disco y la RM en el plano sale con el signo
# contrario al observado en el primer cuadrante (-345 contra +21 rad/m^2
# en l=20-90, |b|<5), con correlación nula contra Oppermann+2012. La RM de
# pulsares y fuentes extragalácticas muestra una inversión de gran escala
# entre el brazo local y el de Sagitario-Carina (Brown et al. 2007; Van
# Eck et al. 2011): adentro el campo es antihorario. Se modela como una
# sola inversión para R < 7 kpc (~1 kpc dentro del círculo solar, donde
# la ubican esos trabajos). Se compararon también el anillo ASS+RING de
# Sun et al. (2008), 6-7.5 kpc literal y escalado a R_sol=8 kpc: dejan el
# primer cuadrante con el signo equivocado (correlación en |b|<10° de
# 0.11-0.19, contra 0.43 con la inversión única).
ANILLOS_INVERSION_CAMPO = ((0.0 * u.kpc, 7.0 * u.kpc),)

# --- Campo regular de halo (Jansson & Farrar 2012, "JF12") ---
# Ver `faradaymr.fields.halo_field`. Sin halo el campo regular es plano y
# simétrico respecto al disco, y el modelo no puede dar la antisimetría
# norte-sur de la RM ni la componente vertical que Planck ve en
# polarización. Valores de la tabla 1 de JF12.
USAR_CAMPO_HALO = True
# Halo toroidal (JF12, ec. 7).
B_HALO_NORTE = 1.4 * u.microgauss
B_HALO_SUR = -1.1 * u.microgauss
R_HALO_NORTE = 9.22 * u.kpc
R_HALO_SUR = 16.7 * u.kpc
ANCHO_HALO = 0.20 * u.kpc
ESCALA_ALTURA_HALO = 5.3 * u.kpc
ALTURA_DISCO_HALO = 0.40 * u.kpc
ANCHO_DISCO_HALO = 0.27 * u.kpc
# Sentido global en las coordenadas de este código (JF12 mide phi con el
# eje x del centro hacia el Sol, al revés que acá). Se fija con el signo
# observado de la RM de Oppermann et al. (2012) hacia la Galaxia interior
# a 10° < |b| < 45°: positiva al norte y negativa al sur en 0° < l < 90°,
# al revés en 270° < l < 360°. Con +1 el campo regular reproduce los
# cuatro signos; con -1, solo dos. El mapa total puede fallar en alguno:
# la turbulencia de escala kpc de esta malla aporta ±10-20 rad/m^2 por
# cuadrante según la semilla (varianza de realización, no del campo regular).
SENTIDO_HALO_TOROIDAL = 1
# Campo en X (JF12, ec. 8-11).
B_CAMPO_X = 4.6 * u.microgauss
ELEVACION_CAMPO_X_DEG = 49.0
R_CRITICO_CAMPO_X = 4.8 * u.kpc
R_ESCALA_CAMPO_X = 2.9 * u.kpc

# --- Turbulencia (componente aleatoria del campo magnético) ---
# RMS local de la literatura: 3 µG (Beck 2001; Sun et al. 2008, ver
# Waelkens et al. 2008 sec. 5.2). El valor efectivo que usa el modelo es
# este por FACTOR_TURBULENCIA (ver arriba y `config.py`).
B0_TURBULENTO = 3.0 * u.microgauss
# Longitud de coherencia de la turbulencia del ISM real (escala externa
# ~0.1 kpc, Haverkorn et al. 2008).
L_COHERENCIA_ISM = 0.1 * u.kpc
# Espectro de Kolmogorov, el estándar adoptado para la turbulencia del
# ISM galáctico (Han et al. 2004; Waelkens et al. 2008 sec. 5.2).
SPECTRAL_INDEX = 5.0 / 3.0
# Envolvente espacial de la turbulencia. Antes tenía la misma amplitud en
# toda la caja, también a 10 kpc sobre el disco, donde su RM (con escalas
# de varios kpc) tapaba la antisimetría norte-sur que da el halo regular.
# Se usa la forma del campo aleatorio de halo de Jansson & Farrar (2012b,
# ApJ 761, L11): exp(-R/r0) exp(-z^2 / 2 z0^2), con r0 = 10.97 kpc y
# z0 = 2.84 kpc. B0_TURBULENTO es el RMS en la vecindad solar
# (7 < R < 9 kpc, |z| < 0.5 kpc).
USAR_ENVOLVENTE_TURBULENCIA = True
ESCALA_RADIAL_TURBULENCIA = 10.97 * u.kpc
ESCALA_ALTURA_TURBULENCIA = 2.84 * u.kpc
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
C = 299792458.0 * (u.m / u.s)
LAMBDA_ONDA = C / NU

# Segunda frecuencia, para validar contra Planck LFI "30 GHz": 28.4 GHz es
# la frecuencia central efectiva de ese canal (Planck 2018 results II,
# tabla 4). A esta frecuencia la rotación de Faraday es despreciable
# (RM ~ 100 rad/m^2 gira ~0.6°), así que Q/U miden la geometría del campo
# sin la despolarización que sí hay a 1.4 GHz.
NU_PLANCK = 28.4e9 * u.Hz
LAMBDA_PLANCK = C / NU_PLANCK
# Temperatura del CMB (Fixsen 2009), para pasar de K_CMB a K_RJ.
T_CMB = 2.7255 * u.K
# Realizaciones de la turbulencia a 28.4 GHz para estimar la varianza
# galáctica (Planck Int. XLII 2016, sec. 3.4.1). No es un parámetro físico:
# más realizaciones dan una estimación menos ruidosa y cuestan más.
N_REALIZACIONES_VARIANZA_GALACTICA = 8

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
# En el perfil exhaustivo cada píxel del lote ocupa ~220 KB de GPU (~1300
# muestras por rayo, cinco campos interpolados más los acumulados), así
# que 16384 píxeles son ~3.6 GB por lote. `Faraday_MR_Colab.ipynb` elige
# el tamaño según la memoria libre de la GPU asignada y lo pasa con esta
# variable de entorno (antes de importar este módulo).
if os.environ.get("FARADAYMR_PIXEL_CHUNK_SIZE"):
    PIXEL_CHUNK_SIZE = int(os.environ["FARADAYMR_PIXEL_CHUNK_SIZE"])
