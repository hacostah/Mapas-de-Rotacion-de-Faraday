"""
Parámetros físicos del filamento WHIM (Proyecto II), con unidades de astropy.

Cada valor lleva su justificación. Los que dependen de la literatura citan la
referencia; los que son decisiones numéricas explican qué error controlan.

Leyes de escala (útiles para leer los resultados con otros parámetros sin
volver a correr nada; ver README):
    sigma_RM  ∝  n0 * B0 * sqrt(Lambda_corr * L_LoS)      (paseo aleatorio)
    N_fuentes ∝  sigma_RM^-4                               (umbral de detección)
"""
import astropy.units as u
import astropy.constants as const
import numpy as np

# ---------------------------------------------------------------------------
# Filamento
# ---------------------------------------------------------------------------

# Densidad electrónica central. Tanimura et al. (2020, A&A 637, A41) miden una
# sobredensidad central de gas delta = 19 (+27/-12) en filamentos de SDSS con
# un modelo beta cilíndrico. Con n_e medio del universo a z=0,
# n_e ≈ 2.4e-7 cm^-3 (Omega_b = 0.049, h = 0.7), eso da n0 ≈ 5e-6 cm^-3, y el
# límite superior a 1 sigma ~1e-5 cm^-3. Se adopta 1e-5 cm^-3, dentro del
# intervalo de Tanimura y del rango 1e-6 - 1e-4 cm^-3 típico del WHIM.
N0 = 1e-5 * (u.cm**-3)

# Índice beta del perfil radial. Se usa beta = 2/3, el mismo que Tanimura et
# al. (2020) ajustan a los perfiles Sunyaev-Zel'dovich y de lente de los
# filamentos. (La versión anterior usaba 0.5 "para diferenciarlo de los
# cúmulos", lo cual no es un argumento físico.)
BETA = 2.0 / 3.0

# Radio de núcleo del perfil. Tanimura et al. (2020) obtienen r_c = 1.5
# (+1.8/-0.7) Mpc para filamentos de 30-100 Mpc de largo, con una fuerte
# degeneración r_c-delta. Para un filamento individual y corto (2 Mpc) se
# adopta un núcleo más compacto, de 300 kpc. Como todos los perfiles del
# proyecto se describen en unidades de d/r_c, cambiar r_c solo reescala el
# eje de distancias (la forma y el exponente p no cambian).
RC = 300.0 * u.kpc

# Longitud física total del filamento (a lo largo de su eje).
LONGITUD_FILAMENTO = 2000.0 * u.kpc

# Campo magnético RMS. 10 nG es el valor que fija el resumen del proyecto y
# el campo medio que predicen Akahori & Ryu (2010, ApJ 723, 476) para
# filamentos con un dínamo turbulento (<B> ~ 10 nG). Como RM es lineal en B,
# todos los resultados de sigma_RM escalan como B0/10 nG (p.ej. los ~40 nG
# de Carretti et al. 2022, usados antes, multiplican sigma_RM por 4).
B0 = 0.010 * u.microgauss

MU = 0.6  # peso molecular medio de un plasma ionizado (no entra en RM)

# ---------------------------------------------------------------------------
# Malla
# ---------------------------------------------------------------------------

# Caja de 160 x 25 kpc = 4000 kpc por lado (+/-2000 kpc). El tamaño lo fija
# la COLA radial del perfil beta a lo largo de la línea de visión, no solo la
# longitud del filamento: para la vista lateral, una línea de visión a una
# distancia d del eje tiene que abarcar la densidad hasta |l| >> d. Con
# beta = 2/3 y DIST_MAX_AJUSTE = 3 r_c, esta caja conserva el 96.5% de la
# varianza de RM en el último bin (99.9% en el eje); con la caja anterior
# (+/-1280 kpc, beta = 0.5, ventana 4 r_c) se perdía ~28% y el perfil lateral
# salía artificialmente empinado. `validacion.verificar_profundidad_radial`
# lo comprueba antes de cada barrido.
N_BASE = 160
DX_BASE = 25.0 * u.kpc

# ---------------------------------------------------------------------------
# Turbulencia magnética (convención de Murgia et al. 2004: |B_k|^2 ∝ k^-n,
# Lambda = pi/k)
# ---------------------------------------------------------------------------

# Escala mínima = tamaño de celda: el corte espectral cae justo en la esfera
# de Nyquist de la malla. Un Lambda_min menor que la celda (antes 10 kpc con
# celdas de 20 kpc) no agrega escalas reales y deja pasar los modos de las
# esquinas del cubo de Fourier, que no son isótropos.
LAMBDA_MIN = DX_BASE
# Escala de inyección: pocos cientos de kpc, como la longitud de coherencia
# de "varios x 100 kpc/h" de Akahori & Ryu (2010).
LAMBDA_MAX = 500.0 * u.kpc
# Índice espectral n de |B_k|^2 ∝ k^-n (Kolmogorov sería n = 11/3; n = 3 da
# más peso a las escalas grandes, dentro del rango 2-4 que explora Murgia).
N_SPEC = 3.0
# Índice de energía de los electrones relativistas (solo sincrotrón).
P_SPEC = 3.0

# ---------------------------------------------------------------------------
# Observación
# ---------------------------------------------------------------------------
NU = 1.4e9 * u.Hz
LAMBDA_ONDA = const.c / NU

# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

# Ventana del perfil transversal sigma_RM(d): 0 - 3 r_c. Es un valor FÍSICO,
# independiente del tamaño de la malla, para que el ajuste no dependa de la
# caja. A 3 r_c la dispersión ya cayó a ~10-20% de su valor en el eje.
DIST_MAX_AJUSTE = 3.0 * RC

# Número de BORDES de bin en 0 - DIST_MAX_AJUSTE (N-1 bins de ~82 kpc, más de
# tres celdas cada uno). Compartido por el barrido y las figuras.
N_BORDES_PERFIL = 12

# Ángulos de visión del barrido (grados entre el eje del filamento y la línea
# de visión): 0 = de frente, 90 = de lado.
THETAS_BARRIDO = np.linspace(0.0, 90.0, 10)

# Realizaciones Monte Carlo del campo turbulento por ángulo. El perfil se
# obtiene apilando las N realizaciones (promedio de RM^2 por bin), y los
# errores con bootstrap sobre semillas.
N_SEMILLAS = 50
