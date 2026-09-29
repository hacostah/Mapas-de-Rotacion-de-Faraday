"""
Parámetros físicos del filamento WHIM (Proyecto II), con unidades de astropy.

Cada valor cita su fuente (literatura) o qué error numérico controla.
"""
import astropy.units as u
import astropy.constants as const
import numpy as np

# ---------------------------------------------------------------------------
# Filamento
# ---------------------------------------------------------------------------

# Densidad electrónica central: límite superior a 1 sigma de Tanimura et al.
# (2020, A&A 637, A41), delta = 19 (+27/-12).
N0 = 1e-5 * (u.cm**-3)

# Índice beta del perfil radial (Tanimura et al. 2020).
BETA = 2.0 / 3.0

# Radio de núcleo del perfil (Tanimura et al. 2020 obtienen 1.5 Mpc para
# filamentos de 30-100 Mpc; aquí, filamento corto de 2 Mpc).
RC = 300.0 * u.kpc

# Longitud física total del filamento (a lo largo de su eje).
LONGITUD_FILAMENTO = 2000.0 * u.kpc

# Campo magnético RMS: 10 nG (resumen del proyecto; Akahori & Ryu 2010,
# ApJ 723, 476).
B0 = 0.010 * u.microgauss

MU = 0.6  # peso molecular medio de un plasma ionizado (no entra en RM)

# ---------------------------------------------------------------------------
# Malla
# ---------------------------------------------------------------------------

# Caja de 160 x 25 kpc = 4000 kpc por lado. Conserva el 96.5% de la varianza
# de RM en el último bin del perfil lateral (99.9% en el eje); lo comprueba
# `validacion.verificar_profundidad_radial` antes de cada barrido.
N_BASE = 160
DX_BASE = 25.0 * u.kpc

# ---------------------------------------------------------------------------
# Turbulencia magnética (convención de Murgia et al. 2004: |B_k|^2 ∝ k^-n,
# Lambda = pi/k)
# ---------------------------------------------------------------------------

# Escala mínima = tamaño de celda (corte en la frecuencia de Nyquist).
LAMBDA_MIN = DX_BASE
# Escala de inyección (Akahori & Ryu 2010: coherencia de varios x 100 kpc/h).
LAMBDA_MAX = 500.0 * u.kpc
# Índice espectral n de |B_k|^2 ∝ k^-n (rango 2-4 de Murgia et al. 2004).
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
# caja.
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
