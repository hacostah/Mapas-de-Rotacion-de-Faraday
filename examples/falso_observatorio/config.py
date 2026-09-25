"""Parámetros de configuración para el Falso Observatorio."""

from __future__ import annotations
import numpy as np

# Configuración de la malla
N_PIXELES = 128
PROFUNDIDAD_KPC = 100.0  # L: Longitud de la línea de visión a través del plasma

# Propiedades de la capa de plasma (Escenario Analítico)
N_E_CM3 = 1e-3           # Densidad de electrones uniforme
B_Z_BASE_UG = 3.0        # Campo magnético base en microGauss (apuntando al observador)
PSI_0_RAD = 0.5          # Ángulo de polarización intrínseco en la fuente

# Configuración del Observatorio (Frecuencias del Abstract)
FRECUENCIAS_GHZ = np.array([5.0, 2.0, 1.4])
VELOCIDAD_LUZ = 299792458.0
LONGITUDES_ONDA_M = VELOCIDAD_LUZ / (FRECUENCIAS_GHZ * 1e9)
LAMBDA_CUADRADO = LONGITUDES_ONDA_M**2