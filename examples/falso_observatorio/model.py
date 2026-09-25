"""Generador del modelo físico y simulación de observaciones de Stokes."""

from __future__ import annotations
import numpy as np

import config as cfg
from faradaymr import get_backend

def construir_mapa_rm_analitico(use_gpu=None):
    """
    Construye un mapa 2D de Medida de Rotación (RM) teórica.
    Combina un fondo de plasma uniforme con una ligera perturbación
    gaussiana para evaluar el desenredado de fase píxel a píxel.
    """
    xp = get_backend(use_gpu)
    
    # Crear malla 2D normalizada de -1 a 1
    eje = xp.linspace(-1, 1, cfg.N_PIXELES)
    xx, yy = xp.meshgrid(eje, eje, indexing="ij")
    r2 = xx**2 + yy**2
    
    # Ecuación analítica de Faraday: RM = 812 * n_e * B_z * L
    # RM resultante estará en rad/m^2
    rm_base = 812.0 * cfg.N_E_CM3 * cfg.B_Z_BASE_UG * cfg.PROFUNDIDAD_KPC
    
    # Agregamos una perturbación espacial para crear gradientes (saltos de fase)
    perturbacion = 50.0 * xp.exp(-r2 / 0.1)
    rm_mapa = rm_base + perturbacion
    
    return rm_mapa.astype(xp.float32)


def observar_stokes_qu(rm_mapa, use_gpu=None):
    """
    Simula la medición de un radiotelescopio generando Q y U 
    para las 3 frecuencias discretas del Falso Observatorio.
    """
    xp = get_backend(use_gpu)
    rm_mapa = xp.asarray(rm_mapa)
    lambda2 = xp.asarray(cfg.LAMBDA_CUADRADO)
    
    # Asumimos intensidad polarizada (P) normalizada a 1 para simplificar
    P = 1.0 
    
    mapas_Q = []
    mapas_U = []
    mapas_psi_obs = []
    
    for l2 in lambda2:
        # Rotación física real (sin límites)
        psi_verdadero = cfg.PSI_0_RAD + rm_mapa * l2
        
        # El telescopio mide intensidades direccionales (Stokes Q y U)
        Q = P * xp.cos(2.0 * psi_verdadero)
        U = P * xp.sin(2.0 * psi_verdadero)
        
        # El ángulo observado sufre la ambigüedad n*pi (queda atrapado en [-pi/2, pi/2])
        psi_obs = 0.5 * xp.arctan2(U, Q)
        
        mapas_Q.append(Q)
        mapas_U.append(U)
        mapas_psi_obs.append(psi_obs)
        
    return xp.array(mapas_Q), xp.array(mapas_U), xp.array(mapas_psi_obs)