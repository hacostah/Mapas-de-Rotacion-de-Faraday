"""
Estudio de estrés estadístico (Monte Carlo) para el Falso Observatorio.
Cuantifica el límite de ruido instrumental y configuración de frecuencias
en el que el algoritmo de fuerza bruta falla debido a la ambigüedad n-pi.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# Importaciones locales y del framework
import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.model import construir_mapa_rm_analitico, observar_stokes_qu
from faradaymr.instrument.noise import add_gaussian_noise
from faradaymr.analysis.rm_fit import resolve_n_pi_ambiguity

def estres_estadistico(rm_verdad, subconjuntos_frecuencias, sigmas_ruido, n_repeticiones=10, umbral_error=5.0):
    """
    Ejecuta el barrido de Monte Carlo inyectando ruido en Stokes Q/U.
    """
    # Generar Stokes ideales (sin ruido)
    Q_ideal, U_ideal, _ = observar_stokes_qu(rm_verdad)
    
    # -----------------------------------------------------------------
    # SOLUCIÓN: Bajar los tensores de la GPU (CuPy) a la RAM (NumPy) 
    # para que sean compatibles con el generador de ruido de NumPy.
    if hasattr(Q_ideal, 'get'):
        Q_ideal = Q_ideal.get()
        U_ideal = U_ideal.get()
        rm_verdad = rm_verdad.get() if hasattr(rm_verdad, 'get') else rm_verdad
    # -----------------------------------------------------------------
    
    # Extraer una muestra aleatoria de píxeles para hacer la prueba factible en tiempo
    n_muestra = 100
    rng = np.random.default_rng(42)
    indices_x = rng.integers(0, cfg.N_PIXELES, n_muestra)
    indices_y = rng.integers(0, cfg.N_PIXELES, n_muestra)
    
    rm_verdad_muestra = rm_verdad[indices_x, indices_y]
    
    resultados = {nombre: [] for nombre in subconjuntos_frecuencias.keys()}
    
    # 1. Bucle de configuración de frecuencias
    for nombre_config, indices_freq in subconjuntos_frecuencias.items():
        print(f"\n--- Evaluando configuración: {nombre_config} ---")
        longitudes_onda = cfg.LONGITUDES_ONDA_M[indices_freq]
        
        # 2. Bucle de niveles de ruido sigma
        for sigma in sigmas_ruido:
            fallos_totales = 0
            
            # 3. Bucle de repeticiones Monte Carlo
            for rep in range(n_repeticiones):
                # Inyectar ruido independiente a Q y U en las frecuencias elegidas
                Q_ruidoso = add_gaussian_noise(Q_ideal[indices_freq], sigma, rng=rng)
                U_ruidoso = add_gaussian_noise(U_ideal[indices_freq], sigma, rng=rng)
                
                # Recalcular el ángulo observado con el ruido incluido
                psi_obs_ruidoso = 0.5 * np.arctan2(U_ruidoso, Q_ruidoso)
                
                # Evaluar cada píxel de la muestra
                for p in range(n_muestra):
                    psi_pixel = psi_obs_ruidoso[:, indices_x[p], indices_y[p]]
                    
                    # Recuperar RM usando el algoritmo del Issue 28
                    rm_estimado = resolve_n_pi_ambiguity(
                        longitudes_onda, 
                        psi_pixel, 
                        n_candidatos=range(-3, 4)
                    )
                    
                    # Criterio de fallo: ¿Saltó a una línea fantasma n-pi?
                    if np.abs(rm_estimado - rm_verdad_muestra[p]) > umbral_error:
                        fallos_totales += 1
            
            # Calcular fracción de fallos (rango 0.0 a 1.0)
            fraccion_falla = fallos_totales / (n_repeticiones * n_muestra)
            resultados[nombre_config].append(fraccion_falla)
            print(f"Sigma: {sigma:.3f} | Fallos: {fraccion_falla*100:.1f}%")
            
    return resultados

def graficar_resultados(sigmas_ruido, resultados):
    """Genera la gráfica de estrés estadístico."""
    plt.figure(figsize=(10, 6))
    
    marcadores = ['o-', 's--', '^-.']
    for (nombre, fallos), marcador in zip(resultados.items(), marcadores):
        plt.plot(sigmas_ruido, fallos, marcador, label=nombre, linewidth=2)
        
    plt.title("Límite de Ruido: Ruptura del Desenredado de Fase (Ambigüedad n-pi)")
    plt.xlabel(r"Ruido Instrumental ($\sigma$) en Stokes Q/U")
    plt.ylabel("Fracción de Píxeles Irrecuperables")
    plt.axhline(0.1, color='red', linestyle=':', label='Umbral de tolerancia (10%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("limite_ruido_montecarlo.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Iniciando Estudio Paramétrico de Monte Carlo...")
    start_time = time.time()
    
    # Obtenemos la verdad física
    rm_verdad = construir_mapa_rm_analitico()
    
    # Definimos qué subconjuntos de nuestras 3 frecuencias (5.0, 2.0, 1.4 GHz) evaluaremos
    # Índices: 0 -> 5.0 GHz | 1 -> 2.0 GHz | 2 -> 1.4 GHz
    configuraciones = {
        "Todas (5.0, 2.0, 1.4)": [0, 1, 2],
        "Banda Alta (5.0, 2.0)": [0, 1],
        "Banda Extrema (5.0, 1.4)": [0, 2]
    }
    
    # Niveles de ruido desde 0 (perfecto) hasta 0.5 (ruido severo)
    sigmas_ruido = np.linspace(0.0, 0.4, 8)
    
    # Ejecutar Monte Carlo
    resultados = estres_estadistico(
        rm_verdad, 
        configuraciones, 
        sigmas_ruido, 
        n_repeticiones=5,   # Aumentar este número para una curva más suave
        umbral_error=5.0
    )
    
    print(f"\nSimulación completada en {time.time() - start_time:.2f} segundos.")
    graficar_resultados(sigmas_ruido, resultados)

if __name__ == "__main__":
    main()