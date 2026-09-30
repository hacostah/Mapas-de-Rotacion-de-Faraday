"""Script de ejecución del Falso Observatorio para el póster.
Genera los mapas 2D de validación del algoritmo con inyección de ruido instrumental.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# Importaciones del framework
from faradaymr.analysis.rm_fit import resolve_n_pi_ambiguity
from faradaymr.instrument.noise import add_gaussian_noise

# Importaciones locales
import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.model import construir_mapa_rm_analitico, observar_stokes_qu

def main():
    print("1. Construyendo mapa de RM analítico (Verdad)...")
    rm_verdad = construir_mapa_rm_analitico()

    print("2. Simulando observaciones del radiotelescopio (Stokes Q/U ideales)...")
    mapas_Q, mapas_U, _ = observar_stokes_qu(rm_verdad)

    # Extraemos a la RAM (NumPy) si se generaron en la GPU (CuPy)
    if hasattr(mapas_Q, 'get'):
        mapas_Q = mapas_Q.get()
        mapas_U = mapas_U.get()
        rm_verdad = rm_verdad.get() if hasattr(rm_verdad, 'get') else rm_verdad

    print("3. Inyectando ruido instrumental ligero (Mundo Real)...")
    sigma = 0.005  # Nivel de ruido ultra-bajo
    rng = np.random.default_rng(42)
    
    Q_ruidoso = add_gaussian_noise(mapas_Q, sigma, rng=rng)
    U_ruidoso = add_gaussian_noise(mapas_U, sigma, rng=rng)
    
    # Recalculamos la fase (ángulo observado) atrapada en la ambigüedad con el ruido incluido
    mapas_psi_obs = 0.5 * np.arctan2(U_ruidoso, Q_ruidoso)

    print("4. Reconstruyendo RM resolviendo la ambigüedad n-pi...")
    start_time = time.time()
    
    rm_recuperado = np.zeros_like(rm_verdad)
    longitudes_onda = cfg.LONGITUDES_ONDA_M
    
    for i in range(cfg.N_PIXELES):
        for j in range(cfg.N_PIXELES):
            psi_pixel = mapas_psi_obs[:, i, j]
            rm_estimado = resolve_n_pi_ambiguity(
                longitudes_onda, 
                psi_pixel, 
                n_candidatos=range(-40, 41)
            )
            rm_recuperado[i, j] = rm_estimado
            
    print(f"   Reconstrucción completada en {time.time() - start_time:.2f} segundos.")

    print("5. Calculando errores...")
    error_absoluto = np.abs(rm_recuperado - rm_verdad)
    print(f"   Error máximo de reconstrucción: {np.max(error_absoluto):.4f} rad/m^2")

    print("6. Generando visualización para el póster...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(rm_verdad, origin='lower', cmap='viridis')
    axes[0].set_title("RM Verdad Analítica")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rm_recuperado, origin='lower', cmap='viridis')
    axes[1].set_title("RM Reconstruido con Ruido")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Configuramos el límite de color para resaltar la textura gaussiana del error
    vmax_error = np.percentile(error_absoluto, 98) 
    im2 = axes[2].imshow(error_absoluto, origin='lower', cmap='inferno', vmin=0, vmax=vmax_error)
    axes[2].set_title(rf"Error Absoluto ($\sigma = {sigma}$)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("mapas_reconstruccion_rm_ruido.pdf", dpi=300, bbox_inches='tight')
    print("   Gráfica guardada como 'mapas_reconstruccion_rm_ruido.pdf'.")
    
    plt.show()

if __name__ == "__main__":
    main()