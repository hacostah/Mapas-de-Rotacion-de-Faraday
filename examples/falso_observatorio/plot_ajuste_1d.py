"""
Script para generar la visualización 1D del ajuste de Faraday (lambda^2 vs Psi).
Ilustra pedagógicamente la ambigüedad n-pi y el fenómeno de aliasing.
Ideal para la sección de Metodología del póster.
"""

import numpy as np
import matplotlib.pyplot as plt

import examples.falso_observatorio.config as cfg
from examples.falso_observatorio.model import construir_mapa_rm_analitico, observar_stokes_qu

def main():
    print("Generando visualización 1D del ajuste de Faraday...")
    
    # 1. Obtener la verdad del modelo y simular observación
    rm_verdad = construir_mapa_rm_analitico()
    mapas_Q, mapas_U, mapas_psi_obs = observar_stokes_qu(rm_verdad)
    
    # Bajar tensores a RAM (NumPy) si están en VRAM (CuPy)
    if hasattr(mapas_psi_obs, 'get'):
        mapas_psi_obs = mapas_psi_obs.get()
        rm_verdad = rm_verdad.get() if hasattr(rm_verdad, 'get') else rm_verdad
        
    # 2. Elegir el píxel central (El más desafiante, donde el RM es máximo)
    ix, iy = cfg.N_PIXELES // 2, cfg.N_PIXELES // 2
    rm_pixel = rm_verdad[ix, iy]
    psi_obs = mapas_psi_obs[:, ix, iy]
    
    lambda_sq = cfg.LONGITUDES_ONDA_M**2
    
    # 3. Reconstruir la recta física real (Desenrollada)
    # Calculamos el ángulo intrínseco asumiendo que el RM simulado es el real
    chi_0 = psi_obs[0] - rm_pixel * lambda_sq[0]
    psi_real_desenrollado = chi_0 + rm_pixel * lambda_sq
    
    # 4. Crear la Recta Fantasma (Aliasing usando solo 5.0 GHz y 2.0 GHz)
    # Forzamos un fallo matemático asumiendo que la antena leyó los ángulos
    # de manera plana sin considerar las vueltas n-pi.
    m_fantasma = (psi_obs[1] - psi_obs[0]) / (lambda_sq[1] - lambda_sq[0])
    b_fantasma = psi_obs[0] - m_fantasma * lambda_sq[0]
    
    # Vectores continuos para trazar las líneas de regresión
    l2_line = np.linspace(0, np.max(lambda_sq) * 1.05, 100)
    recta_real = chi_0 + rm_pixel * l2_line
    recta_falsa = b_fantasma + m_fantasma * l2_line
    
    # 5. Configurar la Gráfica
    plt.figure(figsize=(10, 6))
    
    # A. Trazar límite del instrumento
    plt.axhline(np.pi/2, color='gray', linestyle='-.', alpha=0.5, label='Límite Instrumental')
    plt.axhline(-np.pi/2, color='gray', linestyle='-.', alpha=0.5)
    
    # B. Rectas de regresión
    plt.plot(l2_line, recta_real, 'b-', linewidth=2, 
             label=f'Ajuste Físico Correcto (RM = {rm_pixel:.1f} rad/m²)')
    plt.plot(l2_line, recta_falsa, 'r--', linewidth=2, alpha=0.6, 
             label=f'Ajuste Fantasma (Aliasing, RM = {m_fantasma:.1f} rad/m²)')
    
    # C. Puntos (Observados vs Desenrollados)
    plt.scatter(lambda_sq, psi_obs, color='red', s=120, zorder=5, edgecolor='black',
                label=r'Observado $\Psi_{obs}$ (Atrapado)')
    plt.scatter(lambda_sq, psi_real_desenrollado, color='blue', s=150, marker='*', zorder=5, 
                label='Desenredado Físicamente')
             
    # D. Conectar los puntos para mostrar gráficamente los saltos discretos de n-pi
    for i in range(len(lambda_sq)):
        plt.plot([lambda_sq[i], lambda_sq[i]], [psi_obs[i], psi_real_desenrollado[i]], 'k:', alpha=0.6)
        
        # Etiquetar el tamaño del salto n-pi
        saltos_pi = (psi_real_desenrollado[i] - psi_obs[i]) / np.pi
        if abs(saltos_pi) > 0.5:
            n = round(saltos_pi)
            plt.text(lambda_sq[i] - 0.001, (psi_obs[i] + psi_real_desenrollado[i])/2, 
                     rf'$+{n}\pi$', color='black', fontsize=12, 
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                     
    # 6. Estética final
    plt.title(r'Desenredado de Fase (Ambigüedad $n\pi$) - Sección Transversal 1D', fontsize=14)
    plt.xlabel(r'Longitud de onda al cuadrado ($\lambda^2$) [m$^2$]', fontsize=12)
    plt.ylabel(r'Ángulo de Polarización ($\Psi$) [rad]', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    
    # Exportar para Overleaf
    plt.savefig("ajuste_1d_npi.pdf", dpi=300, bbox_inches='tight')
    print("   Gráfica guardada exitosamente como 'ajuste_1d_npi.pdf'.")
    plt.show()

if __name__ == "__main__":
    main()