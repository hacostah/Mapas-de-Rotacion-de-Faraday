import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from faradaymr.analysis.spatial_stats import transverse_rm_dispersion
from faradaymr.analysis.fitting import fit_transverse_dispersion
from faradaymr.simulation.geometry import filament_axis_from_viewing_angle
from examples.filamento_whim import config_fisica

# Rutas de los datos generados por run.py y run_barrido_theta.py
BASE_DIR = os.path.dirname(__file__)
DIR_MAPAS = os.path.join(BASE_DIR, "results", "mapas_poster")
ARCHIVO_MC = os.path.join(BASE_DIR, "results", "barrido_theta", "barrido_theta_mc.npz")
DIR_PLOTS = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(DIR_PLOTS, exist_ok=True)

# Estilo unificado para figuras académicas
plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

def cargar_mapa_rm(theta_grados):
    """Función auxiliar para leer el mapa RM de la ruta correcta."""
    ruta = os.path.join(DIR_MAPAS, f"theta_{theta_grados:02.0f}deg", "rm_mapa.npy")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Falta el mapa para {theta_grados}°. Ejecuta run.py primero.")
    return np.load(ruta)

def fig1_mapa_rm_ejemplo():
    angulos = [0, 15, 30]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    dx_kpc = config_fisica.DX_BASE.to_value(u.kpc)
    limite = config_fisica.N_BASE / 2 * dx_kpc
    
    for ax, theta in zip(axs, angulos):
        try:
            rm_map = cargar_mapa_rm(theta)
        except FileNotFoundError as e:
            ax.set_title(str(e), fontsize=10)
            continue
            
        im = ax.imshow(
            rm_map, cmap='RdBu_r', origin='lower',
            extent=[-limite, limite, -limite, limite]
        )
        ax.set_title(f"Medida de Rotación - $\\theta = {theta}^\\circ$")
        ax.set_xlabel("x (kpc)")
        if theta == 0:
            ax.set_ylabel("y (kpc)")
            
        fig.colorbar(im, ax=ax, label=r"RM (rad/m$^2$)", fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    ruta_salida = os.path.join(DIR_PLOTS, "fig1_mapas_rm.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    print(f"Generada: {ruta_salida}")

def fig2_perfil_transversal_con_ajuste():
    theta = 30
    try:
        rm_map = cargar_mapa_rm(theta)
    except FileNotFoundError:
        print(f"Omitiendo Fig 2: Falta mapa RM para {theta}°")
        return
        
    dx_kpc = config_fisica.DX_BASE.to_value(u.kpc)
    axis_dir = filament_axis_from_viewing_angle(np.deg2rad(theta))
    
    bordes = np.linspace(0, config_fisica.N_BASE/2 * dx_kpc, 15)
    centros, dispersion = transverse_rm_dispersion(rm_map, axis_dir, dx_kpc, bordes)
    
    if hasattr(centros, 'get'):
        centros = centros.get()
    if hasattr(dispersion, 'get'):
        dispersion = dispersion.get()

    validos = ~np.isnan(dispersion)
    ajuste = fit_transverse_dispersion(centros[validos], dispersion[validos])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(centros, dispersion, color='black', label='Datos (Simulación)')
    
    x_fit = np.linspace(0, max(centros), 100)
    y_fit = ajuste.sigma0 * np.exp(-0.5 * (x_fit / ajuste.width)**2)
    
    plt.plot(x_fit, y_fit, 'r-', lw=2, 
             label=rf'Ajuste Gaussiano\n$\sigma_0={ajuste.sigma0:.1f}$\n$w={ajuste.width:.1f}$ kpc')

    plt.xlabel("Distancia transversal al eje (kpc)")
    plt.ylabel(r"$\sigma_{RM}$ (rad/m$^2$)")
    plt.title(f"Perfil de Dispersión RM Transversal ($\\theta={theta}^\\circ$)")
    plt.legend()
    
    ruta_salida = os.path.join(DIR_PLOTS, "fig2_perfil_transversal.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    print(f"Generada: {ruta_salida}")

def fig3_ancho_vs_theta():
    if not os.path.exists(ARCHIVO_MC):
        print("Omitiendo Fig 3: Falta el archivo del barrido Monte Carlo.")
        return
        
    datos = np.load(ARCHIVO_MC)
    theta = datos["theta_grados"]
    width = datos["width_medio_kpc"]
    width_err = datos["width_std_kpc"]
    
    plt.figure(figsize=(8, 6))
    n_semillas = 50
    error_estandar = np.array(width_err) / np.sqrt(n_semillas)

    plt.errorbar(
            theta, width, yerr=error_estandar, 
            fmt='o-', color='navy', ecolor='darkred', capsize=5, 
            capthick=1.5, markerfacecolor='white', markeredgewidth=1.5
        )
    
    plt.xlabel(r"Ángulo de visión $\theta$ (grados)")
    plt.ylabel("Ancho transversal característico $w$ (kpc)")
    plt.title("Ancho Aparente del Filamento vs. Ángulo de Visión")
    
    ruta_salida = os.path.join(DIR_PLOTS, "fig3_ancho_vs_theta.png")
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    print(f"Generada: {ruta_salida}")

if __name__ == "__main__":
    fig1_mapa_rm_ejemplo()
    fig2_perfil_transversal_con_ajuste()
    fig3_ancho_vs_theta()