"""
Inspección visual manual del escenario cilíndrico del filamento (no es un
test automatizado: no empieza con `test_`, así que pytest no lo recolecta;
esa es la razón por la que se movió aquí desde `tests/test_visual_filamento.py`,
donde además usaba `plt.show()`, que bloquearía cualquier corrida en CI).

Uso: desde la raíz del repo,
    python -m examples.filamento_whim.diagnostico_visual
Genera un PNG en examples/filamento_whim/results/plots/diagnostico_visual.png
en vez de abrir una ventana interactiva.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from examples.filamento_whim.model import construir_escenario
from examples.filamento_whim import config


def correr_prueba_visual():
    print("Construyendo el escenario cilíndrico...")
    # Construir un escenario con el filamento alineado al eje X (1, 0, 0)
    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec=3.0,
        b0_microgauss=0.01,
        axis_direction=[1, 0, 0],  # Eje del filamento apuntando en X
        use_gpu=False,
    )

    print("Generando las gráficas...")
    # ne es un cubo 3D. Vamos a tomar cortes a la mitad del cubo.
    mitad = ne.shape[0] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Corte YZ (Perpendicular al eje X del filamento), tomado en X=0 (centro
    # del filamento, bien dentro de su longitud finita).
    corte_transversal = ne[mitad, :, :]
    im1 = ax1.imshow(corte_transversal, cmap='viridis', origin='lower')
    ax1.set_title("Corte Transversal (Plano YZ)\nDebería verse como un Círculo")
    plt.colorbar(im1, ax=ax1, label='Densidad Electrónica')

    # Corte XY (Paralelo al eje X del filamento). El filamento es de
    # longitud FINITA (ver `model.py` / `config_fisica.LONGITUD_FILAMENTO`),
    # así que el "tubo" debe atenuarse hacia los extremos en X, no
    # extenderse recto hasta el borde de la caja.
    corte_longitudinal = ne[:, :, mitad]
    im2 = ax2.imshow(corte_longitudinal, cmap='viridis', origin='lower')
    ax2.set_title(
        f"Corte Longitudinal (Plano XY)\n"
        f"Tubo que se atenúa más allá de ±{config.LONGITUD_FILAMENTO_KPC / 2:.0f} kpc"
    )
    plt.colorbar(im2, ax=ax2, label='Densidad Electrónica')

    plt.tight_layout()

    ruta_salida = os.path.join(
        os.path.dirname(__file__), "results", "plots", "diagnostico_visual.png"
    )
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Generada: {ruta_salida}")


if __name__ == "__main__":
    correr_prueba_visual()
