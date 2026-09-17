import numpy as np
import pytest
import importlib

from examples.filamento_whim.model import construir_escenario
from examples.filamento_whim import config as cfg

def test_simetria_cilindrica_escenario():
    """
    Verifica que la densidad electrónica generada en el escenario
    tenga simetría cilíndrica perfecta y respete el decaimiento radial.
    """
    # Construimos el escenario forzando el eje Z (0, 0, 1) y usando NumPy (use_gpu=False)
    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec=3.0, 
        b0_microgauss=0.01,
        axis_direction=[0, 0, 1], # Filamento alineado en el eje Z
        use_gpu=False 
    )

    # Si el modelo es un cilindro a lo largo de Z, cualquier corte transversal (plano XY)
    # a diferentes alturas de Z debe ser exactamente idéntico.
    corte_z_inferior = ne[:, :, 0]
    corte_z_medio = ne[:, :, ne.shape[2] // 2]
    corte_z_superior = ne[:, :, -1]

    # Validamos matemáticamente que las capas son iguales
    np.testing.assert_allclose(
        corte_z_inferior, corte_z_superior, 
        err_msg="Error: El perfil de densidad varía a lo largo del eje Z (no es un cilindro infinito)."
    )
    np.testing.assert_allclose(
        corte_z_inferior, corte_z_medio, 
        err_msg="Error: El centro del cilindro difiere de los extremos."
    )

    # 3. Comprobamos la física del BetaModel: la densidad debe ser máxima en el centro 
    # transversal (radio=0) y decrecer hacia los bordes[cite: 3].
    centro_idx = ne.shape[0] // 2
    densidad_centro = ne[centro_idx, centro_idx, 0]
    densidad_borde = ne[0, 0, 0]
    
    assert densidad_centro > densidad_borde, "Error: La densidad no decae radialmente desde el núcleo."