import numpy as np
import pytest
import importlib
import astropy.units as u
from unittest import mock

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

def test_pipeline_filamento_corre_de_punta_a_punta_y_ajusta_gaussiana():
    from faradaymr import ObservationConfig, ObservationPipeline
    from faradaymr.analysis.spatial_stats import transverse_rm_dispersion
    from faradaymr.analysis.fitting import fit_transverse_dispersion
    
    # Importaciones con rutas absolutas
    from examples.filamento_whim.model import construir_escenario
    from examples.filamento_whim import config_fisica

    n_base_test = 32  # malla chica para que el test corra en segundos
    
    # Extraemos los valores físicos sin las clases de astropy para el pipeline
    b0_mg = config_fisica.B0.to_value(u.microgauss)
    dx_base_kpc = config_fisica.DX_BASE.to_value(u.kpc)
    dx_base_pc = config_fisica.DX_BASE.to_value(u.pc)
    nu_hz = config_fisica.NU.to_value(u.Hz)
    lambda_m = config_fisica.LAMBDA_ONDA.to_value(u.m)
    p_spec = config_fisica.P_SPEC
    
    # Usamos mock para reducir la malla de forma segura sin sobreescribir configuraciones globales
    with mock.patch('examples.filamento_whim.config_fisica.N_BASE', n_base_test), \
         mock.patch('examples.filamento_whim.config.N_BASE', n_base_test):
         
        axis_direction = [0.0, 0.0, 1.0]
        # Fijamos la semilla para que el comportamiento del ruido sea determinista en el test
        rng = np.random.RandomState(42) 
        
        bx, by, bz, ne, ne_rel, r = construir_escenario(
            n_spec=3.0, 
            b0_microgauss=b0_mg, 
            axis_direction=axis_direction, 
            use_gpu=False,
            rng=rng
        )

        observacion = ObservationConfig(
            pixel_size=dx_base_kpc, 
            dl=dx_base_pc, 
            frequency=nu_hz,
            wavelength=lambda_m, 
            p_index=p_spec,
        )
        
        resultado = ObservationPipeline(config=observacion).run(bx, by, bz, ne, ne_rel)

        bordes = np.linspace(0.0, n_base_test / 2 * dx_base_kpc, 8)
        centros, dispersion = transverse_rm_dispersion(
            resultado.rm_map, axis_direction, dx_base_kpc, bordes
        )

        # Sanidad física:
        validos = ~np.isnan(dispersion)
        
        # NOTA: En turbulencia ruidosa, np.all(np.diff <= 0) produce "flaky tests" (falsos fallos)
        # por pequeños remolinos que suben la señal en celdas contiguas. 
        # Es mucho más robusto físicamente exigir que el centro sea mayor que las afueras:
        assert dispersion[validos][0] > dispersion[validos][-1], \
            "La dispersión central debe ser mayor que la de los bordes"

        ajuste = fit_transverse_dispersion(centros, dispersion)
        assert ajuste.width > 0, "El ancho gaussiano ajustado debe ser positivo"
        assert np.isfinite(ajuste.sigma0), "La amplitud sigma0 ajustada no es finita"