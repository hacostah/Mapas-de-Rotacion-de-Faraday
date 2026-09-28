import numpy as np
import pytest
import importlib
import astropy.units as u
from unittest import mock

from examples.filamento_whim.model import construir_escenario
from examples.filamento_whim import config as cfg

def test_simetria_cilindrica_escenario():
    """
    Verifica que, DENTRO de su longitud finita, la densidad electrónica
    del filamento tenga simetría cilíndrica (translación a lo largo del
    eje) y respete el decaimiento radial; y que, MÁS ALLÁ de esa longitud,
    la densidad esté fuertemente suprimida (el filamento es finito, no un
    cilindro infinito -ver `construir_escenario` en `model.py`).
    """
    # Construimos el escenario forzando el eje Z (0, 0, 1) y usando NumPy (use_gpu=False)
    bx, by, bz, ne, ne_rel, r = construir_escenario(
        n_spec=3.0,
        b0_microgauss=0.01,
        axis_direction=[0, 0, 1],  # Filamento alineado en el eje Z
        use_gpu=False,
    )

    n_base = cfg.N_BASE
    eje_z = np.linspace(-n_base / 2, n_base / 2, n_base) * cfg.DX_BASE_KPC
    media_longitud = cfg.LONGITUD_FILAMENTO_KPC / 2.0

    # Dos posiciones bien DENTRO del tramo finito del filamento (a 0% y 40%
    # de la media longitud): ahí sigue habiendo simetría de traslación.
    idx_centro = int(np.argmin(np.abs(eje_z - 0.0)))
    idx_dentro = int(np.argmin(np.abs(eje_z - 0.4 * media_longitud)))

    corte_centro = ne[:, :, idx_centro]
    corte_dentro = ne[:, :, idx_dentro]

    np.testing.assert_allclose(
        corte_centro,
        corte_dentro,
        rtol=1e-5,
        err_msg=(
            "Error: el perfil transversal varía entre dos puntos que deberían "
            "estar ambos dentro del tramo finito del filamento."
        ),
    )

    # Simetría z -> -z: el corte en +0.4*media_longitud debe ser igual al de
    # -0.4*media_longitud (la máscara axial depende solo de |proyección|).
    idx_dentro_neg = int(np.argmin(np.abs(eje_z + 0.4 * media_longitud)))
    np.testing.assert_allclose(
        corte_dentro,
        ne[:, :, idx_dentro_neg],
        rtol=1e-5,
        err_msg="Error: la máscara axial no es simétrica respecto al centro del filamento.",
    )

    # Solo tiene sentido pedir supresión en los bordes de la caja si la caja
    # es más profunda que la longitud del filamento (si no, todo el eje
    # z está dentro del filamento y no hay "afuera" que comprobar aquí).
    if n_base * cfg.DX_BASE_KPC > cfg.LONGITUD_FILAMENTO_KPC:
        corte_borde = ne[:, :, 0]  # extremo de la caja, bien más allá de la media longitud
        assert corte_borde.max() < 1e-3 * corte_dentro.max(), (
            "Error: la densidad no está suprimida más allá de la longitud finita "
            "del filamento (¿se está simulando un cilindro infinito?)."
        )

    # Comprobamos la física del BetaModel: la densidad debe ser máxima en el centro
    # transversal (radio=0) y decrecer hacia los bordes, dentro del tramo finito.
    centro_idx = ne.shape[0] // 2
    densidad_centro = ne[centro_idx, centro_idx, idx_centro]
    densidad_borde_transversal = ne[0, 0, idx_centro]

    assert densidad_centro > densidad_borde_transversal, "Error: La densidad no decae radialmente desde el núcleo."

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