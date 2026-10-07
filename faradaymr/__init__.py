"""
faradaymr: framework para simular observaciones de radioastronomía de plasmas
magnetizados (medidas de rotación de Faraday, emisión e imágenes de
polarización sincrotrón).

Este paquete nació como la generalización de una simulación específica del
medio intracúmulo (ver `examples/icm_faraday_rotation`), pero cada pieza
-generación de campos turbulentos, integración de línea de visión, respuesta
instrumental- es independiente del objeto astrofísico que se estudie.
"""

from .backend import get_backend, to_numpy
from .fields import GaussianRandomVectorField
from .io import load_fits, load_hdf5, load_map, save_fits, save_hdf5, save_maps
from .logging_config import configurar_logging, generar_id_simulacion, medir_tiempo_kernel
from .plotting import Plotter
from .simulation import (
    BetaModel,
    DensityProfile,
    DoubleBetaModel,
    NFWModel,
    TabulatedProfile,
    beta_model,
)
from .pipeline import ObservationConfig, ObservationPipeline, ObservationResult
from .analysis.spatial_stats import radial_profile, transverse_rm_dispersion
from .analysis.expected import expected_rm_dispersion_map

__all__ = [
    "get_backend",
    "to_numpy",
    "GaussianRandomVectorField",
    "DensityProfile",
    "BetaModel",
    "DoubleBetaModel",
    "NFWModel",
    "TabulatedProfile",
    "beta_model",
    "ObservationConfig",
    "ObservationPipeline",
    "ObservationResult",
    "configurar_logging",
    "generar_id_simulacion",
    "medir_tiempo_kernel",
    "save_maps",
    "load_map",
    "save_fits",
    "load_fits",
    "save_hdf5",
    "load_hdf5",
    "Plotter",
    "radial_profile",
    "transverse_rm_dispersion",
    "expected_rm_dispersion_map",
]

__version__ = "0.2.0"
