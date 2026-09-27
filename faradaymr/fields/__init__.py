from .gaussian_random_field import GaussianRandomVectorField
from .spiral_field import LogarithmicSpiralField

# Los perfiles de densidad radial (beta, doble beta, NFW, tabulado) vivían
# antes en `fields.profiles`, pero conceptualmente no son "campos"
# (turbulentos, vectoriales): son el modelo del medio sobre el que se
# calculan las integrales de línea de visión. Se movieron a
# `faradaymr.simulation` para que esa distinción quede explícita en la
# estructura del paquete; se re-exportan aquí también para no romper a
# quien ya hacía `from faradaymr.fields import beta_model`.
from ..simulation import (
    BetaModel,
    DensityProfile,
    DoubleBetaModel,
    NFWModel,
    TabulatedProfile,
    beta_model,
)

__all__ = [
    "GaussianRandomVectorField",
    "LogarithmicSpiralField",
    "DensityProfile",
    "BetaModel",
    "DoubleBetaModel",
    "NFWModel",
    "TabulatedProfile",
    "beta_model",
]
