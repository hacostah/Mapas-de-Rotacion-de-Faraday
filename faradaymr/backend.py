from __future__ import annotations

import numpy as _np

_ERROR_CUPY = None
try:
    import cupy as _cp

    # `import cupy` puede funcionar con una instalación rota: si se
    # instalaron dos builds (p. ej. cupy-cuda12x sobre el cupy-cuda13x de
    # Colab) y luego se desinstaló una, queda un paquete `cupy` vacío o con
    # librerías de CUDA que no existen. Se prueba un arreglo, una FFT (usa
    # cuFFT, lo que más falla) y la GPU antes de darlo por bueno.
    _cp.fft.ifftn(_cp.ones((2, 2, 2))).real.sum().item()
    _cp.cuda.runtime.getDeviceCount()
    HAS_GPU = True
except ImportError:  # pragma: no cover - entorno sin cupy
    _cp = None
    HAS_GPU = False
except Exception as error:  # pragma: no cover - cupy instalado pero roto
    _cp = None
    HAS_GPU = False
    _ERROR_CUPY = error


def _mensaje_cupy_roto():
    return (
        f"cupy está instalado pero no funciona ({type(_ERROR_CUPY).__name__}: "
        f"{_ERROR_CUPY}). Suele pasar al tener dos builds de cupy a la vez. "
        "En Colab: `!pip uninstall -y cupy-cuda12x cupy-cuda13x` y luego "
        "`!pip install cupy-cuda13x` (o la build que coincide con "
        "`nvcc --version`), y reiniciar la sesión."
    )


def get_backend(use_gpu: bool | None = None):
    """
    Devuelve el módulo de arreglos a usar (cupy o numpy).

    use_gpu=None  -> usa GPU si está disponible, si no numpy (comportamiento
                      por defecto, transparente para quien llama).
    use_gpu=True  -> exige GPU; falla explícitamente si no hay, para no
                      correr "en silencio" 100x más lento de lo esperado.
    use_gpu=False -> fuerza numpy, útil para pruebas rápidas y reproducibles.
    """
    if use_gpu is False:
        return _np
    if _ERROR_CUPY is not None:
        # Con cupy roto no se cae en silencio a CPU: quien lo instaló
        # esperaba GPU, y la corrida pesada en numpy tarda mucho más.
        raise RuntimeError(_mensaje_cupy_roto())
    if use_gpu is True:
        if not HAS_GPU:
            raise RuntimeError(
                "Se pidió backend GPU pero cupy/CUDA no está disponible en "
                "este entorno."
            )
        return _cp
    return _cp if HAS_GPU else _np


def to_numpy(array):
    """Trae un arreglo a memoria de CPU como numpy, sin importar el backend."""
    if HAS_GPU and isinstance(array, _cp.ndarray):
        return _cp.asnumpy(array)
    return _np.asarray(array)
