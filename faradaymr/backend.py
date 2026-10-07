from __future__ import annotations

import numpy as _np

try:
    import cupy as _cp

    HAS_GPU = True
except Exception:  # pragma: no cover - entorno sin CUDA
    _cp = None
    HAS_GPU = False


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


def backend_de(*arrays):
    """
    Módulo de arreglos (cupy o numpy) que corresponde a los arreglos dados:
    cupy si alguno ya vive en GPU, numpy en cualquier otro caso.

    Se usa como valor por defecto cuando una función recibe `xp=None`. Antes
    varias funciones hacían `try: import cupy` y usaban cupy siempre que
    estuviera instalado (p.ej. en Colab), aunque se les pasara un arreglo de
    numpy, lo que rompía con "'a' must be a cupy.ndarray object".
    """
    if HAS_GPU and any(isinstance(a, _cp.ndarray) for a in arrays):
        return _cp
    return _np
