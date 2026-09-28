"""
Campos vectoriales turbulentos gaussianos (isotrópicos, solenoidales).

Un campo magnético turbulento no se genera "célula por célula" en el espacio
real, porque eso no permite controlar su espectro de potencia. En cambio se
construye en el espacio de Fourier, donde cada modo k tiene una amplitud
aleatoria cuya varianza sigue un espectro de potencia P(k) ∝ k^{-n} (esto es
lo que se conoce como método de Fourier para campos gaussianos aleatorios, el
mismo truco que se usa para generar condiciones iniciales en simulaciones
cosmológicas de materia oscura). Al transformar de vuelta al espacio real con
una FFT inversa se obtiene un campo con la textura turbulenta deseada.

La condición ∇·B = 0 (sin monopolos magnéticos) no se impone "a mano"
recortando la divergencia después: se garantiza analíticamente construyendo
B_k como el rotacional de un potencial vectorial aleatorio A_k,
    B_k = i k × A_k.
El rotacional de cualquier campo es automáticamente libre de divergencia
(∇·(∇×A) = 0 es una identidad vectorial), así que no hay que "limpiar" nada
después. Esta es la misma idea que ya traía `src/grid.py`; aquí solo se
generaliza para que el índice espectral, la caja de escalas y la resolución
sean parámetros del objeto y no constantes globales, de modo que sirva para
cualquier campo turbulento en radioastronomía (medio intracúmulo, medio
interestelar galáctico, viento estelar, etc.), no solo para el caso del ICM.

"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as _np

from ..backend import get_backend, to_numpy
from ..logging_config import medir_tiempo_kernel

_logger = logging.getLogger(__name__)


def power_law_spectrum(xp, k_mag, spectral_index, k_min, k_max):
    """
    Amplitud espectral sigma(k) ∝ k^{-(n+2)/2}, no nula solo entre k_min y
    k_max.

    n es el índice del espectro de potencia de
    energía, P(k) ∝ k^{-n} (n=5/3 sería Kolmogorov, n=2 es lo típico para
    turbulencia de campo magnético en el ICM). La amplitud del campo (no de
    su energía) escala como la raíz de P(k), de ahí el exponente -(n+2)/2:
    el +2 viene de pasar de densidad de energía por modo a densidad de
    energía por intervalo de k en 3D (el volumen de un cascarón esférico en
    k crece como k^2). k_min y k_max representan la escala de inyección
    (remolinos más grandes, Lambda_max) y la escala de disipación (remolinos
    más chicos, Lambda_min): fuera de ese rango no hay turbulencia que
    modelar, así que la amplitud es cero.
    """
    zeta = spectral_index + 2.0
    k_safe = xp.where(k_mag == 0, 1e-20, k_mag)
    return xp.where((k_mag >= k_min) & (k_mag <= k_max), k_safe ** (-zeta / 2.0), 0.0)


@dataclass
class GaussianRandomVectorField:
    """
    Generador de un campo vectorial 3D turbulento, isotrópico y solenoidal.

    Parametros
    ----------
    n : int
        Número de celdas por lado de la malla cúbica.
    dx : float
        Tamaño físico de celda (en las unidades que se quiera trabajar,
        típicamente kpc; el framework no impone unidades, ver `faradaymr.units`).
    spectral_index : float
        Índice n del espectro de potencia P(k) ∝ k^{-n}.
    scale_min, scale_max : float
        Escala de disipación e inyección de la turbulencia (mismas unidades
        que dx).
    spectrum : callable, opcional
        Función sigma(xp, k_mag, spectral_index, k_min, k_max) -> amplitud.
        Por defecto usa `power_law_spectrum`. Se deja como parámetro
        (en vez de subclasificar) porque es la única pieza del cálculo que
        cambiaría entre distintos modelos de turbulencia.
    """

    n: int
    dx: float
    spectral_index: float
    scale_min: float
    scale_max: float
    spectrum: Callable = power_law_spectrum

    def __post_init__(self):
        self._cache = None
        self._cache_backend = None

    def _k_grid(self, xp):
        k_vec = xp.fft.fftfreq(self.n, d=self.dx) * 2.0 * xp.pi
        kx, ky, kz = xp.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
        return kx, ky, kz, xp.sqrt(kx**2 + ky**2 + kz**2)

    def _grid_y_espectro(self, xp):
        """
        Devuelve (kx, ky, kz, sigma_k), calculándolos una sola vez por
        backend y no una vez por realización.

        Físicamente, k y sigma(k) no son parte del "ruido" de una
        realización: son la geometría de la caja (qué frecuencias espaciales
        caben en ella) y la forma del espectro de potencia que se le
        impuso, ambas fijas mientras no cambien n, dx, el índice espectral
        ni las escalas de inyección/disipación. Lo único que de verdad
        cambia entre una realización Monte Carlo y la siguiente es la fase
        y la amplitud aleatorias del ruido gaussiano de entrada (ver
        `sample`). Recalcular la malla de k y el espectro en cada muestra
        es como resolver de nuevo la función de transferencia de un filtro
        cada vez que se le pasa una señal distinta: el filtro no cambió,
        solo la señal. Para un barrido de miles de realizaciones esa parte
        recalculada de balde puede dominar el tiempo total aunque, vista
        sola, cada llamada sea barata.
        """
        if getattr(self, "_cache_backend", None) is not xp:
            kx, ky, kz, k_mag = self._k_grid(xp)
            sigma_k = self.spectrum(
                xp,
                k_mag,
                self.spectral_index,
                xp.pi / self.scale_max,
                xp.pi / self.scale_min,
            )
            self._cache = (kx, ky, kz, sigma_k)
            self._cache_backend = xp
        return self._cache

    @medir_tiempo_kernel
    def sample(self, use_gpu: Optional[bool] = None, rng=None):
        """
        Extrae una realización aleatoria del campo turbulento.

        Devuelve (bx, by, bz) como arreglos reales de forma (n, n, n), con
        <B>=0 y divergencia numéricamente nula por construcción (rotacional
        de A_k). No se normaliza aquí a un B0 dado: esa es una decisión de
        "cuánto vale el campo" que pertenece a quien arma el escenario físico
        (ver `faradaymr.pipeline`), no a quien genera la textura turbulenta.
        """
        xp = get_backend(use_gpu)
        random = xp.random if rng is None else rng

        kx, ky, kz, sigma_k = self._grid_y_espectro(xp)

        potencial_vectorial = []
        for _ in range(3):
            fase = 2.0 * xp.pi * random.random((self.n, self.n, self.n))
            amplitud = random.rayleigh(1.0, (self.n, self.n, self.n))
            potencial_vectorial.append(sigma_k * amplitud * xp.exp(1j * fase))
        ax_k, ay_k, az_k = potencial_vectorial

        bx_k = 1j * (ky * az_k - kz * ay_k)
        by_k = 1j * (kz * ax_k - kx * az_k)
        bz_k = 1j * (kx * ay_k - ky * ax_k)

        ifftn = xp.fft.ifftn
        return ifftn(bx_k).real, ifftn(by_k).real, ifftn(bz_k).real

    def cache_key(self, seed: Optional[int] = None) -> str:
        """
        Identificador determinístico de esta configuración espectral.

        Un estudio paramétrico corre muchas combinaciones de (n, dx,
        spectral_index, scale_min, scale_max) -y potencialmente distintas
        semillas-, así que la caché en disco no puede identificarse solo por
        "existe un archivo": necesita una clave que distinga cada
        combinación de la anterior, o dos corridas con parámetros distintos
        pisarían el mismo archivo. Se usa un hash corto (16 hex) de un JSON
        canónico de los parámetros en vez de, por ejemplo, concatenar
        valores en el nombre del archivo, para no depender de cómo se
        formatean floats (1.0 vs 1 vs 1.00) ni tener nombres de archivo
        arbitrariamente largos con muchos parámetros.
        """
        payload = {
            "n": self.n,
            "dx": self.dx,
            "spectral_index": self.spectral_index,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "spectrum": getattr(self.spectrum, "__name__", repr(self.spectrum)),
            "seed": seed,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def save(self, path, bx, by, bz, seed: Optional[int] = None) -> None:
        """
        Guarda una realización ya generada junto con los parámetros
        espectrales que la produjeron, en un único archivo `.npz`.

        Se usa `numpy.savez` (no tres `.npy` sueltos) para que la malla y
        sus metadatos viajen siempre juntos como una sola unidad atómica en
        disco: así no hay riesgo de, por ejemplo, borrar o sobreescribir
        `bx.npy` de una corrida y dejar `by.npy`/`bz.npy` de otra por error.
        Los parámetros se guardan junto al campo (no solo en el nombre del
        archivo) para poder validar en `load`/`sample_cached` que un archivo
        encontrado en disco de verdad corresponde a esta configuración, en
        vez de confiar ciegamente en que el nombre no miente.
        """
        directorio = os.path.dirname(os.path.abspath(path))
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        _np.savez(
            path,
            bx=to_numpy(bx),
            by=to_numpy(by),
            bz=to_numpy(bz),
            n=self.n,
            dx=self.dx,
            spectral_index=self.spectral_index,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            seed=-1 if seed is None else seed,
        )

    def _matches(self, data, seed: Optional[int] = None) -> bool:
        """Verifica que un `.npz` cargado corresponda a esta configuración
        espectral (y semilla), antes de confiar en su contenido como caché
        válida."""
        try:
            return (
                int(data["n"]) == self.n
                and float(data["dx"]) == float(self.dx)
                and float(data["spectral_index"]) == float(self.spectral_index)
                and float(data["scale_min"]) == float(self.scale_min)
                and float(data["scale_max"]) == float(self.scale_max)
                and int(data["seed"]) == (-1 if seed is None else seed)
            )
        except Exception:
            return False

    @staticmethod
    def load(path, use_gpu: Optional[bool] = None):
        """Carga (bx, by, bz) de un `.npz` guardado con `save`, en el
        backend pedido (numpy o cupy)."""
        data = _np.load(path)
        xp = get_backend(use_gpu)
        bx, by, bz = data["bx"], data["by"], data["bz"]
        if xp is not _np:
            bx, by, bz = xp.asarray(bx), xp.asarray(by), xp.asarray(bz)
        return bx, by, bz

    def sample_cached(
        self,
        cache_dir,
        use_gpu: Optional[bool] = None,
        seed: Optional[int] = None,
        force: bool = False,
    ):
        """
        Como `sample`, pero evitando recomputar la FFT si ya existe en
        disco una malla generada con los mismos parámetros espectrales.

        Este es el punto de entrada pensado para el estudio paramétrico: la
        generación de mallas de alta resolución (N >= 512^3) es cara en
        tiempo de GPU, y repetirla en cada corrida cuando los parámetros
        espectrales (n, dx, spectral_index, scale_min, scale_max, seed) no
        cambiaron es trabajo perdido. `sample_cached` primero busca un
        archivo `grf_<hash>.npz` en `cache_dir` cuya clave (`cache_key`)
        coincida con la configuración actual; si lo encuentra y sus
        metadatos coinciden, lo carga directamente (bypass total de la FFT
        y de la generación aleatoria). Si no, genera la malla como siempre
        y la guarda para la próxima vez.

        seed: fija la semilla del generador aleatorio y además forma parte
        de la clave de caché, para que dos corridas con distinta semilla
        (o una corrida sin semilla fija) nunca compartan malla por
        accidente. Sin `seed`, cada llamada sin caché existente genera una
        realización nueva, igual que `sample`.

        force=True: ignora cualquier caché existente y regenera (útil para
        invalidar una entrada corrupta o forzar una nueva realización con
        la misma clave).
        """
        os.makedirs(cache_dir, exist_ok=True)
        key = self.cache_key(seed=seed)
        cache_path = os.path.join(cache_dir, f"grf_{key}.npz")

        if not force and os.path.exists(cache_path):
            data = _np.load(cache_path)
            if self._matches(data, seed=seed):
                _logger.info(
                    "Malla n=%d encontrada en caché (%s); se omite la FFT.",
                    self.n,
                    cache_path,
                )
                xp = get_backend(use_gpu)
                bx, by, bz = data["bx"], data["by"], data["bz"]
                if xp is not _np:
                    bx, by, bz = xp.asarray(bx), xp.asarray(by), xp.asarray(bz)
                return bx, by, bz

        _logger.info(
            "Sin malla en caché para n=%d (o force=True); generando por FFT.",
            self.n,
        )
        rng = None
        if seed is not None:
            rng = _np.random.RandomState(seed)
        bx, by, bz = self.sample(use_gpu=use_gpu, rng=rng)
        self.save(cache_path, bx, by, bz, seed=seed)
        return bx, by, bz

    @staticmethod
    def normalize_to_rms(bx, by, bz, b_rms_target, xp=None):
        """
        Reescala un campo turbulento para que su valor RMS sea el pedido.

        la generación espectral fija la *forma* del
        campo (cómo se reparte la energía entre escalas) pero no su
        intensidad absoluta; esa se fija aparte con B0, el campo típico
        observado (por ejemplo ~1 microgauss en el ICM). Reescalar por
        B0 / B_rms preserva la forma espectral y solo cambia la amplitud
        global, así que es la operación físicamente correcta (no basta con
        "normalizar" con el máximo, porque eso cambiaría la estadística).
        """
        if xp is None:
            import numpy as xp
        b_rms = xp.sqrt(xp.mean(bx**2 + by**2 + bz**2))
        factor = b_rms_target / b_rms
        return bx * factor, by * factor, bz * factor

    @staticmethod
    def divergence(bx, by, bz, dx, xp=None):
        """
        Divergencia numérica del campo, solo como diagnóstico.

        No es necesaria para que el campo sea solenoidal (eso ya lo
        garantiza construirlo como un rotacional), pero es la manera más
        directa de verificar que la FFT y el recorte espectral no
        introdujeron un error numérico apreciable: si ∇·B se aleja de cero
        más que el error de redondeo, algo está mal en la malla o en el
        gradiente discreto usado para comprobarlo.
        """
        if xp is None:
            import numpy as xp
        dbx_dx = xp.gradient(bx, dx, axis=0)
        dby_dy = xp.gradient(by, dx, axis=1)
        dbz_dz = xp.gradient(bz, dx, axis=2)
        return dbx_dx + dby_dy + dbz_dz
