"""
Componentes de halo del campo magnético galáctico regular, según Jansson &
Farrar (2012, ApJ 757, 14; "JF12"): halo toroidal y campo en X.

Por qué hacen falta además del disco (`LogarithmicSpiralField`): el disco
es plano (B_z = 0) y simétrico respecto al plano, así que por sí solo da
RM(l, b) = RM(l, -b) y ángulos de polarización con el campo siempre
paralelo al disco. El cielo real muestra dos cosas que eso no explica:

1. La antisimetría norte-sur de la RM hacia la Galaxia interior (Han et
   al. 1997; en Oppermann et al. 2012: RM > 0 al norte y < 0 al sur en
   0° < l < 90°, al revés en 270° < l < 360°). La produce un campo
   toroidal de halo que apunta en sentidos opuestos arriba y abajo del
   disco (patrón de dínamo A0).
2. Campo proyectado con una componente vertical fuerte fuera del plano,
   visible en la polarización sincrotrón (WMAP/Planck). JF12 la modelan
   con un campo poloidal en forma de X: líneas que cruzan el plano
   verticalmente en la Galaxia interior y se abren con un ángulo fijo
   hacia afuera.

Ambos se construyen igual que en JF12 (sec. 5.1.2 y 5.1.3). El campo en X
tiene divergencia nula por construcción; el halo toroidal también, porque
es puramente azimutal y su módulo no depende de phi.

Convención de coordenadas: la misma que `LogarithmicSpiralField` (x, y, z
centradas en el centro galáctico, phi = arctan2(y, x) antihorario visto
desde el polo norte). En JF12 el sentido positivo de B_phi es el opuesto
(su eje x apunta del centro al Sol), por eso el sentido global del halo
toroidal es un parámetro (`sentido`) en vez de venir fijo en el signo de
`b_norte`/`b_sur`: se fija en `config_fisica.py` contra el signo observado
de la RM.
"""

from __future__ import annotations

from dataclasses import dataclass


def _transicion_logistica(valor, centro, ancho, xp):
    """L(v, h, w) = 1 / (1 + exp(-2(|v| - h)/w)) de JF12: 0 bien adentro de h, 1 bien afuera."""
    return 1.0 / (1.0 + xp.exp(-2.0 * (xp.abs(valor) - centro) / ancho))


def _a_cartesianas(b_radial, b_azimutal, phi, xp):
    bx = b_radial * xp.cos(phi) - b_azimutal * xp.sin(phi)
    by = b_radial * xp.sin(phi) + b_azimutal * xp.cos(phi)
    return bx, by


@dataclass
class ToroidalHaloField:
    """
    Halo toroidal de JF12 (ecuación 7): campo puramente azimutal, con
    amplitud y radio de corte distintos al norte y al sur del plano,

        B_phi = sentido · exp(-|z|/z0) · L(z, h_disco, w_disco)
                · { b_norte · (1 - L(R, r_norte, w_halo))   si z > 0
                  { b_sur   · (1 - L(R, r_sur,   w_halo))   si z < 0

    El factor L(z, ...) lo apaga dentro del disco (|z| < h_disco), donde
    manda `LogarithmicSpiralField`; exp(-|z|/z0) lo hace decaer hacia
    arriba, y (1 - L(R, ...)) lo corta más allá de r_norte / r_sur.

    b_norte y b_sur con signos opuestos dan la antisimetría norte-sur de
    la RM. Valores de JF12: b_norte = 1.4 µG, b_sur = -1.1 µG,
    r_norte = 9.22 kpc, r_sur = 16.7 kpc, w_halo = 0.20 kpc, z0 = 5.3 kpc,
    h_disco = 0.40 kpc, w_disco = 0.27 kpc.
    """

    b_norte: float
    b_sur: float
    r_norte: float
    r_sur: float
    ancho_halo: float
    escala_altura: float
    altura_disco: float
    ancho_disco: float
    sentido: int = 1

    def __post_init__(self):
        if self.sentido not in (1, -1):
            raise ValueError("sentido debe ser +1 o -1.")

    def sample(self, x, y, z, xp=None):
        if xp is None:
            import numpy as xp

        radio = xp.sqrt(x**2 + y**2)
        phi = xp.arctan2(y, x)
        envolvente_vertical = xp.exp(-xp.abs(z) / self.escala_altura) * _transicion_logistica(
            z, self.altura_disco, self.ancho_disco, xp
        )
        norte = self.b_norte * (1.0 - _transicion_logistica(radio, self.r_norte, self.ancho_halo, xp))
        sur = self.b_sur * (1.0 - _transicion_logistica(radio, self.r_sur, self.ancho_halo, xp))
        b_azimutal = self.sentido * envolvente_vertical * xp.where(z >= 0, norte, sur)

        bx, by = _a_cartesianas(xp.zeros_like(b_azimutal), b_azimutal, phi, xp)
        return bx, by, xp.zeros_like(b_azimutal)


@dataclass
class XField:
    """
    Campo poloidal en X de JF12 (ecuaciones 8-11). Cada línea de campo es
    recta en el plano (R, z) y cruza z = 0 en el radio r_p:

    - Afuera (r_p >= r_critico): las líneas forman un ángulo fijo
      `elevacion` con el plano, r_p = R - |z|/tan(elevacion), y
      |B| = b_x · exp(-r_p/r_x) · r_p/R.
    - Adentro (r_p < r_critico): las líneas se enderezan hasta cruzar el
      plano en vertical, r_p = R·r_critico / (r_critico + |z|/tan(elevacion)),
      ángulo arctan(|z| / (R - r_p)), y |B| = b_x · exp(-r_p/r_x) · (r_p/R)².

    Los factores r_p/R y (r_p/R)² son los que conservan el flujo, y hacen
    el campo solenoidal. B_z tiene el signo de b_x en los dos hemisferios y
    B_R cambia de signo con z: las líneas suben atravesando el disco y se
    abren hacia afuera, de ahí la X vista de canto.

    Valores de JF12: b_x = 4.6 µG, elevacion = 49°, r_critico = 4.8 kpc,
    r_x = 2.9 kpc.
    """

    b_x: float
    elevacion: float
    r_critico: float
    r_x: float

    def sample(self, x, y, z, xp=None):
        if xp is None:
            import numpy as xp

        radio = xp.sqrt(x**2 + y**2)
        phi = xp.arctan2(y, x)
        abs_z = xp.abs(z)
        tan_elevacion = xp.tan(self.elevacion)
        # Evita 0/0 en el eje (R = 0), donde la dirección radial no existe.
        radio_seguro = xp.where(radio == 0, 1e-12, radio)

        r_p_afuera = radio_seguro - abs_z / tan_elevacion
        adentro = r_p_afuera < self.r_critico

        r_p_adentro = radio_seguro * self.r_critico / (self.r_critico + abs_z / tan_elevacion)
        r_p = xp.where(adentro, r_p_adentro, r_p_afuera)
        # En z = 0 dentro de r_critico la línea es vertical: arctan2(0, 0)
        # daría 0, así que se usa arctan2(|z|, R - r_p) solo donde tiene sentido.
        angulo_adentro = xp.where(
            abs_z > 0, xp.arctan2(abs_z, radio_seguro - r_p_adentro), xp.pi / 2
        )
        angulo = xp.where(adentro, angulo_adentro, self.elevacion)

        factor_flujo = xp.where(adentro, (r_p / radio_seguro) ** 2, r_p / radio_seguro)
        b_mag = self.b_x * xp.exp(-r_p / self.r_x) * factor_flujo

        signo_z = xp.where(z >= 0, 1.0, -1.0)
        b_radial = b_mag * xp.cos(angulo) * signo_z
        b_vertical = b_mag * xp.sin(angulo)

        bx, by = _a_cartesianas(b_radial, xp.zeros_like(b_radial), phi, xp)
        return bx, by, b_vertical
