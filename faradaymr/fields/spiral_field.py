"""
Componente regular (de gran escala) del campo magnético galáctico:
espiral logarítmica coherente.

Por qué hace falta además del campo turbulento (`GaussianRandomVectorField`):
la emisión sincrotrón polarizada depende fuertemente de que exista una
dirección de campo coherente sobre trayectos largos. Un campo puramente
turbulento tiene <B>=0 por construcción: el ángulo de polarización
intrínseco (`los.polarization_angle_intrinsic`, que depende de bx, by
locales) varía de forma esencialmente aleatoria de una celda a la siguiente
a lo largo de la línea de visión, y al integrar Q, U se cancelan entre sí
(despolarización por turbulencia). Para que un mapa de cielo tenga un
patrón de polarización organizado -como el que de verdad se observa en la
Vía Láctea- hace falta una componente de campo que no promedie a cero: el
campo regular a gran escala, sostenido por la rotación diferencial del
disco galáctico actuando sobre el gas ionizado (dínamo galáctico).

Geometría: espiral logarítmica.
Se elige espiral logarítmica (y no, por ejemplo, un campo puramente
azimutal/toroidal) porque es la forma que se observa que traza el campo
magnético galáctico real, alineado con los brazos de gas espiral (Jansson &
Farrar 2012; Ferrière 2001; Sun et al. 2008): las líneas de campo no son
círculos concéntricos, cortan las direcciones radiales con un ángulo
constante, el "pitch angle" p. Esa es, de hecho, la propiedad que define a
una espiral logarítmica: r(phi) = r0 * exp((phi - phi0) * tan(p)).

Este módulo modela solo el campo (magnitud + orientación espiral,
axisimétrica en |B|); la modulación en phi que marca brazos discretos de
*densidad* de gas es responsabilidad del futuro modelo de estructura
espiral (issue "Modelo de estructura espiral (densidad de brazos)"). Ambos
comparten el parámetro `pitch_angle` porque físicamente describen la misma
espiral: el campo está congelado al gas (frozen-in) y por lo tanto debe
seguir la misma geometría que los brazos que lo sostienen, aunque cada uno
se implemente por separado.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogarithmicSpiralField:
    """
    Campo magnético regular con geometría de espiral logarítmica,
    confinado al disco galáctico.

    En coordenadas cilíndricas (R, phi, z), con B_z=0 (el campo regular a
    gran escala es predominantemente planar; la componente vertical, mucho
    más débil, no se modela acá): la dirección se fija enteramente por el
    pitch angle,

        B_R   = handedness * B(R, z) * sin(pitch_angle)
        B_phi = handedness * B(R, z) * cos(pitch_angle)

    de modo que en todo punto tan(pitch_angle) = B_R / B_phi, sin importar
    R, phi o z: esa es precisamente la condición que define una espiral
    logarítmica coherente (a diferencia de, por ejemplo, un campo puramente
    toroidal, donde pitch_angle=0 en todas partes). La magnitud B(R, z) sí
    depende de la posición, y se modela con la misma forma funcional
    (exponencial en R y en |z|) que un disco delgado de gas típico, ya que
    el campo está anclado al plasma que lo sostiene.

    No se impone `div(B)=0` construyendo esto como el rotacional de un
    potencial vectorial (a diferencia de `GaussianRandomVectorField`,
    donde sí es exacto por construcción): con un pitch angle y una
    envolvente radial/vertical fijos de antemano, exigir divergencia nula
    exacta requeriría una envolvente muy particular acoplada al pitch
    angle, lo cual complicaría el modelo sin cambiar su conclusión física
    (un campo regular coherente que sostiene polarización). Esta es la
    misma aproximación que usan los modelos de disco tipo ASS/BSS de la
    literatura (p.ej. Sun et al. 2008; el disco de Jansson & Farrar 2012).
    La divergencia numérica residual puede verificarse con
    `GaussianRandomVectorField.divergence(bx, by, bz, dx)`, que es una
    utilidad genérica y no depende de cómo se generó el campo.

    Parámetros
    ----------
    b0 : float
        Magnitud del campo regular en R=r0, z=0 (mismas unidades que se
        use para el campo turbulento, típicamente microgauss).
    r0 : float
        Radio de referencia donde B=b0 (mismas unidades que R; análogo al
        radio galactocéntrico solar, ~8 kpc en la Vía Láctea).
    pitch_angle : float
        Ángulo de paso de la espiral, en radianes. Debe coincidir con el
        `pitch_angle` que use el modelo de densidad de brazos (issue
        dependiente) para que campo y estructura de gas sean consistentes.
        Valores típicos para la Vía Láctea son negativos (espiral trailing),
        del orden de -11° a -13° (p.ej. Jansson & Farrar 2012: p≈-11.5°).
    scale_radial : float
        Longitud de decaimiento exponencial de |B| con el radio R (mismas
        unidades que R).
    scale_height : float
        Longitud de decaimiento exponencial de |B| con |z| (grosor del
        disco magnetizado; mismas unidades que z).
    handedness : int
        +1 o -1: sentido de enrollamiento global de la espiral (permite
        invertir el signo de B_R y B_phi sin cambiar el pitch angle, que
        solo fija el ángulo entre ambas componentes, no su signo global).
    """

    b0: float
    r0: float
    pitch_angle: float
    scale_radial: float
    scale_height: float
    handedness: int = 1

    def __post_init__(self):
        if self.handedness not in (1, -1):
            raise ValueError("handedness debe ser +1 o -1.")

    def sample(self, x, y, z, xp=None):
        """
        Evalúa el campo regular sobre una malla cartesiana (x, y, z).

        Se espera que x, y, z estén centradas en el centro galáctico (el
        mismo origen respecto al cual se posiciona al observador interior
        en `los_raytrace.sky_map`), con la misma convención cartesiana que
        usa el resto del framework. Devuelve (bx, by, bz), listas para
        sumarse celda a celda directamente con la salida de
        `GaussianRandomVectorField.sample()` (misma forma, mismo sistema
        de coordenadas, sin que haya que tocar la interfaz de ninguna de
        las dos clases):

            bx_turb, by_turb, bz_turb = campo_turbulento.sample(...)
            bx_reg, by_reg, bz_reg = campo_regular.sample(x, y, z)
            bx, by, bz = bx_turb + bx_reg, by_turb + by_reg, bz_turb + bz_reg
        """
        if xp is None:
            import numpy as xp

        radio = xp.sqrt(x**2 + y**2)
        phi = xp.arctan2(y, x)
        # Evita 0/0 en el eje z (R=0): ahí la descomposición polar de todos
        # modos pierde sentido físico (no hay una única dirección radial),
        # así que se satura a un radio mínimo positivo en vez de propagar
        # un NaN.
        radio_seguro = xp.where(radio == 0, 1e-12, radio)

        b_mag = (
            self.b0
            * xp.exp(-(radio_seguro - self.r0) / self.scale_radial)
            * xp.exp(-xp.abs(z) / self.scale_height)
        )

        b_radial = self.handedness * b_mag * xp.sin(self.pitch_angle)
        b_azimutal = self.handedness * b_mag * xp.cos(self.pitch_angle)

        bx = b_radial * xp.cos(phi) - b_azimutal * xp.sin(phi)
        by = b_radial * xp.sin(phi) + b_azimutal * xp.cos(phi)
        bz = xp.zeros_like(b_mag)

        return bx, by, bz
