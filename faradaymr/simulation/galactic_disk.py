"""
Densidad de electrones térmicos del disco galáctico, con realce cerca de
los brazos espirales.

Por qué hace falta además de los perfiles de `profiles.py`:
`BetaModel`/`DoubleBetaModel`/`NFWModel` son funciones de un radio esférico
r=sqrt(x²+y²+z²), heredado de que el único caso de uso hasta ahora era un
halo (ICM) con simetría esférica respecto al origen de la caja. La Vía
Láctea no tiene esa simetría: es un disco delgado (la densidad cae con la
altura |z| sobre el plano, no con la distancia al centro galáctico en 3D)
con estructura espiral en el plano (x,y) -brazos donde el gas ionizado se
concentra, no una distribución axisimétrica lisa. Por eso `GalacticDiskProfile`
recibe (xx, yy, zz) por separado -no un `r` escalar- y deliberadamente NO
implementa la interfaz `DensityProfile` de `profiles.py`: esa interfaz
(`density(r, xp=None)`) es físicamente incompatible con una geometría que
necesita saber, además de la distancia, en qué dirección azimutal está cada
punto.

Consistencia con el campo magnético regular:
El campo magnético congela el gas ionizado (frozen-in), así que la espiral
que traza el campo (`faradaymr.fields.LogarithmicSpiralField`) y la espiral
donde se concentra la densidad deben ser la MISMA curva geométrica, con el
mismo `pitch_angle`. `LogarithmicSpiralField` fija la dirección de B tal que
en todo punto tan(pitch_angle) = B_R/B_phi (ver su docstring): la dirección
del campo es tangente, en cada punto, a una familia de espirales
logarítmicas con ese pitch. La curva de un brazo concreto de esa misma
familia, phi_brazo(r), se obtiene integrando esa misma condición de
tangencia,

    dr / (r * dphi) = tan(pitch_angle)
    => d(ln r) = tan(pitch_angle) * dphi
    => phi_brazo(r) = phi0 + ln(r/r0) / tan(pitch_angle),

que es la inversa (en el sentido de r<->phi) de la fórmula más común en la
literatura r(phi) = r0*exp((phi-phi0)*tan(pitch)) -exactamente la misma
espiral, solo despejada para phi en función de r en vez de al revés, porque
acá se necesita evaluar "¿qué tan lejos, en ángulo, está este píxel (r,phi)
del brazo más cercano?" en cada punto de una malla ya dada, no trazar la
curva. (Nota: una versión anterior de este cálculo, documentada en
`plan_faradaymr.md`/`carencias_framework_faradaymr.md`, usaba
`phi_brazo = ln(r/r0) * tan(pitch)` -el pitch multiplicando en vez de
dividir-, que corresponde a medir el pitch angle desde la dirección radial
en vez de la azimutal. Aquí se usa división por tan(pitch), consistente con
la convención ya fijada y probada en `LogarithmicSpiralField`
-tan(pitch)=B_R/B_phi, pitch medido desde la dirección azimutal, la
convención estándar en la literatura de GMF, p.ej. Vallée 2015/2016-, para
que campo y densidad compartan literalmente el mismo parámetro sin
necesidad de convertir unidades de ángulo entre ambos módulos.)
"""

from __future__ import annotations

from dataclasses import dataclass


def spiral_arm_density_factor(xx, yy, pitch_angle, r0, n_arms=4, arm_width=0.5, xp=None):
    """
    Factor multiplicativo (>=1) que realza la densidad cerca de `n_arms`
    brazos espirales logarítmicos igualmente espaciados en fase, vistos
    desde arriba del disco (plano x-y; el centro galáctico está en x=y=0
    -si la malla está centrada en otro punto, hay que restarle ese centro
    a xx, yy antes de llamar a esta función, ver `GalacticDiskProfile`).

    El realce se modela como una gaussiana en la distancia angular
    (convertida a distancia física multiplicando por r, no en radianes
    puros) a la curva del brazo más cercano, no como un escalón abrupto,
    porque la concentración real de gas ionizado decae suavemente al
    alejarse del brazo, no de forma discontinua.

    xx, yy : arreglos de coordenadas cartesianas, YA centradas en el
        centro galáctico (mismas unidades que r0, arm_width, p.ej. kpc).
    pitch_angle : radianes, misma convención que
        `faradaymr.fields.LogarithmicSpiralField` (tan(pitch)=B_R/B_phi,
        medido desde la dirección azimutal; ver derivación en el
        docstring del módulo).
    r0 : radio de referencia (mismas unidades que xx, yy) por el que pasa
        el brazo de fase cero -no cambia la física, solo fija el origen
        angular de la familia de espirales, igual que en
        `LogarithmicSpiralField`.
    n_arms : número de brazos, igualmente espaciados en fase (2*pi/n_arms
        entre uno y el siguiente). Vallée (2016) reporta 4 para la Vía
        Láctea.
    arm_width : ancho (1 sigma) de la gaussiana de realce, en las mismas
        unidades que xx, yy.
    """
    if xp is None:
        import numpy as xp
    r = xp.sqrt(xx**2 + yy**2)
    phi = xp.arctan2(yy, xx)
    # r=0 no tiene una fase espiral bien definida (ln(0) diverge); se
    # satura a un radio mínimo positivo, igual que en `NFWModel.density`
    # y en `LogarithmicSpiralField.sample`, en vez de propagar un NaN.
    r_seguro = xp.where(r <= 0, 1e-6, r)
    tan_pitch = xp.tan(pitch_angle)

    factor = xp.ones_like(r)
    for i in range(n_arms):
        fase = 2.0 * xp.pi * i / n_arms
        phi_brazo = xp.log(r_seguro / r0) / tan_pitch + fase
        # Diferencia angular llevada a (-pi, pi] antes de convertirla a
        # distancia física: sin este paso, un brazo que da varias vueltas
        # completas alrededor del centro (ln(r/r0)/tan(pitch) puede ser
        # arbitrariamente grande) produciría una "distancia angular" que
        # crece sin límite en vez de medir la distancia a la vuelta más
        # cercana del brazo.
        delta = xp.mod(phi - phi_brazo + xp.pi, 2.0 * xp.pi) - xp.pi
        distancia_fisica = xp.abs(delta) * r_seguro
        factor = factor + xp.exp(-0.5 * (distancia_fisica / arm_width) ** 2)
    return factor


@dataclass
class GalacticDiskProfile:
    """
    Densidad de electrones térmicos de un disco galáctico delgado con
    realce de densidad cerca de brazos espirales logarítmicos.

        n_e(R, phi, z) = n_e0 * exp(-(R-r0)/scale_radial) * exp(-|z|/scale_height)
                          * spiral_arm_density_factor(R, phi)

    La envolvente (disco exponencial en R y en |z|) es la forma estándar
    para un disco delgado de gas en equilibrio (misma idea que un perfil
    beta para un halo esférico, pero con la simetría correcta para este
    objeto); el factor espiral multiplicativo superpone la estructura de
    brazos sobre esa envolvente suave, sin cambiar su escala global (en
    r0, z=0, lejos de cualquier brazo, `density` se reduce a `n_e0`).

    Deliberadamente NO implementa `DensityProfile` (`profiles.py`): esa
    interfaz asume un radio esférico escalar, que no alcanza para
    distinguir "cerca de un brazo" de "en el interbrazo" a igual r. En vez
    de forzar la geometría espiral dentro de una interfaz que no la
    contempla, se expone una interfaz propia, coordinate-aware
    (`density(xx, yy, zz)`), y se documenta aquí la incompatibilidad en
    vez de esconderla.

    Parámetros
    ----------
    n_e0 : densidad de referencia en R=r0, z=0, lejos de cualquier brazo
        (mismas unidades que se quiera usar en el resto del pipeline,
        típicamente cm^-3).
    r0 : radio de referencia (mismas unidades que xx, yy, zz; análogo al
        radio galactocéntrico solar).
    scale_radial, scale_height : escalas de decaimiento exponencial en R
        y en |z| respectivamente (mismas unidades que r0).
    pitch_angle : radianes; debe coincidir con el que use
        `LogarithmicSpiralField` para el campo regular, por consistencia
        física (frozen-in): ver docstring del módulo.
    n_arms, arm_width : ver `spiral_arm_density_factor`.
    arm_contrast : si False, `density` devuelve solo la envolvente
        axisimétrica (factor espiral fijo en 1), sin brazos. Útil como
        caso límite para pruebas y para aislar el efecto de los brazos al
        compararlo contra la versión con `arm_contrast=True`.
    """

    n_e0: float
    r0: float
    scale_radial: float
    scale_height: float
    pitch_angle: float
    n_arms: int = 4
    arm_width: float = 0.5
    arm_contrast: bool = True

    def density(self, xx, yy, zz, xp=None):
        if xp is None:
            import numpy as xp
        r = xp.sqrt(xx**2 + yy**2)
        envolvente = (
            self.n_e0
            * xp.exp(-(r - self.r0) / self.scale_radial)
            * xp.exp(-xp.abs(zz) / self.scale_height)
        )
        if not self.arm_contrast:
            return envolvente
        factor = spiral_arm_density_factor(
            xx,
            yy,
            self.pitch_angle,
            self.r0,
            n_arms=self.n_arms,
            arm_width=self.arm_width,
            xp=xp,
        )
        return envolvente * factor

    def __call__(self, xx, yy, zz, xp=None):
        return self.density(xx, yy, zz, xp=xp)
