"""
Chequeos de calibración de orden de magnitud para el escenario de
foreground galáctico (Proyecto III), Fases A y B de `plan_faradaymr.md`.

Antes de este módulo, `run.py` calculaba la RM sintética hacia el polo
galáctico pero la dejaba "sin comparar aún contra un valor publicado" -es
decir, generaba un número sin decir si ese número es razonable. Eso es
precisamente lo que un comité de tesis objetaría primero (criterio 0.1 de
`plan_faradaymr.md`: "todo parámetro/observable debe poder señalar una
referencia concreta"). Este módulo no reimplementa ningún modelo publicado
(JF12, NE2001 completos) -alcanza, para este chequeo, con verificar que el
*orden de magnitud* del modelo de juguete no está desviado del rango
observado, que es el mínimo de rigor que pide el plan (ver
`validar_amplitud_campo_regular` en `plan_faradaymr.md`, sección 3.3, Fase
A, de donde se adapta este código).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResultadoCalibracion:
    """Resultado de un chequeo de orden de magnitud contra un valor de
    referencia publicado: no es un ajuste fino, es un sí/no barato con
    peso ante un comité (ver docstring del módulo)."""

    nombre: str
    valor_modelo: float
    rango_referencia: tuple
    referencia: str
    dentro_de_tolerancia: bool

    def mensaje(self) -> str:
        estado = "OK" if self.dentro_de_tolerancia else "FUERA DE RANGO"
        return (
            f"[{estado}] {self.nombre}: modelo={self.valor_modelo:.3g}, "
            f"referencia publicada={self.rango_referencia} ({self.referencia})"
        )


def validar_rm_polo(
    rm_polo: float,
    rango_referencia=(-10.0, 10.0),
    referencia: str = (
        "Reconstrucción Galactic Faraday depth de Oppermann et al. (2012, "
        "A&A 542, A93): |RM| de origen galáctico hacia latitudes altas es "
        "de unos pocos rad/m^2 hasta ~10 rad/m^2, un orden de magnitud por "
        "debajo de los cientos de rad/m^2 típicos cerca del plano (ver "
        "también Sun et al. 2008, Fig. 1, panel de RM: el mismo contraste "
        "plano/polo que reproduce este framework)."
    ),
) -> ResultadoCalibracion:
    """
    Compara la RM sintética hacia el polo galáctico contra el rango de RM
    de origen galáctico observado a alta latitud (no contra RM
    extragaláctica total, que incluye una componente propia de la fuente
    de fondo que este modelo de juguete no simula).

    Se compara contra un RANGO (no un solo número): la RM real hacia el
    polo varía con la longitud galáctica y con la línea de visión
    particular (estructura de pequeña escala, halo, etc.), así que pedirle
    a un modelo de juguete que reproduzca un valor puntual exacto sería
    exigir más precisión de la que el modelo (disco+espiral liso, sin
    burbuja local ni halo) puede dar honestamente -el objetivo es orden de
    magnitud y signo plausible, no un ajuste fino (ver criterio de
    tolerancia en `plan_faradaymr.md`, Fase A del Proyecto III).
    """
    dentro = rango_referencia[0] <= rm_polo <= rango_referencia[1]
    return ResultadoCalibracion(
        nombre="RM hacia el polo galáctico",
        valor_modelo=rm_polo,
        rango_referencia=rango_referencia,
        referencia=referencia,
        dentro_de_tolerancia=dentro,
    )


def dm_hacia_direccion(ne, observer_pos, direction, dl, dx, box_size, xp=None):
    """
    Medida de dispersión (DM = integral( n_e dl )) a lo largo de un único
    rayo, en pc/cm^3 -la misma cantidad que reportan los catálogos de
    pulsares, y la forma más barata de calibrar `NE0`/`SCALE_HEIGHT_NE`
    contra un dato real sin reimplementar NE2001/YMW16 (Fase B del
    Proyecto III en `plan_faradaymr.md`).

    Es la misma integral de línea de visión que ya usa
    `faradaymr.los_raytrace.sky_map` para RM, pero sin el campo magnético
    (por eso no se reusa `los.rotation_measure`, que multiplica por
    B_parallel): DM es una propiedad únicamente del plasma térmico, no del
    campo magnético que lo atraviesa.

    `dl`, `dx`, `box_size`, y las coordenadas de `observer_pos` deben venir
    en kpc (la convención del resto del framework); se convierte a pc
    justo antes de sumar, que es la unidad estándar en la que se reporta
    DM en la literatura de pulsares.
    """
    if xp is None:
        import numpy as xp

    from . import los_raytrace as lr

    KPC_A_PC = 1000.0
    perfil_ne = lr.sample_fields_along_ray(
        {"ne": ne}, observer_pos, direction, dl, dx, box_size, xp=xp
    )["ne"]
    return float(xp.sum(perfil_ne) * dl * KPC_A_PC)


def validar_dm_polo(
    dm_polo: float,
    rango_referencia=(10.0, 40.0),
    referencia: str = (
        "Columna total de electrones libres del disco Galáctico hacia el "
        "polo según NE2001 (Cordes & Lazio 2002, ArXiv:astro-ph/0207156): "
        "de orden n0*h_disco ~ 0.03 cm^-3 * ~1 kpc ~ 20-30 pc/cm^3, muy por "
        "debajo de las DM de cientos de pc/cm^3 típicas hacia el plano "
        "(ver también catálogo ATNF de pulsares de alta latitud, "
        "Manchester et al. 2005, para ejemplos observados de ese orden)."
    ),
) -> ResultadoCalibracion:
    """
    Compara la DM sintética hacia el polo galáctico contra el orden de
    magnitud esperado de la columna de electrones del disco fino/grueso de
    NE2001 en esa dirección (Fase B del Proyecto III, `plan_faradaymr.md`).
    """
    dentro = rango_referencia[0] <= dm_polo <= rango_referencia[1]
    return ResultadoCalibracion(
        nombre="DM hacia el polo galáctico",
        valor_modelo=dm_polo,
        rango_referencia=rango_referencia,
        referencia=referencia,
        dentro_de_tolerancia=dentro,
    )
