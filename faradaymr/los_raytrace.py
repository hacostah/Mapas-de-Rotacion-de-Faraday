"""
Ray tracing con observador interior a la caja (Proyecto III: Vía Láctea).

`faradaymr.los` integra siempre a lo largo de un eje fijo de la malla
(`axis=-1`): eso equivale a un observador infinitamente lejos, mirando la
caja de frente, con todas las líneas de visión paralelas entre sí. Es
correcto para el ICM o un halo galáctico visto desde fuera (Proyectos I y
II), pero no para la Vía Láctea: el Sol está *dentro* del disco, así que
cada dirección de observación (l, b) atraviesa la caja en un ángulo propio,
con un punto de partida común (la posición del observador) y una longitud
de camino distinta para cada (l, b) -no hay un único eje de la malla que
sirva de línea de visión para todas las direcciones a la vez.

Lo que cambia respecto a `los.py` es *cómo se muestrea la malla a lo largo
del rayo*, no la física de las integrales: una vez que se tiene un perfil
1D de bx, by, bz, ne (y n_rel) muestreado a lo largo de un rayo cualquiera,
`rotation_measure`, `rotation_measure_cumulative`, `stokes_qu`, etc. de
`los.py` se aplican exactamente igual, con `axis=-1` sobre ese perfil -para
esas funciones, "la línea de visión" siempre fue solo "el último eje del
arreglo que se les pasa", sin importar si ese eje es el z fijo de la caja o
un conjunto de puntos muestreados a lo largo de un rayo con dirección
arbitraria.

Deliberadamente NO se implementa un trazador de rayos genérico con
refinamiento adaptativo ni octrees: para un primer modelo de juguete sobre
una malla cartesiana regular alcanza con (1) encontrar dónde sale cada rayo
de la caja mediante álgebra cerrada (intersección rayo-caja tipo "slab",
una sola evaluación por rayo, no una subdivisión espacial), y (2) muestrear
el campo en puntos equiespaciados `dl` a lo largo de ese rayo, interpolando
trilinealmente con `scipy.ndimage.map_coordinates`.

Convención de ejes y unidades:
- `observer_pos`, `direction` y las coordenadas de cada rayo están en las
  mismas unidades físicas que `dx` (el tamaño de celda de la malla, p.ej.
  kpc); dividir entre `dx` es lo que las convierte en índices fraccionarios
  de arreglo que entiende `map_coordinates`.
- Los arreglos 3D (bx, by, bz, ne, ...) deben tener sus tres ejes en el
  mismo orden (x, y, z) que las componentes de `observer_pos`/`direction`
  -la misma convención que ya usa `fields.GaussianRandomVectorField`
  (`meshgrid(..., indexing="ij")`).
- La caja ocupa `[0, box_size]` en cada eje (un origen fijo en una esquina,
  no en el centro); si el observador va cerca del centro de la caja, hay
  que pasar `observer_pos` explícitamente desplazado, no se asume nada.
- Los perfiles muestreados quedan ordenados con el observador en el
  *primer* punto (índice 0) y el borde de la caja en el último -el orden
  natural de "parado en el observador, mirando hacia (l, b)". Como
  `rotation_measure_cumulative`/`stokes_qu` de `los.py` esperan la
  convención opuesta (observador en el *último* índice: así definieron
  "acumulado desde cada punto hasta el borde final de la línea de
  visión"), este módulo invierte el perfil una sola vez, justo después de
  muestrear, antes de llamar a esas funciones -ver `sky_map`.
"""

from __future__ import annotations

import math

from . import los


def _map_coordinates_callable(xp):
    """
    Devuelve la función `map_coordinates` correcta para el backend `xp`:
    `scipy.ndimage` para numpy, `cupyx.scipy.ndimage` para cupy. Se importa
    adentro (no al tope del módulo) para no exigir cupy en entornos sin
    GPU, igual que hace `faradaymr.backend` con el import de cupy.
    """
    if getattr(xp, "__name__", "") == "cupy":
        from cupyx.scipy.ndimage import map_coordinates
    else:
        from scipy.ndimage import map_coordinates
    return map_coordinates


def sample_line_of_sight(
    field_3d, observer_pos, direction, n_samples, dl, dx, xp=None, mode="nearest"
):
    """
    Muestrea `field_3d` en `n_samples` puntos equiespaciados `dl` aparte,
    a lo largo del rayo que parte de `observer_pos` en la dirección
    (unitaria) `direction`, interpolando trilinealmente
    (`map_coordinates(..., order=1)`).

    Parámetros
    ----------
    field_3d : ndarray (nx, ny, nz)
        Campo a interpolar (bx, by, bz, ne, n_rel, ...), en la convención
        de ejes descrita en el docstring del módulo.
    observer_pos : array (3,)
        Posición del observador, en las mismas unidades físicas que `dx`
        (no en índices de malla).
    direction : array (3,)
        Vector unitario de la dirección del rayo (ver
        `direction_from_galactic`).
    n_samples : int
        Número de puntos a muestrear.
    dl : float
        Paso entre puntos consecutivos, en las mismas unidades que
        `observer_pos` y `dx`.
    dx : float
        Tamaño físico de celda de `field_3d` (mismo `dx` para los tres
        ejes: malla cúbica uniforme). Es lo que convierte una posición
        física en el índice fraccionario de arreglo que necesita
        `map_coordinates`.
    mode : str
        Política de `map_coordinates` fuera de los bordes del arreglo.
        "nearest" (por defecto) satura al valor de la celda de borde más
        cercana: razonable aquí porque `n_samples` se calcula (ver
        `sample_fields_along_ray`) para que el rayo no rebase el borde de
        la caja, así que este modo solo puede activarse por un error de
        redondeo en el último punto, no porque el rayo de verdad siga más
        allá de donde hay plasma simulado.

    Devuelve
    --------
    ndarray (n_samples,): el perfil 1D interpolado, con el índice 0 en
    `observer_pos` y el índice n_samples-1 en el punto más lejano del rayo
    (ver nota de convención de ejes en el docstring del módulo).
    """
    if xp is None:
        import numpy as xp
    observer_pos = xp.asarray(observer_pos, dtype=float)
    direction = xp.asarray(direction, dtype=float)

    pasos = xp.arange(n_samples) * dl
    puntos = observer_pos[:, None] + xp.outer(direction, pasos)
    coords_malla = puntos / dx

    map_coordinates = _map_coordinates_callable(xp)
    return map_coordinates(field_3d, coords_malla, order=1, mode=mode)


def direction_from_galactic(l, b, xp=None):
    """
    Vector unitario de dirección a partir de coordenadas galácticas (l, b)
    en radianes:
        x = cos(b) cos(l), y = cos(b) sin(l), z = sin(b)

    Identifica el eje x de la caja con (l=0, b=0) y el eje z con el polo
    galáctico (b=+90°). Es una simplificación deliberada para este primer
    modelo de juguete: no rota la caja simulada a la orientación real del
    disco galáctico en el cielo, solo fija una convención consistente para
    poder generar (y comparar) mapas (l, b) sintéticos; alinear la caja con
    la orientación galáctica real es un problema aparte, no necesario para
    validar que el muestreo del rayo en sí es correcto.

    l, b pueden ser escalares o arreglos de la misma forma (N,); devuelve
    (3,) o (3, N) respectivamente.
    """
    if xp is None:
        import numpy as xp
    cb = xp.cos(b)
    return xp.stack([cb * xp.cos(l), cb * xp.sin(l), xp.sin(b)], axis=0)


def ray_box_exit_distance(observer_pos, direction, box_size, xp=None):
    """
    Distancia `t` a la que el rayo `observer_pos + t*direction` sale de la
    caja `[0, box_size]^3` (o `[0, box_size_eje]` por eje, si `box_size` es
    un arreglo de 3 componentes en vez de un escalar).

    Álgebra de intersección rayo-caja alineada a los ejes ("slab method"):
    para cada eje, el rayo cruza el par de paredes de ese eje en
    `t = (0 - pos)/dir` y `t = (box_size - pos)/dir`; el mayor de los dos
    es el `t` de salida *de ese eje en particular*. El rayo sale de la caja
    completa en el menor de esos tres `t` de salida (el primer par de
    paredes que cruza). Es una sola evaluación algebraica por rayo -no un
    trazador de rayos genérico, ni requiere refinamiento adaptativo/octrees.

    Se asume que `observer_pos` está estrictamente dentro de la caja (si no
    lo está, el resultado puede ser negativo o sin sentido físico; no se
    valida aquí porque el caso de uso previsto -Proyecto III- siempre tiene
    al observador dentro).
    """
    if xp is None:
        import numpy as xp
    observer_pos = xp.asarray(observer_pos, dtype=float)
    direction = xp.asarray(direction, dtype=float)
    box_size = xp.asarray(box_size, dtype=float) * xp.ones(3)

    # dir_i == 0 -> el rayo nunca avanza en ese eje, así que ese eje no debe
    # acotar la salida; se satura a un valor casi nulo (en vez de dividir
    # por cero de verdad) para que t_salida_por_eje quede enorme en ese eje
    # y nunca sea el mínimo.
    dir_seguro = xp.where(direction == 0, 1e-30, direction)
    t_pared_0 = (0.0 - observer_pos) / dir_seguro
    t_pared_1 = (box_size - observer_pos) / dir_seguro
    t_salida_por_eje = xp.maximum(t_pared_0, t_pared_1)
    return float(xp.min(t_salida_por_eje))


def sample_fields_along_ray(
    fields,
    observer_pos,
    direction,
    dl,
    dx,
    box_size,
    margin=1e-6,
    xp=None,
    mode="nearest",
):
    """
    Muestrea varios campos 3D (típicamente bx, by, bz, ne, n_rel) a lo
    largo del mismo rayo, con el mismo `n_samples` para todos, calculado
    una sola vez a partir de dónde sale el rayo de la caja
    (`ray_box_exit_distance`) -así los perfiles resultantes quedan
    alineados punto a punto entre sí.

    `margin` encoge levemente la distancia de salida (factor `1 - margin`)
    antes de calcular `n_samples` por `floor`: es la manera simple de
    garantizar que el último punto muestreado quede estrictamente dentro
    de la caja, sin necesidad de manejar el caso borde de un rayo que cae
    exactamente sobre la pared exterior.

    Parámetros
    ----------
    fields : dict[str, ndarray]
        Nombre -> campo 3D (todos con la misma forma y `dx`).

    Devuelve
    --------
    dict[str, ndarray] : mismo nombre -> perfil 1D de longitud `n_samples`
    (índice 0 = observador, ver convención de ejes en el docstring del
    módulo).
    """
    if xp is None:
        import numpy as xp
    t_salida = ray_box_exit_distance(observer_pos, direction, box_size, xp=xp)
    n_samples = max(1, int(math.floor((t_salida * (1.0 - margin)) / dl)))
    return {
        nombre: sample_line_of_sight(
            campo, observer_pos, direction, n_samples, dl, dx, xp=xp, mode=mode
        )
        for nombre, campo in fields.items()
    }


def sky_map(
    bx,
    by,
    bz,
    ne,
    ne_rel,
    observer_pos,
    dx,
    box_size,
    l_grid,
    b_grid,
    dl,
    frequency,
    wavelength,
    p_index,
    margin=1e-6,
    xp=None,
    mode="nearest",
):
    """
    Mapa de cielo (l, b) en grilla regular -RM, I, Q, U- visto por un
    observador *dentro* de la caja, en vez de un observador externo mirando
    de frente (ver `faradaymr.pipeline.ObservationPipeline`, que asume eso
    último).

    Por cada píxel (l, b): calcula la dirección del rayo
    (`direction_from_galactic`), muestrea bx, by, bz, ne, ne_rel a lo largo
    de ese rayo hasta el borde de la caja (`sample_fields_along_ray`), y
    aplica sobre esos perfiles 1D las mismas funciones de integración de
    `faradaymr.los` que ya se usan para el ICM -sin modificarlas: para esas
    funciones un perfil muestreado a lo largo de un rayo es indistinguible
    de un corte a lo largo del eje fijo de la caja, ambos son solo "un
    arreglo con la línea de visión en el último eje".

    No se usa healpix ni ninguna proyección esférica: `l_grid`/`b_grid` son
    simplemente los valores (en radianes) de una grilla rectangular
    l x b, adecuada para un primer modelo de juguete. Es un doble bucle
    explícito sobre los píxeles (no vectorizado entre píxeles, cada uno
    tiene su propio `n_samples` porque la distancia al borde de la caja
    depende de la dirección) -para el tamaño de grilla de un modelo de
    juguete es explícito y suficientemente rápido; vectorizar el bucle de
    píxeles (rellenando con padding hasta el n_samples máximo) es una
    optimización a futuro, no una corrección de física.

    Devuelve
    --------
    (rm_map, i_map, q_map, u_map) : cada uno ndarray (len(l_grid),
    len(b_grid)).
    """
    if xp is None:
        import numpy as xp

    n_l = len(l_grid)
    n_b = len(b_grid)
    rm_map = xp.zeros((n_l, n_b))
    i_map = xp.zeros((n_l, n_b))
    q_map = xp.zeros((n_l, n_b))
    u_map = xp.zeros((n_l, n_b))

    for i_l in range(n_l):
        for i_b in range(n_b):
            direction = direction_from_galactic(l_grid[i_l], b_grid[i_b], xp=xp)
            perfiles = sample_fields_along_ray(
                {"bx": bx, "by": by, "bz": bz, "ne": ne, "ne_rel": ne_rel},
                observer_pos,
                direction,
                dl,
                dx,
                box_size,
                margin=margin,
                xp=xp,
                mode=mode,
            )

            # los.rotation_measure_cumulative/stokes_qu esperan al
            # observador en el *último* índice (ver docstring del módulo);
            # sample_line_of_sight lo deja en el primero, así que se
            # invierte una sola vez, aquí, antes de usar esas funciones.
            perfiles_obs_al_final = {
                nombre: xp.flip(perfil, axis=-1) for nombre, perfil in perfiles.items()
            }
            pbx = perfiles_obs_al_final["bx"]
            pby = perfiles_obs_al_final["by"]
            pbz = perfiles_obs_al_final["bz"]
            pne = perfiles_obs_al_final["ne"]
            pne_rel = perfiles_obs_al_final["ne_rel"]

            b_perp = los.perpendicular_field_magnitude(pbx, pby, pbz, xp=xp)
            j_nu = los.synchrotron_emissivity(b_perp, pne_rel, frequency, p_index, xp=xp)
            i_map[i_l, i_b] = los.synchrotron_intensity(j_nu, dl, axis=-1, xp=xp)

            psi_0 = los.polarization_angle_intrinsic(pbx, pby, xp=xp)
            rm_cumulative = los.rotation_measure_cumulative(pne, pbz, dl, axis=-1, xp=xp)
            q_val, u_val = los.stokes_qu(
                j_nu, psi_0, rm_cumulative, wavelength, p_index, dl, axis=-1, xp=xp
            )
            q_map[i_l, i_b] = q_val
            u_map[i_l, i_b] = u_val

            rm_map[i_l, i_b] = los.rotation_measure(pne, pbz, dl, axis=-1, xp=xp)

    return rm_map, i_map, q_map, u_map
