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

Un punto que NO es solo cambiar el eje de integración: en `los.py`, "bz" no
es "la componente z del campo" en abstracto, es *la componente paralela a
la línea de visión*, y eso solo coincide con bz porque ahí la LOS siempre
es el eje z fijo de la caja. Con un observador interior la LOS es la
dirección (l, b), que para una línea de visión en el plano galáctico
(b≈0) casi no tiene componente z -usar bz directamente ahí sería casi
siempre cero, sin que eso signifique que no hay señal física. Por eso,
antes de llamar a `los.perpendicular_field_magnitude`/
`rotation_measure`/`stokes_qu`, `sky_map` reexpresa el campo muestreado
(bx, by, bz de la caja) en el marco local de cada rayo -dos ejes en el
plano del cielo perpendicular al rayo, y uno a lo largo de él- con
`project_field_to_los_frame`. Esa proyección es la única parte nueva; las
funciones de `los.py` en sí no cambian, solo dejan de recibir bz "crudo" y
reciben la componente ya proyectada a lo largo de la LOS que de verdad les
corresponde en su firma.

Elección de los dos ejes del "plano del cielo" (e1, e2) perpendiculares a
la LOS: para RM e I (que no dependen de ángulo, solo de magnitudes) esa
elección es irrelevante. Pero para `psi_0`/Q/U sí importa *qué* dirección
del plano del cielo se llama "ángulo cero": una base construida de forma
arbitraria por rayo (`_base_perpendicular_al_rayo`, un producto cruz con
un vector de referencia fijo) no varía suavemente de un píxel (l, b) al
vecino, y además tiene un salto explícito de convención cerca de los
polos (donde cambia el vector de referencia para no dividir por un
producto cruz casi nulo). `sky_map` usa en cambio `los_frame_from_galactic`,
la base tangente estándar de la esfera en (l, b) -(e_l, e_b), las
direcciones de longitud y latitud crecientes- que varía suavemente en
(l, b) en todas partes salvo en el polo mismo (una singularidad de
coordenadas inherente a usar (l, b) en absoluto, no un defecto de esta
implementación). `_base_perpendicular_al_rayo` se conserva como utilidad
genérica (y como valor por defecto de `project_field_to_los_frame`) para
quien muestree a lo largo de una `direction` que no venga de una grilla
(l, b), donde no hay una base "natural" con la que compararla.
"""

from __future__ import annotations

import math

from . import los
from .backend import to_numpy


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


def los_frame_from_galactic(l, b, xp=None):
    """
    Base ortonormal tangente estándar de la esfera celeste en el punto
    (l, b): (e_l, e_b, direction), con `direction` el vector radial (la
    propia línea de visión, igual que `direction_from_galactic`), `e_l`
    la dirección de longitud galáctica creciente y `e_b` la de latitud
    creciente -la convención habitual en polarimetría para definir un
    "plano del cielo" local en cada punto de un mapa (l, b) (a diferencia
    de `_base_perpendicular_al_rayo`, que no sabe nada de (l, b) y elige
    sus dos ejes de forma arbitraria).

    Se obtienen derivando `direction(l, b)` respecto a cada coordenada:
        e_l = d(direction)/dl, normalizado -> (-sin(l), cos(l), 0)
        e_b = d(direction)/db, normalizado -> (-sin(b)cos(l), -sin(b)sin(l), cos(b))
    Ambas ya salen unitarias y perpendiculares entre sí y a `direction`
    sin necesidad de normalizar aparte (se puede verificar por sustitución
    directa); a diferencia de la base arbitraria, esta varía de forma
    continua en (l, b) -sin ningún salto de convención- excepto justo en
    el polo (b=±90°), donde `l` deja de tener sentido (una singularidad de
    coordenadas de la esfera misma, no algo que dependa de esta
    implementación).

    l, b pueden ser escalares o arreglos de la misma forma; se devuelve
    (3,) o (3, N) respectivamente, igual que `direction_from_galactic`.
    """
    if xp is None:
        import numpy as xp
    l = xp.asarray(l, dtype=float)
    b = xp.asarray(b, dtype=float)
    direction = direction_from_galactic(l, b, xp=xp)
    cero = xp.zeros_like(l)
    e_l = xp.stack([-xp.sin(l), xp.cos(l), cero], axis=0)
    e_b = xp.stack(
        [-xp.sin(b) * xp.cos(l), -xp.sin(b) * xp.sin(l), xp.cos(b)], axis=0
    )
    return e_l, e_b, direction


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

    Acepta tanto un solo rayo (`direction` de forma `(3,)`, devuelve un
    escalar) como un lote de rayos (`direction` de forma `(3, N)` o
    `(3, ...)` en general, devuelve un arreglo con la forma de las
    dimensiones extra). Es la misma álgebra en ambos casos -solo cambia si
    `observer_pos`/`box_size` se difunden (broadcast) sobre dimensiones
    extra o no- por lo que no hace falta una función separada para el caso
    vectorizado (necesario para procesar todos los píxeles de `sky_map` en
    un solo lote, en vez de un rayo a la vez, ver nota de paralelismo en el
    docstring del módulo).
    """
    if xp is None:
        import numpy as xp
    observer_pos = xp.asarray(observer_pos, dtype=float)
    direction = xp.asarray(direction, dtype=float)
    box_size = xp.asarray(box_size, dtype=float) * xp.ones(3)

    es_lote = direction.ndim > 1
    if es_lote:
        # direction tiene forma (3, ...N); observer_pos/box_size son (3,) y
        # deben difundirse sobre las dimensiones extra agregando ejes (1,)
        # de más, para que la división elemento a elemento que sigue
        # broadcastee correctamente contra cada rayo del lote.
        forma_extra = (1,) * (direction.ndim - 1)
        observer_pos = observer_pos.reshape(3, *forma_extra)
        box_size = box_size.reshape(3, *forma_extra)

    # dir_i == 0 -> el rayo nunca avanza en ese eje, así que ese eje no debe
    # acotar la salida; se satura a un valor casi nulo (en vez de dividir
    # por cero de verdad) para que t_salida_por_eje quede enorme en ese eje
    # y nunca sea el mínimo.
    dir_seguro = xp.where(direction == 0, 1e-30, direction)
    t_pared_0 = (0.0 - observer_pos) / dir_seguro
    t_pared_1 = (box_size - observer_pos) / dir_seguro
    t_salida_por_eje = xp.maximum(t_pared_0, t_pared_1)
    t_salida = xp.min(t_salida_por_eje, axis=0)
    return t_salida if es_lote else float(t_salida)


def _base_perpendicular_al_rayo(direction, xp=None):
    """
    Construye una base ortonormal (e1, e2) del plano perpendicular a
    `direction`, para poder expresar un campo vectorial muestreado a lo
    largo del rayo en un marco local donde la línea de visión juega el
    papel del eje z -el mismo papel que el eje z de la caja jugaba en
    `faradaymr.los` para el caso de observador externo con LOS fija (ver
    nota al final del docstring del módulo).

    Se elige un vector de referencia no paralelo a `direction` (el eje z
    de la caja, salvo que `direction` ya esté casi alineado con z, en
    cuyo caso se usa el eje x -si no, el producto cruz de dos vectores
    casi paralelos sería casi nulo y la base saldría mal condicionada) y
    se construye e1 = normalizar(referencia x direction),
    e2 = direction x e1: (e1, e2, direction) queda ortonormal por
    construcción, para cualquier `direction` unitaria.
    """
    if xp is None:
        import numpy as xp
    direction = xp.asarray(direction, dtype=float)
    referencia = xp.array([0.0, 0.0, 1.0])
    if abs(float(direction[2])) > 0.99:
        referencia = xp.array([1.0, 0.0, 0.0])
    e1 = xp.cross(referencia, direction)
    e1 = e1 / xp.linalg.norm(e1)
    e2 = xp.cross(direction, e1)
    return e1, e2


def project_field_to_los_frame(
    bx_perfil, by_perfil, bz_perfil, direction, xp=None, basis=None
):
    """
    Reexpresa un campo magnético ya muestreado a lo largo de un rayo (en
    las coordenadas x, y, z de la caja) en el marco local *de ese rayo*:
    (b1, b2) en el plano del cielo perpendicular a la línea de visión, y
    b_parallel a lo largo de la línea de visión -exactamente los papeles
    que `faradaymr.los` espera de (bx, by, bz) cuando asume que z es la
    LOS fija (ver `perpendicular_field_magnitude`, `inclination_angle`,
    `polarization_angle_intrinsic`, `rotation_measure*`).

    Como `direction` es constante a lo largo de todo el rayo (el
    observador no cambia de dirección de un punto muestreado al
    siguiente), la proyección es la misma combinación lineal fija en cada
    punto: no hace falta una rotación por muestra, una sola base (e1, e2,
    direction) sirve para el rayo completo.

    basis : tupla opcional (e1, e2) ya calculada. Si no se da, se
    construye aquí mismo con `_base_perpendicular_al_rayo` (una base
    válida pero arbitraria, sin relación con ninguna grilla (l, b)). Para
    un mapa de cielo -donde sí existe una base "natural" en cada píxel-
    `sky_map` pasa en cambio `los_frame_from_galactic(l, b)`, que varía
    suavemente en (l, b) en vez de tener un salto de convención cerca de
    los polos (ver la nota correspondiente en el docstring del módulo).

    Devuelve (b1, b2, b_parallel), cada uno con la misma forma que
    `bx_perfil`/`by_perfil`/`bz_perfil` (típicamente perfiles 1D).
    """
    if xp is None:
        import numpy as xp
    direction = xp.asarray(direction, dtype=float)
    if basis is None:
        e1, e2 = _base_perpendicular_al_rayo(direction, xp=xp)
    else:
        e1, e2 = basis
    b1 = e1[0] * bx_perfil + e1[1] * by_perfil + e1[2] * bz_perfil
    b2 = e2[0] * bx_perfil + e2[1] * by_perfil + e2[2] * bz_perfil
    b_parallel = (
        direction[0] * bx_perfil + direction[1] * by_perfil + direction[2] * bz_perfil
    )
    return b1, b2, b_parallel


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


def _sky_map_chunk(
    bx,
    by,
    bz,
    ne,
    ne_rel,
    observer_pos,
    dx,
    box_size,
    l_chunk,
    b_chunk,
    dl,
    frequency,
    wavelength,
    p_index,
    margin,
    xp,
    mode,
    length_unit_pc=1.0,
):
    """
    Calcula (RM, I, Q, U) para un lote de píxeles (l_chunk, b_chunk) -ambos
    arreglos 1D de la misma longitud `n_pix`- en un solo paso vectorizado:
    todas las direcciones del lote se muestrean con una única llamada a
    `map_coordinates` por campo, en vez de una llamada por píxel.

    Física idéntica a integrar un píxel a la vez (ver docstring de
    `sky_map`); lo único que cambia es que cada rayo del lote tiene su
    propio alcance dentro de la caja (`n_samples_por_rayo`, distinto para
    cada dirección), así que se muestrea hasta el máximo del lote
    (`n_max`) y se ponen a cero las muestras de `ne`/`ne_rel` más allá del
    alcance real de cada rayo -esas muestras son, físicamente, puntos fuera
    de la caja (o repetidos por el recorte `mode="nearest"` de
    `map_coordinates`), no plasma real, así que no deben contribuir a
    ninguna integral. Basta con anular `ne` y `ne_rel` ahí: RM y la
    RM-acumulada dependen de `ne`, y la emisividad sincrotrón j_nu es
    proporcional a `ne_rel`, así que un cero en cualquiera de las dos
    anula exactamente la contribución de esa muestra a todas las integrales
    (RM, I, Q, U) sin necesidad de tocar bx/by/bz.
    """
    n_pix = l_chunk.shape[0]

    e_l, e_b, direction = los_frame_from_galactic(l_chunk, b_chunk, xp=xp)  # (3, n_pix)

    t_salida = ray_box_exit_distance(observer_pos, direction, box_size, xp=xp)  # (n_pix,)
    n_samples_por_rayo = xp.maximum(
        1, xp.floor(t_salida * (1.0 - margin) / dl).astype(xp.int64)
    )
    n_max = int(to_numpy(n_samples_por_rayo).max())

    pasos = xp.arange(n_max) * dl  # (n_max,)
    # puntos: (3, n_pix, n_max) = observer_pos + direction * pasos, para
    # todos los píxeles y todas las muestras del lote a la vez.
    puntos = (
        observer_pos.reshape(3, 1, 1)
        + direction[:, :, None] * pasos[None, None, :]
    )
    coords = (puntos / dx).reshape(3, n_pix * n_max)

    map_coordinates = _map_coordinates_callable(xp)
    campos = {"bx": bx, "by": by, "bz": bz, "ne": ne, "ne_rel": ne_rel}
    perfiles = {
        nombre: map_coordinates(campo, coords, order=1, mode=mode).reshape(n_pix, n_max)
        for nombre, campo in campos.items()
    }

    # Máscara de validez en el orden de muestreo original (índice 0 =
    # observador): anula las muestras más allá del alcance real de cada
    # rayo (ver docstring de esta función).
    indice_muestra = xp.arange(n_max)[None, :]
    valido = indice_muestra < n_samples_por_rayo[:, None]
    perfiles["ne"] = xp.where(valido, perfiles["ne"], 0.0)
    perfiles["ne_rel"] = xp.where(valido, perfiles["ne_rel"], 0.0)

    # los.rotation_measure_cumulative/stokes_qu esperan al observador en el
    # *último* índice (ver docstring del módulo); el muestreo lo deja en
    # el primero, así que se invierte una sola vez, aquí, antes de usar
    # esas funciones. Las muestras anuladas (más allá del alcance de cada
    # rayo) quedan al principio del arreglo invertido -siguen sumando
    # exactamente cero a cualquier suma o suma acumulada, sin importar en
    # qué extremo del eje caigan.
    perfiles = {nombre: xp.flip(perfil, axis=-1) for nombre, perfil in perfiles.items()}
    pne = perfiles["ne"]
    pne_rel = perfiles["ne_rel"]

    # bz de la caja NO es "la componente paralela a la línea de visión"
    # aquí -eso solo es cierto cuando la LOS es el eje z fijo (caso
    # `pipeline.py`). Con observador interior la LOS es `direction`, así
    # que el campo se reexpresa primero en el marco local de cada rayo
    # (ver `project_field_to_los_frame`): b1/b2 hacen el papel de bx/by
    # (plano del cielo) y b_par el de bz (a lo largo de la LOS) en las
    # funciones de `los.py`. `direction`/`e_l`/`e_b` tienen forma (3,
    # n_pix); se agrega un eje de más (`[..., None]`) para que
    # broadcasteen contra los perfiles (n_pix, n_max).
    b1, b2, b_par = project_field_to_los_frame(
        perfiles["bx"],
        perfiles["by"],
        perfiles["bz"],
        direction[:, :, None],
        xp=xp,
        basis=(e_l[:, :, None], e_b[:, :, None]),
    )

    b_perp = los.perpendicular_field_magnitude(b1, b2, b_par, xp=xp)
    j_nu = los.synchrotron_emissivity(b_perp, pne_rel, frequency, p_index, xp=xp)
    i_chunk = los.synchrotron_intensity(j_nu, dl, axis=-1, xp=xp)

    psi_0 = los.polarization_angle_intrinsic(b1, b2, xp=xp)
    # La geometría (posición de las muestras, alcance del rayo) va en las
    # unidades de la caja (`dl`, p.ej. kpc), pero la constante 0.812 de
    # `los.rotation_measure*` solo vale con dl en pc: se convierte aquí,
    # solo para RM. `j_nu` (u.a.) no depende de esta unidad.
    dl_pc = dl * length_unit_pc
    rm_cumulative = los.rotation_measure_cumulative(pne, b_par, dl_pc, axis=-1, xp=xp)
    q_chunk, u_chunk = los.stokes_qu(
        j_nu, psi_0, rm_cumulative, wavelength, p_index, dl, axis=-1, xp=xp
    )
    rm_chunk = los.rotation_measure(pne, b_par, dl_pc, axis=-1, xp=xp)

    return rm_chunk, i_chunk, q_chunk, u_chunk


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
    pixel_chunk_size=4096,
    length_unit_pc=1.0,
):
    """
    Mapa de cielo (l, b) en grilla regular -RM, I, Q, U- visto por un
    observador *dentro* de la caja, en vez de un observador externo mirando
    de frente (ver `faradaymr.pipeline.ObservationPipeline`, que asume eso
    último).

    Por cada píxel (l, b): calcula la dirección del rayo junto con la base
    tangente local del cielo (`los_frame_from_galactic`, ver nota sobre
    Q/U en el docstring del módulo), muestrea bx, by, bz, ne, ne_rel a lo
    largo de ese rayo hasta el borde de la caja, y aplica sobre esos
    perfiles las mismas funciones de integración de `faradaymr.los` que ya
    se usan para el ICM -sin modificarlas: para esas funciones un perfil
    muestreado a lo largo de un rayo es indistinguible de un corte a lo
    largo del eje fijo de la caja, ambos son solo "un arreglo con la línea
    de visión en el último eje".

    Paralelismo (importante para correr en GPU): a diferencia de una
    versión anterior de esta función, que hacía un doble bucle explícito
    en Python sobre cada píxel (l, b) -correcto, pero con un lanzamiento de
    kernel de `map_coordinates` por píxel, que en una GPU desperdicia casi
    todo el tiempo en overhead de lanzamiento en vez de cómputo real-, esta
    versión resuelve TODOS los píxeles de un lote (`pixel_chunk_size` de
    ellos a la vez) en una sola llamada vectorizada a `map_coordinates` por
    campo (bx, by, bz, ne, ne_rel): construye las coordenadas de muestreo
    de los `n_pix` rayos del lote como un único arreglo (3, n_pix*n_max) y
    dispara un solo kernel. Como cada rayo tiene un alcance distinto dentro
    de la caja, se muestrea hasta el máximo del lote (`n_max`) y se anulan
    (ver `_sky_map_chunk`) las muestras que caen más allá del alcance real
    de cada rayo particular -el resultado es matemáticamente idéntico al
    del bucle explícito, célda por célda, solo que expresado como álgebra
    de arreglos en vez de un bucle interpretado por Python. `pixel_chunk_size`
    acota cuántos rayos se resuelven a la vez (y por lo tanto la memoria
    pico: `pixel_chunk_size * n_max * 3` flotantes por campo), para que un
    mapa de alta resolución no intente reservar de golpe más memoria de GPU
    de la que hay disponible; un solo lote (`pixel_chunk_size >= n_l*n_b`)
    es la opción más rápida cuando la memoria alcanza.

    No se usa healpix ni ninguna proyección esférica: `l_grid`/`b_grid` son
    simplemente los valores (en radianes) de una grilla rectangular l x b,
    adecuada para un primer modelo de juguete (ver `faradaymr.plotting_sky`
    para desplegar esta grilla con una proyección de igual área tipo
    Mollweide, como las figuras de Waelkens et al. 2008).

    length_unit_pc : cuántos pc vale una unidad de longitud de la caja
        (`dl`, `dx`, `box_size`, `observer_pos`). Solo afecta a RM y a la
        rotación de Faraday de Q/U, porque la constante 0.812 de
        `los.rotation_measure` exige dl en pc. Default 1.0 (la caja ya está
        en pc, comportamiento histórico); para una caja en kpc pasar
        1000.0. Con la unidad equivocada RM sale mal por ese mismo factor.

    Devuelve
    --------
    (rm_map, i_map, q_map, u_map) : cada uno ndarray (len(l_grid),
    len(b_grid)).
    """
    if xp is None:
        import numpy as xp

    n_l = len(l_grid)
    n_b = len(b_grid)
    n_pix = n_l * n_b

    observer_pos = xp.asarray(observer_pos, dtype=float)
    l_grid = xp.asarray(l_grid, dtype=float)
    b_grid = xp.asarray(b_grid, dtype=float)

    ll, bb = xp.meshgrid(l_grid, b_grid, indexing="ij")  # (n_l, n_b) cada uno
    l_flat = ll.reshape(-1)
    b_flat = bb.reshape(-1)

    rm_flat = xp.empty(n_pix)
    i_flat = xp.empty(n_pix)
    q_flat = xp.empty(n_pix)
    u_flat = xp.empty(n_pix)

    for inicio in range(0, n_pix, pixel_chunk_size):
        fin = min(inicio + pixel_chunk_size, n_pix)
        rm_chunk, i_chunk, q_chunk, u_chunk = _sky_map_chunk(
            bx,
            by,
            bz,
            ne,
            ne_rel,
            observer_pos,
            dx,
            box_size,
            l_flat[inicio:fin],
            b_flat[inicio:fin],
            dl,
            frequency,
            wavelength,
            p_index,
            margin,
            xp,
            mode,
            length_unit_pc,
        )
        rm_flat[inicio:fin] = rm_chunk
        i_flat[inicio:fin] = i_chunk
        q_flat[inicio:fin] = q_chunk
        u_flat[inicio:fin] = u_chunk

    return (
        rm_flat.reshape(n_l, n_b),
        i_flat.reshape(n_l, n_b),
        q_flat.reshape(n_l, n_b),
        u_flat.reshape(n_l, n_b),
    )
