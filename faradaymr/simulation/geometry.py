from ..backend import backend_de


def cylindrical_radius(xx, yy, zz, axis_direction, xp=None):
    """
    Calcula la distancia perpendicular de cada punto en la malla 
    al eje del filamento (simetría cilíndrica).
    
    Parámetros:
    -----------
    xx, yy, zz : array_like
        Mallas de coordenadas 3D.
    axis_direction : list o array_like
        Vector de dirección del eje del filamento (ej. [0, 0, 1]).
    xp : module, opcional
        Módulo de array (numpy o cupy). Si es None, intenta inferirlo.
    """
    if xp is None:
        xp = backend_de(xx, yy, zz)

    # Apilar las coordenadas en un solo arreglo vectorial
    pos = xp.stack([xx, yy, zz], axis=-1)
    
    # Normalizar el vector director del eje
    eje = xp.asarray(axis_direction, dtype=float)
    eje = eje / xp.linalg.norm(eje)
    
    # Calcular la proyección de cada punto sobre el eje (producto punto)
    proyeccion = xp.tensordot(pos, eje, axes=([-1], [0]))
    
    # Vector perpendicular = Vector original - Vector proyectado
    perp = pos - proyeccion[..., None] * eje
    
    # La distancia cilíndrica es la norma del vector perpendicular
    return xp.linalg.norm(perp, axis=-1)

def axial_projection(xx, yy, zz, axis_direction, xp=None):
    """
    Proyección (con signo) de cada punto de la malla sobre el eje del
    filamento -la coordenada "a lo largo" del filamento, complementaria a
    `cylindrical_radius` (la coordenada "perpendicular" al filamento).

    Se usa para truncar axialmente un filamento de longitud finita: un
    punto pertenece al filamento si `abs(axial_projection(...)) <= L/2`,
    donde L es la extensión física total del objeto (`LONGITUD_FILAMENTO`
    en `examples/filamento_whim/config_fisica.py`).

    Parámetros
    -----------
    xx, yy, zz : array_like
        Mallas de coordenadas 3D.
    axis_direction : list o array_like
        Vector de dirección del eje del filamento.
    xp : module, opcional
        Módulo de array (numpy o cupy).
    """
    if xp is None:
        xp = backend_de(xx, yy, zz)

    pos = xp.stack([xx, yy, zz], axis=-1)
    eje = xp.asarray(axis_direction, dtype=float)
    eje = eje / xp.linalg.norm(eje)
    return xp.tensordot(pos, eje, axes=([-1], [0]))


def sky_footprint_mask(ne, umbral_relativo=1e-3, axis=-1, xp=None):
    """
    Máscara 2D booleana: True donde la línea de visión atraviesa densidad
    apreciable del filamento en algún punto de su profundidad (por encima
    de `umbral_relativo * max(ne)`).

    Por qué hace falta: con un filamento de longitud FINITA (ver
    `model.construir_escenario`, que aplica un corte axial suave sobre
    `ne`), algunos píxeles del mapa 2D quedan "más allá de las puntas" del
    filamento proyectado sobre el cielo y tienen RM≈0 en toda su línea de
    visión -no porque ahí no haya señal por ruido estadístico, sino porque
    genuinamente no hay filamento en ese punto del cielo. Si esos píxeles
    se incluyen al calcular la dispersión transversal de RM
    (`faradaymr.analysis.spatial_stats.transverse_rm_dispersion`), diluyen
    el `std` de cada bin con ceros que no son parte de la física que se
    quiere medir.

    Se calcula directamente sobre la densidad 3D ya construida (`ne`, con
    el corte axial ya aplicado) en vez de con una fórmula geométrica 2D
    aproximada, porque una aproximación así (proyectar el eje al plano del
    cielo y cortar por `|proyección| <= L/2`) solo es exacta cuando el eje
    del filamento no tiene componente a lo largo de la línea de visión
    (theta=90°): para cualquier otro ángulo, la posición a lo largo del eje
    de un punto en la línea de visión cambia con la profundidad, así que
    "está dentro o fuera de la banda proyectada" no es una propiedad fija
    de (x, y) sola. Usar `ne` directamente (que ya incorpora el corte axial
    real, `axial_projection`, vía `model.construir_escenario`) es exacto
    para cualquier ángulo sin necesitar esa aproximación.

    El umbral (0.1% de la densidad pico por defecto) es deliberadamente
    bajo: no debe recortar el perfil radial normal del filamento (a
    `DIST_MAX_AJUSTE = 4*r_c`, la densidad de un perfil beta típico sigue
    siendo ~5-10% del pico), solo excluir las regiones genuinamente vacías
    más allá del corte axial.

    Parámetros
    ----------
    ne : ndarray 3D
        Densidad electrónica ya con el corte axial aplicado (ver
        `model.construir_escenario`). El eje de línea de visión debe ser
        el último (`axis=-1`), la misma convención que usa todo el resto
        del framework (`faradaymr.pipeline.ObservationPipeline`).
    umbral_relativo : float
        Fracción del máximo de `ne` por debajo de la cual se considera que
        un punto no tiene densidad apreciable.
    axis : int
        Eje de integración de la línea de visión (por convención, -1).
    xp : module, opcional
        numpy o cupy.

    Devuelve
    --------
    mascara : ndarray 2D de booleanos, con la forma de `ne` sin el eje LOS.
    """
    if xp is None:
        xp = backend_de(ne)

    pico = xp.max(ne)
    if pico <= 0:
        forma_2d = tuple(n for i, n in enumerate(ne.shape) if i != (axis % ne.ndim))
        return xp.zeros(forma_2d, dtype=bool)
    umbral = umbral_relativo * pico
    return xp.any(ne > umbral, axis=axis)


def filament_body_mask(shape, filament_axis_3d, pixel_size, longitud, r_core, xp=None):
    """
    Máscara 2D: True en los píxeles cuya línea de visión cruza el núcleo del
    filamento COMPLETO dentro de su longitud finita (el "cuerpo"), excluyendo
    las regiones de las puntas.

    Por qué hace falta (además de `sky_footprint_mask`): con un perfil beta
    de caída lenta (beta=0.5, n ~ r^-1.5) la huella por umbral de densidad
    cubre prácticamente todo el mapa para theta <= 60°, así que no excluye
    nada. Los píxeles más allá de las puntas proyectadas tienen una
    sigma_RM mucho menor y, al mezclarse en los bins de distancia
    transversal, diluyen sigma0 justo en la cantidad que depende de theta.

    Derivación: con el eje a = (sin t, 0, cos t) y la LoS a lo largo de z,
    la LoS del píxel con coordenada u a lo largo del eje proyectado pasa lo
    más cerca del eje en la posición axial s* = u / sin t, y cruza el núcleo
    (|r| <= r_c) en un tramo axial de +/- r_c * cot t alrededor de s*. Para
    que ese cruce quede entero dentro de |s| <= L/2:
        |u| <= L sin(t)/2 - r_c cos(t).
    A t=90° es |u| <= L/2 (toda la longitud). Cerca de la vista de frente
    (tan t < 2 r_c / L) ningún píxel cumple la condición; ahí se usa la
    mitad central de la huella proyectada, |u| <= L sin(t)/4. A t=0 la
    proyección es un punto y la máscara es todo el mapa (distancia radial).
    """
    if xp is None:
        import numpy as xp
    nx, ny = shape
    x = (xp.arange(nx) - nx // 2) * pixel_size
    y = (xp.arange(ny) - ny // 2) * pixel_size
    xx, yy = xp.meshgrid(x, y, indexing="ij")
    eje = xp.asarray(filament_axis_3d, dtype=float)
    eje = eje / xp.linalg.norm(eje)
    sin_t = float(xp.hypot(eje[0], eje[1]))
    if sin_t < 1e-12:
        return xp.ones(shape, dtype=bool)
    cos_t = abs(float(eje[2]))
    u = (xx * eje[0] + yy * eje[1]) / sin_t
    # Para ángulos casi de frente (ninguna LoS completa el cruce del
    # núcleo) se conserva la mitad central de la huella proyectada, donde
    # la LoS recorre el cuerpo del filamento, no sus extremos.
    limite = max(longitud * sin_t / 2.0 - r_core * cos_t, longitud * sin_t / 4.0, pixel_size)
    return xp.abs(u) <= limite


def filament_axis_from_viewing_angle(theta_rad):
    """
    Vector unitario del eje del filamento para un ángulo de vista theta_rad
    respecto a la línea de visión (que faradaymr.los asume siempre fija en
    el eje Z de la caja, axis=-1).

    En vez de rotar la línea de visión se rota el objeto: 
    se gira el eje del filamento dentro de la misma caja cúbica
    regular, dejando la geometría de integración exactamente como está.

    theta_rad = 0    -> eje = (0, 0, 1): filamento "de frente" (paralelo a la LoS).
    theta_rad = pi/2 -> eje = (1, 0, 0): filamento "de lado" (perpendicular a la LoS).

    Rotación 2D en (x, z), no 3D genérica: por simetría cilíndrica el
    "roll" del filamento no es observable.
    """
    import numpy as np
    return np.array([np.sin(theta_rad), 0.0, np.cos(theta_rad)])

def projected_axis_distance(shape, filament_axis_3d, pixel_size, xp=None):
    """
    Distancia perpendicular, en el plano del mapa (x, y), de cada píxel al
    eje proyectado de un filamento 3D.

    El observable final de este proyecto no es RM en sí (la topología
    caótica del campo turbulento anula el <RM> promedio, ver
    `faradaymr.analysis.spatial_stats.transverse_rm_dispersion`), sino cómo
    se dispersa RM en cortes perpendiculares al eje del filamento *tal como
    se ve proyectado en el mapa 2D*. Esa distancia no es la misma que usa
    `cylindrical_radius`: aquella opera sobre el cubo 3D completo (antes de
    integrar la línea de visión, para construir el perfil de densidad del
    medio); esta opera sobre el mapa 2D ya integrado, donde la componente z
    del eje del filamento se ignora a propósito -un mapa observado no tiene
    forma de "ver" la profundidad de ese eje, solo su proyección sobre el
    cielo.

    Parámetros
    ----------
    shape : tuple (nx, ny)
        Forma del mapa 2D, en la convención (x, y) que usa el resto del
        framework (`faradaymr.pipeline.ObservationPipeline`: el eje de línea
        de visión es siempre el último eje de la caja 3D, así que al
        integrarlo el mapa resultante queda con eje 0 = x, eje 1 = y).
    filament_axis_3d : array_like de 3 componentes
        Vector de dirección 3D del filamento (p.ej. [sin(theta), 0,
        cos(theta)] para un barrido en el ángulo de inclinación theta
        respecto a la línea de visión).
    pixel_size : float
        Tamaño físico de un píxel del mapa; la distancia devuelta queda en
        esas mismas unidades.
    xp : module, opcional
        numpy o cupy.

    Devuelve
    --------
    distance_map : ndarray de forma `shape`
        Distancia perpendicular de cada píxel a la recta que pasa por el
        centro del mapa con la dirección proyectada del filamento.
    """
    if xp is None:
        import numpy as xp

    nx, ny = shape
    x = (xp.arange(nx) - nx // 2) * pixel_size
    y = (xp.arange(ny) - ny // 2) * pixel_size
    xx, yy = xp.meshgrid(x, y, indexing="ij")

    axis_2d = xp.asarray(filament_axis_3d[:2], dtype=float)
    norma = xp.linalg.norm(axis_2d)
    if norma == 0:
        # Filamento paralelo a la línea de visión (theta=0): su proyección
        # sobre el cielo es un punto, no una recta, así que la vista es
        # circularmente simétrica alrededor del centro del mapa. La
        # distancia relevante es entonces la radial sqrt(x^2+y^2), NO la
        # distancia a una recta con dirección arbitraria [1,0] -esa
        # convención anterior medía |y| en vez de un radio, rompiendo la
        # simetría circular esperada justo en el caso theta=0 (ver
        # discusión de Issue: sesgo del estimador a theta pequeño).
        return xp.sqrt(xx**2 + yy**2)

    axis_2d = axis_2d / norma

    # Vector normal al eje proyectado (rotación de 90°): la distancia
    # perpendicular de un punto a una recta que pasa por el origen es la
    # proyección de ese punto sobre la normal de la recta.
    normal_2d = xp.array([-axis_2d[1], axis_2d[0]])

    return xp.abs(xx * normal_2d[0] + yy * normal_2d[1])