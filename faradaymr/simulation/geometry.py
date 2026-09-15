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
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp

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
        try:
            import cupy as xp
        except ImportError:
            import numpy as xp

    axis_2d = xp.asarray(filament_axis_3d[:2], dtype=float)
    norma = xp.linalg.norm(axis_2d)
    if norma > 0:
        axis_2d = axis_2d / norma
    else:
        # Filamento paralelo a la línea de visión: su proyección sobre el
        # cielo es un punto, no una recta, así que "distancia al eje
        # proyectado" no tiene una dirección física privilegiada. Se elige
        # [1, 0] como convención arbitraria pero determinística -para que
        # un barrido en theta que pase por este caso límite (theta=0) no
        # se caiga por una división por cero.
        axis_2d = xp.array([1.0, 0.0])

    # Vector normal al eje proyectado (rotación de 90°): la distancia
    # perpendicular de un punto a una recta que pasa por el origen es la
    # proyección de ese punto sobre la normal de la recta.
    normal_2d = xp.array([-axis_2d[1], axis_2d[0]])

    nx, ny = shape
    x = (xp.arange(nx) - nx // 2) * pixel_size
    y = (xp.arange(ny) - ny // 2) * pixel_size
    xx, yy = xp.meshgrid(x, y, indexing="ij")

    return xp.abs(xx * normal_2d[0] + yy * normal_2d[1])