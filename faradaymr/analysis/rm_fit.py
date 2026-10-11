"""Ajuste de la medida de rotación a partir de ángulos observados.

La rotación de Faraday obedece psi(lambda^2) = psi_0 + RM * lambda**2.
Por tanto, un ajuste lineal de psi contra lambda**2 recupera la
pendiente RM y la ordenada al origen psi_0.
"""

from __future__ import annotations
import itertools


def unwrap_polarization_angle(psi_obs_rad, axis=-1, xp=None):
    """Desenrolla saltos artificiales de pi en el ángulo de polarización.
    
    Aprovecha la simetría de 180° de la polarización multiplicando el 
    ángulo por 2, aplicando el unwrap estándar (2*pi) y dividiendo a la mitad.
    
    Parameters
    ----------
    psi_obs_rad : array-like
        Ángulos de polarización observados en radianes.
    axis : int, optional
        Eje a lo largo del cual desenrollar. Por defecto es el último (-1).
    xp : module, optional
        Backend de arreglos compatible con NumPy. Si se omite se usa NumPy.
        
    Returns
    -------
    array-like
        Ángulos de polarización desenrollados continuamente.
    """
    if xp is None:
        import numpy as xp
        
    psi_obs_rad = xp.asarray(psi_obs_rad)
    return xp.unwrap(2.0 * psi_obs_rad, axis=axis) / 2.0


def resolve_n_pi_ambiguity(wavelengths_m, psi_obs_rad, n_candidatos=range(-3, 4), xp=None):
    """API histórica: enumera enteros relativos del primer par y ajusta LS.

    Conserva n_candidatos, añade alternativas completas para otros canales.
    Para ventanas físicas y mapas, usar resolve_n_pi_ambiguity_batch.
    Esta API no restringe RM y no declara unicidad de su solución.
    """
    if xp is None:
        import numpy as xp
    w = xp.asarray(wavelengths_m, dtype=float)
    p = xp.asarray(psi_obs_rad, dtype=float)
    if w.ndim != 1 or p.ndim != 1 or w.size != p.size or w.size < 2:
        raise ValueError('Se requieren vectores 1D de igual longitud y al menos dos canales.')
    if not bool(xp.all(xp.isfinite(w))) or bool(xp.any(w <= 0)) or not bool(xp.all(xp.isfinite(p))):
        raise ValueError('Longitudes positivas y datos finitos son obligatorios.')
    if xp.unique(w).size != w.size:
        raise ValueError('Las longitudes de onda deben ser distintas.')
    ks = list(n_candidatos)
    if not ks or any(int(k) != k for k in ks):
        raise ValueError('n_candidatos debe contener enteros y no estar vacío.')
    x = w**2
    xc = x-x.mean()
    t = (x[2:]-x[0])/(x[1]-x[0])
    # Cota SSE de la recta del primer par con interiores redondeados:
    # (m-2)*pi²/4. Limita la enumeración incluso si los canales vienen desordenados.
    limits = [int(xp.ceil(.5*xp.sqrt(w.size-2)*(1+xp.abs(1-v)+xp.abs(v))))+1 for v in t]
    offsets = list(itertools.product(*(range(-lim,lim+1) for lim in limits)))
    best_cost, best_rm = float('inf'), float('inf')
    for k in ks:
        endpoint = p[1]+k*xp.pi
        predicted = p[0]+t*(endpoint-p[0])
        closest = xp.rint((predicted-p[2:])/xp.pi)
        for offset in offsets:
            y = p.copy()
            y[1] = endpoint
            if w.size > 2:
                y[2:] += xp.pi*(closest+xp.asarray(offset))
            rm = float(xp.sum(xc*y)/xp.sum(xc**2))
            intercept = xp.mean(y)-rm*xp.mean(x)
            cost = float(xp.sum((y-intercept-rm*x)**2))
            if cost < best_cost-1e-12 or (abs(cost-best_cost) <= 1e-12 and abs(rm) < abs(best_rm)):
                best_cost, best_rm = cost, rm
    return best_rm


def resolve_n_pi_ambiguity_batch(
    wavelengths_m,
    psi_obs_rad,
    rm_min,
    rm_max,
    xp=None,
    return_diagnostics=False,
):
    """Desenredado n-pi y mínimos cuadrados con ventana física explícita.

    Enumera ramas del par de mayor separación en lambda² y las alternativas
    enteras de los canales interiores. Cada rama se ajusta por LS, restringiendo
    la pendiente a [rm_min, rm_max]. No usa RM verdadera ni suavizado espacial.
    Es global para el costo angular LS en esta ventana: en un mínimo global
    ningún residuo puede superar pi/2 (cambiar su entero reduciría el costo).
    Por ello el entero relativo de un canal interior difiere como máximo en
    uno del entero más cercano a la interpolación entre los extremos.

    El resultado es (n_pixeles,). Los diagnósticos incluyen los ángulos
    observados más enteros*pi, NO puntos reemplazados por la recta del ajuste.
    La incertidumbre local de una rama no resuelve aliases entre ramas.
    """
    if xp is None:
        import numpy as xp
    psi = xp.asarray(psi_obs_rad, dtype=float)
    if psi.ndim != 2:
        raise ValueError("psi_obs_rad debe tener forma (n_frecuencias, n_pixeles).")
    best_cost = xp.full(psi.shape[1], xp.inf)
    best_rm = xp.full(psi.shape[1], xp.inf)
    second_cost = xp.full(psi.shape[1], xp.inf)
    best_psi0 = xp.zeros(psi.shape[1])
    best_unwrapped = xp.zeros_like(psi)
    for rm, psi0, cost, unwrapped in iter_rm_branch_fits(
        wavelengths_m, psi, rm_min, rm_max, xp=xp
    ):
        # Un empate físico se decide por menor |RM|, nunca por RM verdadera.
        mask = (cost < best_cost - 1e-12) | (
            (xp.abs(cost - best_cost) <= 1e-12) & (xp.abs(rm) < xp.abs(best_rm))
        )
        second_cost = xp.where(mask, best_cost, xp.minimum(second_cost, cost))
        best_cost = xp.where(mask, cost, best_cost)
        best_rm = xp.where(mask, rm, best_rm)
        best_psi0 = xp.where(mask, psi0, best_psi0)
        best_unwrapped = xp.where(mask[None, :], unwrapped, best_unwrapped)
    if return_diagnostics:
        return {
            "rm": best_rm, "psi0": best_psi0, "cost": best_cost,
            "second_cost": second_cost,
            "unique": second_cost - best_cost > 1e-12,
            "psi_unwrapped": best_unwrapped,
            "n_pi": xp.rint((best_unwrapped - psi) / xp.pi).astype(int),
            "at_boundary": (xp.abs(best_rm - rm_min) < 1e-9) |
                           (xp.abs(best_rm - rm_max) < 1e-9),
        }
    return best_rm


def iter_rm_branch_fits(wavelengths_m, psi_obs_rad, rm_min, rm_max, xp=None):
    """Itera ajustes (RM, psi0, SSE angular, ángulos levantados) por rama.

    Ordena internamente los canales y devuelve ángulos en el orden de entrada.
    Consume memoria O(n_frecuencias*n_pixeles), sin cubos gigantes de RM.
    La enumeración crece con el número de canales: destinada a bandas escasas.
    """
    if xp is None:
        import numpy as xp
    w = xp.asarray(wavelengths_m, dtype=float)
    psi = xp.asarray(psi_obs_rad, dtype=float)
    if w.ndim != 1 or psi.ndim != 2 or psi.shape[0] != w.size or psi.shape[1] == 0:
        raise ValueError("psi_obs_rad debe tener forma (n_frecuencias, n_pixeles) no vacía.")
    if w.size < 2 or xp.unique(w).size != w.size:
        raise ValueError("Se requieren al menos dos longitudes de onda distintas.")
    if not bool(xp.all(xp.isfinite(w))) or bool(xp.any(w <= 0)) or not bool(xp.all(xp.isfinite(psi))):
        raise ValueError("Longitudes positivas y ángulos finitos son obligatorios.")
    import math
    if not math.isfinite(rm_min) or not math.isfinite(rm_max) or rm_min >= rm_max:
        raise ValueError("La ventana RM debe ser finita y rm_min < rm_max.")
    # Canonicalizar observaciones equivalentes módulo pi.
    canonical = (psi + xp.pi / 2) % xp.pi - xp.pi / 2
    order_sorted = xp.argsort(w ** 2)
    order = xp.concatenate((order_sorted[:1], order_sorted[-1:], order_sorted[1:-1]))
    inv = xp.argsort(order)
    x = (w ** 2)[order]
    p = canonical[order]
    delta = float(x[1] - x[0])
    diff = p[1] - p[0]
    k_min = int(xp.floor((rm_min * delta - xp.max(diff)) / xp.pi)) - 1
    k_max = int(xp.ceil((rm_max * delta - xp.min(diff)) / xp.pi)) + 1
    xc = x - xp.mean(x)
    denominator = xp.sum(xc ** 2)
    t = (x[2:] - x[0]) / delta
    offsets = list(itertools.product((-1, 0, 1), repeat=w.size - 2))
    for k in range(k_min, k_max + 1):
        y_end = p[1] + k * xp.pi
        prediction = p[0][None, :] + t[:, None] * (y_end - p[0])[None, :]
        n_mid = xp.rint((prediction - p[2:]) / xp.pi)
        for offset in offsets:
            y = p.copy()
            y[1] = y_end
            if w.size > 2:
                y[2:] += xp.pi * (n_mid + xp.asarray(offset)[:, None])
            ym = xp.mean(y, axis=0)
            rm = xp.clip(xp.sum(xc[:, None] * y, axis=0) / denominator, rm_min, rm_max)
            psi0 = ym - rm * xp.mean(x)
            residual = y - psi0[None, :] - rm[None, :] * x[:, None]
            cost = xp.sum(residual ** 2, axis=0)
            # Recuperar también la representación equivalente del ángulo de entrada.
            yield rm, psi0, cost, y[inv]

def estimate_rm_unwrapped_lsq(
    wavelengths_m,
    psi_obs_rad,
    rm_min,
    rm_max,
    xp=None,
):
    """Desenreda la fase en una ventana física de RM y aplica mínimos cuadrados.

    La ventana ``[rm_min, rm_max]`` es parte explícita del experimento: fuera de
    ella existen soluciones alias igualmente compatibles con la periodicidad de
    la polarización. Usa la misma enumeración completa que la versión batch;
    fija a cero el entero del primer canal para eliminar la redundancia global.
    """
    if xp is None:
        import numpy as xp
    psi = xp.asarray(psi_obs_rad, dtype=float)
    if psi.ndim != 1:
        raise ValueError("psi_obs_rad debe ser un vector 1D.")
    return float(resolve_n_pi_ambiguity_batch(
        wavelengths_m, psi[:, None], rm_min, rm_max, xp=xp
    )[0])


def estimate_rm_lsq(wavelengths_m, psi_obs_rad, xp=None):
    """Estima (RM, psi_0) mediante mínimos cuadrados lineales.

    Parameters
    ----------
    wavelengths_m : array-like, shape (n,)
        Longitudes de onda en metros.
    psi_obs_rad : array-like, shape (n,)
        Ángulos de polarización observados, en radianes y ya desenrollados
        respecto de la ambigüedad n*pi.
    xp : module, optional
        Backend de arreglos compatible con NumPy. Si se omite se usa NumPy.

    Returns
    -------
    rm : scalar
        Medida de rotación estimada, en rad/m^2.
    psi_0 : scalar
        Ángulo de polarización intrínseco estimado, en radianes.
    """
    if xp is None:
        import numpy as xp

    wavelengths_m = xp.asarray(wavelengths_m, dtype=float)
    psi_obs_rad = xp.asarray(psi_obs_rad, dtype=float)

    if wavelengths_m.ndim != 1 or psi_obs_rad.ndim != 1:
        raise ValueError("wavelengths_m y psi_obs_rad deben ser vectores 1D.")
    if wavelengths_m.size != psi_obs_rad.size:
        raise ValueError("wavelengths_m y psi_obs_rad deben tener la misma longitud.")
    if wavelengths_m.size < 2:
        raise ValueError("Se requieren al menos dos longitudes de onda.")
    if not bool(xp.all(xp.isfinite(wavelengths_m))) or not bool(
        xp.all(xp.isfinite(psi_obs_rad))
    ):
        raise ValueError("wavelengths_m y psi_obs_rad deben contener valores finitos.")
    if bool(xp.any(wavelengths_m < 0)):
        raise ValueError("wavelengths_m no puede contener longitudes de onda negativas.")

    wavelengths_squared = wavelengths_m**2

    if not bool(xp.all(xp.isfinite(wavelengths_squared))):
        raise ValueError("wavelengths_m**2 debe contener valores finitos.")
    if xp.unique(wavelengths_squared).size < 2:
        raise ValueError("Se requieren al menos dos valores distintos de lambda**2.")

    rm, psi_0 = xp.polyfit(wavelengths_squared, psi_obs_rad, deg=1)
    return rm, psi_0
