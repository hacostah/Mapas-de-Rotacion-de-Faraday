import numpy as np
import warnings

def verificar_caja_suficiente(n_base: int, dx_base_kpc: float, longitud_filamento_kpc: float, thetas_grados: list, logger=None) -> bool:
    """
    Para el ángulo más oblicuo del barrido, el filamento proyecta su
    longitud completa dentro de la caja con un factor 1/cos(theta).
    Si la caja no alcanza para el peor caso, distintos theta del barrido
    verían un filamento truncado de forma distinta, contaminando la
    comparación entre ángulos.
    """
    if not thetas_grados:
        return True
        
    theta_max = max(abs(t) for t in thetas_grados)
    lado_caja_kpc = n_base * dx_base_kpc
    
    # Prevenir división por cero si theta es exactamente 90 grados
    factor_proyeccion = 1.0 / max(np.cos(np.deg2rad(theta_max)), 1e-6)
    longitud_necesaria = longitud_filamento_kpc * factor_proyeccion

    if longitud_necesaria > lado_caja_kpc:
        msg = (
            f"La caja ({lado_caja_kpc:.0f} kpc) no alcanza para theta={theta_max}° "
            f"sin truncar el filamento (se necesitan ~{longitud_necesaria:.0f} kpc). "
            "Aumentar N_BASE o DX_BASE, o acotar el rango de theta."
        )
        if logger is not None:
            logger.warning(msg)
        else:
            warnings.warn(msg)
            
    return bool(longitud_necesaria <= lado_caja_kpc)