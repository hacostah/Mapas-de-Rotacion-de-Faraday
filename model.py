"""
Construcción del escenario físico: Vía Láctea de juguete vista por un
observador dentro del disco (Proyecto III).

A diferencia de `icm_faraday_rotation/model.py` (una nube esférica
homogénea vista desde fuera), acá hay dos piezas nuevas que combinar,
ambas ya existentes en el framework pero nunca ensambladas juntas en un
escenario completo:

1. Densidad de electrones: disco delgado + brazos espirales
   (`faradaymr.simulation.GalacticDiskProfile`), en vez de un perfil
   esférico.
2. Campo magnético: componente REGULAR coherente, espiral logarítmica
   (`faradaymr.fields.LogarithmicSpiralField`) -sin la cual la turbulencia
   sola despolarizaría casi toda la señal (ver docstring de ese módulo)-,
   el halo toroidal y el campo en X de Jansson & Farrar 2012
   (`faradaymr.fields.halo_field`), más la componente turbulenta de
   siempre (`GaussianRandomVectorField`), sumadas celda a celda.

Ambas piezas comparten `pitch_angle` y `r0` (radio solar): físicamente
describen la misma espiral (el campo está congelado al gas que lo
sostiene), así que usar el mismo parámetro en las dos no es una
coincidencia de conveniencia, es la consistencia que exige el modelo.
"""

from __future__ import annotations

import config as cfg
from faradaymr import GaussianRandomVectorField, get_backend
from faradaymr.fields import LogarithmicSpiralField, ToroidalHaloField, XField
from faradaymr.simulation import GalacticDiskProfile


def construir_escenario(use_gpu=None, rng=None, arm_contrast=True, return_components=False):
    """
    Arma bx, by, bz, ne, ne_rel para el escenario de foreground galáctico,
    junto con la posición del observador y el tamaño de caja que
    `faradaymr.los_raytrace.sky_map` necesita.

    Convención de coordenadas (ver docstring de `faradaymr.los_raytrace`):
    la caja ocupa [0, box_size] en cada eje, con el índice 0 de cada
    arreglo en la esquina física (0,0,0) -no en el centro-. El centro
    galáctico se coloca en el centro geométrico de la caja
    (`centro_caja`), y el observador se desplaza desde ahí `R_SOLAR_KPC`
    a lo largo de -x, como el Sol respecto al centro de la Vía Láctea
    (misma convención ya validada en
    `tests/test_los_raytrace_visual.py`). `xx_gc, yy_gc, zz_gc` (con "gc"
    de "galactic center") son las coordenadas usadas para evaluar la
    física (R, phi, z respecto al centro galáctico); son un desplazamiento
    rígido de las coordenadas de caja, no un sistema distinto -el arreglo
    resultante (bx, by, bz, ne, ne_rel) sigue indexado exactamente como
    `los_raytrace` espera.

    arm_contrast : ver `GalacticDiskProfile`; False da un disco puramente
        axisimétrico (sin brazos), útil para aislar el efecto de la
        estructura espiral comparando dos corridas.
    return_components : si True, además de la tupla habitual, devuelve
        `((bx_reg, by_reg, bz_reg), (bx_turb, by_turb, bz_turb))` -las
        componentes regular y turbulenta SIN sumar, útil solo para
        graficar la comparación regular-vs-total (ver
        `faradaymr.galactic_structure_plots.comparacion_campo_regular_vs_total`);
        ningún cálculo físico las necesita por separado (por eso no se
        devuelven por defecto: cambiaría la forma de la tupla para quien
        ya llama a esta función sin pedirlas).

    Devuelve
    --------
    bx, by, bz, ne, ne_rel : arreglos (N_BASE, N_BASE, N_BASE).
    observer_pos : arreglo (3,), posición del observador en las mismas
        unidades físicas que `box_size`/`dx` (kpc).
    box_size, dx : escalares (kpc).
    """
    xp = get_backend(use_gpu)

    dx = cfg.DX_BASE_KPC
    box_size = cfg.N_BASE * dx
    centro_caja = box_size / 2.0

    eje = xp.arange(cfg.N_BASE) * dx  # origen en la esquina, ver los_raytrace
    xx, yy, zz = xp.meshgrid(eje, eje, eje, indexing="ij")
    xx_gc, yy_gc, zz_gc = xx - centro_caja, yy - centro_caja, zz - centro_caja

    perfil_disco = GalacticDiskProfile(
        n_e0=cfg.NE0_CM3,
        r0=cfg.R_SOLAR_KPC,
        scale_radial=cfg.SCALE_RADIAL_NE_KPC,
        scale_height=cfg.SCALE_HEIGHT_NE_KPC,
        pitch_angle=cfg.PITCH_ANGLE_RAD,
        n_arms=cfg.N_ARMS,
        arm_width=cfg.ARM_WIDTH_KPC,
        phase0=cfg.ARM_PHASE0_RAD,
        arm_contrast=arm_contrast,
        radial_profile=cfg.NE_RADIAL_PROFILE,
        radial_cutoff=cfg.NE_RADIAL_CUTOFF_KPC,
        arm_r_min=cfg.ARM_R_MIN_KPC,
    )
    ne = perfil_disco.density(xx_gc, yy_gc, zz_gc, xp=xp).astype(xp.float32)
    ne_rel = ne * cfg.NE_REL_FRACCION

    campo_regular = LogarithmicSpiralField(
        b0=cfg.B0_REGULAR_MG,
        r0=cfg.R_SOLAR_KPC,
        pitch_angle=cfg.PITCH_ANGLE_RAD,
        scale_radial=cfg.SCALE_RADIAL_B_KPC,
        scale_height=cfg.SCALE_HEIGHT_B_KPC,
        handedness=cfg.HANDEDNESS,
        anillos_invertidos=cfg.ANILLOS_INVERSION_CAMPO_KPC,
        ancho_vertical=cfg.ANCHO_VERTICAL_B_KPC,
        radio_nucleo=cfg.RADIO_NUCLEO_B_KPC,
        radio_sin_campo=cfg.RADIO_SIN_CAMPO_B_KPC,
    )
    bx_reg, by_reg, bz_reg = campo_regular.sample(xx_gc, yy_gc, zz_gc, xp=xp)

    if cfg.USAR_CAMPO_HALO:
        halo_toroidal = ToroidalHaloField(
            b_norte=cfg.B_HALO_NORTE_MG,
            b_sur=cfg.B_HALO_SUR_MG,
            r_norte=cfg.R_HALO_NORTE_KPC,
            r_sur=cfg.R_HALO_SUR_KPC,
            ancho_halo=cfg.ANCHO_HALO_KPC,
            escala_altura=cfg.ESCALA_ALTURA_HALO_KPC,
            altura_disco=cfg.ALTURA_DISCO_HALO_KPC,
            ancho_disco=cfg.ANCHO_DISCO_HALO_KPC,
            sentido=cfg.SENTIDO_HALO_TOROIDAL,
        )
        campo_x = XField(
            b_x=cfg.B_CAMPO_X_MG,
            elevacion=cfg.ELEVACION_CAMPO_X_RAD,
            r_critico=cfg.R_CRITICO_CAMPO_X_KPC,
            r_x=cfg.R_ESCALA_CAMPO_X_KPC,
        )
        # El "campo regular" que se devuelve por separado (figuras de
        # estructura) es disco + halo: todo lo que no es turbulencia.
        for componente in (halo_toroidal, campo_x):
            bx_h, by_h, bz_h = componente.sample(xx_gc, yy_gc, zz_gc, xp=xp)
            bx_reg, by_reg, bz_reg = bx_reg + bx_h, by_reg + by_h, bz_reg + bz_h

    # Amplitudes de literatura -> efectivas (ver config_fisica.py, sección
    # de campo magnético): un solo factor para todo el campo regular.
    bx_reg, by_reg, bz_reg = (
        cfg.FACTOR_CAMPO_REGULAR * bx_reg,
        cfg.FACTOR_CAMPO_REGULAR * by_reg,
        cfg.FACTOR_CAMPO_REGULAR * bz_reg,
    )

    campo_turbulento = GaussianRandomVectorField(
        n=cfg.N_BASE,
        dx=dx,
        spectral_index=cfg.SPECTRAL_INDEX,
        scale_min=cfg.LAMBDA_MIN_KPC,
        scale_max=cfg.LAMBDA_MAX_KPC,
    )
    if cfg.USAR_ENVOLVENTE_TURBULENCIA:
        radio_gc = xp.sqrt(xx_gc**2 + yy_gc**2)
        envolvente = xp.exp(-radio_gc / cfg.ESCALA_RADIAL_TURBULENCIA_KPC) * xp.exp(
            -(zz_gc**2) / (2.0 * cfg.ESCALA_ALTURA_TURBULENCIA_KPC**2)
        )
        bx_turb, by_turb, bz_turb = campo_turbulento.sample_with_envelope(
            envolvente, use_gpu=use_gpu, rng=rng
        )
        # B0_TURBULENTO es el RMS en la posición del Sol. Como B ≈ f·b con b
        # estadísticamente homogéneo, <b²> = ΣB² / Σf², y el RMS en el Sol
        # es f(R_sol, 0)·sqrt(<b²>). Así no hace falta que la caja contenga
        # celdas cerca del Sol (las mallas chicas de los tests no las tienen).
        envolvente_sol = xp.exp(-cfg.R_SOLAR_KPC / cfg.ESCALA_RADIAL_TURBULENCIA_KPC)
        b_rms_sol = envolvente_sol * xp.sqrt(
            xp.sum(bx_turb**2 + by_turb**2 + bz_turb**2) / xp.sum(envolvente**2)
        )
        factor = cfg.B0_TURBULENTO_MG / b_rms_sol
        bx_turb, by_turb, bz_turb = bx_turb * factor, by_turb * factor, bz_turb * factor
    else:
        bx_turb, by_turb, bz_turb = campo_turbulento.sample(use_gpu=use_gpu, rng=rng)
        bx_turb, by_turb, bz_turb = GaussianRandomVectorField.normalize_to_rms(
            bx_turb, by_turb, bz_turb, cfg.B0_TURBULENTO_MG, xp=xp
        )

    bx = bx_reg + bx_turb
    by = by_reg + by_turb
    bz = bz_reg + bz_turb

    observer_pos = xp.array(
        [centro_caja - cfg.R_SOLAR_KPC, centro_caja, centro_caja]
    )

    resultado = (bx, by, bz, ne, ne_rel, observer_pos, box_size, dx)
    if return_components:
        resultado = resultado + (
            (bx_reg, by_reg, bz_reg),
            (bx_turb, by_turb, bz_turb),
        )
    return resultado
