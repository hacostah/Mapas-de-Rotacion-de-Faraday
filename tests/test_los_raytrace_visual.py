"""
Prueba de inspección visual del mapa de cielo con observador interior.

A diferencia de `test_los_raytrace.py` (que valida números contra fórmulas
analíticas o solo formas de arreglo), esta prueba arma un disco galáctico
de juguete -densidad y campo magnético con estructura espacial reconocible,
no un campo uniforme- y guarda un PNG del mapa de cielo (l, b) resultante,
para poder mirarlo y confirmar a ojo que la forma cualitativa es la
esperada (brillo/RM concentrados cerca del plano galáctico, b~0, cayendo
hacia los polos). También deja un par de aserciones cuantitativas mínimas
(sin NaN/Inf, y la intensidad del plano por encima de la de los polos por
un margen amplio) para que siga siendo una prueba automática de verdad y no
solo un script que hay que mirar para saber si pasó.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from faradaymr import los_raytrace as lr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _campo_disco_de_juguete(n_celdas, dx, escala_radial, escala_vertical):
    """
    bx, by, bz, ne, ne_rel de un disco galáctico de juguete:

    - n_e y n_rel caen exponencialmente con el radio cilíndrico R y la
      altura |z| sobre el plano (perfil típico de un disco delgado, no de
      un halo esférico como el ICM).
    - B es puramente azimutal, B_phi(R) = B0 * exp(-R/escala_radial) (un
      campo galáctico coherente a gran escala "de juguete", tipo espiral
      logarítmica sin brazos), para que el signo de RM cambie de forma
      reconocible según de qué lado del centro pase la línea de visión.

    El centro del disco queda en el centro geométrico de la caja; el
    observador se desplaza de ahí a lo largo de x (ver la prueba de más
    abajo), como el Sol respecto al centro galáctico.
    """
    coords = (np.arange(n_celdas) - (n_celdas - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    radio = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    perfil_disco = np.exp(-radio / escala_radial) * np.exp(-np.abs(z) / escala_vertical)
    ne = 1e-2 * perfil_disco
    ne_rel = perfil_disco.copy()  # mismo perfil espacial, por simplicidad

    b_mag = 3.0 * np.exp(-radio / escala_radial)
    bx = -b_mag * np.sin(phi)
    by = b_mag * np.cos(phi)
    bz = np.zeros_like(bx)

    return bx, by, bz, ne, ne_rel


def test_sky_map_disco_galactico_de_juguete_genera_png_de_inspeccion(tmp_path):
    n_celdas = 64
    dx = 0.5  # kpc/celda -> caja de 32 kpc de lado
    box_size = n_celdas * dx
    escala_radial = 3.5  # kpc
    escala_vertical = 0.3  # kpc (disco delgado)

    bx, by, bz, ne, ne_rel = _campo_disco_de_juguete(
        n_celdas, dx, escala_radial, escala_vertical
    )

    centro = box_size / 2.0
    radio_solar = 8.0  # kpc, análogo de juguete al radio galactocéntrico solar
    observer_pos = np.array([centro - radio_solar, centro, centro])

    l_grid = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    b_grid = np.linspace(-np.pi / 2 * 0.98, np.pi / 2 * 0.98, 37)

    rm_map, i_map, q_map, u_map = lr.sky_map(
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
        dl=0.25,
        frequency=1.4,
        wavelength=0.21,
        p_index=3.0,
        xp=np,
    )

    assert np.all(np.isfinite(rm_map))
    assert np.all(np.isfinite(i_map))

    # La intensidad tiene que concentrarse cerca del plano galáctico
    # (b~0): si el promedio ahí no supera por mucho al de cerca de los
    # polos, algo está mal en el muestreo/geometría, no solo "se ve feo".
    centro_b = i_map.shape[1] // 2
    intensidad_plano = i_map[:, centro_b - 2 : centro_b + 3].mean()
    intensidad_polos = np.concatenate(
        [i_map[:, :5].ravel(), i_map[:, -5:].ravel()]
    ).mean()
    assert intensidad_plano > 5 * intensidad_polos

    # Ángulo de polarización observado (mod pi), 0.5*atan2(U,Q): con la
    # base tangente esférica de `los_frame_from_galactic` (la que usa
    # `sky_map` desde el fix de continuidad) este mapa debe variar suave
    # en (l, b); con la base arbitraria anterior habría mostrado una
    # discontinuidad visible cerca de |b| ~ 81.4° (donde esa base cambiaba
    # de vector de referencia). Se incluye aquí precisamente para dejar
    # esa comprobación a la vista, no solo en un assert numérico.
    psi_obs = 0.5 * np.arctan2(u_map, q_map)

    fig, ejes = plt.subplots(1, 3, figsize=(14, 4))
    extent = [
        np.degrees(l_grid[0]),
        np.degrees(l_grid[-1]),
        np.degrees(b_grid[0]),
        np.degrees(b_grid[-1]),
    ]
    for ax, mapa, titulo, etiqueta, cmap in (
        (ejes[0], rm_map, "RM", "rad/m$^2$", "RdBu_r"),
        (ejes[1], i_map, "Intensidad sincrotrón", "u.a.", "inferno"),
        (ejes[2], psi_obs, "Ángulo de polarización", "rad", "twilight"),
    ):
        im = ax.imshow(mapa.T, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.set_title(f"{titulo}\ndisco de juguete, observador interior", fontsize=10)
        ax.set_xlabel("l [grados]")
        ax.set_ylabel("b [grados]")
        fig.colorbar(im, ax=ax, label=etiqueta)
    fig.tight_layout()

    ruta_tmp = tmp_path / "sky_map_disco_de_juguete.png"
    fig.savefig(ruta_tmp, dpi=150)

    # Copia también a tests/output/ (si el directorio del repo es
    # escribible) para poder abrirla a mano después de correr la prueba
    # localmente; en un entorno de CI de solo lectura esto simplemente no
    # se hace, sin que la prueba falle por eso.
    try:
        directorio_repo = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(directorio_repo, exist_ok=True)
        fig.savefig(os.path.join(directorio_repo, "sky_map_disco_de_juguete.png"), dpi=150)
    except OSError:
        pass

    plt.close(fig)

    assert ruta_tmp.exists()
