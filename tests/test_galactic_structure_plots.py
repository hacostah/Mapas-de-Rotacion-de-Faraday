"""
Tests de `faradaymr.galactic_structure_plots`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from faradaymr.galactic_structure_plots import generar_graficos_estructura


def test_generar_graficos_estructura_crea_el_directorio_si_no_existe(tmp_path):
    """
    Regresión de un `FileNotFoundError` real visto en una corrida en Colab
    (clon nuevo del repo, sin `results/foreground_galactico/` todavía):
    `run.py` llama a `generar_graficos_estructura_3d` ANTES de
    `faradaymr.io.save_maps` (que es lo que normalmente crea el
    directorio), así que esta función tiene que crear su propio
    directorio de destino si no existe -no asumir que ya lo creó otra
    cosa antes.
    """
    n = 6
    campo = np.ones((n, n, n))
    dx = 0.5
    box_size = n * dx
    observer_pos = np.array([box_size / 2.0 - 2.0, box_size / 2.0, box_size / 2.0])

    ruta_destino = tmp_path / "no_existe_todavia" / "foreground_galactico"
    assert not ruta_destino.exists()

    rutas = generar_graficos_estructura(
        str(ruta_destino), campo, campo, campo, campo, dx, box_size,
        observer_pos, r0=2.0, pitch_angle=np.radians(12.0), n_arms=4,
    )

    assert ruta_destino.is_dir()
    assert len(rutas) > 0
    for ruta in rutas:
        assert os.path.isfile(ruta)


def test_figura_turbulencia_y_perfiles_con_componentes_completas(tmp_path):
    from faradaymr import GaussianRandomVectorField

    n, dx = 24, 0.5
    box_size = n * dx
    turb = GaussianRandomVectorField(
        n=n, dx=dx, spectral_index=5.0 / 3.0, scale_min=1.0, scale_max=6.0
    ).sample(use_gpu=False, rng=np.random.RandomState(0))
    ne = np.ones((n, n, n))
    observer_pos = np.array([box_size / 2.0 - 2.0, box_size / 2.0, box_size / 2.0])

    rutas = generar_graficos_estructura(
        str(tmp_path), ne, *turb, dx, box_size, observer_pos,
        r0=2.0, pitch_angle=np.radians(12.0), n_arms=4,
        componentes_regular=turb, componentes_turbulento=turb,
        ne_axisimetrico=ne, escalas_b={"b0": 1.0, "escala_radial_b": 5.0, "escala_altura_b": 1.0},
        k_banda=(np.pi / 6.0, np.pi / 1.0),
    )

    nombres = sorted(os.path.basename(r) for r in rutas)
    assert nombres == [
        "estructura_campo.png", "estructura_disco.png",
        "estructura_perfiles.png", "estructura_turbulencia.png",
    ]
    for ruta in rutas:
        assert os.path.getsize(ruta) > 0
