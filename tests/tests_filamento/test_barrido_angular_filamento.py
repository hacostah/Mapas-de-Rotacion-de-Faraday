import os
import sys
import importlib
import numpy as np

# Importaciones absolutas directas al paquete del ejemplo
from examples.filamento_whim import config as cfg
from examples.filamento_whim import run

def test_barrido_angular_produce_orientaciones_distintas_del_filamento(tmp_path):
    importlib.reload(cfg)
    # Perspectiva de físico: el objetivo de este ticket es poder recorrer
    # theta sin tocar los.py. Se usa una malla chica (corre en segundos)
    # y solo 3 ángulos representativos: 0 (de frente), 45 y 90 (de lado).
    cfg.N_BASE = 12

    angulos = np.array([0.0, np.pi / 4, np.pi / 2])
    resultados = run.ejecutar_barrido_angular(
        angulos_rad=angulos,
        ruta_destino=str(tmp_path),
        use_gpu=False,
        seed=7,
    )

    assert len(resultados) == 3

    # 1) Los ejes del filamento reportados coinciden con la geometría
    # esperada para cada theta.
    np.testing.assert_allclose(resultados[0]["eje_filamento"], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(resultados[2]["eje_filamento"], [1.0, 0.0, 0.0], atol=1e-12)

    # 2) Con la MISMA semilla de turbulencia, cambiar theta debe cambiar el
    # campo de densidad n_e (la firma de que el filamento realmente giró
    # dentro de la caja, no solo que cambió un parámetro cosmético).
    ne_frente = resultados[0]["ne"]
    ne_lado = resultados[2]["ne"]
    assert not np.allclose(ne_frente, ne_lado)

    # 3) La dispersión de RM (el observable físico de interés) también
    # debe diferir entre orientaciones: es justo lo que el barrido existe
    # para poder estudiar.
    sigmas = [r["sigma_rm"] for r in resultados]
    assert len(set(np.round(sigmas, 10))) == len(sigmas)

    # 4) Cada corrida debe haber guardado sus propios mapas, en carpetas
    # separadas por ángulo (nada se pisa entre corridas del barrido).
    carpetas = sorted(os.listdir(tmp_path))
    carpetas_theta = [c for c in carpetas if c.startswith("theta_")]
    assert len(carpetas_theta) == 3
    for carpeta in carpetas_theta:
        assert os.path.exists(os.path.join(tmp_path, carpeta, "rm_mapa.npy"))

    # 5) El csv resumen del barrido debe existir y tener una fila por ángulo.
    ruta_csv = os.path.join(tmp_path, "sigma_rm_vs_theta.csv")
    assert os.path.exists(ruta_csv)
    contenido = np.genfromtxt(ruta_csv, delimiter=",", skip_header=1)
    assert contenido.shape[0] == 3


def test_barrido_angular_no_modifica_el_eje_de_integracion_de_los(tmp_path):
    # Chequeo explícito del criterio "sin modificar los.py": el barrido no
    # le pasa ningún ángulo ni parámetro nuevo al pipeline observacional;
    # `ObservationPipeline` sigue llamando a `faradaymr.los` exactamente
    # igual que en el ejemplo del ICM (mismo axis=-1 implícito).
    import inspect

    from faradaymr import los

    firmas_originales = {
        "rotation_measure": "(ne, b_parallel, dl, axis=-1, xp=None)",
        "rotation_measure_cumulative": "(ne, b_parallel, dl, axis=-1, xp=None)",
        "synchrotron_intensity": "(j_nu, dl, axis=-1, xp=None)",
        "stokes_qu": "(j_nu, psi_0, rm_cumulative, wavelength, p_index, dl, axis=-1, xp=None)",
    }
    for nombre, firma_esperada in firmas_originales.items():
        firma_real = str(inspect.signature(getattr(los, nombre)))
        assert firma_real == firma_esperada, (
            f"{nombre} cambió de firma: el barrido angular no debería haber "
            "requerido tocar faradaymr.los."
        )