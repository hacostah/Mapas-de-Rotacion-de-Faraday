import logging

from faradaymr.logging_config import (
    configurar_logging,
    generar_id_simulacion,
    medir_tiempo_kernel,
)


def test_generar_id_simulacion_es_unico_incluso_en_el_mismo_segundo():
    # Un estudio paramétrico lanza varias corridas casi simultáneas en un
    # clúster; si dos corridas del mismo segundo generaran el mismo id, sus
    # archivos de log se pisarían entre sí.
    ids = {generar_id_simulacion() for _ in range(20)}
    assert len(ids) == 20


def test_configurar_logging_escribe_un_archivo_ligado_al_id_de_simulacion(tmp_path):
    # El requisito físico real es poder encontrar, después de una corrida
    # larga, el log que corresponde exactamente a esa corrida -de ahí que
    # el nombre del archivo deba coincidir con el id que se le dio.
    id_simulacion = "corrida_de_prueba"
    logger = configurar_logging(
        directorio_logs=str(tmp_path), id_simulacion=id_simulacion
    )
    logger.info("mensaje de prueba")

    ruta_log = tmp_path / f"{id_simulacion}.log"
    assert ruta_log.exists()
    contenido = ruta_log.read_text()
    assert "mensaje de prueba" in contenido

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_configurar_logging_no_duplica_handlers_en_llamadas_sucesivas(tmp_path):
    # Un estudio paramétrico puede llamar configurar_logging una vez por
    # corrida dentro del mismo proceso; sin la limpieza de handlers, cada
    # llamada adicional haría que las líneas de log se repitieran N veces.
    logger = configurar_logging(nivel=logging.INFO)
    configurar_logging(nivel=logging.INFO)
    configurar_logging(nivel=logging.INFO)
    assert len(logger.handlers) == 1


def test_medir_tiempo_kernel_registra_la_duracion(caplog):
    # No importa el número exacto (depende de la máquina), solo que el
    # decorador de verdad deja un registro cuantitativo del tiempo del
    # kernel, que es el punto del ticket: poder ver dónde se va el tiempo
    # de GPU en un estudio paramétrico con muchas corridas.
    @medir_tiempo_kernel
    def kernel_de_juguete(x):
        return x * 2

    with caplog.at_level(logging.INFO):
        resultado = kernel_de_juguete(21)

    assert resultado == 42
    assert any("kernel_de_juguete" in m for m in caplog.messages)
