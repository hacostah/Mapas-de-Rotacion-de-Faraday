# Mapas de Rotación de Faraday — Proyecto III: foreground galáctico

Simulación de juguete de la Vía Láctea vista **desde dentro** (un
observador situado en el radio solar, no un objeto externo visto de
frente): disco delgado de electrones térmicos con brazos espirales,
campo magnético regular en espiral logarítmica + componente turbulenta, y
un *ray tracer* que integra cada línea de visión (l, b) desde el
observador hasta el borde de la caja para producir un mapa de cielo
completo de Medida de Rotación (RM), intensidad sincrotrón total, e
intensidad/ángulo de polarización.

Este README cubre el estado de la rama `V4-Foreground-CMB` (Proyecto
III). El resto del framework (`faradaymr/`) es más general -nació para
simular el medio intracúmulo (ICM) visto desde fuera- y se documenta en
los docstrings de cada módulo.

## Por qué un observador interior

A diferencia de una nube esférica externa (el caso ICM, que resuelve
`faradaymr.pipeline.ObservationPipeline`), la Vía Láctea solo se puede
observar desde un punto dentro de su propio disco. Eso exige una pieza
distinta del framework, `faradaymr.los_raytrace.sky_map`: en vez de mirar
la caja de frente a lo largo de un eje fijo, traza un haz de rayos desde
`observer_pos` hacia cada dirección (l, b) de una grilla de cielo
completo, y aplica sobre cada rayo las mismas integrales de RM/Stokes
Q,U que ya usa el resto del framework.

## Estructura del modelo físico

| Pieza | Módulo | Parámetros (`config_fisica.py`) |
|---|---|---|
| Densidad de electrones térmicos: disco delgado + 4 brazos espirales logarítmicos | `faradaymr.simulation.GalacticDiskProfile` | `NE0`, `SCALE_RADIAL_NE`, `SCALE_HEIGHT_NE`, `N_ARMS`, `ARM_WIDTH`, `PITCH_ANGLE_DEG` |
| Campo magnético regular: espiral logarítmica coherente (mismo `pitch_angle` que los brazos, por consistencia *frozen-in*) | `faradaymr.fields.LogarithmicSpiralField` | `B0_REGULAR`, `SCALE_RADIAL_B`, `SCALE_HEIGHT_B`, `HANDEDNESS` |
| Campo magnético turbulento: espectro de Kolmogorov | `faradaymr.fields.GaussianRandomVectorField` | `B0_TURBULENTO`, `SPECTRAL_INDEX`, `LAMBDA_MIN`, `LAMBDA_MAX` |
| Electrones relativistas (emisión sincrotrón): fracción *ad hoc* de `n_e` | — | `NE_REL_FRACCION`, `P_SPEC` |
| Ray tracing desde observador interior | `faradaymr.los_raytrace.sky_map` | `N_L`, `N_B`, `B_MAX_DEG`, `DL`, `NU` |

`model.construir_escenario()` arma las cinco piezas sobre una malla común
y devuelve `(bx, by, bz, ne, ne_rel, observer_pos, box_size, dx)` (más las
componentes regular/turbulenta por separado si se llama con
`return_components=True`, para las figuras de estructura 3D).

Todo parámetro físico en `config_fisica.py` cita su referencia
(Cordes & Lazio 2002, Vallée 2015/2016, Jansson & Farrar 2012, Sun et al.
2008, Beck 2001, Han et al. 2004...) -ver los comentarios de ese archivo
para el detalle de cada uno.

## Cómo correrlo

```bash
pip install -r requirements.txt

python run.py                    # simulación + mapas de cielo + estructura 3D
python data/external/descargar_datos.py   # una vez, para la comparación observacional
python compare_observaciones.py  # comparación/resta contra datos reales
```

`run.py` deja todo en `results/foreground_galactico/`: los mapas
(`rm_mapa.npy`, `intensidad.npy`, `stokes_q.npy`, `stokes_u.npy`,
`l_grid.npy`, `b_grid.npy`), un log por corrida en `results/logs/`, y las
figuras:

- `figura_1_mapa_de_cielo.png` — panel combinado (I, P, ángulo de
  polarización, RM), estilo Waelkens et al. (2008, hammurabi).
- `figura_1_{intensidad,intensidad_polarizada,angulo_polarizacion,rm}.png`
  — los mismos cuatro observables, cada uno en su propio archivo a mayor
  tamaño/detalle.
- `figura_2_perfil_latitud.png`, `figura_4_histograma_rm.png` —
  diagnósticos 1D del mapa de cielo.
- `figura_estructura_*.png` — la estructura 3D del modelo vista desde
  afuera: densidad y campo magnético cara-on (con las curvas analíticas de
  los brazos superpuestas), corte de canto (espesor del disco), perfiles
  radial/vertical, y comparación campo-regular-vs-total (por qué hace
  falta la componente regular: sin ella, la turbulencia despolariza casi
  toda la señal).

Al final de `run.py`, `faradaymr.calibration` compara la RM y la DM
sintéticas hacia el polo galáctico contra el rango publicado
(Oppermann & Enßlin 2012; NE2001) y avisa si la corrida queda fuera de
rango -ver *Estado de calibración* más abajo.

## Comparación contra datos reales (`compare_observaciones.py`)

`faradaymr.observational` carga tres conjuntos de datos observacionales
reales (no versionados en git, ver `data/external/descargar_datos.py`):

1. **Oppermann & Enßlin (2012, A&A 542, A93)** — reconstrucción Bayesiana
   de cielo completo de la Faraday depth galáctica, HEALPix nside=128.
   Comparación de estructura de gran escala (perfil RMS(RM) vs. |b|,
   distribución global de valores) en las mismas unidades físicas
   (rad/m²), sin reescalar nada.
2. **Taylor, Stil & Sunstrum (2009, ApJ 702, 1230)** — catálogo NVSS de
   37 543 fuentes puntuales con RM medida. Comparación punto a punto,
   interpolando el modelo en la posición real de cada fuente.
3. **Haslam et al. (1982), 408 MHz** — comparación *morfológica* (forma
   espacial, normalizada) de la intensidad sincrotrón: `sky_map` devuelve
   I en unidades arbitrarias (no hay, en este framework, una emisividad
   sincrotrón calibrada en unidades físicas), así que no es honesto
   comparar la escala absoluta contra los Kelvin de Haslam.

`compare_observaciones.py` corre las tres comparaciones sobre la última
corrida de `run.py`, guarda las figuras de comparación y un resumen
numérico en `results/foreground_galactico/resumen_comparacion_observacional.json`.

**Limitación de fondo, la misma en las tres comparaciones** (ver
docstring de `faradaymr.los_raytrace.direction_from_galactic` y de
`faradaymr.observational`): la caja no está rotada a la orientación real
de la Vía Láctea -l=0 sí apunta al centro galáctico, pero la *fase* de los
4 brazos espirales del modelo en longitud galáctica es arbitraria, no la
de Norma/Scutum-Centaurus/Sagitario/Perseo reales. Por eso ninguna
comparación resta punto a punto en (l, b) y lo reporta como "error del
modelo"; las comparaciones válidas son estadísticas (perfiles en |b|,
histogramas) o punto a punto contra el catálogo (válida precisamente
porque no depende de la fase, ver docstring del módulo).

## Estado de calibración (última corrida)

La comparación contra Oppermann+2012 y el catálogo de Taylor+2009 muestra
que el modelo **sobreestima la amplitud de RM en todo el rango de
latitud**, cada vez más lejos del plano:

| | Modelo | Real | Factor |
|---|---|---|---|
| RM global (RMS) | 372 rad/m² | 58 rad/m² | 6.4× |
| RM cerca del plano (\|b\|<5°) | ~1466 rad/m² | ~186 rad/m² | 7.9× |
| RM cerca del polo (\|b\|~82°) | ~61 rad/m² | ~6.2 rad/m² | 9.8× (peor caso) |
| RM en fuentes reales del catálogo NVSS (37 370 comparadas) | 237 rad/m² | 59 rad/m² | 4× |
| Morfología sincrotrón vs. Haslam 408 MHz (correlación log-log) | r = 0.65 | — | (razonable) |

El chequeo de `faradaymr.calibration.validar_rm_polo` marca la corrida
como **FUERA DE RANGO** (RM sintética hacia el polo ≈ -75 rad/m² contra un
rango publicado de -10 a 10).

**Causa raíz identificada** (confirmada visualmente en
`figura_estructura_densidad_cara_on.png`): el observador cae casi
exactamente sobre la cresta de un brazo espiral en vez de en una región
interbrazo como el Sol real -con `N_ARMS=4` las fases quedan en
0°/90°/180°/270°, y el observador está en `phi=180°` por construcción de
la caja, una coincidencia estructural, no una calibración deliberada.
Mirar hacia `l=0` (centro galáctico) equivale entonces a mirar a lo largo
del propio brazo del observador, lo que infla la densidad/campo integrados
en gran parte del mapa. Lo alentador: la *forma* general (disco + brazos,
r=0.65 contra Haslam) ya es cualitativamente razonable; lo que falta es
desacoplar la fase de los brazos de la posición del observador y
re-calibrar la amplitud (`NE0`, `B0_REGULAR`, `B0_TURBULENTO`) una vez
resuelto eso.

## Tests

```bash
python -m pytest -q
```

`tests/test_observational.py` corre contra datos sintéticos por defecto;
los tests que necesitan los archivos reales de `data/external/` se
saltan automáticamente (`skipif`) si no están descargados.

## Estructura del repositorio

```
config_fisica.py         Parámetros físicos con unidades (astropy.units) y referencias
config.py                Los mismos parámetros, convertidos a valores planos (kpc, µG, cm^-3, ...)
model.py                 Ensambla el escenario 3D (densidad + campo) para Proyecto III
run.py                   Corrida completa: simulación -> mapas de cielo -> figuras -> calibración
plots.py                 Figuras del mapa de cielo (combinado, individuales, histograma)
compare_observaciones.py Comparación/resta contra datos observacionales reales (issue #37)
faradaymr/               Framework general (campos, ray tracing, io, calibración, plotting)
  observational.py         Carga y comparación contra Oppermann+2012 / Taylor+2009 / Haslam
  galactic_structure_plots.py  Vistas 3D del modelo (cara-on, de canto, perfiles)
  plotting_sky.py           Mapas de cielo en proyección Mollweide
  plotting_comparacion.py   Figuras de comparación modelo-vs-observación
data/external/            Datos reales descargados (no versionados, ver descargar_datos.py)
results/                  Salida de cada corrida (mapas, figuras, logs)
tests/                    Suite de pytest
```
