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
| Densidad de electrones térmicos: disco grueso de NE2001 + 4 brazos espirales logarítmicos que nacen en el extremo de la barra (3 kpc) | `faradaymr.simulation.GalacticDiskProfile` | `NE0`, `NE_RADIAL_PROFILE`, `SCALE_HEIGHT_NE`, `N_ARMS`, `ARM_R_MIN`, `ARM_WIDTH`, `PITCH_ANGLE_DEG` |
| Campo regular de disco: espiral logarítmica (mismo `pitch_angle` que los brazos, *frozen-in*), horario en el Sol e invertido dentro de R = 7 kpc (Van Eck et al. 2011), constante dentro de 5 kpc (Sun et al. 2008) y nulo dentro de 3 kpc (JF12), con el corte vertical de JF12 | `faradaymr.fields.LogarithmicSpiralField` | `B0_REGULAR`, `SCALE_RADIAL_B`, `SCALE_HEIGHT_B`, `ANCHO_VERTICAL_B`, `HANDEDNESS`, `ANILLOS_INVERSION_CAMPO`, `RADIO_NUCLEO_B`, `RADIO_SIN_CAMPO_B` |
| Campo regular de halo: toroidal norte/sur y campo en X de Jansson & Farrar (2012) | `faradaymr.fields.ToroidalHaloField`, `XField` | `B_HALO_*`, `R_HALO_*`, `B_CAMPO_X`, ... |
| Campo turbulento: espectro de Kolmogorov, ∇·B = 0, envolvente en R y z de JF12b | `faradaymr.fields.GaussianRandomVectorField` | `B0_TURBULENTO`, `L_COHERENCIA_ISM`, `SPECTRAL_INDEX`, `LAMBDA_MIN`, `LAMBDA_MAX` |
| Electrones relativistas (emisión sincrotrón): fracción *ad hoc* de `n_e` | — | `NE_REL_FRACCION`, `P_SPEC` |
| Ray tracing desde observador interior | `faradaymr.los_raytrace.sky_map` | `N_L`, `N_B`, `B_MAX_DEG`, `DL`, `NU` |

Las amplitudes de campo son las publicadas (Beck 2001, JF12) sin
reescalar: `FACTOR_CAMPO_REGULAR = 1.0` sale de la calibración contra
Oppermann+2012 y NVSS (`calibrar_amplitud_campo.py`). La turbulencia usa
una amplitud *efectiva* (3 µG × 0.25 = 0.74 µG en el Sol) porque la malla
solo resuelve escalas de kpc y la RM aleatoria crece como
B·sqrt(L_coherencia): es la amplitud que da la RM de una turbulencia real de
3 µG con L_coh = 0.1 kpc, no el campo turbulento real.

`model.construir_escenario()` arma las piezas sobre una malla común
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

Así corre el perfil "rápido" (malla 64³, CPU, ~1 min), que es el de los
números de este README. La corrida pesada (perfil "exhaustivo": malla 256³
con celdas de 0.125 kpc, cielo de 1°) está en `Faraday_MR_Colab.ipynb`, para
GPU. El notebook trae el código (de GitHub o de un zip subido) y comprueba
que cupy y numpy den el mismo cielo. Después elige el tamaño de lote según
la memoria de la GPU, recalibra `FACTOR_CAMPO_REGULAR` a esa resolución,
corre la simulación y la comparación, y deja en
`results/foreground_galactico/parametros_corrida.json` el factor usado, la
GPU y los tiempos.

`run.py` deja todo en `results/foreground_galactico/`: los mapas
(`rm_mapa.npy`, `intensidad.npy`, `stokes_q.npy`, `stokes_u.npy`, los
mismos a 28.4 GHz y el ensamble de realizaciones para Planck,
`l_grid.npy`, `b_grid.npy`), un log por corrida en `results/logs/`, y las
figuras:

- `mapa_de_cielo.png` — I, P, ángulo de polarización (IAU) y RM en
  Mollweide, estilo Waelkens et al. (2008, hammurabi).
- `estadistica_mapa.png` — I(b) contra la ley 1/sin|b| de un disco
  plano-paralelo, e histograma y perfil en |b| de P/I (con el tope teórico
  (p+1)/(p+7/3)).
- `estructura_disco.png` — n_e en z=0 y contraste de brazos (n_e / n_e sin
  brazos) con las curvas analíticas de los brazos encima.
- `estructura_perfiles.png` — n_e, B regular y B turbulento contra R y |z|,
  con sus envolventes de referencia.
- `estructura_campo.png` — líneas de campo: solo regular vs regular +
  turbulento (por qué hace falta la componente regular).
- `estructura_turbulencia.png` — corte de |B_turb| del modelo, y espectro
  E(k) contra k^-5/3, gaussianidad y distribución de |B| del generador sin
  envolvente.

`compare_observaciones.py` agrega en la misma carpeta
`comparacion_{mapa_rm,perfil_latitud,rm_vs_longitud,histograma_rm,
catalogo_nvss,morfologia_sincrotron}.png` y
`validacion_planck_030ghz.png` (ver *Validación contra Planck 30 GHz*) y
`remocion_foreground_planck.png` (ver *Remoción del foreground polarizado
de Planck 30 GHz*).

Al final de `run.py`, `faradaymr.calibration` compara la RM y la DM
sintéticas hacia el polo galáctico contra el rango publicado
(Oppermann & Enßlin 2012; NE2001) y avisa si la corrida queda fuera de
rango -ver *Resultados* más abajo.

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

## Resultados (corrida de referencia, semilla 0)

Estadísticas ponderadas por área (cos b). Números de
`results/foreground_galactico/resumen_comparacion_observacional.json`.

### Amplitud de la RM contra latitud

| \|b\| | Modelo | Oppermann+2012 | NVSS (ruido restado) |
|---|---|---|---|
| 2.5° | 229 rad/m² | 186 | 152 |
| 7.5° | 100 | 99 | 112 |
| 22.5° | 38 | 52 | 62 |
| 42.5° | 22 | 17 | 23 |
| 62.5° | 16 | 9.4 | 20 |
| 82.5° | 14 | 6.2 | 15 |
| Global | 79 | 70 | — |

Los dos conjuntos de datos encierran la RM galáctica real: la
reconstrucción de Oppermann suaviza escalas pequeñas (cota inferior) y el
catálogo NVSS incluye la RM intrínseca de cada fuente, ~6-10 rad/m²
(Schnitzeler 2010; cota superior). El modelo sigue a NVSS a |b| > 40°, coincide
con ambos en |b| < 15° (1.2-1.5× en la banda del plano) y queda ~30 % por
debajo de los dos entre 15° y 40°. `run.py` marca la calota |b| ≥ 60° como
FUERA DE RANGO porque su criterio (fijado antes) usa solo Oppermann; contra
NVSS está dentro.

### Signo de la RM (estructura del campo regular)

RM media por región, para el modelo como media ± dispersión de las 8
realizaciones de la turbulencia (`signo_rm_por_region`; el cielo real es
una sola realización, así que una sola semilla no basta):

| Región | Modelo | Oppermann+2012 | Signo |
|---|---|---|---|
| Plano, l = 20-90° (Galaxia interior, Q1) | +99 ± 52 | +21 | ✓ |
| Plano, l = 90-180° | −82 ± 21 | −83 | ✓ |
| Plano, l = 180-270° | +115 ± 20 | +88 | ✓ |
| Plano, l = 270-340° (Galaxia interior, Q4) | −0.3 ± 39 | +34 | ✗ (compatible con 0) |
| 10° < b < 45°, l = 0-90° | +3.5 ± 20 | +31 | ✓ (marginal) |
| −45° < b < −10°, l = 0-90° | −42 ± 21 | −37 | ✓ |
| 10° < b < 45°, l = 270-360° | −22 ± 19 | −22 | ✓ |
| −45° < b < −10°, l = 270-360° | +65 ± 19 | +21 | ✓ |

7 de 8 signos. Correlación píxel a píxel con Oppermann (semilla 0): 0.56 en
|b| < 10° y 0.39 en |b| > 10°; con las 37 370 fuentes NVSS: r = 0.37. La
Galaxia exterior (90° < l < 270°) coincide en signo y amplitud.

### Otros observables

- DM hacia el polo: 31 pc cm⁻³ (NE2001: 20-40). OK.
- Morfología sincrotrón contra Haslam 408 MHz: r = 0.83 en log-log. Sale
  sobre todo del contraste plano-polo; fuera del plano el modelo cae más
  rápido que Haslam (falta un halo de electrones relativistas más grueso
  que `NE_REL_FRACCION` × n_e, y Haslam tiene un fondo isótropo).
- P/I a 1.4 GHz: 0.3-0.6 de mediana, nunca por encima de
  (p+1)/(p+7/3) = 0.75. El cielo real está muy por debajo: la turbulencia
  de escala sub-kpc, que no resuelve esta malla, es la que despolariza.

## Validación contra Planck 30 GHz

`run.py` integra también el cielo a 28.4 GHz (frecuencia efectiva de LFI
30 GHz, donde la rotación de Faraday es despreciable) para 8 realizaciones
de la turbulencia, y `validar_contra_planck_030ghz` sigue el procedimiento
de Planck Int. XLII (2016): solo polarización, varianza galáctica entre
realizaciones y máscaras de loops/spurs, centro galáctico y Fan. Planck se
pasa a K_RJ y a la convención IAU (U_IAU = −U_COSMO) y se degrada a
nside=32. Criterios fijados antes de ver el resultado:

| Prueba | Criterio | Modelo | Resultado |
|---|---|---|---|
| Ángulo de polarización, ⟨cos 2Δψ⟩ en píxeles con P/σ ≥ 5 | ≥ 0.5 y mayor que la plantilla ψ=0 (0.15) | 0.27 ± 0.07 | no pasa |
| Forma de P, R² con fondo libre | mayor que un disco plano-paralelo (0.31) | 0.40 ± 0.04 | pasa |
| Perfiles de P en latitud (informativa) | — | χ²_red = 2.5 (Galaxia interior), 1.3 (3er cuadrante); 97-100 % de bandas a ≤ 3σ | — |

**La validación global no pasa** por el ángulo: el modelo mejora la
plantilla trivial, pero queda lejos de 0.5. Las estructuras locales que
dominan la polarización a 30 GHz (Loop I / North Polar Spur, Fan) están
enmascaradas, pero el campo a gran escala del modelo no tiene la
componente fuera del plano con la orientación correcta en todo el cielo.

## Remoción del foreground polarizado de Planck 30 GHz

Es el objetivo del proyecto: usar el cielo sintético como plantilla para
sustraer la emisión galáctica de Planck. `remover_foreground_polarizado_planck`
hace un ajuste de plantilla (template fitting, Planck 2015 X) sobre Q y U a
28.4 GHz: en cada región ajusta una amplitud libre a ≥ 0 (la emisividad
del modelo no está en unidades físicas) y resta a·(Q, U) del modelo. La
plantilla es la media de las 8 realizaciones de la turbulencia. La métrica
es la fracción de la potencia polarizada de Planck removida, con el ruido
restado y sin las estructuras locales enmascaradas (figura
`remocion_foreground_planck.png`):

| Región | % de la potencia de Planck | Modelo | Una realización | Plantilla trivial (campo ∥ plano) |
|---|---|---|---|---|
| Alta latitud, \|b\| > 20° | 4 % | **36 %** | 25 ± 5 % | 3 % |
| Plano exterior, 90° < l < 270°, \|b\| < 20° | 16 % | **50 %** | 37 ± 14 % | 46 % |
| Plano interior, \|l\| < 90°, 3° < \|b\| < 20° | 5 % | 0 % | 0 % | 0 % |
| Franja interior, \|l\| < 90°, \|b\| < 3° | 75 % | 0 % | 0 % | 0 % |

Con una sola amplitud para todo el cielo (sin la franja) se remueve el
12 %: 31 % a alta latitud y 23 % en el plano exterior, pero se agrega
potencia en la Galaxia interior. El modelo reparte mal la emisión entre la
Galaxia interior y la exterior.

Lectura:

- **A alta latitud el modelo remueve ~12 veces más que la plantilla
  trivial.** Ahí es donde se observa el CMB, y es el resultado que sostiene
  el objetivo del proyecto. Lo aporta la geometría del campo regular
  (halo de JF12 y espiral): el ángulo del modelo coincide con el de Planck
  con ⟨cos 2Δψ⟩ = 0.6-0.7 a |b| > 10°.
- **En el plano exterior empata con la plantilla trivial:** el campo es
  casi paralelo al plano y eso ya lo da una plantilla sin física.
- **En la Galaxia interior no remueve nada.** En la franja |b| < 3°, Q y U de
  Planck siguen a la intensidad total (r = −0.76 y −0.89 en |l| < 60°,
  |b| < 2°, con Q/I ≈ U/I ≈ −1.5 %), aunque a 30 GHz esa intensidad es
  sobre todo free-free y emisión anómala, que no polarizan: es la firma de
  la fuga de intensidad a polarización de LFI (Planck 2018 II), no de
  sincrotrón. Entre 3° y 20° el fallo es del modelo: le falta la geometría
  real de la Galaxia interior (varias inversiones, tangencias de brazos).

Una réplica exacta no se alcanza: es un modelo de juguete con fase de
brazos arbitraria y turbulencia de escala kpc. Lo que sí queda medido es
cuánto foreground remueve y dónde, frente a una plantilla trivial.

## Limitaciones

- **Turbulencia sub-resuelta.** La malla (0.5 kpc) solo resuelve escalas de
  1-6 kpc contra ~0.1 kpc en el ISM real. La amplitud efectiva (0.74 µG)
  reproduce la RM aleatoria, pero no la despolarización: P/I a 1.4 GHz
  sale demasiado alto, y la turbulencia de kpc da una varianza de
  realización grande en la RM de cuadrantes enteros. El perfil
  `FARADAYMR_PERFIL_RESOLUCION=exhaustivo` (celdas de 0.125 kpc) resuelve
  4 veces más fino; está pensado para GPU (`Faraday_MR_Colab.ipynb`) y no
  se ha corrido para esta versión del modelo (la malla de 256³ con 8
  realizaciones necesita del orden de varios GB de memoria).
- **Galaxia interior en polarización.** La plantilla no remueve nada en
  |l| < 90° (ver *Remoción*), que concentra ~80 % de la potencia
  polarizada de Planck a 30 GHz.
- **Galaxia interior simplificada.** Una sola inversión de campo (R < 7 kpc)
  en vez de las varias que proponen los modelos de pulsares (Han et al.
  2006, 2018). La banda del plano queda 1.2-1.5× alta y la RM de
  l = 270-340° sale con signo negativo (observado: +33).
- **Fase de los brazos arbitraria.** La caja no está orientada como la Vía
  Láctea real (ver *Comparación contra datos reales*): las comparaciones
  válidas son estadísticas o en signo por cuadrante.
- **Sin estructuras locales** (Loop I, Gum, burbuja local), que dominan
  parte de la RM entre 15° y 40° y la polarización de Planck.
- **Emisividad sincrotrón en unidades arbitrarias**, con n_rel ∝ n_e.

## Correcciones de la revisión física

- **Sentido de la espiral y del campo.** Los brazos estaban enrollados al
  revés (leading) y el campo local apuntaba hacia l≈270° en vez de l≈90°.
  Ahora `PITCH_ANGLE_DEG=+12` (con phi antihorario, brazos trailing;
  verificado con las tangentes de Scutum-Crux) y `HANDEDNESS=-1` (campo
  horario, B_R<0). Test: `tests/test_convenciones_galacticas.py`.
- **Densidad radial.** La exponencial normalizada en R☉ daba >1 cm⁻³ en el
  centro y RM ~2000 rad/m² hacia l=0; se usa la forma del disco grueso de
  NE2001 (`NE_RADIAL_PROFILE="ne2001"`). La escala vertical (1 kpc) se
  atribuye ahora a NE2001; Gaensler et al. (2008) dan 1.8 kpc con
  n0 = 0.014 cm⁻³, la misma columna n0·h.
- **Halo de JF12.** Halo toroidal norte/sur y campo en X: dan la
  antisimetría norte-sur de la RM y la componente vertical.
- **Inversión del campo en la Galaxia interior.** Sin ella la RM del plano
  en l = 20-90° salía −345 rad/m² (observado: +21) y la correlación en
  |b| < 10° era 0.00; con ella, 0.56. Se compararon las alternativas
  (anillo ASS+RING de Sun et al. 2008, 6-7.5 kpc y escalado a R☉ = 8 kpc):
  dejan el primer cuadrante con el signo equivocado.
- **Campo en la región de la barra.** La exponencial en R hacía crecer el
  campo del disco a ~9 µG en el centro (RM ±300-400 rad/m² hacia
  |l| < 20°). Ahora es constante dentro de 5 kpc (Sun et al. 2008) y nulo
  dentro de 3 kpc (JF12), y los brazos de densidad nacen en 3 kpc (antes
  se amontonaban hasta n_e = 0.2 cm⁻³ en R < 1 kpc).
- **Transición disco-halo.** El disco se apagaba con exp(−|z|/0.4 kpc) y el
  halo se encendía con la logística de JF12, dejando un hueco; ahora el
  disco usa el complemento 1 − L(z) de JF12. El mínimo de |B| que queda en
  |z| ≈ 0.5 kpc al norte es físico: B_φ cambia de signo entre el disco
  (horario) y el halo norte (antihorario).
- **Calibración.** Con la geometría completa, `FACTOR_CAMPO_REGULAR` sale
  1.0 (0.80 contra Oppermann, 1.25 contra NVSS): las amplitudes de la
  literatura no necesitan reescalarse. La turbulencia no se ajusta: su
  factor (0.25) se deriva de la longitud de coherencia.
- **Estadística.** Promedios, histogramas y el ajuste contra Planck pesan
  cada píxel por cos b (ángulo sólido); antes sobrerrepresentaban los polos.
- **Chequeo de RM polar.** Se valida la dispersión en la calota |b|≥60° en
  vez de un solo píxel (que además estaba en l=-180°, no en la dirección
  usada para la DM).
- **Figuras.** La escala simétrica de RM usa el percentil 99 (barra con
  `extend`), el perfil radial usa solo el lado del Sol, y se corrigió la
  documentación del espectro (E(k)∝k^-n). La gaussianidad y el espectro de
  la turbulencia se miden sobre el generador sin envolvente (la envolvente
  mezclaba amplitudes y daba una curtosis falsa de +16). El B turbulento
  se rotula como efectivo. I y P del mapa de cielo van normalizados a
  I_max. La velocidad de la luz es ahora 299 792 458 m/s.

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
