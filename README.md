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

Por defecto corre la versión pesada (perfil "exhaustivo", ver
`config_fisica.py`): malla de 256³ celdas de 0.125 kpc, cielo de 1°, paso de
0.025 kpc y 8 realizaciones de la turbulencia. Si cupy está instalado (en
Colab ya viene; no instalar otro, ver `requirements.txt`) `run.py` integra en GPU y elige solo el tamaño de lote
según la memoria libre; sin GPU corre en numpy, más lento y con ~5 GB de
RAM. Para probar cambios rápido (malla 64³, ~1 min en CPU, el perfil de los
tests):

```bash
FARADAYMR_PERFIL_RESOLUCION=rapido python run.py
FARADAYMR_PERFIL_RESOLUCION=rapido python compare_observaciones.py
```

`run.py` deja todo en `results/foreground_galactico/`: los mapas
(`rm_mapa.npy`, `intensidad.npy`, `stokes_q.npy`, `stokes_u.npy`, los
mismos a 28.4 GHz y el ensamble de realizaciones para Planck,
`l_grid.npy`, `b_grid.npy`), un log por corrida en `results/logs/`, y las
figuras:

Figuras de `run.py` (colores comunes en `faradaymr/estilo_figuras.py`: el
modelo en naranja, las observaciones en azul, NVSS en verde agua, las
referencias sin física en gris):

- `mapa_de_cielo.png` — el cielo sintético en Mollweide: (a) intensidad a
  1.4 GHz, (b) RM, (c) intensidad polarizada a 1.4 GHz (los huecos son
  despolarización por Faraday) y (d) intensidad polarizada a 28.4 GHz con
  la orientación del campo magnético proyectado en trazos, como en los
  mapas de Planck.
- `estadistica_mapa.png` — I(b) contra un disco plano-paralelo, y la
  fracción de polarización contra |b| a 1.4 y 28.4 GHz con el tope
  (p+1)/(p+7/3). Cerca del plano Faraday despolariza; a latitud alta, con
  |RM| ~ 10 rad/m², la rotación puede repolarizar un poco al realinear el
  campo del disco con el del halo (repolarización por rotación de Faraday
  diferencial; Sokoloff et al. 1998).
- `estructura_disco.png` — n_e en z=0 y contraste de brazos (n_e / n_e sin
  brazos), con las espirales analíticas en línea discontinua.
- `estructura_perfiles.png` — n_e, B regular y B turbulento efectivo contra
  R y |z|, con sus envolventes de referencia. El mínimo de |B| en
  |z| ≈ 0.5 kpc es el cambio de sentido entre el disco y el halo norte.
- `estructura_campo.png` — líneas de campo en z=0, solo regular y regular +
  turbulento, con los radios de la inversión (7 kpc) y de la barra (3 kpc).
- `estructura_turbulencia.png` — corte de |B_turb| del modelo, y espectro
  E(k) contra k^-5/3, gaussianidad y distribución de |B| del generador sin
  envolvente.

`python run.py --solo-estructura` rehace solo las cuatro figuras de
estructura (misma semilla, sin integrar el cielo).

`compare_observaciones.py` agrega en la misma carpeta:

- `comparacion_mapa_rm.png` — RM del modelo, de Oppermann y su residuo,
  con la misma escala de color.
- `comparacion_perfil_latitud.png` — RMS(RM) contra |b|: modelo, Oppermann,
  NVSS y la franja entre las dos estimaciones observacionales.
- `comparacion_rm_vs_longitud.png` — RM con signo contra l: el plano, y
  cada banda de latitud separada en norte y sur (el halo invierte el
  signo entre hemisferios; promediarlos lo cancela).
- `comparacion_catalogo_nvss.png` — RM del modelo contra la de 37 370
  fuentes NVSS, como densidad, con la mediana del modelo y la diagonal 1:1;
  plano y fuera del plano por separado.
- `comparacion_histograma_rm.png` — distribución de la RM, plano y fuera
  del plano.
- `comparacion_morfologia_sincrotron.png` — forma de la intensidad contra
  Haslam 408 MHz.
- `validacion_planck_030ghz.png` — ver *Validación contra Planck 30 GHz*.
- `remocion_foreground_planck.png` — ver *Remoción del foreground
  polarizado de Planck 30 GHz*.

En los mapas comparados con Planck, las estructuras locales excluidas de
las estadísticas (loops, centro galáctico, Fan) van con contorno punteado,
sin tapar los datos.

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

## Resultados (perfil exhaustivo en GPU, semilla 0)

Corrida de producción: malla de 256³, cielo de 1°, 8 realizaciones de la
turbulencia. Estadísticas ponderadas por área (cos b). Números de
`results/foreground_galactico/resumen_comparacion_observacional.json`.
El perfil rápido (64³) da números parecidos salvo a alta latitud, donde
su turbulencia más gruesa sube la RM (14 contra 8.7 rad/m² en el polo).

### Amplitud de la RM contra latitud

| \|b\| | Modelo | Oppermann+2012 | NVSS (ruido restado) |
|---|---|---|---|
| 2.5° | 241 rad/m² | 179 | 152 |
| 7.5° | 107 | 100 | 112 |
| 22.5° | 38 | 52 | 62 |
| 42.5° | 21 | 17 | 23 |
| 62.5° | 13 | 9.4 | 20 |
| 82.5° | 8.7 | 6.4 | 15 |
| Global | 85 | 70 | — |

Los dos conjuntos de datos encierran la RM galáctica real: la
reconstrucción de Oppermann suaviza escalas pequeñas (cota inferior) y el
catálogo NVSS incluye la RM intrínseca de cada fuente, ~6-10 rad/m²
(Schnitzeler 2010; cota superior). A |b| > 40° el modelo queda entre los
dos; en |b| < 15° coincide con ambos salvo la banda del plano (1.3-1.6×), y
entre 15° y 40° queda ~30 % por debajo de los dos (faltan las estructuras
locales). La RMS en la calota |b| ≥ 60° es 10.9 rad/m², dentro del rango
del chequeo de `run.py` (2-12 rad/m²).

### Signo de la RM (estructura del campo regular)

RM media por región, para el modelo como media ± dispersión de las 8
realizaciones de la turbulencia (`signo_rm_por_region`; el cielo real es
una sola realización, así que una sola semilla no basta):

| Región | Modelo | Oppermann+2012 | Signo |
|---|---|---|---|
| Plano, l = 20-90° (Galaxia interior, Q1) | +94 ± 18 | +23 | ✓ |
| Plano, l = 90-180° | −74 ± 24 | −80 | ✓ |
| Plano, l = 180-270° | +104 ± 20 | +85 | ✓ |
| Plano, l = 270-340° (Galaxia interior, Q4) | −41 ± 48 | +35 | ✗ (compatible con 0 a 1σ) |
| 10° < b < 45°, l = 0-90° | +15 ± 10 | +32 | ✓ |
| −45° < b < −10°, l = 0-90° | −52 ± 14 | −37 | ✓ |
| 10° < b < 45°, l = 270-360° | −19 ± 11 | −21 | ✓ |
| −45° < b < −10°, l = 270-360° | +40 ± 18 | +21 | ✓ |

7 de 8 signos, incluida la antisimetría norte-sur del halo en las cuatro
regiones de latitud media. Correlación píxel a píxel con Oppermann: 0.58 en
|b| < 10° y 0.41 en |b| > 10°; con las 37 370 fuentes NVSS: r = 0.36. La
Galaxia exterior (90° < l < 270°) coincide en signo y amplitud.

### Otros observables

- DM hacia el polo: 31 pc cm⁻³ (NE2001: 20-40). OK.
- Morfología sincrotrón contra Haslam 408 MHz: r = 0.83 en log-log. Sale
  sobre todo del contraste plano-polo. Fuera del plano el modelo cae más
  que Haslam: a alta latitud queda en ~7 % del valor del plano, contra
  ~20 % en Haslam. Parte de esa diferencia es el fondo isótropo de Haslam;
  el resto, un halo de electrones relativistas más grueso que
  `NE_REL_FRACCION` × n_e.
- P/I a 1.4 GHz: mediana 0.25 en |b| < 10° y 0.53 en 30° < |b| < 60°, nunca
  por encima de (p+1)/(p+7/3) = 0.75. El cielo real está muy por debajo:
  la turbulencia de escala menor que 0.25 kpc, que no resuelve esta malla,
  es la que despolariza.

## Validación contra Planck 30 GHz

`run.py` integra también el cielo a 28.4 GHz (frecuencia efectiva de LFI
30 GHz, donde la rotación de Faraday es despreciable) para 8 realizaciones
de la turbulencia, y `validar_contra_planck_030ghz` sigue el procedimiento
de Planck Int. XLII (2016): solo polarización, varianza galáctica entre
realizaciones y máscaras de loops/spurs, centro galáctico y Fan (en las
figuras, contorno punteado). Planck se pasa a K_RJ y a la convención IAU
(U_IAU = −U_COSMO) y se degrada a nside=32. Criterios fijados antes de ver
el resultado:

| Prueba | Criterio | Modelo | Resultado |
|---|---|---|---|
| Ángulo de polarización, ⟨cos 2Δψ⟩ en píxeles con P/σ ≥ 5 | ≥ 0.5 y mayor que la plantilla ψ=0 (0.17) | 0.28 ± 0.07 | no pasa |
| Forma de P, R² con fondo libre | mayor que un disco plano-paralelo (0.37) | 0.38 ± 0.05 | pasa (por poco) |
| Perfiles de P en latitud (informativa) | — | Galaxia interior: χ²_red = 1.9, 100 % de bandas a ≤ 3σ, datos/modelo = 1.03; 3er cuadrante: χ²_red = 2.9, el modelo subestima el pico del plano | — |

**La validación global no pasa** por el ángulo: el modelo mejora la
plantilla trivial, pero queda lejos de 0.5. El promedio lo hunde el plano:
fuera de él el acuerdo es bueno (⟨cos 2Δψ⟩ = 0.75-0.85 a |b| > 10°, ver
*Remoción*), mientras que en |b| < 10° el ángulo de Planck no tiene la
orientación de un campo paralelo al disco (ver la franja interior más
abajo). Las estructuras locales que
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
| Alta latitud, \|b\| > 20° | 5 % | **38 %** | 23 ± 10 % | 2 % |
| Plano exterior, 90° < l < 270°, \|b\| < 20° | 18 % | **51 %** | 42 ± 7 % | 38 % |
| Plano interior, \|l\| < 90°, 3° < \|b\| < 20° | 4 % | 0 % | 0 % | 0 % |
| Franja interior, \|l\| < 90°, \|b\| < 3° | 73 % | 0 % | 0 % | 0 % |

Con una sola amplitud para todo el cielo (sin la franja) se remueve el
10 %: 24 % a alta latitud y 20 % en el plano exterior, pero se agrega
potencia en la Galaxia interior. El modelo reparte mal la emisión entre la
Galaxia interior y la exterior.

Lectura:

- **A alta latitud el modelo remueve ~16 veces más que la plantilla
  trivial (38 % contra 2 %).** Ahí es donde se observa el CMB, y es el
  resultado que sostiene el objetivo del proyecto. Lo aporta la geometría
  del campo regular (halo de JF12 y espiral): fuera de las máscaras y
  donde Planck detecta polarización (P/σ ≥ 5), el ángulo del modelo
  coincide con el de Planck con ⟨cos 2Δψ⟩ = 0.79 (10°-20°), 0.85
  (20°-40°) y 0.75 (|b| > 40°).
- **En el plano exterior supera a la plantilla trivial (51 % contra 38 %)**,
  aunque ahí el campo es casi paralelo al plano y una plantilla sin física
  ya remueve bastante.
- **En la Galaxia interior no remueve nada** (rayado en la figura). En la
  franja |b| < 3°, Q y U de Planck siguen a la intensidad total
  (r = −0.76 y −0.89 en |l| < 60°, |b| < 2°, con Q/I ≈ U/I ≈ −1.5 %), aunque
  a 30 GHz esa intensidad es sobre todo free-free y emisión anómala, que no
  polarizan. Además, el ángulo de Planck en esa franja es casi el mismo
  (ψ ≈ −50° a −70°, campo a ~25-40° de la vertical) a lo largo de 80° de
  longitud, cuando el sincrotrón del disco daría un campo paralelo al plano
  (como sí ocurre en el plano exterior, ψ ≈ 0°). Un ángulo fijo con P ∝ I es
  lo que produce la fuga de intensidad a polarización por desajuste de
  banda de LFI (Planck 2018 II). Es la explicación más probable, pero no
  está demostrada: confirmarla requiere las plantillas de corrección de
  fuga de Planck. Entre 3° y 20° el fallo es del
  modelo: le falta la geometría real de la Galaxia interior (varias
  inversiones, tangencias de brazos).
- **Lo que queda en el residuo a alta latitud son las estructuras locales**
  (North Polar Spur y los demás loops), excluidas del ajuste y fuera del
  alcance de un modelo de gran escala.

Una réplica exacta no se alcanza: es un modelo de juguete con fase de
brazos arbitraria y turbulencia de escala sub-kpc sin resolver. Lo que sí
queda medido es cuánto foreground remueve y dónde, frente a una plantilla
trivial.

## Limitaciones

- **Turbulencia sub-resuelta.** La malla resuelve escalas de 0.25-6 kpc
  (perfil exhaustivo; 1-6 kpc en el rápido) contra ~0.1 kpc en el ISM real. La amplitud efectiva (0.74 µG)
  reproduce la RM aleatoria, pero no la despolarización: P/I a 1.4 GHz
  sale demasiado alto, y la turbulencia de kpc da una varianza de
  realización grande en la RM de cuadrantes enteros. El perfil exhaustivo
  (celdas de 0.125 kpc) baja la escala mínima a 0.25 kpc, todavía mayor que
  la del ISM real.
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
