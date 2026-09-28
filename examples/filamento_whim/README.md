# Filamento WHIM — Rotación de Faraday en un filamento cósmico aislado

Modelo idealizado (*toy model*) de un filamento del Medio Intergaláctico
Tibio-Caliente (WHIM), pensado para predecir su firma observacional en
Medida de Rotación (RM) y estimar un umbral teórico para campañas de
observación con instrumentos de próxima generación (SKA). Ver la
descripción completa del proyecto en la propuesta.

## Resumen físico

El filamento se modela como un cilindro de densidad electrónica con perfil
beta (`ne(r) = n0 (1 + r²/r_c²)^(-3β/2)`, r = distancia perpendicular al
eje) y **longitud finita** a lo largo de su propio eje (ver
`config_fisica.LONGITUD_FILAMENTO`; el corte axial se aplica en
`model.construir_escenario`). Sobre esa densidad se superpone un campo
magnético turbulento sintético, con espectro de potencias tipo ley de
potencia y RMS ~10-40 nG, generado con
`faradaymr.fields.GaussianRandomVectorField`. La topología caótica del
campo anula el RM promedio a lo largo del filamento, así que el
observable central del proyecto no es RM en sí, sino cómo se **dispersa**
RM en cortes perpendiculares al eje proyectado del filamento
(`faradaymr.analysis.spatial_stats.transverse_rm_dispersion`), y cómo esa
dispersión cambia con el ángulo de visión θ entre el eje del filamento y
la línea de visión del observador.

## Flujo de trabajo

```
config_fisica.py (parámetros físicos, con unidades de astropy)
        │
        ▼
    config.py (los mismos valores, ya en kpc/µG/Hz "desnudos",
               listos para pasarle al pipeline numérico)
        │
        ▼
    model.construir_escenario()  → B turbulento + densidad ne del filamento finito
        │
        ▼
faradaymr.ObservationPipeline.run()  → mapas 2D: RM, I, Q, U
        │
        ▼
faradaymr.analysis.transverse_rm_dispersion()  → sigma_RM(d) por bins
        │
        ▼
faradaymr.analysis.fitting.{fit_transverse_dispersion, fit_beta_dispersion}
        → ancho/r_c característico + R² (dos formas funcionales, se compara
          cuál ajusta mejor en vez de asumir la gaussiana a ciegas)
        │
        ▼
      plots.py  → figuras finales (ver abajo)
```

Antes de barrer en θ, `validacion.verificar_caja_suficiente` chequea que la
caja simulada sea suficientemente profunda a lo largo de la línea de
visión para el filamento finito en el ángulo más cercano a 0° del barrido
(el caso más exigente: ver el docstring de esa función para la derivación
completa). Si la caja no alcanza, `run_barrido_theta.barrer_angulos` lanza
un error en vez de continuar silenciosamente — un barrido corrido con la
caja truncando el filamento no es válido y no debe usarse para reportar
resultados.

## Cómo correr el pipeline completo

Desde la raíz del repositorio (todos los scripts se importan como paquete,
`examples.filamento_whim.*`, así que se ejecutan con `python -m`):

```bash
# 1. Mapas de RM para 3 ángulos de ejemplo (usa las figuras 1 y 2)
python -m examples.filamento_whim.run

# 2. Barrido Monte Carlo en theta (usa las figuras 3 y 4; tarda varios
#    minutos en CPU con los parámetros por defecto — bajar n_semillas
#    para una corrida rápida de prueba)
python -m examples.filamento_whim.run_barrido_theta

# 3. Figuras finales, a partir de lo generado en 1 y 2
python -m examples.filamento_whim.plots
```

Las figuras quedan en `examples/filamento_whim/results/plots/`:

- **fig1_mapas_rm.png**: mapas de RM simulados para θ = 0°, 15°, 30°.
- **fig2_perfil_transversal.png**: σ_RM(d) para θ = 30°, con ambos ajustes
  (gaussiano y forma beta) superpuestos y su R².
- **fig3_ancho_vs_theta.png**: resultado central del proyecto — ancho/r_c
  característico transversal vs. θ, con barras de error del Monte Carlo.
- **fig4_sigma0_vs_theta.png**: amplitud de RM en el propio eje del
  filamento vs. θ (ver nota física en `plots.fig4_sigma0_vs_theta`: esta es
  la cantidad con la dependencia angular más directa de interpretar, ya
  que depende de cuánta longitud del filamento finito atraviesa la línea
  de visión, no de la forma del perfil transversal).

`results/` no se versiona (ver `.gitignore`): son salidas de cada corrida
local, no código fuente.

## Nota sobre "ray-tracing"

La propuesta del proyecto menciona "algoritmos de trazado de rayos
(ray-tracing)". Conviene distinguir dos sentidos del término, porque el
código de este repositorio implementa solo uno de ellos:

- **Líneas de visión paralelas** (lo que implementa
  `faradaymr/los.py` y usa `ObservationPipeline`): válido para un
  observador *externo*, a distancia cosmológica del objeto, donde todos
  los rayos que llegan al instrumento pueden tratarse como paralelos entre
  sí. Este es el caso físicamente correcto para un filamento del WHIM
  observado desde la Tierra, y es lo que usa este ejemplo.
- **Ray-tracing con observador interior** (rayos genuinamente
  divergentes desde un punto dentro o cerca de la estructura extendida):
  no está implementado en este repositorio (requeriría un módulo del
  estilo `los_raytrace.py`, que no existe). Es el caso relevante para un
  proyecto futuro sobre estructuras extendidas vistas desde dentro, no
  para este.

Se deja esta nota explícita para que, frente a quien evalúe el proyecto,
el uso del término en la propuesta no se lea como una promesa incumplida:
la integración de línea de visión paralela es la que corresponde a este
escenario, y es la que está implementada y validada por los tests.
