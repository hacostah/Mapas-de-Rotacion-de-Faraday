# Filamento WHIM — Rotación de Faraday en un filamento cósmico aislado

Modelo idealizado (*toy model*) de un filamento del Medio Intergaláctico
Tibio-Caliente (WHIM) que predice su firma observacional en Medida de
Rotación (RM) en función del ángulo de visión, y a partir de ella un umbral
teórico de detección para campañas con RM-grids (POSSUM, SKA).

## Resumen físico

- **Densidad.** Cilindro de longitud finita L = 2 Mpc con perfil beta
  radial, `n_e(r) = n0 (1 + r²/r_c²)^(-3β/2)`, con n0 = 10⁻⁵ cm⁻³, β = 2/3 y
  r_c = 300 kpc (β y el rango de n0 según Tanimura et al. 2020).
- **Campo magnético.** Turbulento, gaussiano, con divergencia nula por
  construcción (B = ∇×A en Fourier), espectro |B_k|² ∝ k⁻³ entre 25 y
  500 kpc, y RMS de **10 nG** (Akahori & Ryu 2010).
- **Observación.** Integración de RM = 0.812 ∫ n_e B_∥ dl a lo largo de
  líneas de visión paralelas, con el filamento inclinado un ángulo θ
  respecto a la línea de visión (θ = 0° de frente, θ = 90° de lado).

La RM de cada línea de visión es una variable gaussiana de media cero, así
que el promedio neto se anula y la información está en su **dispersión**
σ_RM. El observable del proyecto es el perfil transversal σ_RM(d) en función
de la distancia d al eje proyectado del filamento, y cómo cambia con θ.

## Resultados (B = 10 nG, 50 realizaciones)

| θ | σ₀ (rad/m²) | p | d½ (kpc) | dispersión de 1 filamento | R² gaussiana |
|---|---|---|---|---|---|
| 0° | 0.0311 ± 0.0012 | 0.99 ± 0.03 | 303 ± 7 | 27% | 0.59 |
| 30° | 0.0210 ± 0.0006 | 0.83 ± 0.03 | 343 ± 8 | 21% | 0.82 |
| 60° | 0.0162 ± 0.0003 | 0.75 ± 0.01 | 371 ± 6 | 13% | 0.86 |
| 90° | 0.0149 ± 0.0002 | 0.72 ± 0.01 | 380 ± 5 | 10% | 0.86 |

1. **La RM es gaussiana** en cada línea de visión (asimetría y exceso de
   curtosis < 0.02 sobre 1.7 × 10⁶ píxeles; fig. 5).
2. **El perfil de dispersión NO es gaussiano**: sigue la ley beta
   σ_RM(d) = σ₀ (1 + d²/r_c²)^(−p) con R² > 0.997 en todos los ángulos,
   mientras que la gaussiana da R² = 0.59–0.87 (fig. 2).
3. **La forma depende del ángulo** (fig. 3). El exponente pasa de
   p = 3β/2 = 1 (de frente: cada línea de visión recorre el filamento a
   radio constante, σ ∝ n_e(d)) a p = (3β − ½)/2 = 0.75 (de lado:
   σ² ∝ ∫ n_e² dl, la misma cuenta que Murgia et al. 2004 para cúmulos).
   El semiancho a media altura crece 25%, de 303 a 380 kpc.
4. **La amplitud cae un factor 2.1** de frente a de lado (fig. 4), porque de
   frente la línea de visión recorre toda la longitud del filamento.
5. **Umbral** (fig. 6): σ₀ es 225–470 veces menor que la dispersión
   intrínseca de 7 rad/m² de las fuentes de fondo. Detectar 10 nG a 3σ
   requiere ~10¹¹ fuentes (RM-grid ideal) o ~10¹² (ruido tipo POSSUM); con
   las 54 fuentes de Stuardi et al. (2026) el campo mínimo detectable es de
   2–4 µG. Un filamento individual con B ~ 10 nG no es detectable por este
   método.

A 90° el exponente simulado (0.72) queda 0.03 por debajo del límite de
fórmula cerrada (0.75). La diferencia se explica porque la longitud de
correlación del campo a lo largo de la línea de visión (≈250 kpc) no es
mucho menor que r_c (300 kpc), como supone la fórmula de paseo aleatorio.

### Leyes de escala

Todo σ_RM es lineal en n0 y en B, y el número de fuentes escala como σ₀⁻⁴:

    σ₀ ≈ 0.016 rad/m² · (n0 / 10⁻⁵ cm⁻³) · (B / 10 nG) · (Λ / 250 kpc)^½ · (L_LoS / 470 kpc)^½

con Λ ≈ 250 kpc la longitud de correlación del campo a lo largo de la línea
de visión y L_LoS la longitud efectiva atravesada (≈ π r_c/2 de lado, L de
frente). Por ejemplo, con los ~40 nG que usaba la versión anterior del modelo
(Carretti et al. 2022) σ₀ se multiplica por 4 y N se divide entre 256. Las distancias escalan con r_c: p y
d½/r_c no dependen de r_c.

## Cómo reproducir

Desde la raíz del repositorio (tarda ~3 min en CPU; usa GPU con cupy si está
disponible):

```bash
python -m examples.filamento_whim.run_barrido_theta   # barrido Monte Carlo en θ
python -m examples.filamento_whim.run                 # mapas de ejemplo (fig. 1)
python -m examples.filamento_whim.plots               # figuras 1-6
python -m examples.filamento_whim.umbral_deteccion    # reporte del umbral (texto)
python -m pytest                                      # tests
```

Salidas en `examples/filamento_whim/results/` (no se versiona):

| Figura | Contenido |
|---|---|
| `fig1_mapas_rm.png` | Mapas de RM a θ = 0°, 30°, 60°, 90° (misma realización, escala común) |
| `fig2_perfiles_transversales.png` | σ_RM(d) apilado con ajuste beta; comparación beta vs gaussiana |
| `fig3_forma_vs_theta.png` | **Resultado central**: exponente p(θ) y semiancho d½(θ) |
| `fig4_sigma0_vs_theta.png` | Amplitud en el eje σ₀(θ) y dispersión de un solo filamento |
| `fig5_gaussianidad_rm.png` | Histograma de RM/σ esperado contra N(0, 1) |
| `fig6_umbral_deteccion.png` | Fuentes necesarias vs θ y campo mínimo detectable vs N |

## Flujo de trabajo

```
config_fisica.py (parámetros con unidades y su justificación)
      │
      ▼
model.construir_escenario()      → n_e del filamento finito, B turbulento
      │
      ▼
faradaymr.los.rotation_measure() → mapa de RM por realización
      │
      ▼
transverse_rm_dispersion()       → σ_RM(d) por semilla (estimador RMS)
      │
      ▼
run_barrido_theta.analizar_perfiles()
      → apilado de semillas, ajustes (beta con r_c fijo, beta libre,
        gaussiana), errores por bootstrap, dispersión de un filamento
      │
      ▼
plots.py, umbral_deteccion.py
```

Antes del barrido, `validacion.py` comprueba que la caja contenga tanto la
longitud del filamento proyectada en la línea de visión
(`verificar_caja_suficiente`) como su cola radial
(`verificar_profundidad_radial`): si la caja corta la cola, los bins
exteriores pierden varianza y el perfil lateral sale más empinado.

## Decisiones de análisis

- **Estimador RMS respecto de cero, no `np.std`.** Como ⟨RM⟩ = 0, el
  promedio de RM² en un bin es un estimador insesgado de σ_RM². `np.std`
  resta la media del bin, que se come la varianza cuando el bin cabe en una
  longitud de correlación (el disco central a θ = 0): subestimaba σ₀ hasta
  20% y creaba un "bache" espurio en el ancho a θ ≈ 10°.
- **Apilado y bootstrap.** Se promedian los RM² de todas las semillas y se
  ajusta ese perfil (promediar ajustes individuales sesga los parámetros).
  Las barras de error son bootstrap sobre semillas. La banda de la fig. 4 es
  la dispersión entre realizaciones individuales: lo que vería un
  observador con un solo filamento.
- **Ajuste con r_c fijo.** En una observación real r_c se conoce del perfil
  de rayos X o SZ del filamento; fijarlo deja p(θ) como observable limpio.
  El ajuste con r_c libre recupera r_c = 314–332 kpc (esperado 300; la
  diferencia viene de la longitud de correlación finita del campo).
- **Validación del código** (`faradaymr.analysis.expected`, no se
  dibuja en las figuras): para un campo gaussiano,
  ⟨RM²⟩ = (0.812 dl)² Σ n_e(a) n_e(b) C_zz(a − b), con C_zz la correlación
  de B_z que se obtiene del mismo espectro del generador. Es el promedio de
  ensamble del MISMO modelo, así que no confirma la física: comprueba que
  la generación del campo, la integración de RM, las máscaras, el
  estimador y los ajustes son insesgados. `test_expected.py` verifica que
  el Monte Carlo converge a él; los valores quedan en las llaves `esp_*`
  del `.npz`. Aparte, con β = 0.8 y β = 1.0 (valores nunca usados al
  desarrollar), el exponente simulado reproduce los límites de fórmula
  cerrada dentro de 1σ de frente y 2–3% por debajo de lado, el mismo efecto
  de correlación finita que a β = 2/3.
- **Sistemático menor sin cerrar:** σ₀ simulado queda 2–3% por debajo del
  valor esperado en todos los ángulos (1–2σ). La hipótesis más probable es
  la normalización de cada realización a B_rms exacto; no está verificada.

## Limitaciones del modelo

- n_e y B son independientes; no se incluye B ∝ n_e^η (Murgia et al. 2004),
  que haría más empinado el perfil (el exponente pasaría a depender de η).
- El filamento está aislado: no incluye otras estructuras en la línea de
  visión ni el foreground galáctico, que en la práctica domina (Stuardi et
  al. 2026).
- r_c = 300 kpc es más compacto que el ajuste de Tanimura et al. (2020)
  para filamentos de 30–100 Mpc (r_c ≈ 1.5 Mpc); los resultados en unidades
  de d/r_c no cambian.
- El umbral usa σ₀ en el eje (caso más favorable) y supone ruido gaussiano.

## Nota sobre "ray-tracing"

La propuesta menciona "algoritmos de trazado de rayos (ray-tracing)". Aquí
se usan **líneas de visión paralelas** (`faradaymr/los.py`), que es lo
correcto para un observador externo a distancia cosmológica: todos los
rayos que llegan al telescopio desde el filamento son, en la práctica,
paralelos. El ray-tracing con observador interior y rayos divergentes (un
futuro `los_raytrace.py`) corresponde a otro proyecto, sobre estructuras
vistas desde dentro.
