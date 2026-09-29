# Filamento WHIM — Rotación de Faraday en un filamento cósmico aislado

Modelo idealizado (*toy model*) de un filamento del Medio Intergaláctico
Tibio-Caliente (WHIM): mapas simulados de Medida de Rotación (RM) en función
del ángulo de visión, perfil transversal de su dispersión y umbral de
detección con RM-grids (POSSUM, SKA).

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

Observable: el perfil transversal de la dispersión σ_RM(d), con d la
distancia al eje proyectado del filamento, para cada θ.

## Resultados (B = 10 nG, 50 realizaciones)

| θ | σ₀ (rad/m²) | p | d½ (kpc) | dispersión de 1 filamento | χ²_ν ley beta | χ²_ν gaussiana |
|---|---|---|---|---|---|---|
| 0° | 0.0311 ± 0.0012 | 0.99 ± 0.03 | 303 ± 7 | 27% | 0.74 | 66 |
| 30° | 0.0210 ± 0.0006 | 0.83 ± 0.03 | 343 ± 8 | 21% | 0.79 | 35 |
| 60° | 0.0162 ± 0.0003 | 0.75 ± 0.01 | 371 ± 6 | 13% | 0.85 | 60 |
| 90° | 0.0149 ± 0.0002 | 0.72 ± 0.01 | 380 ± 5 | 10% | 0.91 | 86 |

χ²_ν es el chi² reducido de cada ajuste, ponderado por los errores bootstrap
(9 grados de libertad). Los puntos de distintos θ usan las mismas 50
realizaciones del campo, así que sus fluctuaciones están correlacionadas
entre ángulos.

1. **Momentos de RM/σ de ensamble por píxel** (fig. 5), sobre los
   1.7 × 10⁶ píxeles de todos los ángulos: media +0.000, desviación 0.987,
   asimetría +0.002, exceso de curtosis −0.019. Por ángulo: desviación
   0.95–1.00; |asimetría| y |exceso de curtosis| < 0.08.
2. **Forma del perfil** (fig. 2): ley beta σ_RM(d) = σ₀ (1 + d²/r_c²)^(−p),
   χ²_ν = 0.5–0.9; gaussiana, χ²_ν = 25–86. (R² sin pesos: ≥ 0.997 beta,
   0.94–0.95 gaussiana.)
3. **Exponente p(θ)** (fig. 3): de 0.99 ± 0.03 (0°) a 0.72 ± 0.01 (90°).
   Límites de paseo aleatorio: p = 3β/2 = 1.00 (0°) y p = (3β − ½)/2 = 0.75
   (90°). Semiancho d½: de 303 a 380 kpc (+25%).
4. **Amplitud** (fig. 4): σ₀(0°)/σ₀(90°) = 2.09.
5. **Umbral** (fig. 6, detección a 3σ): σ_intr/σ₀ = 225–470. Fuentes
   necesarias para 10 nG: 9.3 × 10¹⁰ – 1.8 × 10¹² (solo σ_intr = 7 rad/m²)
   y 1.4 × 10¹² – 2.7 × 10¹³ (POSSUM, 7 + 12 rad/m²). Campo mínimo con 54
   fuentes (θ = 0°): 2.0 µG y 4.0 µG.
6. **Perfil de ensamble** (`esp_*`): p = 0.98 (0°) a 0.735 (90°);
   σ₀(MC)/σ₀(ensamble) = 0.97–0.99.
7. **Ajuste con r_c libre**: r_c = 281–316 kpc (MC), 314–332 kpc (ensamble).

### Leyes de escala

    σ₀ ≈ 0.016 rad/m² · (n0 / 10⁻⁵ cm⁻³) · (B / 10 nG) · (Λ / 250 kpc)^½ · (L_LoS / 470 kpc)^½
    N ∝ σ₀⁻⁴

Λ ≈ 250 kpc: longitud de correlación de B_z en la línea de visión.
L_LoS ≈ π r_c/2 de lado y ≈ L de frente. p y d½/r_c no dependen de r_c.

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
| `fig3_forma_vs_theta.png` | Exponente p(θ) y semiancho d½(θ) |
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

Antes del barrido, `validacion.py` comprueba que la caja contenga la
longitud del filamento proyectada en la línea de visión
(`verificar_caja_suficiente`) y su cola radial
(`verificar_profundidad_radial`).

## Decisiones de análisis

- **Estimador RMS respecto de cero** (sqrt⟨RM²⟩), no `np.std`, que resta la
  media del bin.
- **Apilado y bootstrap.** Se promedian los RM² de todas las semillas y se
  ajusta ese perfil; errores por bootstrap sobre semillas. La banda de la
  fig. 4 es la dispersión del ajuste entre realizaciones individuales.
  Todos los θ usan las mismas semillas.
- **Ajuste con r_c fijo** al del perfil de densidad (modo principal); el
  ajuste con r_c libre también se guarda.
- **Bondad de ajuste:** χ² reducido de ajustes ponderados por los errores
  bootstrap; el R² se guarda sin pesos.
- **Validación del código** (`faradaymr.analysis.expected`):
  ⟨RM²⟩ = (0.812 dl)² Σ n_e(a) n_e(b) C_zz(a − b), con C_zz del mismo
  espectro del generador. `test_expected.py` verifica que el Monte Carlo
  converge a ese valor; el perfil de ensamble queda en las llaves `esp_*`.
- **σ₀(MC)/σ₀(ensamble) = 0.97–0.99 con 50 semillas.** Con 200 semillas
  (θ = 0° y 90°) el cociente ⟨RM²⟩/⟨RM²⟩_ensamble es 0.970 ± 0.021 a
  1.000 ± 0.061; con y sin normalizar cada realización a B_rms exacto da
  lo mismo (B_rms² varía 0.7% entre realizaciones).

## Limitaciones del modelo

- n_e y B independientes (sin B ∝ n_e^η).
- Filamento aislado, sin foreground galáctico.
- r_c = 300 kpc (Tanimura et al. 2020: ≈ 1.5 Mpc para filamentos de 30–100 Mpc).
- El umbral usa σ₀ en el eje y supone ruido gaussiano.

## Nota sobre "ray-tracing"

Se usan **líneas de visión paralelas** (`faradaymr/los.py`), para un
observador externo a distancia cosmológica. El trazado con observador
interior (rayos divergentes) corresponde a otro proyecto.
