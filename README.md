# Dashboard — Simulación Monte Carlo MPI-DUAE

Panel interactivo de una sola página (HTML + JavaScript) que visualiza los resultados de la simulación Monte Carlo del diseño cuasiexperimental de **réplica cruzada contrabalanceada (crossover AB/BA)** del estudio doctoral sobre el **Modelo Pedagógico Integrado DUA-Experiencial (MPI-DUAE)** en el IESP DYC (Lima Metropolitana, 2026-2027).

> **Aviso importante:** los datos mostrados son **simulados** bajo los supuestos del diseño (d = 0.30, ρ = 0.50, N = 120), no observaciones reales de estudiantes. Su función es **validar la capacidad del diseño metodológico** para detectar el efecto esperado, no anticipar resultados empíricos.

---

## Contenido

- [Características](#características)
- [Estructura del panel](#estructura-del-panel)
- [Diseño metodológico simulado](#diseño-metodológico-simulado)
- [Datos embebidos](#datos-embebidos)
- [Requisitos](#requisitos)
- [Uso](#uso)
- [Parámetros de la simulación](#parámetros-de-la-simulación)
- [Interpretación de los paneles](#interpretación-de-los-paneles)
- [Estructura del código](#estructura-del-código)
- [Personalización](#personalización)
- [Referencias metodológicas](#referencias-metodológicas)

---

## Características

- **Una sola página autocontenida** (`index.html`) — sin build, sin backend.
- **Chart.js 4.5.1** (vía CDN con SRI) para los gráficos: Gantt, potencia, forest plot, medias, robustez.
- **8 secciones navegables** con scroll suave y barra de navegación *sticky*.
- **Diagrama interactivo embebido** (iframe a `Diagramas/crossover_cronograma_mpi_duae.html`, generado con Archify).
- **Tablas ordenables** por columna (cliente, sin dependencias).
- **Heatmap ρ×d** con paleta Viridis calculada en tiempo real.
- **Toggle "solo significativas"** (Bonferroni α/5 = .01) que filtra la tabla de efectos.
- **Selector "resaltar variable"** que atenúa las no seleccionadas en los gráficos de efectos y medias.
- **Cajas desplegables (`<details>`)** con la función, lectura e impacto en la tesis de cada gráfico/tabla.
- **Estilos listos para impresión** (`@media print`).
- **Diseño responsive** (grid adaptable, colapsos en ≤ 768 px).

---

## Estructura del panel

El panel se organiza en secciones ancladas (`id="sec-*"`):

| Sección | `id` | Contenido |
|---|---|---|
| Investigación | `sec-investigacion` | Pregunta, diseño, sistema de hipótesis (HG, HE1, HE1a-e, HE2, HE3). |
| Cronograma | `sec-cronograma` | Gantt del semestre 2026-2027 (18 semanas) + 5 fichas de hitos (T1, T2, T3, S18). |
| Diagrama interactivo | `sec-diagrama` | iframe con el diagrama Archify del crossover AB/BA. |
| Potencia | `sec-power` | Curva de potencia empírica vs. N. |
| Tamaños de efecto | `sec-efectos` | Forest plot con d<sub>z</sub>, IC 95 % y p. |
| Medias | `sec-medias` | Medias por condición (A = MPI-DUAE, B = comparación). |
| Robustez | `sec-robustez` | Potencia bajo sesgos (outliers, heterocedasticidad, atrición). |
| Sensibilidad ρ×d | `sec-heatmap` | Heatmap Viridis con celda destacada (ρ = 0.50, d = 0.30). |
| Tablas | `sec-tablas` | Pestañas: Efectos, Descriptivos, Supuestos, HE3 (logística). |

---

## Diseño metodológico simulado

- **Tipo:** cuasiexperimental de réplica cruzada contrabalanceada (crossover AB/BA), **intrasujeto** — cada estudiante es su propio control.
- **Secciones:** 6 (3 con secuencia AB, 3 con BA); 4 programas de estudio como factor de bloqueo.
- **Covariables:** línea de base de cada VD, rendimiento académico previo.
- **Controles:** periodo (P1/P2), secuencia (arrastre), forma del TTCT.
- **Muestra:** N = 120 casos completos (de 145 plazas totales).
- **Instrumentos:** TTCT Figural (5 dimensiones) y PSSM (pertenencia global).
- **Hipótesis:**
  - **HG** — diferencia global asociada al MPI-DUAE.
  - **HE1** — contraste conjunto de las 5 dimensiones del TTCT (Hotelling T²).
    - **HE1a-e** — subhipótesis por dimensión (Bonferroni α/5 = .01, *procedimiento de puerta cerrada*).
  - **HE2** — diferencia intrasujeto en PSSM global.
  - **HE3 (exploratoria)** — PSSM → permanencia (regresión logística).

---

## Datos embebidos

Todos los resultados están embebidos en el objeto `DATA` (al final del HTML), provenientes de `simulacion_mpi_duae.py` (semilla fija **2026**) y sus CSV en `resultados/`:

| Clave en `DATA` | Origen | Usada en |
|---|---|---|
| `descriptivos` | `resultados/descriptivos.csv` | Tabla "Descriptivos", gráfico de medias |
| `supuestos` | `resultados/supuestos.csv` | Tabla "Supuestos" |
| `efectos` | `resultados/efectos.csv` | Forest plot, tabla "Efectos", KPIs |
| `he3` | `resultados/he3_logit.csv` | Tabla "HE3" |
| `potencia_n` | `resultados/potencia_n.csv` | Curva de potencia, KPI N=120 |
| `sensibilidad` | `resultados/sensibilidad_rho_d.csv` | Heatmap ρ×d |
| `robustez` | `resultados/robustez.csv` | Panel de robustez |

### Resultados principales (bajo los supuestos simulados)

- **HE1 ómnibus (Hotelling T²):** F(5,115) = 8.836, p < .0001 ✅
- **HE2 (PSSM):** t = 2.373, p = .019, d<sub>z</sub> = 0.217 ✅
- **Significativas con Bonferroni (α/5 = .01):** Elaboración, Abstracción de títulos, Resistencia al cierre y PSSM.
- **No significativas:** Fluidez (p = .066) y Originalidad (p = .044, pero > .01).
- **Potencia empírica (N = 120):** 90.2 % (objetivo ≥ 80 %).
- **HE3:** PSSM promedio predice permanencia (OR ≈ 3.21, p = .001). Ningún programa difiere significativamente frente a *Comunicación Audiovisual* (categoría de referencia).

---

## Requisitos

- **Navegador moderno** (Chrome/Edge/Firefox/Safari recientes). No requiere servidor.
- **Conexión a Internet** para cargar Chart.js desde el CDN (jsDelivr). Para uso 100 % *offline*, descargue `chart.umd.min.js` y ajuste el `<script src>`.
- **Archivo complementario requerido** para la sección "Diagrama interactivo":
  `Diagramas/crossover_cronograma_mpi_duae.html` (relativo a `index.html`). Sin él, el iframe aparecerá vacío — el resto del panel funciona con normalidad.

---

## Uso

1. Coloque los archivos en la siguiente estructura:
   ```
   proyecto/
   ├── index.html
   └── Diagramas/
       └── crossover_cronograma_mpi_duae.html
   ```
2. Abra `index.html` con doble clic o sírvalo con:
   ```bash
   python -m http.server 8000
   # luego visita http://localhost:8000/
   ```
3. Navegue con la barra superior. Haga **clic en las barras del Gantt** para saltar al panel correspondiente.

### Interacciones rápidas

| Elemento | Acción |
|---|---|
| Barra/hito del Gantt | Clic → salta a la sección vinculada. |
| Selector "Resaltar variable" | Atenúa las series no seleccionadas en efectos y medias. |
| Toggle "Solo significativas" | Filtra la tabla de efectos por Bonferroni. |
| Encabezados de tabla | Clic → ordena asc/desc por esa columna. |
| Cajas `<details>` | Clic en el título → despliega función, lectura e impacto en la tesis. |
| Pestañas "Tablas de detalle" | Cambia entre Efectos / Descriptivos / Supuestos / HE3. |

---

## Parámetros de la simulación

| Parámetro | Valor |
|---|---|
| Iteraciones por escenario (principal) | B = 10 000 |
| Iteraciones por escenario (robustez) | B = 3 000 |
| Semilla | 2026 (fija) |
| N planificado | 120 |
| ρ (correlación intrasujeto) | 0.50 |
| d de Cohen esperado | 0.30 |
| α | 0.05 (Bonferroni α/5 = .01) |
| Potencia objetivo | 0.80 |

---

## Interpretación de los paneles

Cada gráfico y tabla del panel incluye una caja `<details>` con tres apartados:

1. **Función** — qué verifica la simulación en ese punto del diseño.
2. **Lectura del resultado** — qué dicen los datos simulados y por qué.
3. **Impacto en la tesis doctoral** — qué sección del documento sustenta (p. ej. 1.7 viabilidad poblacional, 2.3.1.2 análisis de potencia, 2.3.2.2 procedimiento de puerta cerrada, 2.3.3 amenazas a la validez).

Esto convierte el panel en una herramienta de **defensa metodológica**, no solo de visualización.

---

## Estructura del código

```
index.html
├── <style>          → Variables CSS, layout, componentes, @media print
├── <body>
│   ├── header       → Título + filtros (Resaltar variable, Solo significativas)
│   ├── nav.page-nav → Barra sticky con anclas a cada sección
│   ├── .notice      → Aviso metodológico (datos simulados)
│   └── secciones    → Investigación, Cronograma, Diagrama, KPIs,
│                      charts (potencia, efectos, medias, robustez),
│                      heatmap, tablas, footer
└── <script>
    ├── DATA         → Resultados de la simulación (JSON embebido)
    ├── VAR_LABELS   → Mapeo variable técnica → etiqueta legible
    ├── GANTT_ROWS   → Cronograma del semestre (18 semanas)
    ├── Utilidades   → viridis(), lerpColor(), fmtPct(), fmtP()
    └── class Dashboard
        ├── renderKPIs()          ├── renderPowerChart()
        ├── renderGanttChart()    ├── renderEffectsChart()
        ├── renderMeansChart()    ├── renderRobustnessChart()
        ├── renderHeatmap()       ├── renderTables()
        ├── buildSortableTable()  ├── setupTabs()
        ├── toggleSigOnly()       └── setHighlight()
```

La instancia global `dash` se crea al final del script y queda accesible desde la consola del navegador (p. ej. `dash.setHighlight('elaboracion')`).

---

## Personalización

### Reemplazar los datos simulados por datos reales

Sustituya el objeto `DATA` por los CSV reales (misma estructura de claves y campos). Los nombres de campo sensibles son:

- `efectos`: `variable`, `dz`, `ic95_lo`, `ic95_hi`, `p`, `significativo_bonferroni_.01`.
- `descriptivos`: `variable`, `condicion`, `n`, `media`, `de`, `mediana`, `ric`, `asimetria`, `curtosis`.
- `supuestos`: `shapiro_p_A`, `shapiro_p_B`, `levene_F`, `levene_p`.
- `he3`: `termino`, `OR`, `IC95_lo`, `IC95_hi`, `p`.
- `potencia_n`: `N`, `potencia`.
- `sensibilidad`: `rho`, `d`, `potencia`.
- `robustez`: `escenario`, `potencia_empirica`.

### Cambiar el cronograma

Edite el arreglo `GANTT_ROWS`: cada fila tiene `label`, `start`, `end` (en semanas del semestre), `type` (`phaseA` | `phaseB` | `milestone` | `perm`), `detail`, `link` y `linkLabel`.

### Cambiar los colores

Los colores están centralizados en el bloque `:root` del `<style>` y en las constantes `VIRIDIS` y `GANTT_COLORS` del script.

---

## Referencias metodológicas

- **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences* (2.ª ed.).
- **G*Power 3.1** — cálculo a priori (t pareada bilateral, d = 0.30, α = .05, 1−β = .80).
- **Hotelling T²** — contraste conjunto de las 5 dimensiones del TTCT (HE1).
- **Bonferroni α/5 = .01** — control de error tipo I en HE1a-e (*procedimiento de puerta cerrada*).
- **Hausmann, L. R. M., et al.** (2007, 2009) — cadena teórica pertenencia → compromiso → permanencia (base de HE3).
- **TTCT Figural** (Torrance) — dimensiones: fluidez, originalidad, elaboración, abstracción de títulos, resistencia al cierre.
- **PSSM** — *Psychological Sense of School Membership* (Goodenow).

---

## Licencia y autoría

- **Autor del panel y del estudio:** Mg. Mario Rafael Quiroz Martínez.
- **Programa:** Doctorado en Educación — IESP DYC, Lima Metropolitana, 2026-2027.
- **Script generador:** `simulacion_mpi_duae.py` (semilla fija 2026).
- **Visualización del diagrama:** Archify.

Este panel es una **instantánea** de la simulación, no un sistema de datos en vivo. Los resultados cambian solo si se vuelve a ejecutar el script con una semilla o parámetros distintos y se regenera el objeto `DATA`.