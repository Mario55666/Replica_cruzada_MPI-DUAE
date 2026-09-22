# Resumen de la simulacion Monte Carlo -- MPI-DUAE

**Semilla:** 2026 | **N principal:** 120 | **rho:** 0.5 | **d esperado:** 0.3 |
**alpha:** 0.05 (Bonferroni HE1a-e: 0.01) | **potencia objetivo:** 0.8 |
**iteraciones Monte Carlo (contraste principal):** 10000

## 1. Verificacion del analisis de potencia a priori (G*Power 3.1)

- Efecto minimo detectable teorico (formula cerrada, N=120, rho=0.5): **0.256**
  (rango reportado en la tesis: 0.22-0.31).
- Potencia empirica por Monte Carlo para d=0.3: **0.907**.

## 2. Resultado del contraste principal (una replica representativa)

- **HE1 (omnibus, Hotelling T2, 5 dimensiones TTCT):**
  F(5, 115) =
  8.836, p = 0.0000
- **HE2 (PSSM):** t = 2.373, p = 0.0193,
  dz = 0.217 [0.036, 0.398]

Ver `03_tamanos_efecto.csv` para las subhipotesis HE1a-HE1e (Bonferroni alpha/5=0.01).

## 3. Robustez (Monte Carlo, B=3000 por escenario)

| escenario           |   potencia_empirica |
|:--------------------|--------------------:|
| base                |               0.832 |
| outliers_5pct       |               0.708 |
| heterocedastico     |               0.620 |
| attrition_historica |               0.799 |

## 4. Archivos generados

- `01_descriptivos.csv`, `02_pruebas_supuestos.csv`, `03_tamanos_efecto.csv`
- `04_prueba_omnibus_HE1.txt`, `05_modelo_mixto.txt`, `06_HE3_regresion_logistica_OR.csv`
- `07_verificacion_potencia_apriori.txt`, `08_potencia_vs_N.csv`, `09_sensibilidad_rho_d.csv`
- `10_robustez_sesgos.csv`, `dataset_simulado_representativo.csv`
- Figuras en `../figuras/`: a) distribucion/densidad, b) curva de potencia vs N,
  c) forest plot + regresion pre-post, d) violin+boxplot+puntos, e) mapa de
  sensibilidad rho x d (figura suplementaria).

## 5. Limitaciones de la simulacion (declaradas)

- Las seis variables de resultado comparten el mismo componente "entre
  sujetos" (rho) por simplicidad computacional; en la practica cada
  instrumento (TTCT y PSSM) tendria su propia estructura de varianza.
- Los tamanos de efecto por dimension del TTCT (`D_BY_DIM`) son supuestos
  de la simulacion para ilustrar el forest plot, no estimaciones
  empiricas: la tesis no fija a priori un d distinto por dimension.
- El barrido de potencia usa la equivalencia cerrada de la prueba t
  pareada (Var(diferencia) = 2(1-rho)) en lugar de reajustar un modelo
  mixto en cada una de las 10000 iteraciones, por eficiencia
  computacional; el modelo mixto completo se ajusta una vez sobre un
  dataset representativo (`05_modelo_mixto.txt`) para verificar
  consistencia.
