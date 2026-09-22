"""
Simulacion Monte Carlo y analisis de potencia para la tesis doctoral:
"Modelo Pedagogico Integrado DUA-Experiencial (MPI-DUAE) frente al enfoque
por competencias, en el pensamiento creativo divergente y el sentido de
pertenencia inclusiva de ingresantes a programas de Diseno y Comunicacion
en un instituto publico de educacion superior no universitaria de Lima
Metropolitana (IESP DYC), 2026-2027."

Diseno real de la tesis (Capitulo I y II, secciones 1.7, 2.3.1-2.3.2):
- Cuasiexperimental de replica cruzada contrabalanceada (crossover AB/BA),
  intrasujeto: cada estudiante es su propio control.
- 6 secciones (3 con secuencia AB, 3 con secuencia BA), programa de estudios
  (Diseno Publicitario, Diseno de Interiores, Comunicacion Audiovisual,
  Diseno de Modas) como factor de bloqueo cruzado con la secuencia.
- N = 120 casos completos (capacidad total 145 plazas).
- Correlacion entre medidas repetidas rho = 0.50 y tamano de efecto
  esperado d = 0.30 (por debajo de 0.50 dado que ambas condiciones operan
  bajo el mismo enfoque por competencias), alpha = 0.05, potencia objetivo
  (1-beta) = 0.80, calculados a priori con G*Power 3.1 (Faul et al., 2007).
  Efecto minimo detectable reportado en la tesis: entre 0.22 y 0.31.
- VD1 Pensamiento creativo divergente: TTCT Figural, 5 dimensiones
  (fluidez, originalidad, elaboracion, abstraccion de titulos, resistencia
  al cierre prematuro). HE1 es un contraste conjunto (puerta cerrada);
  las subhipotesis HE1a-HE1e solo se examinan si HE1 es significativa,
  con correccion de Bonferroni alpha/5 = .01.
- VD2 Sentido de pertenencia inclusiva: escala PSSM (18 items), puntaje
  global. HE2 es un contraste univariado simple.
- VD3 (HE3, exploratoria, sin factor de condicion): permanencia academica
  (0 = retirado, 1 = permanece) ~ PSSM + rendimiento previo + programa,
  via regresion logistica binaria (razones de momios, IC 95%).
- Covariables del contraste principal: linea de base de cada VD y
  rendimiento academico previo. Terminos de control: periodo, secuencia
  (arrastre/carryover), forma del TTCT (alternada) y programa (bloqueo).

Enfoque estadistico de la simulacion
------------------------------------
Todas las variables continuas se trabajan en unidades estandarizadas
(SD = 1 en la poblacion total) para que el tamano de efecto d se
interprete directamente como diferencia de medias en desviaciones
estandar (Cohen, 1988). Para un diseno de medidas repetidas de 2
periodos, la varianza total se descompone en un componente entre
sujetos (varianza = rho) y un componente intrasujeto residual
(varianza = 1 - rho), de modo que:

    Var(diferencia A - B) = 2 * (1 - rho)

Esta es la relacion cerrada que usa G*Power para la prueba t de
medidas dependientes, y es la que se reproduce aqui por simulacion
para verificar el efecto minimo detectable reportado en la tesis
(0.22-0.31 con N=120, rho=0.50).

Para el contraste principal HE1 (conjunto de 5 dimensiones) se usa la
T2 de Hotelling para diferencias pareadas multivariadas, que es el
analogo multivariado exacto de un t pareado y evita inflar el error
Tipo I por comparaciones multiples no corregidas antes del "cierre"
del procedimiento de puerta cerrada. Las subhipotesis HE1a-HE1e y HE2
se evaluan con pruebas t pareadas (equivalentes, bajo normalidad, al
termino de tratamiento de un modelo mixto de medidas repetidas que
controla periodo y secuencia, dado que period y secuencia se
contrabalancean de forma cruzada con la condicion en este diseno).
Un modelo mixto (statsmodels MixedLM) se ajusta ademas una vez sobre
un dataset simulado representativo, para reportar los coeficientes de
periodo, secuencia y condicion tal como se reportarian en la tesis.

Autor: generado para apoyar el diseno metodologico de M. Quiroz
        (Programa de Doctorado en Educacion).
Semilla fija: 2026 (reproducibilidad exacta).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Configuracion global: reproducibilidad, rutas de salida y estilo grafico
# ---------------------------------------------------------------------------

SEED = 2026
RNG = np.random.default_rng(SEED)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figuras"
RES_DIR = OUT_DIR / "resultados"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# Paleta viridis (accesible para daltonismo) y estilo de publicacion (APA-like)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PALETTE = sns.color_palette("viridis", 8)
COND_PALETTE = {"A_MPI-DUAE": PALETTE[1], "B_Comparacion": PALETTE[5]}
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

# ---------------------------------------------------------------------------
# 1. Parametros del diseno y de la simulacion (tomados de la tesis)
# ---------------------------------------------------------------------------

N_MAIN = 120                 # casos completos (1.7; 2.3.1.2)
N_CAPACITY = 145             # capacidad total de plazas (2.1.1)
N_SWEEP = np.arange(60, 151, 10)   # barrido de sensibilidad de N (60-150)

RHO = 0.50                   # correlacion entre medidas repetidas (2.3.1.2)
RHO_SWEEP = np.round(np.arange(0.30, 0.71, 0.05), 2)  # sensibilidad de rho

D_EFFECT = 0.30               # d de Cohen esperado (2.3.1.2)
D_SWEEP = np.round(np.arange(0.20, 0.51, 0.05), 2)    # sensibilidad de d
MDE_BAND = (0.22, 0.31)       # efecto minimo detectable reportado en la tesis

ALPHA = 0.05
ALPHA_BONF = ALPHA / 5        # correccion para HE1a-HE1e (2.3.2.2)
POWER_TARGET = 0.80

B_ITER = 10_000               # iteraciones Monte Carlo (contraste principal)
B_SWEEP = 3_000                # iteraciones por punto en los barridos (N y rho/d)

# Dimensiones de la variable dependiente 1 (TTCT Figural, Tabla 2)
TTCT_DIMS = [
    "fluidez",
    "originalidad",
    "elaboracion",
    "abstraccion_titulos",
    "resistencia_cierre",
]
# Pequenas diferencias plausibles de d por dimension alrededor de 0.30,
# unicamente para que el forest plot no muestre cinco puntos identicos;
# se documentan como supuesto de la simulacion, no como dato empirico.
D_BY_DIM = {
    "fluidez": 0.32,
    "originalidad": 0.24,
    "elaboracion": 0.30,
    "abstraccion_titulos": 0.28,
    "resistencia_cierre": 0.33,
}
DV2 = "pssm_global"
D_BY_DIM[DV2] = D_EFFECT

PROGRAMAS = [
    "Diseno Publicitario",
    "Diseno de Interiores",
    "Comunicacion Audiovisual",
    "Diseno de Modas",
]
# Seis secciones -> tres con secuencia AB, tres con BA, cruzadas con programa
# y turno (Tabla 8). Se aproxima un tamano de seccion uniforme para la
# simulacion; la asignacion real varia por la capacidad de cada programa.
SECCIONES = pd.DataFrame(
    {
        "seccion": [f"S{i}" for i in range(1, 7)],
        "programa": [
            "Diseno Publicitario",
            "Diseno Publicitario",
            "Diseno de Interiores",
            "Diseno de Interiores",
            "Comunicacion Audiovisual",
            "Diseno de Modas",
        ],
        "secuencia": ["AB", "BA", "BA", "AB", "AB", "BA"],
    }
)

# Tasa historica de retiro en primer semestre, IESP DYC 2023-2025 (1.2):
# de 1/261 (~0.4%) a 19/178 (~10.7%). Se usa el valor mas reciente y
# critico como escenario de attrition en el analisis de sensibilidad.
ATTRITION_RATE_HISTORICA = 19 / 178


# ---------------------------------------------------------------------------
# 2. Generador del dataset simulado (una replica del diseno crossover)
# ---------------------------------------------------------------------------

def simular_dataset(
    n_subjects: int,
    rho: float,
    d_by_var: dict,
    rng: np.random.Generator,
    period_effect: float = 0.10,
    carryover_effect: float = 0.05,
    program_effect_sd: float = 0.08,
    contaminar_outliers: bool = False,
    heterocedastico: bool = False,
    attrition_rate: float = 0.0,
) -> pd.DataFrame:
    """Simula un dataset en formato largo (2 filas por estudiante) para el
    diseno de replica cruzada contrabalanceada.

    Todas las variables de resultado se generan en unidades estandarizadas
    (SD poblacional = 1). La varianza total de cada resultado se
    descompone en un componente estable entre sujetos (varianza = rho,
    que representa el "trait" psicometrico del estudiante) y un
    componente residual intrasujeto (varianza = 1 - rho). El efecto de
    la condicion (MPI-DUAE vs comparacion) se anade como un desplazamiento
    fijo de magnitud d (Cohen) entre A y B.
    """
    # Asignacion de secciones (programa x secuencia), aprox. n/6 por seccion
    n_per_section = int(np.ceil(n_subjects / 6))
    seccion_rows = SECCIONES.loc[np.tile(SECCIONES.index, n_per_section)].reset_index(drop=True)
    seccion_rows = seccion_rows.iloc[:n_subjects].copy()
    seccion_rows["subject_id"] = np.arange(1, n_subjects + 1)

    # Efecto de bloqueo por programa (aleatorio pequeno, fijo dentro del programa)
    programa_effect = {
        p: rng.normal(0, program_effect_sd) for p in PROGRAMAS
    }
    seccion_rows["programa_effect"] = seccion_rows["programa"].map(programa_effect)

    # Covariable: rendimiento academico previo (estandarizado), correlacionado
    # moderadamente con el resultado (Tabachnick y Fidell, 2013)
    seccion_rows["rendimiento_previo"] = rng.normal(0, 1, size=n_subjects)

    # Componente estable entre sujetos (mismo para las 6 variables por
    # simplicidad computacional; en la practica cada instrumento tendria su
    # propio "trait", pero la correlacion rho es la misma por diseno)
    b_subject = rng.normal(0, np.sqrt(rho), size=n_subjects)

    # Formato largo: 2 periodos por sujeto
    long_rows = []
    for _, row in seccion_rows.iterrows():
        secuencia = row["secuencia"]
        orden_condiciones = ["A", "B"] if secuencia == "AB" else ["B", "A"]
        for periodo, condicion in zip([1, 2], orden_condiciones):
            long_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "seccion": row["seccion"],
                    "programa": row["programa"],
                    "programa_effect": row["programa_effect"],
                    "secuencia": secuencia,
                    "rendimiento_previo": row["rendimiento_previo"],
                    "periodo": periodo,
                    "condicion": condicion,
                    "forma_ttct": "A" if periodo == 1 else "B",  # alternada y
                    # contrabalanceada, cruzada con la condicion (2.3.3)
                }
            )
    df = pd.DataFrame(long_rows)
    df["b_subject"] = df["subject_id"].map(dict(zip(seccion_rows["subject_id"], b_subject)))

    n_long = len(df)
    sigma_e = np.sqrt(1 - rho)

    # Termino de arrastre (carryover): afecta solo al segundo periodo de la
    # secuencia AB (arrastre del efecto de A hacia el periodo de B)
    carry = np.where(
        (df["secuencia"] == "AB") & (df["periodo"] == 2), carryover_effect, 0.0
    )
    period_fx = np.where(df["periodo"] == 2, period_effect, 0.0)

    for var, d in d_by_var.items():
        cond_effect = np.where(df["condicion"] == "A", d / 2, -d / 2)
        if heterocedastico:
            # Periodo 2 con mayor dispersion residual (p. ej. fatiga / fin de semestre)
            sd_resid = np.where(df["periodo"] == 2, sigma_e * 1.6, sigma_e)
            eps = rng.normal(0, 1, n_long) * sd_resid
        else:
            eps = rng.normal(0, sigma_e, n_long)

        y = (
            df["b_subject"]
            + df["programa_effect"]
            + 0.15 * df["rendimiento_previo"]
            + cond_effect
            + period_fx
            + carry
            + eps
        )

        if contaminar_outliers:
            # 5% de observaciones con ruido de cola pesada (t de Student, df=2)
            mask = rng.random(n_long) < 0.05
            y[mask] += stats.t.rvs(df=2, size=mask.sum(), random_state=rng.integers(1e9))

        df[var] = y

    # Linea de base (semana 1): correlacionada con el "trait" del sujeto,
    # medida antes de la exposicion a cualquier condicion
    for var in d_by_var:
        df[f"{var}_baseline"] = (
            df.groupby("subject_id")["b_subject"].transform("first")
            + rng.normal(0, sigma_e, n_long)
        )

    # Permanencia academica (HE3): depende de PSSM promedio del estudiante y
    # del rendimiento previo, con la tasa base historica del IESP DYC
    pssm_mean_subject = df.groupby("subject_id")[DV2].transform("mean")
    logit_p = (
        np.log(1 / (1 - ATTRITION_RATE_HISTORICA) - 1) * -1  # intercepto ~ tasa base
        + 0.9 * pssm_mean_subject
        + 0.4 * df["rendimiento_previo"]
    )
    prob_permanece = 1 / (1 + np.exp(-logit_p))
    df["permanece"] = rng.binomial(1, prob_permanece)

    if attrition_rate > 0:
        drop_subjects = rng.choice(
            df["subject_id"].unique(),
            size=int(round(attrition_rate * n_subjects)),
            replace=False,
        )
        df = df[~df["subject_id"].isin(drop_subjects)].reset_index(drop=True)

    return df


def a_ancho(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Convierte el dataset largo a formato ancho (una fila por sujeto) con
    columnas A y B para una variable de resultado dada."""
    wide = df.pivot_table(index="subject_id", columns="condicion", values=var)
    wide.columns = [f"{var}_{c}" for c in wide.columns]
    meta = df.drop_duplicates("subject_id").set_index("subject_id")[
        ["programa", "secuencia", "rendimiento_previo"]
    ]
    return meta.join(wide)


# ---------------------------------------------------------------------------
# 3. Estadistica descriptiva
# ---------------------------------------------------------------------------

def descriptivos(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    filas = []
    for var in variables:
        for cond, sub in df.groupby("condicion")[var]:
            filas.append(
                {
                    "variable": var,
                    "condicion": cond,
                    "n": sub.shape[0],
                    "media": sub.mean(),
                    "de": sub.std(ddof=1),
                    "mediana": sub.median(),
                    "ric": sub.quantile(0.75) - sub.quantile(0.25),
                    "asimetria": stats.skew(sub),
                    "curtosis": stats.kurtosis(sub),
                }
            )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 4. Pruebas de supuestos
# ---------------------------------------------------------------------------

def pruebas_supuestos(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    filas = []
    for var in variables:
        a = df.loc[df["condicion"] == "A", var]
        b = df.loc[df["condicion"] == "B", var]
        sw_a = stats.shapiro(a)
        sw_b = stats.shapiro(b)
        levene = stats.levene(a, b)
        filas.append(
            {
                "variable": var,
                "shapiro_W_A": sw_a.statistic,
                "shapiro_p_A": sw_a.pvalue,
                "shapiro_W_B": sw_b.statistic,
                "shapiro_p_B": sw_b.pvalue,
                "levene_F": levene.statistic,
                "levene_p": levene.pvalue,
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 5. Pruebas de hipotesis principales (HE1 omnibus, HE1a-e, HE2)
# ---------------------------------------------------------------------------

def hotelling_t2_pareado(diffs: np.ndarray):
    """T2 de Hotelling para un vector de diferencias pareadas (n x p).
    Devuelve (F, gl1, gl2, p)."""
    n, p = diffs.shape
    mean_d = diffs.mean(axis=0)
    cov_d = np.cov(diffs, rowvar=False)
    t2 = n * mean_d @ np.linalg.solve(cov_d, mean_d)
    f_stat = (n - p) / (p * (n - 1)) * t2
    gl1, gl2 = p, n - p
    p_val = 1 - stats.f.cdf(f_stat, gl1, gl2)
    return f_stat, gl1, gl2, p_val


def cohen_dz_ic(diffs: np.ndarray, alpha: float = 0.05):
    """d de Cohen para muestras pareadas (dz) con IC aproximado (delta method
    sobre la t no central)."""
    n = len(diffs)
    dz = diffs.mean() / diffs.std(ddof=1)
    se = np.sqrt(1 / n + dz**2 / (2 * n))
    z = stats.norm.ppf(1 - alpha / 2)
    return dz, dz - z * se, dz + z * se


def contraste_hipotesis(df: pd.DataFrame) -> dict:
    """Aplica el procedimiento de puerta cerrada de la tesis sobre UN dataset
    simulado: HE1 omnibus (Hotelling T2) -> si es significativa, HE1a-e con
    Bonferroni; HE2 univariada sobre PSSM."""
    resultados = {}

    diffs_ttct = np.column_stack(
        [
            a_ancho(df, dim).eval(f"{dim}_A - {dim}_B")
            for dim in TTCT_DIMS
        ]
    )
    f_stat, gl1, gl2, p_omnibus = hotelling_t2_pareado(diffs_ttct)
    resultados["HE1_omnibus"] = {"F": f_stat, "gl1": gl1, "gl2": gl2, "p": p_omnibus}

    subhipotesis = {}
    if p_omnibus < ALPHA:
        for i, dim in enumerate(TTCT_DIMS):
            d = diffs_ttct[:, i]
            t_stat, p_val = stats.ttest_1samp(d, 0)
            dz, lo, hi = cohen_dz_ic(d)
            subhipotesis[dim] = {
                "t": t_stat,
                "p": p_val,
                "significativo_bonferroni": p_val < ALPHA_BONF,
                "dz": dz,
                "ic95_lo": lo,
                "ic95_hi": hi,
            }
    resultados["HE1_subhipotesis"] = subhipotesis

    wide_pssm = a_ancho(df, DV2)
    diff_pssm = (wide_pssm[f"{DV2}_A"] - wide_pssm[f"{DV2}_B"]).values
    t_stat, p_val = stats.ttest_1samp(diff_pssm, 0)
    dz, lo, hi = cohen_dz_ic(diff_pssm)
    resultados["HE2"] = {
        "t": t_stat,
        "p": p_val,
        "significativo": p_val < ALPHA,
        "dz": dz,
        "ic95_lo": lo,
        "ic95_hi": hi,
    }
    return resultados


# ---------------------------------------------------------------------------
# 6. Modelo mixto de medidas repetidas (una vez, sobre dataset representativo)
# ---------------------------------------------------------------------------

def ajustar_modelo_mixto(df: pd.DataFrame, var: str):
    """Ajusta un modelo mixto lineal Y ~ condicion + periodo + secuencia +
    rendimiento_previo + baseline, con intercepto aleatorio por sujeto.
    Reproduce el termino de tratamiento que la tesis reporta como HE1/HE2."""
    datos = df.copy()
    datos["condicion_A"] = (datos["condicion"] == "A").astype(int)
    datos["periodo_2"] = (datos["periodo"] == 2).astype(int)
    datos["secuencia_AB"] = (datos["secuencia"] == "AB").astype(int)
    formula = (
        f"{var} ~ condicion_A + periodo_2 + secuencia_AB + rendimiento_previo + "
        f"{var}_baseline + C(programa)"
    )
    modelo = smf.mixedlm(formula, datos, groups=datos["subject_id"])
    return modelo.fit(reml=True)


# ---------------------------------------------------------------------------
# 7. Potencia empirica: barrido vectorizado (paired t-test) por N, rho y d
# ---------------------------------------------------------------------------

def potencia_empirica_vectorizada(
    n: int, rho: float, d: float, alpha: float, b_iter: int, rng: np.random.Generator
) -> float:
    """Calcula la potencia empirica de una prueba t pareada (equivalente,
    bajo normalidad y diseno balanceado, al termino de condicion de un
    modelo mixto que controla periodo/secuencia) mediante simulacion
    vectorizada: Var(diferencia) = 2*(1-rho) en la escala estandarizada."""
    sd_diff = np.sqrt(2 * (1 - rho))
    diffs = rng.normal(loc=d, scale=sd_diff, size=(b_iter, n))
    se = diffs.std(axis=1, ddof=1) / np.sqrt(n)
    t_stat = diffs.mean(axis=1) / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 1))
    return float(np.mean(p_vals < alpha))


def barrido_potencia_n(rho: float, d: float, alpha: float, rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    for n in N_SWEEP:
        pot = potencia_empirica_vectorizada(n, rho, d, alpha, B_SWEEP, rng)
        filas.append({"N": n, "potencia": pot})
    return pd.DataFrame(filas)


def barrido_sensibilidad(rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    for rho in RHO_SWEEP:
        for d in D_SWEEP:
            pot = potencia_empirica_vectorizada(N_MAIN, rho, d, ALPHA, B_SWEEP, rng)
            filas.append({"rho": rho, "d": d, "potencia": pot})
    return pd.DataFrame(filas)


def efecto_minimo_detectable(n: int, rho: float, alpha: float, power_target: float) -> float:
    """Despeja d tal que la potencia teorica (t pareada, aprox normal) sea
    power_target, dado N y rho. Se usa como verificacion analitica del
    rango 0.22-0.31 reportado en la tesis con N=120 y rho=0.50."""
    sd_diff = np.sqrt(2 * (1 - rho))
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power_target)
    return (z_alpha + z_beta) * sd_diff / np.sqrt(n)


# ===========================================================================
# EJECUCION PRINCIPAL
# ===========================================================================

def main():
    print(f"Semilla fija: {SEED}. N principal = {N_MAIN}, rho = {RHO}, d = {D_EFFECT}.")

    variables = TTCT_DIMS + [DV2]

    # -- 1. Dataset representativo para descriptivos, supuestos, modelo mixto,
    #       forest plot y graficos de distribucion / violin -----------------
    df_rep = simular_dataset(N_MAIN, RHO, D_BY_DIM, RNG)
    df_rep.to_csv(RES_DIR / "dataset_simulado_representativo.csv", index=False)

    desc = descriptivos(df_rep, variables)
    desc.to_csv(RES_DIR / "01_descriptivos.csv", index=False, float_format="%.3f")

    supuestos = pruebas_supuestos(df_rep, variables)
    supuestos.to_csv(RES_DIR / "02_pruebas_supuestos.csv", index=False, float_format="%.4f")

    # -- 2. Contraste de hipotesis (procedimiento de puerta cerrada) --------
    resultados_h = contraste_hipotesis(df_rep)

    filas_efecto = []
    for dim in TTCT_DIMS:
        wide = a_ancho(df_rep, dim)
        diff = (wide[f"{dim}_A"] - wide[f"{dim}_B"]).values
        dz, lo, hi = cohen_dz_ic(diff)
        sub = resultados_h["HE1_subhipotesis"].get(dim, {})
        filas_efecto.append(
            {
                "variable": dim,
                "dz": dz,
                "ic95_lo": lo,
                "ic95_hi": hi,
                "p": sub.get("p", np.nan),
                "significativo_bonferroni_.01": sub.get("significativo_bonferroni", np.nan),
            }
        )
    wide_pssm = a_ancho(df_rep, DV2)
    diff_pssm = (wide_pssm[f"{DV2}_A"] - wide_pssm[f"{DV2}_B"]).values
    dz, lo, hi = cohen_dz_ic(diff_pssm)
    filas_efecto.append(
        {
            "variable": DV2,
            "dz": dz,
            "ic95_lo": lo,
            "ic95_hi": hi,
            "p": resultados_h["HE2"]["p"],
            "significativo_bonferroni_.01": resultados_h["HE2"]["p"] < ALPHA,
        }
    )
    tabla_efectos = pd.DataFrame(filas_efecto)
    tabla_efectos.to_csv(RES_DIR / "03_tamanos_efecto.csv", index=False, float_format="%.3f")

    with open(RES_DIR / "04_prueba_omnibus_HE1.txt", "w", encoding="utf-8") as f:
        o = resultados_h["HE1_omnibus"]
        f.write(
            f"HE1 (omnibus, Hotelling T2 pareada, 5 dimensiones TTCT):\n"
            f"F({o['gl1']}, {o['gl2']}) = {o['F']:.3f}, p = {o['p']:.4f}\n"
        )
        f.write(
            f"\nHE2 (PSSM, t pareada): t = {resultados_h['HE2']['t']:.3f}, "
            f"p = {resultados_h['HE2']['p']:.4f}, "
            f"dz = {resultados_h['HE2']['dz']:.3f} "
            f"[{resultados_h['HE2']['ic95_lo']:.3f}, {resultados_h['HE2']['ic95_hi']:.3f}]\n"
        )

    # -- 3. Modelo mixto de medidas repetidas (demostrativo, una dimension +
    #       PSSM) ------------------------------------------------------------
    resumen_mixto = []
    for var in ["fluidez", DV2]:
        try:
            ajuste = ajustar_modelo_mixto(df_rep, var)
            resumen_mixto.append((var, ajuste.summary().as_text()))
        except Exception as exc:  # modelos mixtos pueden no converger en reps chicas
            resumen_mixto.append((var, f"No convergio: {exc}"))
    with open(RES_DIR / "05_modelo_mixto.txt", "w", encoding="utf-8") as f:
        for var, texto in resumen_mixto:
            f.write(f"===== Modelo mixto: {var} =====\n{texto}\n\n")

    # -- 4. Regresion logistica HE3 (exploratoria) --------------------------
    df_he3 = df_rep.drop_duplicates("subject_id").copy()
    df_he3["pssm_prom"] = df_rep.groupby("subject_id")[DV2].mean().values
    logit_formula = "permanece ~ pssm_prom + rendimiento_previo + C(programa)"
    modelo_logit = smf.logit(logit_formula, data=df_he3).fit(disp=False)
    or_table = pd.DataFrame(
        {
            "coef": modelo_logit.params,
            "OR": np.exp(modelo_logit.params),
            "IC95_lo": np.exp(modelo_logit.conf_int()[0]),
            "IC95_hi": np.exp(modelo_logit.conf_int()[1]),
            "p": modelo_logit.pvalues,
        }
    )
    or_table.to_csv(RES_DIR / "06_HE3_regresion_logistica_OR.csv", float_format="%.4f")

    # -- 5. Potencia empirica: verificacion del efecto minimo detectable ----
    mde_teorico = efecto_minimo_detectable(N_MAIN, RHO, ALPHA, POWER_TARGET)
    potencia_en_d030 = potencia_empirica_vectorizada(N_MAIN, RHO, D_EFFECT, ALPHA, B_ITER, RNG)
    with open(RES_DIR / "07_verificacion_potencia_apriori.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Efecto minimo detectable teorico con N={N_MAIN}, rho={RHO}, "
            f"alpha={ALPHA}, potencia={POWER_TARGET}: d = {mde_teorico:.3f}\n"
            f"(la tesis reporta un rango de 0.22 a 0.31 con G*Power 3.1; "
            f"el valor puntual aqui depende del extremo exacto de la "
            f"aproximacion normal vs. t usada por G*Power).\n\n"
            f"Potencia empirica por simulacion Monte Carlo (B={B_ITER}) "
            f"para d={D_EFFECT}, N={N_MAIN}, rho={RHO}: {potencia_en_d030:.4f}\n"
        )

    # -- 6. Barridos de sensibilidad -----------------------------------------
    tabla_potencia_n = barrido_potencia_n(RHO, D_EFFECT, ALPHA, RNG)
    tabla_potencia_n.to_csv(RES_DIR / "08_potencia_vs_N.csv", index=False, float_format="%.4f")

    tabla_sensibilidad = barrido_sensibilidad(RNG)
    tabla_sensibilidad.to_csv(RES_DIR / "09_sensibilidad_rho_d.csv", index=False, float_format="%.4f")

    # -- 7. Condiciones de sensibilidad: outliers, heterocedasticidad, attrition
    filas_robustez = []
    escenarios = {
        "base": dict(),
        "outliers_5pct": dict(contaminar_outliers=True),
        "heterocedastico": dict(heterocedastico=True),
        "attrition_historica": dict(attrition_rate=ATTRITION_RATE_HISTORICA),
    }
    for nombre, kwargs in escenarios.items():
        rechazos = 0
        for _ in range(B_SWEEP):
            df_s = simular_dataset(N_MAIN, RHO, {DV2: D_EFFECT}, RNG, **kwargs)
            wide = a_ancho(df_s, DV2)
            diff = (wide[f"{DV2}_A"] - wide[f"{DV2}_B"]).dropna().values
            _, p = stats.ttest_1samp(diff, 0)
            rechazos += p < ALPHA
        filas_robustez.append({"escenario": nombre, "potencia_empirica": rechazos / B_SWEEP})
    tabla_robustez = pd.DataFrame(filas_robustez)
    tabla_robustez.to_csv(RES_DIR / "10_robustez_sesgos.csv", index=False, float_format="%.4f")

    # -----------------------------------------------------------------------
    # FIGURAS (a-d), 300 DPI, paleta viridis
    # -----------------------------------------------------------------------

    df_rep_plot = df_rep.copy()
    df_rep_plot["condicion_lbl"] = df_rep_plot["condicion"].map(
        {"A": "A_MPI-DUAE", "B": "B_Comparacion"}
    )

    # (a) Distribucion / densidad con IC 95% de la media --------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, var, titulo in zip(
        axes,
        ["fluidez", DV2],
        [r"TTCT Figural $-$ Fluidez ($z$)", "PSSM $-$ puntaje global ($z$)"],
    ):
        for cond, color in COND_PALETTE.items():
            datos = df_rep_plot.loc[df_rep_plot["condicion_lbl"] == cond, var]
            sns.kdeplot(datos, ax=ax, color=color, fill=True, alpha=0.25, linewidth=2, label=cond)
            m = datos.mean()
            se = datos.std(ddof=1) / np.sqrt(len(datos))
            ci = stats.norm.ppf(0.975) * se
            ax.axvline(m, color=color, linestyle="--", linewidth=1.2)
            ax.axvspan(m - ci, m + ci, color=color, alpha=0.12)
        ax.set_title(titulo)
        ax.set_xlabel(r"Puntaje estandarizado ($z$)")
        ax.set_ylabel("Densidad")
    axes[0].legend(title="Condicion", loc="upper left", fontsize=8)
    fig.suptitle(
        "Distribucion simulada por condicion, con media e IC 95% (N = 120, B = 1 replica ilustrativa)",
        fontsize=11,
    )
    fig.savefig(FIG_DIR / "a_distribucion_densidad_IC95.png", bbox_inches="tight")
    plt.close(fig)

    # (b) Curva de potencia empirica vs. N -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(
        tabla_potencia_n["N"], tabla_potencia_n["potencia"], marker="o", color=PALETTE[2], linewidth=2
    )
    ax.axhline(POWER_TARGET, color="gray", linestyle=":", linewidth=1.2, label=r"Potencia objetivo $1-\beta=0.80$")
    ax.axvline(N_MAIN, color=PALETTE[6], linestyle="--", linewidth=1.2, label=f"N = {N_MAIN} (tesis)")
    ax.axvspan(
        *sorted([N_MAIN - 5, N_MAIN + 5]), color=PALETTE[6], alpha=0.08
    )
    ax.set_xlabel(r"Tamano de muestra ($N$ casos completos)")
    ax.set_ylabel(r"Potencia empirica ($\hat{1-\beta}$)")
    ax.set_title(
        rf"Curva de potencia empirica vs. $N$ ($d={D_EFFECT}$, $\rho={RHO}$, $\alpha={ALPHA}$)"
    )
    ax.legend(fontsize=9)
    fig.savefig(FIG_DIR / "b_curva_potencia_vs_N.png", bbox_inches="tight")
    plt.close(fig)

    # (c) Forest plot de tamanos de efecto + regresion pre-post -------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    ax = axes[0]
    orden = tabla_efectos["variable"].tolist()
    y_pos = np.arange(len(orden))
    ax.errorbar(
        tabla_efectos["dz"],
        y_pos,
        xerr=[
            tabla_efectos["dz"] - tabla_efectos["ic95_lo"],
            tabla_efectos["ic95_hi"] - tabla_efectos["dz"],
        ],
        fmt="o",
        color=PALETTE[3],
        ecolor=PALETTE[3],
        capsize=3,
        markersize=6,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvspan(*MDE_BAND, color=PALETTE[5], alpha=0.10, label="Banda EMD (0.22-0.31)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(orden)
    ax.invert_yaxis()
    ax.set_xlabel(r"$d_z$ de Cohen (IC 95%)")
    ax.set_title("Tamanos de efecto: 5 dimensiones TTCT + PSSM")
    ax.legend(fontsize=8, loc="lower right")

    ax2 = axes[1]
    wide_f = a_ancho(df_rep, "fluidez")
    base_mean = df_rep.groupby("subject_id")["fluidez_baseline"].first()
    post_a = wide_f["fluidez_A"]
    sns.regplot(
        x=base_mean.values,
        y=post_a.values,
        ax=ax2,
        color=COND_PALETTE["A_MPI-DUAE"],
        scatter_kws={"alpha": 0.5, "s": 22},
        line_kws={"linewidth": 2},
        label="A_MPI-DUAE",
        ci=95,
    )
    post_b = wide_f["fluidez_B"]
    sns.regplot(
        x=base_mean.values,
        y=post_b.values,
        ax=ax2,
        color=COND_PALETTE["B_Comparacion"],
        scatter_kws={"alpha": 0.5, "s": 22},
        line_kws={"linewidth": 2},
        label="B_Comparacion",
        ci=95,
    )
    ax2.set_xlabel(r"Fluidez, linea de base ($z$)")
    ax2.set_ylabel(r"Fluidez, puntaje del periodo ($z$)")
    ax2.set_title("Regresion pre-post con banda de confianza 95%")
    ax2.legend(fontsize=8)

    fig.savefig(FIG_DIR / "c_forest_y_regresion_prepost.png", bbox_inches="tight")
    plt.close(fig)

    # (d) Violin + boxplot + puntos individuales -----------------------------
    df_long_plot = df_rep_plot.melt(
        id_vars=["subject_id", "condicion_lbl"],
        value_vars=variables,
        var_name="variable",
        value_name="puntaje",
    )
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    sns.violinplot(
        data=df_long_plot,
        x="variable",
        y="puntaje",
        hue="condicion_lbl",
        split=True,
        inner=None,
        palette=COND_PALETTE,
        linewidth=1,
        ax=ax,
        cut=0,
    )
    sns.boxplot(
        data=df_long_plot,
        x="variable",
        y="puntaje",
        hue="condicion_lbl",
        width=0.15,
        showcaps=True,
        boxprops={"zorder": 3, "facecolor": "white"},
        showfliers=False,
        whiskerprops={"zorder": 3},
        ax=ax,
        dodge=True,
        legend=False,
    )
    sns.stripplot(
        data=df_long_plot,
        x="variable",
        y="puntaje",
        hue="condicion_lbl",
        dodge=True,
        alpha=0.25,
        size=2.5,
        palette=COND_PALETTE,
        ax=ax,
        legend=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Condicion", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(r"Puntaje estandarizado ($z$)")
    ax.set_title("Comparacion por condicion: TTCT Figural (5 dimensiones) y PSSM")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIG_DIR / "d_violin_boxplot_puntos.png", bbox_inches="tight")
    plt.close(fig)

    # -- Figura suplementaria: mapa de calor de sensibilidad rho x d --------
    pivot = tabla_sensibilidad.pivot(index="d", columns="rho", values="potencia")
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    sns.heatmap(
        pivot.sort_index(ascending=False),
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Potencia empirica"},
        ax=ax,
    )
    ax.set_xlabel(r"$\rho$ (correlacion entre medidas repetidas)")
    ax.set_ylabel(r"$d$ de Cohen esperado")
    ax.set_title(f"Sensibilidad de la potencia (N = {N_MAIN}, alpha = {ALPHA})")
    fig.savefig(FIG_DIR / "e_sensibilidad_rho_d_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Resumen ejecutivo en Markdown
    # -----------------------------------------------------------------------
    with open(RES_DIR / "00_RESUMEN.md", "w", encoding="utf-8") as f:
        f.write(
            f"""# Resumen de la simulacion Monte Carlo -- MPI-DUAE

**Semilla:** {SEED} | **N principal:** {N_MAIN} | **rho:** {RHO} | **d esperado:** {D_EFFECT} |
**alpha:** {ALPHA} (Bonferroni HE1a-e: {ALPHA_BONF}) | **potencia objetivo:** {POWER_TARGET} |
**iteraciones Monte Carlo (contraste principal):** {B_ITER}

## 1. Verificacion del analisis de potencia a priori (G*Power 3.1)

- Efecto minimo detectable teorico (formula cerrada, N={N_MAIN}, rho={RHO}): **{mde_teorico:.3f}**
  (rango reportado en la tesis: 0.22-0.31).
- Potencia empirica por Monte Carlo para d={D_EFFECT}: **{potencia_en_d030:.3f}**.

## 2. Resultado del contraste principal (una replica representativa)

- **HE1 (omnibus, Hotelling T2, 5 dimensiones TTCT):**
  F({resultados_h['HE1_omnibus']['gl1']}, {resultados_h['HE1_omnibus']['gl2']}) =
  {resultados_h['HE1_omnibus']['F']:.3f}, p = {resultados_h['HE1_omnibus']['p']:.4f}
- **HE2 (PSSM):** t = {resultados_h['HE2']['t']:.3f}, p = {resultados_h['HE2']['p']:.4f},
  dz = {resultados_h['HE2']['dz']:.3f} [{resultados_h['HE2']['ic95_lo']:.3f}, {resultados_h['HE2']['ic95_hi']:.3f}]

Ver `03_tamanos_efecto.csv` para las subhipotesis HE1a-HE1e (Bonferroni alpha/5={ALPHA_BONF}).

## 3. Robustez (Monte Carlo, B={B_SWEEP} por escenario)

{tabla_robustez.to_markdown(index=False, floatfmt='.3f')}

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
  mixto en cada una de las {B_ITER} iteraciones, por eficiencia
  computacional; el modelo mixto completo se ajusta una vez sobre un
  dataset representativo (`05_modelo_mixto.txt`) para verificar
  consistencia.
"""
        )

    print("Simulacion completa. Resultados en:", RES_DIR)
    print("Figuras en:", FIG_DIR)


if __name__ == "__main__":
    main()
