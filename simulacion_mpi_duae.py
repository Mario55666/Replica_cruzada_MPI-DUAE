"""
Simulacion Monte Carlo y analisis de potencia para la tesis doctoral:
"Modelo Pedagogico Integrado DUA-Experiencial (MPI-DUAE) frente al enfoque
por competencias, en el pensamiento creativo divergente y el sentido de
pertenencia inclusiva de ingresantes a programas de Diseno y Comunicacion
en un instituto publico de educacion superior no universitaria de Lima
Metropolitana (IESP DYC), 2026-2027."

Version 2 (2026-10-08): alineada al Capitulo III (Metodologia).
La version 1 (2026-09-22, basada en los Capitulos I y II) se conserva en
_version_anterior_2026-09-22/. Cambios principales:

- Poblacion y muestra (3.3): estudio censal del ingreso 2027-I; capacidad
  de 145 vacantes en seis secciones (25/25/25/25/25/20, Tabla 11); ingreso
  estimado de 110; meta de 95 casos con datos completos (86,4 %). La v1
  usaba N = 120.
- Potencia y sensibilidad (3.3.3, Tabla 12): prueba t para medias pareadas,
  bilateral, alfa = .05, potencia .80, rho = .50 entre las dos mediciones
  posteriores. Con n = 95 la diferencia minima detectable es d = 0.29
  (HE1, HE2) y d = 0.36 con alfa = .01 (HE1a-e). Son UMBRALES de deteccion,
  no efectos esperados: el escenario base simula un efecto verdadero igual
  al umbral (d = 0.29) en las seis variables.
- Diseno (3.2.1, Tabla 9): O1 (semana 1, linea base), periodo 1 (semanas
  2-8), O2 (semana 9), periodo 2 (semanas 10-16), O3 (semana 16), registro
  de permanencia R (semana 18). La linea base es covariable, no parte del
  contraste.
- Formas del TTCT (3.6.1.1, Tabla 15): asignacion por estudiante; dentro de
  cada seccion, la mitad elegida al azar sigue A-B-A y la otra B-A-B. La
  forma entra como covariable (la v1 asociaba forma = periodo).
- Modelo de analisis (3.8.3):
      Y_ij = b0 + tau*Cond + pi*Per + lambda*Sec + gamma*Base + delta*Forma
             + phi*Prog + beta*Rend + u_i + v_s + e_ij
  Para HE1 se apilan las cinco dimensiones (estandarizadas con la DE de la
  linea base) y se aplica la prueba de Wald conjunta de los cinco
  coeficientes de condicion (5 gl); como robustez, la T2 de Hotelling sobre
  las diferencias entre periodos comparadas entre AB y BA (Hills y
  Armitage, 1979). HE1a-e solo si HE1 es significativa, alfa = .01 e IC 99 %.
  El modelo apilado incluye estudiante x medicion como efecto aleatorio:
  sin el, la Wald conjunta tiene error tipo I inflado (archivo 20).
- Tamanos del efecto (3.8.4): dz, dav y diferencia ajustada estandarizada
  con la DE de la linea base.
- HE3 (3.8.6): regresion logistica penalizada de Firth (1993) con el PSSM
  de linea base (estandarizado) y el rendimiento previo, sin programa ni
  condicion; todos los estudiantes con linea base; IC 95 % por verosimilitud
  perfilada; con menos de 10 retiros solo se informa de forma descriptiva.
- Reglas psicometricas (3.6.1.3, 3.6.1.5), RFI-MPI-DUAE (3.6.3), datos
  faltantes (3.8.8) y analisis de sensibilidad (Tabla 20).

Enfoque de la simulacion
------------------------
Todas las variables se generan en unidades de DE (SD poblacional = 1), de
modo que d se lee como diferencia de medias en desviaciones estandar
(Cohen, 1988). La varianza de cada resultado se descompone en un componente
estable entre sujetos (varianza rho; incluye programa, seccion y
rendimiento previo) y un componente residual por ocasion (varianza 1 - rho):

    Var(diferencia intrasujeto) = 2 * (1 - rho)

En un crossover de dos periodos, el coeficiente de condicion del modelo
mixto se estima solo con informacion intrasujeto: con datos completos, los
terminos que no varian dentro del estudiante (linea base, programa,
rendimiento, u_i, v_s) se cancelan y tau coincide exactamente con la
regresion de las diferencias entre periodos (P2 - P1) sobre la secuencia y
la forma. Ese estimador rapido se usa en los barridos Monte Carlo; su
equivalencia con el modelo mixto completo se verifica en B_MIXTO replicas
(archivo 19).

Software de la tesis: R (lme4, lmerTest con Kenward-Roger, logistf, mice).
Este script reproduce el plan en Python como verificacion de diseno; los
grados de libertad del modelo mixto se aproximan con los de la regresion
intrasujeto (n - 3) en lugar de Kenward-Roger.

Autor: generado para apoyar el diseno metodologico de M. Quiroz
        (Programa de Doctorado en Educacion).
Semilla fija: 2026 (reproducibilidad exacta).
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import optimize, stats
from scipy.special import expit

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
# 1. Parametros del diseno (Capitulo III) y supuestos de la simulacion
# ---------------------------------------------------------------------------

N_CAPACIDAD = 145            # vacantes aprobadas (3.3.1, Tabla 11)
N_INGRESANTES = 110          # ingreso estimado (3.3.1; IESP DYC, 2025)
N_META = 95                  # casos con datos completos (3.3.3)
PERDIDA_ADMITIDA = 1 - N_META / N_INGRESANTES   # 13,6 %

RHO = 0.50                   # correlacion entre las dos mediciones posteriores
RHO_ALT = (0.30, 0.70)       # escenarios de la Tabla 12
ALPHA = 0.05
ALPHA_SUB = ALPHA / 5        # HE1a-e (Bonferroni, 3.5)
POWER_TARGET = 0.80

B_ITER = 10_000              # replicas del escenario principal
B_SWEEP = 3_000              # replicas por punto en barridos y escenarios
B_HE3 = 2_000                # replicas para HE3 (Firth)
B_MIXTO = 200                # replicas para verificar modelo mixto vs. estimador rapido
B_WALD = 400                 # replicas bajo H0 para el error tipo I de la Wald apilada
B_BOOT = 1_000               # bootstrap del CCI

TTCT_DIMS = [
    "fluidez",
    "originalidad",
    "elaboracion",
    "abstraccion_titulos",
    "resistencia_cierre",
]
DV2 = "pssm_global"
VARIABLES = TTCT_DIMS + [DV2]
N_VARS = len(VARIABLES)

PROGRAMAS = [
    "Diseno Publicitario",
    "Diseno de Interiores",
    "Comunicacion Audiovisual",
    "Diseno de Modas",
]
# Tabla 11: seccion, programa, turno, secuencia y vacantes
SECCIONES = pd.DataFrame(
    {
        "seccion": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "programa": [
            "Diseno Publicitario",
            "Diseno Publicitario",
            "Diseno de Interiores",
            "Diseno de Interiores",
            "Comunicacion Audiovisual",
            "Diseno de Modas",
        ],
        "turno": ["Diurno", "Nocturno", "Diurno", "Nocturno", "Nocturno", "Diurno"],
        "secuencia": ["AB", "BA", "BA", "AB", "AB", "BA"],
        "vacantes": [25, 25, 25, 25, 25, 20],
    }
)
SEC_PROG = SECCIONES["programa"].map(PROGRAMAS.index).to_numpy()
SEC_NOCT = (SECCIONES["turno"] == "Nocturno").to_numpy()
SEC_BA = (SECCIONES["secuencia"] == "BA").to_numpy()

# Tabla 12 del Capitulo III, tal como esta impresa (para contrastarla)
TABLA12_N = [110, 95, 85, 75, 65, 55]
TABLA12_CAP3 = {
    "rho50_a05": [0.27, 0.29, 0.31, 0.33, 0.35, 0.39],
    "rho50_a01": [0.33, 0.36, 0.38, 0.40, 0.44, 0.48],
    "rho30_a05": [0.32, 0.34, 0.36, 0.39, 0.42, 0.46],
    "rho70_a05": [0.21, 0.23, 0.24, 0.25, 0.27, 0.30],
}
TABLA12_ESCENARIOS = {
    "rho50_a05": (0.50, 0.05),
    "rho50_a01": (0.50, 0.01),
    "rho30_a05": (0.30, 0.05),
    "rho70_a05": (0.70, 0.05),
}

# Supuestos de generacion (no provienen de la tesis; se declaran)
PERIOD_EFFECT = 0.10          # avance del semestre (maduracion), periodo 2
FORM_EFFECT = 0.10            # forma A del TTCT algo mas facil que la B (0 en PSSM)
BETA_REND = 0.15              # peso del rendimiento previo en el componente estable
SD_PROGRAMA = 0.08            # bloque de programa
SD_SECCION = 0.10             # efecto aleatorio de seccion (v_s)
FORM_VEC = np.array([FORM_EFFECT] * 5 + [0.0])

# Correlaciones entre variables: componente estable (rasgo) y residual por ocasion.
# Fluidez-originalidad alta por el sesgo de fluidez (Acar, 2022); PSSM-TTCT baja.
R_TRAIT = np.array(
    [
        [1.00, 0.75, 0.45, 0.40, 0.50, 0.15],
        [0.75, 1.00, 0.45, 0.45, 0.55, 0.15],
        [0.45, 0.45, 1.00, 0.40, 0.50, 0.15],
        [0.40, 0.45, 0.40, 1.00, 0.40, 0.15],
        [0.50, 0.55, 0.50, 0.40, 1.00, 0.15],
        [0.15, 0.15, 0.15, 0.15, 0.15, 1.00],
    ]
)
R_RESID = np.full((6, 6), 0.35)
R_RESID[:5, 5] = R_RESID[5, :5] = 0.05
R_RESID[0, 1] = R_RESID[1, 0] = 0.50
np.fill_diagonal(R_RESID, 1.0)
assert np.all(np.linalg.eigvalsh(R_TRAIT) > 0) and np.all(np.linalg.eigvalsh(R_RESID) > 0)

# Permanencia (HE3): tasa reciente de retiro 19/178 = 10,7 % (IESP DYC, 2025)
TASA_RETIRO = 19 / 178
OR_PSSM_PERMANENCIA = 2.0     # supuesto: OR por DE del PSSM de linea base
OR_REND_PERMANENCIA = 1.3     # supuesto: OR por DE del rendimiento previo
PROP_RETIRO_ANTES_O2 = 0.5    # mitad de los retiros ocurre antes de la semana 9
# Inasistencia a cada medicion posterior calibrada para que, en promedio,
# 95 de 110 ingresantes tengan datos completos (perdida del 13,6 %).
P_INASISTENCIA = 1 - np.sqrt(1 - (PERDIDA_ADMITIDA - TASA_RETIRO) / (1 - TASA_RETIRO))

# Confiabilidad entre calificadores del TTCT (3.6.1.3): DE del error de cada
# calificador alrededor del puntaje promedio; produce CCI(2,1) de ~.80 a ~.96.
SD_CALIFICADOR = np.array([0.15, 0.28, 0.33, 0.30, 0.25])
SESGO_CALIFICADOR = 0.05

# Escala bruta ilustrativa para el dataset representativo (media, DE).
ESCALA_BRUTA = {
    "fluidez": (17.0, 6.0),
    "originalidad": (12.0, 5.0),
    "elaboracion": (9.0, 3.5),
    "abstraccion_titulos": (6.0, 3.0),
    "resistencia_cierre": (10.0, 3.5),
    "pssm_global": (3.70, 0.55),
}

# Escenarios de robustez: amenazas de la Tabla 10 y analisis de la Tabla 20
ESCENARIOS = {
    "base": dict(),
    "heterogeneidad_seccion": dict(het_seccion_sd=0.15),
    "contaminacion_B": dict(contaminacion=0.30),
    "arrastre": dict(arrastre=0.30),
    "dosis_incompleta": dict(asistencia_p=0.80, efecto_por_dosis=True),
    "outliers_5pct": dict(outliers=True),
    "heterocedasticidad_nocturno": dict(hetero_nocturno=1.5),
    "ingreso_85": dict(n_ingresantes=85),
}

# ---------------------------------------------------------------------------
# 2. Potencia analitica (t no central, equivalente a G*Power 3.1)
# ---------------------------------------------------------------------------


def potencia_t_pareada(dz: float, n: int, alpha: float) -> float:
    """Potencia exacta de la prueba t bilateral para medias pareadas."""
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    nc = dz * np.sqrt(n)
    return float(stats.nct.sf(tc, df, nc) + np.nan_to_num(stats.nct.cdf(-tc, df, nc)))


def dz_minimo(n: int, alpha: float, power: float = POWER_TARGET) -> float:
    return optimize.brentq(lambda dz: potencia_t_pareada(dz, n, alpha) - power, 0.01, 1.5)


def d_minimo(n: int, rho: float, alpha: float, power: float = POWER_TARGET) -> float:
    """Diferencia minima detectable en unidades d: d = dz * sqrt(2(1 - rho))."""
    return dz_minimo(n, alpha, power) * np.sqrt(2 * (1 - rho))


MDE_95 = d_minimo(N_META, RHO, ALPHA)          # 0.29
MDE_95_SUB = d_minimo(N_META, RHO, ALPHA_SUB)  # 0.36
D_UMBRAL = round(MDE_95, 2)                     # efecto verdadero del escenario base

# ---------------------------------------------------------------------------
# 3. Generador de cohortes (version vectorizada con numpy)
# ---------------------------------------------------------------------------


def tamanos_seccion(n_ingresantes: int) -> np.ndarray:
    """Reparte el ingreso entre las seis secciones en proporcion a sus vacantes."""
    cuota = n_ingresantes * SECCIONES["vacantes"].to_numpy() / N_CAPACIDAD
    tam = np.floor(cuota).astype(int)
    resto = n_ingresantes - tam.sum()
    tam[np.argsort(-(cuota - tam))[:resto]] += 1
    return tam


L_TRAIT = np.linalg.cholesky(R_TRAIT)
L_RESID = np.linalg.cholesky(R_RESID)


def _calibrar_intercepto_permanencia() -> float:
    """Intercepto del modelo logistico de permanencia que reproduce, en
    promedio, la tasa de retiro historica (10,7 %)."""
    rng = np.random.default_rng(1)
    rend = rng.normal(size=400_000)
    pssm0 = BETA_REND * rend + rng.normal(0, np.sqrt(1 - BETA_REND**2), size=rend.size)
    bp, br = np.log(OR_PSSM_PERMANENCIA), np.log(OR_REND_PERMANENCIA)
    return optimize.brentq(
        lambda b0: expit(b0 + bp * pssm0 + br * rend).mean() - (1 - TASA_RETIRO), -5, 10
    )


B0_PERMANENCIA = _calibrar_intercepto_permanencia()


def simular_cohorte(
    rng: np.random.Generator,
    n_ingresantes: int = N_INGRESANTES,
    d: float = D_UMBRAL,
    rho: float = RHO,
    het_seccion_sd: float = 0.0,
    contaminacion: float = 0.0,
    arrastre: float = 0.0,
    asistencia_p: float = 0.90,
    efecto_por_dosis: bool = False,
    outliers: bool = False,
    hetero_nocturno: float = 1.0,
    con_faltantes: bool = True,
) -> dict:
    """Simula una cohorte completa del diseno de replica cruzada (Tabla 9).

    Devuelve arreglos por estudiante: seccion, secuencia, orden de formas,
    rendimiento previo, puntajes O1 (Y0), O2 (Y1) y O3 (Y2) en unidades de DE
    (n x 6), asistencia por condicion, permanencia y disponibilidad de O2/O3.

    Parametros de escenario:
      het_seccion_sd  efecto de la condicion distinto por seccion (par docente)
      contaminacion   fraccion del efecto que aparece en la comparacion (B)
      arrastre        fraccion del efecto de A que persiste en el periodo 2 de AB
      efecto_por_dosis el efecto de A es proporcional a las sesiones asistidas
      outliers        5 % de observaciones con ruido de cola pesada (t, gl = 2)
      hetero_nocturno multiplica la DE residual en el turno nocturno
    """
    tam = tamanos_seccion(n_ingresantes)
    sec = np.repeat(np.arange(6), tam)
    n = sec.size
    seq_ba = SEC_BA[sec]
    nocturno = SEC_NOCT[sec]

    # Orden de formas por estudiante (Tabla 15): mitad aleatoria de cada seccion A-B-A
    orden1 = np.zeros(n, dtype=bool)
    inicio = 0
    for t in tam:
        k = t // 2 + (rng.random() < 0.5 if t % 2 else 0)
        orden1[inicio + rng.choice(t, size=k, replace=False)] = True
        inicio += t
    forma_a = np.column_stack([orden1, ~orden1, orden1]).astype(float)  # O1, O2, O3

    rend = rng.normal(size=n)
    var_b = rho - BETA_REND**2 - SD_PROGRAMA**2 - SD_SECCION**2
    b = (rng.standard_normal((n, N_VARS)) @ L_TRAIT.T) * np.sqrt(var_b)
    prog_fx = rng.normal(0, SD_PROGRAMA, size=(4, N_VARS))[SEC_PROG[sec]]
    sec_fx = rng.normal(0, SD_SECCION, size=(6, N_VARS))[sec]
    estable = b + prog_fx + sec_fx + BETA_REND * rend[:, None]

    escala_resid = np.sqrt(1 - rho) * np.where(nocturno, hetero_nocturno, 1.0)[:, None]

    def residual():
        e = (rng.standard_normal((n, N_VARS)) @ L_RESID.T) * escala_resid
        if outliers:
            mask = rng.random((n, N_VARS)) < 0.05
            e[mask] += rng.standard_t(2, size=mask.sum())
        return e

    # Asistencia (0-7 sesiones por condicion) y efecto individual de la condicion
    ses_a = rng.binomial(7, asistencia_p, size=n)
    ses_b = rng.binomial(7, asistencia_p, size=n)
    tau_sec = d + (rng.normal(0, het_seccion_sd, size=6) if het_seccion_sd > 0 else 0.0)
    tau_sec = np.broadcast_to(tau_sec, (6,))[sec]
    tau_i = tau_sec * (ses_a / 7 if efecto_por_dosis else 1.0)
    efecto_b = contaminacion * tau_sec

    cond_a_p1 = ~seq_ba
    cond_a_p2 = seq_ba
    ef_p1 = np.where(cond_a_p1, tau_i, efecto_b)
    ef_p2 = np.where(cond_a_p2, tau_i, efecto_b + arrastre * tau_i)

    y0 = estable + FORM_VEC * forma_a[:, [0]] + residual()
    y1 = estable + FORM_VEC * forma_a[:, [1]] + ef_p1[:, None] + residual()
    y2 = estable + PERIOD_EFFECT + FORM_VEC * forma_a[:, [2]] + ef_p2[:, None] + residual()

    # Permanencia (HE3) y datos faltantes (3.8.8)
    logit = (
        B0_PERMANENCIA
        + np.log(OR_PSSM_PERMANENCIA) * y0[:, 5]
        + np.log(OR_REND_PERMANENCIA) * rend
    )
    permanece = rng.random(n) < expit(logit)
    retiro_antes_o2 = ~permanece & (rng.random(n) < PROP_RETIRO_ANTES_O2)
    if con_faltantes:
        obs1 = ~retiro_antes_o2 & (rng.random(n) >= P_INASISTENCIA)
        obs2 = permanece & (rng.random(n) >= P_INASISTENCIA)
    else:
        obs1 = np.ones(n, dtype=bool)
        obs2 = np.ones(n, dtype=bool)

    return dict(
        n=n, sec=sec, seq_ba=seq_ba, nocturno=nocturno, orden1=orden1, forma_a=forma_a,
        rend=rend, y0=y0, y1=y1, y2=y2, ses_a=ses_a, ses_b=ses_b, permanece=permanece,
        retiro_antes_o2=retiro_antes_o2, obs1=obs1, obs2=obs2,
    )


# ---------------------------------------------------------------------------
# 4. Estimador rapido intrasujeto (equivalente al modelo mixto con datos completos)
# ---------------------------------------------------------------------------


def analisis_intrasujeto(c: dict, mascara: np.ndarray | None = None) -> dict:
    """Regresion de las diferencias entre periodos (P2 - P1) sobre la secuencia
    (c = +1 si BA, -1 si AB) y la forma (f = +1 si la forma A cae en O3).
    El coeficiente de c es tau; el intercepto es el efecto de periodo.
    HE1 omnibus: prueba multivariante de la fila de tau en las cinco
    dimensiones (F exacta, equivalente a la T2 de Hotelling con covariable)."""
    comp = c["obs1"] & c["obs2"]
    if mascara is not None:
        comp = comp & mascara
    n = int(comp.sum())
    dif = c["y2"][comp] - c["y1"][comp]
    cvec = np.where(c["seq_ba"][comp], 1.0, -1.0)
    fvec = np.where(c["orden1"][comp], 1.0, -1.0)

    def ols(Y, X):
        xtx_inv = np.linalg.inv(X.T @ X)
        coef = xtx_inv @ X.T @ Y
        resid = Y - X @ coef
        gl = n - X.shape[1]
        se = np.sqrt((resid**2).sum(axis=0) / gl * xtx_inv[1, 1])
        return coef, resid, gl, se, xtx_inv[1, 1]

    # La forma solo entra en el TTCT (VD1); el PSSM no tiene formas (3.8.3)
    coef_t, res_t, gl, se_t, c11 = ols(dif[:, :5], np.column_stack([np.ones(n), cvec, fvec]))
    coef_p, _, gl_p, se_p, _ = ols(dif[:, 5:], np.column_stack([np.ones(n), cvec]))
    tau = np.concatenate([coef_t[1], coef_p[1]])
    se_tau = np.concatenate([se_t, se_p])
    gl_vec = np.array([gl] * 5 + [gl_p])
    t_tau = tau / se_tau
    p_tau = 2 * stats.t.sf(np.abs(t_tau), gl_vec)

    p_dim = len(TTCT_DIMS)
    E = res_t.T @ res_t
    t2 = gl * coef_t[1] @ np.linalg.solve(E, coef_t[1]) / c11
    f_omni = (gl - p_dim + 1) / (p_dim * gl) * t2
    gl2 = gl - p_dim + 1
    p_omni = stats.f.sf(f_omni, p_dim, gl2)
    return dict(
        n=n, gl=gl, gl_vec=gl_vec, tau=tau, periodo=np.concatenate([coef_t[0], coef_p[0]]),
        forma=coef_t[2], se=se_tau, t=t_tau, p=p_tau, F_omni=f_omni, gl1_omni=p_dim,
        gl2_omni=gl2, p_omni=p_omni,
    )


def decisiones_puerta_cerrada(r: dict) -> dict:
    omni = r["p_omni"] < ALPHA
    return dict(
        omni=omni,
        sub=omni & (r["p"][:5] < ALPHA_SUB),
        he2=r["p"][5] < ALPHA,
        individual_05=r["p"] < ALPHA,
    )


def potencia_vectorizada(n: int, rho: float, d: float, alpha: float, b: int,
                         rng: np.random.Generator) -> float:
    """Potencia del analisis intrasujeto planificado (periodo + forma, gl = n - 3)
    simulando directamente las diferencias entre periodos:
    P2 - P1 = pi + tau*c + delta*f + e, Var(e) = 2(1 - rho)."""
    cvec = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    fvec = np.where((np.arange(n) // 2) % 2 == 0, 1.0, -1.0)
    X = np.column_stack([np.ones(n), cvec, fvec])
    xtx_inv = np.linalg.inv(X.T @ X)
    proy = xtx_inv @ X.T
    media = PERIOD_EFFECT + d * cvec + FORM_EFFECT * fvec
    dif = media + rng.normal(0, np.sqrt(2 * (1 - rho)), size=(b, n))
    coef = dif @ proy.T
    resid = dif - coef @ X.T
    s2 = (resid**2).sum(axis=1) / (n - 3)
    t_tau = coef[:, 1] / np.sqrt(s2 * xtx_inv[1, 1])
    return float(np.mean(2 * stats.t.sf(np.abs(t_tau), n - 3) < alpha))


def potencia_t_pareada_mc(n: int, dz: float, alpha: float, b: int,
                          rng: np.random.Generator) -> float:
    dif = rng.normal(dz, 1.0, size=(b, n))
    t_stat = dif.mean(axis=1) / (dif.std(axis=1, ddof=1) / np.sqrt(n))
    return float(np.mean(2 * stats.t.sf(np.abs(t_stat), n - 1) < alpha))


# ---------------------------------------------------------------------------
# 5. Regresion logistica de Firth con IC por verosimilitud perfilada (HE3)
# ---------------------------------------------------------------------------


def _firth_pll(beta, X, y):
    eta = X @ beta
    p = expit(eta)
    info = X.T @ ((p * (1 - p))[:, None] * X)
    sign, logdet = np.linalg.slogdet(info)
    ll = np.sum(y * eta - np.logaddexp(0, eta))
    return ll + 0.5 * logdet if sign > 0 else -np.inf


def _firth_score(beta, X, y):
    p = expit(X @ beta)
    w = p * (1 - p)
    info = X.T @ (w[:, None] * X)
    h = w * np.einsum("ij,ij->i", X @ np.linalg.inv(info), X)
    return X.T @ (y - p + h * (0.5 - p))


def firth_ajuste(X, y, fijo: int | None = None, valor: float = 0.0, x0=None):
    """Maximiza la log-verosimilitud penalizada de Firth (1993). Si `fijo` se
    indica, el coeficiente correspondiente se fija en `valor` (perfil)."""
    k = X.shape[1]
    libres = [j for j in range(k) if j != fijo]
    inicio = np.zeros(k) if x0 is None else np.array(x0, dtype=float)

    def completo(bl):
        beta = inicio.copy()
        beta[libres] = bl
        if fijo is not None:
            beta[fijo] = valor
        return beta

    res = optimize.minimize(
        lambda bl: -_firth_pll(completo(bl), X, y),
        inicio[libres],
        jac=lambda bl: -_firth_score(completo(bl), X, y)[libres],
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 500},
    )
    beta = completo(res.x)
    return beta, _firth_pll(beta, X, y)


def firth_inferencia(X, y, nombres: list) -> pd.DataFrame:
    beta, pll_max = firth_ajuste(X, y)
    p = expit(X @ beta)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ ((p * (1 - p))[:, None] * X))))
    corte = stats.chi2.ppf(0.95, 1)
    filas = []
    for j, nombre in enumerate(nombres):
        def desvio(v, j=j):
            _, pll = firth_ajuste(X, y, fijo=j, valor=v, x0=beta)
            return 2 * (pll_max - pll) - corte

        limites = []
        for signo in (-1, 1):
            paso = 2 * se[j]
            extremo = beta[j] + signo * paso
            while desvio(extremo) < 0 and abs(extremo - beta[j]) < 50 * se[j]:
                extremo += signo * paso
            limites.append(optimize.brentq(desvio, *sorted([beta[j], extremo])))
        _, pll0 = firth_ajuste(X, y, fijo=j, valor=0.0, x0=beta)
        chi = max(2 * (pll_max - pll0), 0.0)
        filas.append(
            {
                "termino": nombre,
                "coef": beta[j],
                "EE": se[j],
                "OR": np.exp(beta[j]),
                "IC95_lo": np.exp(limites[0]),
                "IC95_hi": np.exp(limites[1]),
                "chi2_LR_pen": chi,
                "p": stats.chi2.sf(chi, 1),
            }
        )
    return pd.DataFrame(filas)


def firth_p_lr(X, y, j: int) -> float:
    beta, pll_max = firth_ajuste(X, y)
    _, pll0 = firth_ajuste(X, y, fijo=j, valor=0.0, x0=beta)
    return float(stats.chi2.sf(max(2 * (pll_max - pll0), 0.0), 1))


# ---------------------------------------------------------------------------
# 6. Utilidades de reporte: IC de dz, CCI(2,1), kappa ponderado, Fisher z
# ---------------------------------------------------------------------------


def ic_dz(dz: float, n: int, nivel: float) -> tuple:
    """IC exacto de dz invirtiendo la t no central."""
    t_obs = dz * np.sqrt(n)
    gl = n - 1
    a = (1 - nivel) / 2

    def f(nc, q):
        p = stats.nct.cdf(t_obs, gl, nc)
        if np.isnan(p):
            p = 0.0 if nc > t_obs else 1.0
        return p - q

    lo = optimize.brentq(f, t_obs - 6, t_obs + 6, args=(1 - a,))
    hi = optimize.brentq(f, t_obs - 6, t_obs + 6, args=(a,))
    return lo / np.sqrt(n), hi / np.sqrt(n)


def influyentes_cook(datos: pd.DataFrame, var: str) -> pd.Index:
    """Estudiantes con distancia de Cook > 4/n en la regresion intrasujeto
    (P2 - P1 sobre secuencia y forma), que es la parte del modelo mixto que
    estima tau."""
    ancho = datos.pivot_table(index=["subject_id", "secuencia", "orden_forma"],
                              columns="periodo", values=var).dropna()
    dif = (ancho[2] - ancho[1]).to_numpy()
    idx = ancho.index
    X = np.column_stack([np.ones(len(dif)),
                         np.where(idx.get_level_values("secuencia") == "BA", 1.0, -1.0),
                         np.where(idx.get_level_values("orden_forma") == "A-B-A", 1.0, -1.0)])
    xtx_inv = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    e = dif - X @ (xtx_inv @ X.T @ dif)
    p = X.shape[1]
    s2 = (e**2).sum() / (len(dif) - p)
    cook = e**2 / (p * s2) * h / (1 - h) ** 2
    return idx.get_level_values("subject_id")[cook > 4 / len(dif)]


def cci_21(puntajes: np.ndarray) -> float:
    """CCI(2,1) de acuerdo absoluto (Shrout y Fleiss; McGraw y Wong) para una
    matriz n sujetos x k calificadores."""
    n, k = puntajes.shape
    gm = puntajes.mean()
    msr = k * np.sum((puntajes.mean(axis=1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((puntajes.mean(axis=0) - gm) ** 2) / (k - 1)
    sse = np.sum((puntajes - puntajes.mean(axis=1, keepdims=True)
                  - puntajes.mean(axis=0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def kappa_ponderado(a: np.ndarray, b: np.ndarray, categorias=(0, 1, 2)) -> float:
    k = len(categorias)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[x, y] += 1
    obs /= obs.sum()
    esp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = 1 - np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    return 1 - (1 - (w * obs).sum()) / (1 - (w * esp).sum())


def r_ic(x, y, nivel=0.95):
    r = np.corrcoef(x, y)[0, 1]
    z = np.arctanh(r)
    se = 1 / np.sqrt(len(x) - 3)
    q = stats.norm.ppf(1 - (1 - nivel) / 2)
    return r, np.tanh(z - q * se), np.tanh(z + q * se)


# ---------------------------------------------------------------------------
# 7. Dataset representativo en formato de analisis (pandas)
# ---------------------------------------------------------------------------


def a_tablas(c: dict, rng: np.random.Generator, bruto: bool = True) -> tuple:
    """Convierte una cohorte en (estudiantes, largo): una fila por estudiante
    y una fila por estudiante x periodo observado. Agrega la escala bruta
    ilustrativa (redondeada a puntajes posibles si bruto=True), los dos
    calificadores del TTCT en la linea base y el momento del retiro."""
    n = c["n"]
    est = pd.DataFrame(
        {
            "subject_id": np.arange(1, n + 1),
            "seccion": SECCIONES["seccion"].to_numpy()[c["sec"]],
            "programa": SECCIONES["programa"].to_numpy()[c["sec"]],
            "turno": SECCIONES["turno"].to_numpy()[c["sec"]],
            "secuencia": np.where(c["seq_ba"], "BA", "AB"),
            "orden_forma": np.where(c["orden1"], "A-B-A", "B-A-B"),
            "rendimiento_previo": c["rend"],
            "sesiones_A": c["ses_a"],
            "sesiones_B": c["ses_b"],
            "permanece": c["permanece"].astype(int),
            "retiro": np.where(
                c["permanece"], "no", np.where(c["retiro_antes_o2"], "antes de O2", "entre O2 y O3")
            ),
            "obs_O2": c["obs1"],
            "obs_O3": c["obs2"],
        }
    )
    for j, var in enumerate(VARIABLES):
        media, de = ESCALA_BRUTA[var]
        for m, y in zip(["O1", "O2", "O3"], [c["y0"], c["y1"], c["y2"]]):
            valor = media + de * y[:, j]
            if bruto and var == DV2:
                valor = np.clip(np.round(valor * 13) / 13, 1, 5)
            elif bruto:
                valor = np.clip(np.round(valor), 0, None)
            est[f"{var}_{m}"] = valor
    est.loc[~c["obs1"], [f"{v}_O2" for v in VARIABLES]] = np.nan
    est.loc[~c["obs2"], [f"{v}_O3" for v in VARIABLES]] = np.nan

    # Dos calificadores del TTCT alrededor del puntaje final (promedio) en O1
    for j, var in enumerate(TTCT_DIMS):
        de = ESCALA_BRUTA[var][1]
        err = SESGO_CALIFICADOR + rng.normal(0, SD_CALIFICADOR[j], size=n)
        est[f"{var}_O1_cal1"] = est[f"{var}_O1"] + de * err
        est[f"{var}_O1_cal2"] = est[f"{var}_O1"] - de * err

    # Estandarizacion con la media y DE de la linea base (3.8.3)
    for var in VARIABLES:
        m0, s0 = est[f"{var}_O1"].mean(), est[f"{var}_O1"].std(ddof=1)
        for m in ["O1", "O2", "O3"]:
            est[f"z_{var}_{m}"] = (est[f"{var}_{m}"] - m0) / s0

    filas = []
    for periodo, m, cond_a in [(1, "O2", ~c["seq_ba"]), (2, "O3", c["seq_ba"])]:
        bloque = est[["subject_id", "seccion", "programa", "turno", "secuencia",
                      "orden_forma", "rendimiento_previo", "sesiones_A", "sesiones_B"]].copy()
        bloque["periodo"] = periodo
        bloque["semana"] = 9 if periodo == 1 else 16
        bloque["condicion"] = np.where(cond_a, "A", "B")
        bloque["forma"] = np.where(c["forma_a"][:, periodo] == 1, "A", "B")
        bloque["observado"] = (c["obs1"] if periodo == 1 else c["obs2"])
        for var in VARIABLES:
            bloque[var] = est[f"z_{var}_{m}"]
            bloque[f"{var}_base"] = est[f"z_{var}_O1"]
        filas.append(bloque)
    largo = pd.concat(filas, ignore_index=True)
    largo = largo[largo["observado"]].drop(columns="observado")
    largo["cond_A"] = (largo["condicion"] == "A").astype(int)
    largo["periodo_2"] = (largo["periodo"] == 2).astype(int)
    largo["secuencia_BA"] = (largo["secuencia"] == "BA").astype(int)
    largo["forma_A"] = (largo["forma"] == "A").astype(int)
    largo["nocturno"] = (largo["turno"] == "Nocturno").astype(int)
    return est, largo


def completos(largo: pd.DataFrame) -> pd.DataFrame:
    cuenta = largo.groupby("subject_id")["periodo"].transform("count")
    return largo[cuenta == 2].copy()


def a_ancho(largo: pd.DataFrame, var: str) -> pd.DataFrame:
    w = largo.pivot_table(index="subject_id", columns="condicion", values=var)
    return w.dropna()


# ---------------------------------------------------------------------------
# 8. Modelos del plan de analisis sobre el dataset representativo (3.8.3)
# ---------------------------------------------------------------------------


def formula_modelo(var: str, extra: str = "") -> str:
    forma = " + forma_A" if var in TTCT_DIMS else ""
    return (f"{var} ~ cond_A + periodo_2 + secuencia_BA + {var}_base{forma}"
            f" + C(programa) + rendimiento_previo{extra}")


def _formula_estimable(datos: pd.DataFrame, formula: str) -> tuple:
    """Con subconjuntos de secciones, la secuencia (o el turno) puede quedar
    colineal con el programa (2.4.1.3). Se omite el termino entre sujetos
    redundante; tau, que es intrasujeto, no cambia."""
    omitidos = []
    for termino in ("", " + secuencia_BA", " + nocturno"):
        if termino:
            if termino not in formula:
                continue
            formula = formula.replace(termino, "")
            omitidos.append(termino.strip(" +"))
        X = patsy.dmatrix(formula.split("~", 1)[1], datos, return_type="dataframe")
        if np.linalg.matrix_rank(X.to_numpy()) == X.shape[1]:
            break
    return formula, omitidos


def ajustar_mixto(datos: pd.DataFrame, var: str, extra: str = "",
                  coef_extra: str | None = None) -> dict:
    """Modelo lineal mixto con efecto aleatorio de estudiante y de seccion.
    Si la varianza de seccion es nula o no converge, se retira v_s (3.8.3).
    `coef_extra` devuelve ademas la estimacion de un termino agregado
    (p. ej., la interaccion condicion x turno)."""
    formula, omitidos = _formula_estimable(datos, formula_modelo(var, extra))
    estructura = "estudiante + seccion"
    try:
        modelo = smf.mixedlm(formula, datos, groups="seccion", re_formula="1",
                             vc_formula={"estudiante": "0 + C(subject_id)"})
        ajuste = modelo.fit(reml=True)
        if (not ajuste.converged) or float(ajuste.cov_re.iloc[0, 0]) < 1e-4:
            raise ValueError("varianza de seccion nula")
    except Exception:
        estructura = "estudiante (v_s retirado)"
        ajuste = smf.mixedlm(formula, datos, groups="subject_id").fit(reml=True)
    n_est = datos["subject_id"].nunique()
    # gl de la regresion intrasujeto: periodo, condicion y (en el TTCT) forma
    gl = max(n_est - (3 if var in TTCT_DIMS else 2), 1)
    tau, ee = ajuste.fe_params["cond_A"], ajuste.bse_fe["cond_A"]
    salida = dict(var=var, estructura=estructura, ajuste=ajuste, n=n_est, gl=gl,
                  tau=tau, ee=ee, t=tau / ee, p=2 * stats.t.sf(abs(tau / ee), gl),
                  formula=formula, omitidos=omitidos)
    for nivel in (0.95, 0.99):
        q = stats.t.ppf(1 - (1 - nivel) / 2, gl)
        salida[f"ic{int(nivel * 100)}"] = (tau - q * ee, tau + q * ee)
    for termino in ("periodo_2", "secuencia_BA"):
        if termino not in ajuste.fe_params:
            salida[termino] = (np.nan, np.nan, np.nan)
            continue
        b, s = ajuste.fe_params[termino], ajuste.bse_fe[termino]
        q = stats.t.ppf(0.975, gl)
        salida[termino] = (b, b - q * s, b + q * s)
    if coef_extra is not None:
        b, s = ajuste.fe_params[coef_extra], ajuste.bse_fe[coef_extra]
        q = stats.t.ppf(0.975, gl)
        salida["extra"] = (b, b - q * s, b + q * s, 2 * stats.t.sf(abs(b / s), gl))
    return salida


def wald_apilado(datos: pd.DataFrame, ocasion: bool = True) -> dict:
    """HE1: cinco dimensiones apiladas, interceptos, coeficientes de condicion y
    de linea base propios de cada dimension; prueba de Wald conjunta (5 gl).

    Efectos aleatorios: estudiante y estudiante x dimension; con ocasion=True,
    ademas estudiante x medicion, que recoge la correlacion entre las cinco
    dimensiones medidas el mismo dia. Sin ese termino la Wald conjunta tiene
    error tipo I inflado (verificado en la seccion K de main)."""
    filas = []
    for var in TTCT_DIMS:
        b = datos[["subject_id", "seccion", "programa", "rendimiento_previo", "cond_A",
                   "periodo", "periodo_2", "secuencia_BA", "forma_A"]].copy()
        b["dim"] = var
        b["y"] = datos[var]
        b["base"] = datos[f"{var}_base"]
        filas.append(b)
    ap = pd.concat(filas, ignore_index=True)
    formula = ("y ~ 0 + C(dim) + C(dim):cond_A + C(dim):base + periodo_2 + secuencia_BA"
               " + forma_A + C(programa) + rendimiento_previo")
    vc = {"dim": "0 + C(dim)"}
    if ocasion:
        vc["ocasion"] = "0 + C(periodo)"
    ajuste = smf.mixedlm(formula, ap, groups="subject_id", re_formula="1",
                         vc_formula=vc).fit(reml=True)
    nombres = [f"C(dim)[{v}]:cond_A" for v in TTCT_DIMS]
    b = ajuste.fe_params[nombres].to_numpy()
    V = ajuste.cov_params().loc[nombres, nombres].to_numpy()
    chi2 = float(b @ np.linalg.solve(V, b))
    n_est = ap["subject_id"].nunique()
    gl2 = n_est - 3
    return dict(chi2=chi2, gl=5, p_chi2=stats.chi2.sf(chi2, 5), F=chi2 / 5, gl2=gl2,
                p_F=stats.f.sf(chi2 / 5, 5, gl2), tau=dict(zip(TTCT_DIMS, b)),
                ajuste=ajuste)


def hotelling_dos_muestras(datos: pd.DataFrame) -> dict:
    """Robustez de HE1: T2 de Hotelling sobre las diferencias entre periodos
    (P2 - P1) de cada estudiante, comparadas entre AB y BA."""
    ancho = datos.pivot_table(index=["subject_id", "secuencia"], columns="periodo",
                              values=TTCT_DIMS)
    dif = pd.DataFrame({v: ancho[(v, 2)] - ancho[(v, 1)] for v in TTCT_DIMS}).dropna()
    sec = dif.index.get_level_values("secuencia")
    x1, x2 = dif[sec == "BA"].to_numpy(), dif[sec == "AB"].to_numpy()
    n1, n2, p = len(x1), len(x2), len(TTCT_DIMS)
    dm = x1.mean(axis=0) - x2.mean(axis=0)
    sp = ((n1 - 1) * np.cov(x1, rowvar=False) + (n2 - 1) * np.cov(x2, rowvar=False)) / (n1 + n2 - 2)
    t2 = n1 * n2 / (n1 + n2) * dm @ np.linalg.solve(sp, dm)
    f_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * t2
    gl2 = n1 + n2 - p - 1
    return dict(T2=t2, F=f_stat, gl1=p, gl2=gl2, p=stats.f.sf(f_stat, p, gl2), n_BA=n1, n_AB=n2)


def simular_rfi(rng: np.random.Generator) -> tuple:
    """RFI-MPI-DUAE (3.6.3): 12 items 0-2 en cuatro dimensiones; sesiones 2, 4 y
    6 de cada periodo en las dos unidades de cada seccion (36 sesiones); 9
    sesiones con doble observador. Devuelve indices por sesion y acuerdo."""
    dims = {"adherencia": 3, "mediacion": 3, "participacion": 3, "diferenciacion": 3}
    filas, dobles = [], []
    propension = rng.normal(0, 0.6, size=6)  # fidelidad propia del docente de A
    sesiones = [(s, cond, k) for s in range(6) for cond in ("A", "B") for k in (2, 4, 6)]
    idx_doble = set(rng.choice(len(sesiones), size=9, replace=False).tolist())
    for i, (s, cond, k) in enumerate(sesiones):
        seq = SECCIONES.loc[s, "secuencia"]
        periodo = 1 if (cond == "A") == (seq == "AB") else 2
        puntajes = {}
        for dim, m in dims.items():
            if cond == "A":
                p2 = expit(1.6 + propension[s])
                probs = [0.04, 1 - 0.04 - p2 * 0.96, p2 * 0.96]
            else:
                # contaminacion: mas rasgos del modelo en la comparacion posterior al modelo
                extra = 0.6 if (seq == "AB" and periodo == 2) else 0.0
                p0 = expit(1.2 - extra) if dim in ("adherencia", "diferenciacion") else 0.35
                probs = [p0, (1 - p0) * 0.7, (1 - p0) * 0.3]
            puntajes[dim] = rng.choice(3, size=m, p=probs)
        total = sum(v.sum() for v in puntajes.values())
        contam = (puntajes["adherencia"].sum() + puntajes["diferenciacion"].sum()) / 12 * 100
        filas.append(dict(seccion=f"S{s + 1}", secuencia=seq, condicion=cond, periodo=periodo,
                          sesion=k, indice=total / 24 * 100, contaminacion=contam))
        if i in idx_doble:
            for dim, v in puntajes.items():
                cambio = rng.random(v.size) < 0.12
                obs2 = np.clip(v + np.where(cambio, rng.choice([-1, 1], size=v.size), 0), 0, 2)
                dobles.extend({"dim": dim, "o1": a, "o2": b} for a, b in zip(v, obs2))
    sesiones_df = pd.DataFrame(filas)
    dob = pd.DataFrame(dobles)
    acuerdo = {d: kappa_ponderado(g["o1"].to_numpy(), g["o2"].to_numpy()) for d, g in dob.groupby("dim")}
    return sesiones_df, acuerdo


DASHBOARD_HTML = OUT_DIR / "dashboard_simulacion_mpi_duae.html"


def actualizar_dashboard(datos_json: str) -> bool:
    """Reemplaza en el dashboard el bloque de datos entre /*DATA*/ y /*FIN_DATA*/."""
    if not DASHBOARD_HTML.exists():
        return False
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    ini, fin = "/*DATA*/", "/*FIN_DATA*/"
    a, b = html.find(ini), html.find(fin)
    if a < 0 or b < a:
        return False
    DASHBOARD_HTML.write_text(html[: a + len(ini)] + datos_json + html[b:], encoding="utf-8")
    return True


# ===========================================================================
# EJECUCION PRINCIPAL
# ===========================================================================


def main():
    print(f"Semilla {SEED} | ingreso {N_INGRESANTES} | meta {N_META} casos completos | "
          f"rho {RHO} | d umbral {D_UMBRAL} | inasistencia por medicion {P_INASISTENCIA:.4f}")

    # -----------------------------------------------------------------------
    # A. Tabla 12: replica analitica (t no central) y verificacion Monte Carlo
    # -----------------------------------------------------------------------
    filas_t12 = []
    for clave, (rho, alpha) in TABLA12_ESCENARIOS.items():
        for n, d_cap3 in zip(TABLA12_N, TABLA12_CAP3[clave]):
            d_exacto = d_minimo(n, rho, alpha)
            dz_cap3 = d_cap3 / np.sqrt(2 * (1 - rho))
            filas_t12.append(
                {
                    "n": n,
                    "escenario": clave,
                    "rho": rho,
                    "alpha": alpha,
                    "d_cap3": d_cap3,
                    "d_exacto": d_exacto,
                    "d_exacto_redondeado": round(d_exacto + 1e-12, 2),
                    "coincide": round(d_exacto + 1e-12, 2) == d_cap3,
                    "potencia_analitica_en_d_cap3": potencia_t_pareada(dz_cap3, n, alpha),
                    "potencia_mc_en_d_cap3": potencia_t_pareada_mc(n, dz_cap3, alpha, B_ITER, RNG),
                }
            )
    tabla12 = pd.DataFrame(filas_t12)
    tabla12.to_csv(RES_DIR / "11_tabla12_replica.csv", index=False, float_format="%.4f")
    print("Tabla 12 replicada; celdas que difieren:",
          tabla12.loc[~tabla12["coincide"], ["n", "escenario", "d_cap3", "d_exacto"]].to_dict("records"))

    # -----------------------------------------------------------------------
    # B. Escenario principal Monte Carlo (110 ingresantes, perdida realista)
    # -----------------------------------------------------------------------
    acum = dict(omni=0, sub=np.zeros(5), alguna=0, he2=0, ind=np.zeros(6), n=[], tau=[])
    for _ in range(B_ITER):
        c = simular_cohorte(RNG)
        r = analisis_intrasujeto(c)
        dec = decisiones_puerta_cerrada(r)
        acum["omni"] += dec["omni"]
        acum["sub"] += dec["sub"]
        acum["alguna"] += dec["sub"].any()
        acum["he2"] += dec["he2"]
        acum["ind"] += dec["individual_05"]
        acum["n"].append(r["n"])
        acum["tau"].append(r["tau"])
    n_comp = np.array(acum["n"])
    tau_mc = np.array(acum["tau"])
    principal = {
        "B": B_ITER,
        "d_verdadero": D_UMBRAL,
        "n_completos_media": float(n_comp.mean()),
        "n_completos_p05": float(np.percentile(n_comp, 5)),
        "n_completos_p95": float(np.percentile(n_comp, 95)),
        "prob_n_menor_95": float(np.mean(n_comp < N_META)),
        "potencia_HE1_omnibus": acum["omni"] / B_ITER,
        "potencia_HE1_sub_puerta": dict(zip(TTCT_DIMS, (acum["sub"] / B_ITER).round(4).tolist())),
        "potencia_HE1_sub_alguna": acum["alguna"] / B_ITER,
        "potencia_HE2": acum["he2"] / B_ITER,
        "potencia_individual_05": dict(zip(VARIABLES, (acum["ind"] / B_ITER).round(4).tolist())),
        "sesgo_tau_medio": dict(zip(VARIABLES, (tau_mc.mean(axis=0) - D_UMBRAL).round(4).tolist())),
        "potencia_analitica_n95": potencia_t_pareada(D_UMBRAL / np.sqrt(2 * (1 - RHO)), N_META, ALPHA),
        "potencia_analitica_sub_n95_d029": potencia_t_pareada(D_UMBRAL / np.sqrt(2 * (1 - RHO)), N_META, ALPHA_SUB),
    }

    # Potencia con exactamente 95 casos completos (sin perdida), referencia de la Tabla 12
    pot95 = potencia_vectorizada(N_META, RHO, D_UMBRAL, ALPHA, B_ITER, RNG)
    principal["potencia_HE2_n95_exacto"] = pot95

    # -----------------------------------------------------------------------
    # C. Barridos: potencia vs. n completos y sensibilidad rho x d (n = 95)
    # -----------------------------------------------------------------------
    n_barrido = np.arange(55, 146, 5)
    curvas = {
        "rho50_a05": (0.50, ALPHA),
        "rho50_a01": (0.50, ALPHA_SUB),
        "rho30_a05": (0.30, ALPHA),
        "rho70_a05": (0.70, ALPHA),
    }
    filas_n = []
    for n in n_barrido:
        fila = {"N": int(n)}
        for clave, (rho, alpha) in curvas.items():
            fila[clave] = potencia_vectorizada(int(n), rho, D_UMBRAL, alpha, B_SWEEP, RNG)
            fila[f"{clave}_analitica"] = potencia_t_pareada(D_UMBRAL / np.sqrt(2 * (1 - rho)), int(n), alpha)
        filas_n.append(fila)
    tabla_potencia_n = pd.DataFrame(filas_n)
    tabla_potencia_n.to_csv(RES_DIR / "08_potencia_vs_N.csv", index=False, float_format="%.4f")

    rho_barrido = np.round(np.arange(0.30, 0.71, 0.05), 2)
    d_barrido = np.round(np.arange(0.20, 0.51, 0.03), 2)
    filas_s = []
    for rho in rho_barrido:
        for d in d_barrido:
            filas_s.append(
                {
                    "rho": rho,
                    "d": d,
                    "potencia": potencia_vectorizada(N_META, rho, d, ALPHA, B_SWEEP, RNG),
                    "potencia_analitica": potencia_t_pareada(d / np.sqrt(2 * (1 - rho)), N_META, ALPHA),
                }
            )
    tabla_sens = pd.DataFrame(filas_s)
    tabla_sens.to_csv(RES_DIR / "09_sensibilidad_rho_d.csv", index=False, float_format="%.4f")

    # -----------------------------------------------------------------------
    # D. Escenarios de robustez (Tablas 10 y 20)
    # -----------------------------------------------------------------------
    filas_r = []
    for nombre, kwargs in ESCENARIOS.items():
        r_he2 = r_omni = 0
        r_pp = 0
        sesgo, ns = [], []
        for _ in range(B_SWEEP):
            c = simular_cohorte(RNG, **kwargs)
            r = analisis_intrasujeto(c)
            r_he2 += r["p"][5] < ALPHA
            r_omni += r["p_omni"] < ALPHA
            sesgo.append(r["tau"][5] - D_UMBRAL)
            ns.append(r["n"])
            if nombre == "dosis_incompleta":
                pp = analisis_intrasujeto(c, mascara=(c["ses_a"] >= 5) & (c["ses_b"] >= 5))
                r_pp += pp["p"][5] < ALPHA
        filas_r.append(
            {
                "escenario": nombre,
                "potencia_HE2": r_he2 / B_SWEEP,
                "potencia_HE1_omnibus": r_omni / B_SWEEP,
                "sesgo_tau_PSSM": float(np.mean(sesgo)),
                "n_completos_media": float(np.mean(ns)),
                "potencia_HE2_por_protocolo": (r_pp / B_SWEEP) if nombre == "dosis_incompleta" else np.nan,
            }
        )
    tabla_robustez = pd.DataFrame(filas_r)
    tabla_robustez.to_csv(RES_DIR / "10_robustez_sesgos.csv", index=False, float_format="%.4f")

    # -----------------------------------------------------------------------
    # E. HE3: eventos esperados y comportamiento de la regresion de Firth
    # -----------------------------------------------------------------------
    eventos, rechazos, rechazos_10, n_10 = [], 0, 0, 0
    for _ in range(B_HE3):
        c = simular_cohorte(RNG)
        y = c["permanece"].astype(float)
        z_pssm = (c["y0"][:, 5] - c["y0"][:, 5].mean()) / c["y0"][:, 5].std(ddof=1)
        X = np.column_stack([np.ones(c["n"]), z_pssm, c["rend"]])
        ev = int((1 - y).sum())
        eventos.append(ev)
        rech = firth_p_lr(X, y, 1) < ALPHA
        rechazos += rech
        if ev >= 10:
            n_10 += 1
            rechazos_10 += rech
    eventos = np.array(eventos)
    he3_mc = {
        "B": B_HE3,
        "OR_supuesto_permanencia_por_DE": OR_PSSM_PERMANENCIA,
        "eventos_media": float(eventos.mean()),
        "eventos_p05": float(np.percentile(eventos, 5)),
        "eventos_p95": float(np.percentile(eventos, 95)),
        "prob_menos_de_10_eventos": float(np.mean(eventos < 10)),
        "prob_menos_de_10_binomial": float(stats.binom.cdf(9, N_INGRESANTES, TASA_RETIRO)),
        "prob_IC_excluye_1": rechazos / B_HE3,
        "prob_IC_excluye_1_si_10_o_mas": rechazos_10 / max(n_10, 1),
        "eventos_por_predictor_media": float(eventos.mean() / 2),
        "histograma_eventos": {int(k): int(v) for k, v in zip(*np.unique(eventos, return_counts=True))},
    }

    # -----------------------------------------------------------------------
    # F. Equivalencia: modelo mixto completo vs. estimador rapido intrasujeto
    # -----------------------------------------------------------------------
    filas_eq = []
    for rep in range(B_MIXTO):
        c = simular_cohorte(RNG)
        r = analisis_intrasujeto(c)
        est_eq, largo_eq = a_tablas(c, RNG, bruto=False)
        comp_eq = completos(largo_eq)
        for j, var in [(0, "fluidez"), (5, DV2)]:
            formula = formula_modelo(var)
            a_cc = smf.mixedlm(formula, comp_eq, groups="subject_id").fit(reml=True)
            a_all = smf.mixedlm(formula, largo_eq, groups="subject_id").fit(reml=True)
            # mismo dato en otra escala: el modelo trabaja con z (DE de la
            # linea base), el estimador rapido con la escala latente
            escala = est_eq[f"{var}_O1"].std(ddof=1) / ESCALA_BRUTA[var][1]
            filas_eq.append(
                {
                    "replica": rep,
                    "variable": var,
                    "tau_rapido": r["tau"][j] / escala,
                    "p_rapido": r["p"][j],
                    "tau_mixto_completos": a_cc.fe_params["cond_A"],
                    "p_mixto_completos": 2 * stats.t.sf(abs(a_cc.tvalues["cond_A"]), r["gl_vec"][j]),
                    "tau_mixto_disponibles": a_all.fe_params["cond_A"],
                    "p_mixto_disponibles": 2 * stats.t.sf(abs(a_all.tvalues["cond_A"]), r["gl_vec"][j]),
                    "n_completos": r["n"],
                    "n_disponibles": largo_eq["subject_id"].nunique(),
                }
            )
    tabla_eq = pd.DataFrame(filas_eq)
    tabla_eq.to_csv(RES_DIR / "19_equivalencia_mixto_vs_rapido.csv", index=False, float_format="%.5f")
    resumen_eq = []
    for var, g in tabla_eq.groupby("variable"):
        resumen_eq.append(
            {
                "variable": var,
                "r_tau_rapido_vs_mixto": float(np.corrcoef(g["tau_rapido"], g["tau_mixto_completos"])[0, 1]),
                "max_dif_abs_tau": float(np.max(np.abs(g["tau_rapido"] - g["tau_mixto_completos"]))),
                "max_dif_abs_p": float(np.max(np.abs(g["p_rapido"] - g["p_mixto_completos"]))),
                "acuerdo_decision_05": float(np.mean((g["p_rapido"] < ALPHA) == (g["p_mixto_completos"] < ALPHA))),
                "rechazo_rapido": float(np.mean(g["p_rapido"] < ALPHA)),
                "rechazo_mixto_completos": float(np.mean(g["p_mixto_completos"] < ALPHA)),
                "rechazo_mixto_disponibles": float(np.mean(g["p_mixto_disponibles"] < ALPHA)),
            }
        )
    resumen_eq = pd.DataFrame(resumen_eq)

    # -----------------------------------------------------------------------
    # K. Error tipo I de la prueba omnibus HE1 bajo H0 (d = 0)
    # -----------------------------------------------------------------------
    rech = {"wald_sin_ocasion": 0, "wald_con_ocasion": 0, "multivariante_no_estructurado": 0}
    for _ in range(B_WALD):
        c = simular_cohorte(RNG, d=0.0)
        _, largo_h0 = a_tablas(c, RNG, bruto=False)
        comp_h0 = completos(largo_h0)
        rech["wald_sin_ocasion"] += wald_apilado(comp_h0, ocasion=False)["p_chi2"] < ALPHA
        rech["wald_con_ocasion"] += wald_apilado(comp_h0, ocasion=True)["p_chi2"] < ALPHA
        rech["multivariante_no_estructurado"] += analisis_intrasujeto(c)["p_omni"] < ALPHA
    tipo1 = pd.DataFrame(
        [
            {
                "prueba": k,
                "error_tipo_I": v / B_WALD,
                "ic95_lo": stats.binomtest(int(v), B_WALD).proportion_ci().low,
                "ic95_hi": stats.binomtest(int(v), B_WALD).proportion_ci().high,
            }
            for k, v in rech.items()
        ]
    )
    tipo1.to_csv(RES_DIR / "20_error_tipo_I_HE1.csv", index=False, float_format="%.4f")
    print("Error tipo I HE1:", tipo1[["prueba", "error_tipo_I"]].to_dict("records"))

    # -----------------------------------------------------------------------
    # G. Dataset representativo: una cohorte analizada con el plan completo
    # -----------------------------------------------------------------------
    c_rep = simular_cohorte(RNG)
    est, largo = a_tablas(c_rep, RNG)
    est.to_csv(RES_DIR / "dataset_simulado_representativo.csv", index=False, float_format="%.4f")
    comp = completos(largo)
    n_rep = comp["subject_id"].nunique()

    # G1. Flujo de participantes por seccion (3.8.2)
    flujo = (
        est.groupby("seccion")
        .agg(
            ingresantes=("subject_id", "size"),
            retiros_antes_O2=("retiro", lambda s: (s == "antes de O2").sum()),
            retiros_O2_O3=("retiro", lambda s: (s == "entre O2 y O3").sum()),
            con_O2=("obs_O2", "sum"),
            con_O3=("obs_O3", "sum"),
            completos=("obs_O3", lambda s: (s & est.loc[s.index, "obs_O2"]).sum()),
        )
        .reset_index()
    )
    flujo = SECCIONES[["seccion", "programa", "turno", "secuencia", "vacantes"]].merge(flujo)
    total = flujo.drop(columns=["seccion", "programa", "turno", "secuencia"]).sum()
    flujo = pd.concat([flujo, pd.DataFrame([{"seccion": "Total", **total.to_dict()}])], ignore_index=True)
    flujo.to_csv(RES_DIR / "12_flujo_participantes.csv", index=False)

    # G2. Equivalencia de secuencias en la linea base (diferencia estandarizada)
    filas_eqb = []
    for var in VARIABLES:
        ab = est.loc[est["secuencia"] == "AB", f"z_{var}_O1"]
        ba = est.loc[est["secuencia"] == "BA", f"z_{var}_O1"]
        sp = np.sqrt((ab.var(ddof=1) + ba.var(ddof=1)) / 2)
        filas_eqb.append({"variable": var, "media_AB": ab.mean(), "media_BA": ba.mean(),
                          "dif_estandarizada": (ba.mean() - ab.mean()) / sp})
    equiv_base = pd.DataFrame(filas_eqb)
    equiv_base.to_csv(RES_DIR / "14_equivalencia_linea_base.csv", index=False, float_format="%.3f")

    # G3. Reglas psicometricas del TTCT (3.6.1.3 y 3.6.1.5)
    filas_psi = []
    icc_2k = {}
    for var in TTCT_DIMS:
        m = est[[f"{var}_O1_cal1", f"{var}_O1_cal2"]].to_numpy()
        icc = cci_21(m)
        boot = [cci_21(m[RNG.integers(0, len(m), len(m))]) for _ in range(B_BOOT)]
        icc_2k[var] = 2 * icc / (1 + icc)  # Spearman-Brown: promedio de dos calificadores
        filas_psi.append({"dimension": var, "CCI_21": icc, "IC95_lo": np.percentile(boot, 2.5),
                          "IC95_hi": np.percentile(boot, 97.5), "CCI_2k": icc_2k[var],
                          "cumple_75": icc >= 0.75, "cumple_80_capacitacion": icc >= 0.80})
    psico = pd.DataFrame(filas_psi)
    base_z = est[[f"z_{v}_O1" for v in TTCT_DIMS]].to_numpy()
    r_base = np.corrcoef(base_z, rowvar=False)
    rel = np.array([icc_2k[v] for v in TTCT_DIMS])
    r_corr = r_base / np.sqrt(np.outer(rel, rel))
    np.fill_diagonal(r_corr, np.nan)
    r_corr_max = float(np.nanmax(r_corr))
    par_max = np.unravel_index(np.nanargmax(r_corr), r_corr.shape)
    regla_a = bool(psico["cumple_75"].all())
    regla_b = r_corr_max <= 0.85
    psico.to_csv(RES_DIR / "13_psicometria_reglas.csv", index=False, float_format="%.3f")
    pd.DataFrame(r_base, index=TTCT_DIMS, columns=TTCT_DIMS).to_csv(
        RES_DIR / "13b_correlaciones_linea_base_TTCT.csv", float_format="%.3f")
    # rho estimado como lo preve 3.3.3: linea base y semana 9
    rho_est = {var: float(est[[f"z_{var}_O1", f"z_{var}_O2"]].dropna().corr().iloc[0, 1])
               for var in VARIABLES}
    rho_post = {var: float(est[[f"z_{var}_O2", f"z_{var}_O3"]].dropna().corr().iloc[0, 1])
                for var in VARIABLES}

    # G4. Descriptivos por condicion y supuestos
    filas_desc = []
    for var in VARIABLES:
        for cond, sub in comp.groupby("condicion")[var]:
            filas_desc.append({"variable": var, "condicion": cond, "n": sub.shape[0],
                               "media": sub.mean(), "de": sub.std(ddof=1), "mediana": sub.median(),
                               "ric": sub.quantile(0.75) - sub.quantile(0.25),
                               "asimetria": stats.skew(sub), "curtosis": stats.kurtosis(sub)})
    desc = pd.DataFrame(filas_desc)
    desc.to_csv(RES_DIR / "01_descriptivos.csv", index=False, float_format="%.3f")

    # G5. Modelos mixtos HE1 (por dimension) y HE2 (3.8.3)
    modelos = {var: ajustar_mixto(comp, var) for var in VARIABLES}
    wald = wald_apilado(comp, ocasion=True)
    wald_sin = wald_apilado(comp, ocasion=False)
    hot = hotelling_dos_muestras(comp)
    rapido_rep = analisis_intrasujeto(c_rep)
    he1_sig = wald["p_F"] < ALPHA

    filas_sup = []
    for var in VARIABLES:
        res = np.asarray(modelos[var]["ajuste"].resid)
        a = comp.loc[comp["condicion"] == "A", var]
        b = comp.loc[comp["condicion"] == "B", var]
        lev_cond = stats.levene(a, b)
        lev_sec = stats.levene(*[res[(comp["seccion"] == s).to_numpy()] for s in SECCIONES["seccion"]])
        sw = stats.shapiro(res)
        filas_sup.append({"variable": var, "shapiro_W_residuos": sw.statistic,
                          "shapiro_p_residuos": sw.pvalue,
                          "levene_F_condicion": lev_cond.statistic, "levene_p_condicion": lev_cond.pvalue,
                          "levene_F_seccion": lev_sec.statistic, "levene_p_seccion": lev_sec.pvalue,
                          "casos_influyentes_cook": int(len(influyentes_cook(comp, var)))})
    supuestos = pd.DataFrame(filas_sup)
    supuestos.to_csv(RES_DIR / "02_pruebas_supuestos.csv", index=False, float_format="%.4f")

    # G6. Tamanos del efecto (3.8.4)
    filas_ef = []
    for var in VARIABLES:
        w = a_ancho(comp, var)
        dif = (w["A"] - w["B"]).to_numpy()
        n = len(dif)
        dz = dif.mean() / dif.std(ddof=1)
        dav = dif.mean() / ((w["A"].std(ddof=1) + w["B"].std(ddof=1)) / 2)
        nivel = 0.99 if var in TTCT_DIMS else 0.95
        lo, hi = ic_dz(dz, n, nivel)
        m = modelos[var]
        clave_ic = "ic99" if var in TTCT_DIMS else "ic95"
        alfa_ap = ALPHA_SUB if var in TTCT_DIMS else ALPHA
        sig = (he1_sig and m["p"] < ALPHA_SUB) if var in TTCT_DIMS else (m["p"] < ALPHA)
        filas_ef.append(
            {
                "variable": var, "n": n, "dif_media_A_B": dif.mean(), "dz": dz,
                "dz_ic_lo": lo, "dz_ic_hi": hi, "dav": dav, "tau_ajustado": m["tau"],
                "tau_ic_lo": m[clave_ic][0], "tau_ic_hi": m[clave_ic][1], "ee": m["ee"],
                "p": m["p"], "alfa": alfa_ap, "nivel_ic": nivel, "significativo": bool(sig),
                "estructura_aleatoria": m["estructura"],
            }
        )
    efectos = pd.DataFrame(filas_ef)
    efectos.to_csv(RES_DIR / "03_tamanos_efecto.csv", index=False, float_format="%.4f")

    # G7. Periodo y secuencia (3.8.5)
    filas_ps = []
    for var in VARIABLES:
        m = modelos[var]
        filas_ps.append({"variable": var, "periodo": m["periodo_2"][0], "periodo_lo": m["periodo_2"][1],
                         "periodo_hi": m["periodo_2"][2], "secuencia_BA": m["secuencia_BA"][0],
                         "secuencia_lo": m["secuencia_BA"][1], "secuencia_hi": m["secuencia_BA"][2]})
    periodo_sec = pd.DataFrame(filas_ps)

    with open(RES_DIR / "04_prueba_omnibus_HE1.txt", "w", encoding="utf-8") as f:
        f.write(
            f"HE1 (omnibus): prueba de Wald conjunta de los 5 coeficientes de condicion\n"
            f"en el modelo mixto apilado (dimensiones estandarizadas con la DE de la linea base;\n"
            f"efectos aleatorios: estudiante, estudiante x dimension, estudiante x medicion):\n"
            f"  chi2(5) = {wald['chi2']:.3f}, p = {wald['p_chi2']:.4f};  F(5, {wald['gl2']}) = "
            f"{wald['F']:.3f}, p = {wald['p_F']:.4f}\n"
            f"  Sin el termino estudiante x medicion: chi2(5) = {wald_sin['chi2']:.3f}, "
            f"p = {wald_sin['p_chi2']:.4f} (error tipo I inflado; ver 20_error_tipo_I_HE1.csv)\n"
            f"Robustez (Hills y Armitage multivariante): T2 de Hotelling sobre P2 - P1, AB vs BA:\n"
            f"  T2 = {hot['T2']:.3f}, F({hot['gl1']}, {hot['gl2']}) = {hot['F']:.3f}, p = {hot['p']:.4f}"
            f" (n_BA = {hot['n_BA']}, n_AB = {hot['n_AB']})\n"
            f"Estimador intrasujeto con forma como covariable: F({rapido_rep['gl1_omni']}, "
            f"{rapido_rep['gl2_omni']}) = {rapido_rep['F_omni']:.3f}, p = {rapido_rep['p_omni']:.4f}\n\n"
            f"Decision HE1 (alfa = .05): {'se rechaza H0' if he1_sig else 'no se rechaza H0'}; "
            f"HE1a-e {'se examinan' if he1_sig else 'no se examinan'} con alfa = .01 e IC 99 %.\n\n"
            f"HE2 (PSSM, modelo mixto): tau = {modelos[DV2]['tau']:.3f} "
            f"[{modelos[DV2]['ic95'][0]:.3f}, {modelos[DV2]['ic95'][1]:.3f}], "
            f"t({modelos[DV2]['gl']}) = {modelos[DV2]['t']:.3f}, p = {modelos[DV2]['p']:.4f}\n"
        )

    with open(RES_DIR / "05_modelo_mixto.txt", "w", encoding="utf-8") as f:
        for var in VARIABLES:
            m = modelos[var]
            f.write(f"===== Modelo mixto: {var} ({m['estructura']}) =====\n")
            f.write(f"Formula: {m['formula']}\n")
            f.write(m["ajuste"].summary().as_text() + "\n\n")
        f.write("===== Modelo apilado HE1 (Wald 5 gl) =====\n")
        f.write(wald["ajuste"].summary().as_text() + "\n")

    # G8. HE3 con Firth (3.8.6)
    y_perm = est["permanece"].to_numpy(dtype=float)
    pssm0 = est["z_pssm_global_O1"].to_numpy()
    X3 = np.column_stack([np.ones(len(est)), (pssm0 - pssm0.mean()) / pssm0.std(ddof=1),
                          est["rendimiento_previo"].to_numpy()])
    he3 = firth_inferencia(X3, y_perm, ["Intercepto", "PSSM linea base (por DE)", "Rendimiento previo (por DE)"])
    n_retiros = int((1 - y_perm).sum())
    he3.to_csv(RES_DIR / "06_HE3_firth_OR.csv", index=False, float_format="%.4f")
    he3_desc = est.groupby("permanece")["pssm_global_O1"].agg(["count", "mean", "std"]).reset_index()

    # G9. Relacion descriptiva PSSM-TTCT (3.8.7)
    filas_cor = []
    for var in TTCT_DIMS:
        for m in ["O1", "O2", "O3"]:
            d2 = est[[f"z_pssm_global_{m}", f"z_{var}_{m}"]].dropna()
            r, lo, hi = r_ic(d2.iloc[:, 0], d2.iloc[:, 1])
            filas_cor.append({"dimension": var, "momento": m, "n": len(d2), "r": r, "IC95_lo": lo, "IC95_hi": hi})
        wa, wp = a_ancho(comp, var), a_ancho(comp, DV2)
        r, lo, hi = r_ic((wa["A"] - wa["B"]).to_numpy(), (wp["A"] - wp["B"]).to_numpy())
        filas_cor.append({"dimension": var, "momento": "cambio A-B", "n": len(wa), "r": r, "IC95_lo": lo, "IC95_hi": hi})
    correl = pd.DataFrame(filas_cor)
    correl.to_csv(RES_DIR / "15_correlaciones_pssm_ttct.csv", index=False, float_format="%.3f")

    # G10. RFI-MPI-DUAE (3.6.3)
    rfi_ses, rfi_kappa = simular_rfi(RNG)
    rfi_sec = (
        rfi_ses.pivot_table(index=["seccion", "secuencia"], columns="condicion",
                            values=["indice", "contaminacion"], aggfunc="mean")
    )
    rfi = pd.DataFrame({
        "seccion": [i[0] for i in rfi_sec.index],
        "secuencia": [i[1] for i in rfi_sec.index],
        "fidelidad_A": rfi_sec[("indice", "A")].to_numpy(),
        "contaminacion_B": rfi_sec[("contaminacion", "B")].to_numpy(),
    })
    rfi["adecuada_85"] = rfi["fidelidad_A"] >= 85
    rfi.to_csv(RES_DIR / "18_rfi_fidelidad.csv", index=False, float_format="%.1f")
    cont_ba_p1 = rfi_ses.query("condicion == 'B' and secuencia == 'BA'")["contaminacion"].mean()
    cont_ab_p2 = rfi_ses.query("condicion == 'B' and secuencia == 'AB'")["contaminacion"].mean()

    # G11. Heterogeneidad por seccion (Tabla 20): efecto en cada seccion
    comp["ttct_compuesto"] = comp[TTCT_DIMS].mean(axis=1)
    filas_het = []
    for s in SECCIONES["seccion"]:
        sub = comp[comp["seccion"] == s]
        for var, etiqueta in [("ttct_compuesto", "TTCT (compuesto z)"), (DV2, "PSSM")]:
            w = a_ancho(sub, var)
            dif = (w["A"] - w["B"]).to_numpy()
            n = len(dif)
            se = dif.std(ddof=1) / np.sqrt(n)
            q = stats.t.ppf(0.975, n - 1)
            filas_het.append({"seccion": s, "variable": etiqueta, "n": n, "dif_A_B": dif.mean(),
                              "ic95_lo": dif.mean() - q * se, "ic95_hi": dif.mean() + q * se})
    heterog = pd.DataFrame(filas_het)
    heterog.to_csv(RES_DIR / "17_heterogeneidad_seccion.csv", index=False, float_format="%.3f")

    # G12. Analisis de sensibilidad (Tabla 20) para HE2 y fluidez
    secciones_ok = rfi.loc[rfi["adecuada_85"], "seccion"].tolist()
    filas_t20 = []
    for var in ["fluidez", DV2]:
        base_m = modelos[var]
        variantes = {
            "Principal (casos completos)": base_m,
            "Turno como covariable": ajustar_mixto(comp, var, extra=" + nocturno"),
            "Por protocolo (>= 5 de 7 sesiones)": ajustar_mixto(
                completos(largo[(largo["sesiones_A"] >= 5) & (largo["sesiones_B"] >= 5)]), var),
            "Fidelidad (secciones con RFI >= 85 %)": ajustar_mixto(comp[comp["seccion"].isin(secciones_ok)], var)
            if len(secciones_ok) < 6 else base_m,
            "Datos disponibles (modelo mixto con incompletos)": ajustar_mixto(largo, var),
        }
        # Valores atipicos: sin casos influyentes (Cook > 4/n)
        influyentes = influyentes_cook(comp, var)
        variantes["Sin casos influyentes"] = ajustar_mixto(comp[~comp["subject_id"].isin(influyentes)], var)
        for nombre, m in variantes.items():
            filas_t20.append({"variable": var, "analisis": nombre, "n": m["n"], "tau": m["tau"],
                              "ic95_lo": m["ic95"][0], "ic95_hi": m["ic95"][1], "p": m["p"],
                              "terminos_omitidos": ", ".join(m["omitidos"])})
        # El turno es constante dentro del estudiante: como covariable no puede
        # mover tau; para examinarlo hace falta la interaccion condicion x turno.
        mi = ajustar_mixto(comp, var, extra=" + nocturno + cond_A:nocturno", coef_extra="cond_A:nocturno")
        filas_t20.append({"variable": var, "analisis": "Interaccion condicion x turno (nocturno - diurno)",
                          "n": mi["n"], "tau": mi["extra"][0], "ic95_lo": mi["extra"][1],
                          "ic95_hi": mi["extra"][2], "p": mi["extra"][3],
                          "terminos_omitidos": ", ".join(mi["omitidos"])})
    sens_t20 = pd.DataFrame(filas_t20)
    sens_t20.to_csv(RES_DIR / "16_sensibilidad_tabla20.csv", index=False, float_format="%.4f")

    # -----------------------------------------------------------------------
    # H. Verificacion de potencia a priori (texto)
    # -----------------------------------------------------------------------
    difieren = tabla12.loc[~tabla12["coincide"]]
    with open(RES_DIR / "07_verificacion_potencia_apriori.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Diferencia minima detectable (t pareada bilateral, t no central), n = {N_META}, rho = {RHO}:\n"
            f"  alfa = .05 (HE1, HE2): d = {MDE_95:.4f}  (Cap. III: 0,29)\n"
            f"  alfa = .01 (HE1a-e):  d = {MDE_95_SUB:.4f}  (Cap. III: 0,36; seccion 1.7 del trabajo final: 0,35)\n\n"
            f"Tabla 12: {int(tabla12['coincide'].sum())} de {len(tabla12)} celdas coinciden al redondear a 2 decimales.\n"
        )
        for _, fila in difieren.iterrows():
            f.write(f"  Difiere: n = {fila['n']}, {fila['escenario']}: Cap. III {fila['d_cap3']:.2f}, "
                    f"exacto {fila['d_exacto']:.4f}\n")
        f.write(
            f"\nEscenario principal Monte Carlo (B = {B_ITER}; 110 ingresantes; retiro 10,7 %; "
            f"inasistencia por medicion {P_INASISTENCIA:.3f}; efecto verdadero d = {D_UMBRAL}):\n"
            f"  casos completos: media {principal['n_completos_media']:.1f} "
            f"[P5 {principal['n_completos_p05']:.0f}, P95 {principal['n_completos_p95']:.0f}]; "
            f"P(n < 95) = {principal['prob_n_menor_95']:.3f}\n"
            f"  potencia HE2 = {principal['potencia_HE2']:.4f}; HE1 omnibus = {principal['potencia_HE1_omnibus']:.4f}\n"
            f"  potencia con 95 casos completos exactos = {pot95:.4f}\n"
        )

    # -----------------------------------------------------------------------
    # I. FIGURAS (300 DPI, paleta viridis)
    # -----------------------------------------------------------------------
    comp_plot = comp.copy()
    comp_plot["condicion_lbl"] = comp_plot["condicion"].map({"A": "A_MPI-DUAE", "B": "B_Comparacion"})

    # (a) Distribucion / densidad con IC 95 %
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, var, titulo in zip(axes, ["fluidez", DV2],
                               [r"TTCT Figural $-$ Fluidez ($z$)", r"PSSM $-$ puntaje global ($z$)"]):
        for cond, color in COND_PALETTE.items():
            datos = comp_plot.loc[comp_plot["condicion_lbl"] == cond, var]
            sns.kdeplot(datos, ax=ax, color=color, fill=True, alpha=0.25, linewidth=2, label=cond)
            m = datos.mean()
            ci = stats.norm.ppf(0.975) * datos.std(ddof=1) / np.sqrt(len(datos))
            ax.axvline(m, color=color, linestyle="--", linewidth=1.2)
            ax.axvspan(m - ci, m + ci, color=color, alpha=0.12)
        ax.set_title(titulo)
        ax.set_xlabel(r"Puntaje estandarizado con la DE de la linea base ($z$)")
        ax.set_ylabel("Densidad")
    axes[0].legend(title="Condicion", loc="upper left", fontsize=8)
    fig.suptitle(f"Distribucion simulada por condicion, media e IC 95 % (n = {n_rep} casos completos, una cohorte ilustrativa)",
                 fontsize=11)
    fig.savefig(FIG_DIR / "a_distribucion_densidad_IC95.png", bbox_inches="tight")
    plt.close(fig)

    # (b) Curvas de potencia vs. n completos
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    estilos = {
        "rho50_a05": (r"$\rho$=.50, $\alpha$=.05 (HE1, HE2)", PALETTE[1], "-"),
        "rho50_a01": (r"$\rho$=.50, $\alpha$=.01 (HE1a-e)", PALETTE[4], "-"),
        "rho30_a05": (r"$\rho$=.30, $\alpha$=.05", PALETTE[6], "--"),
        "rho70_a05": (r"$\rho$=.70, $\alpha$=.05", PALETTE[2], "--"),
    }
    for clave, (lbl, color, ls) in estilos.items():
        ax.plot(tabla_potencia_n["N"], tabla_potencia_n[clave], marker="o", markersize=3.5,
                color=color, linestyle=ls, linewidth=2, label=lbl)
    ax.axhline(POWER_TARGET, color="gray", linestyle=":", linewidth=1.2)
    ax.axvline(N_META, color="#c44e52", linestyle="--", linewidth=1.2, label=f"Meta: {N_META} casos completos")
    ax.axvline(N_INGRESANTES, color="#c44e52", linestyle=":", linewidth=1, label=f"Ingreso estimado: {N_INGRESANTES}")
    ax.set_xlabel(r"Casos con datos completos ($n$)")
    ax.set_ylabel("Potencia empirica")
    ax.set_title(rf"Potencia del contraste intrasujeto con $d$ = {D_UMBRAL} (umbral de deteccion)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIG_DIR / "b_curva_potencia_vs_N.png", bbox_inches="tight")
    plt.close(fig)

    # (c) Forest plot de tamanos de efecto + regresion pre-post
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ax = axes[0]
    y_pos = np.arange(len(efectos))
    colores = [PALETTE[3] if s else "#9aa0a6" for s in efectos["significativo"]]
    for i, fila in efectos.iterrows():
        ax.errorbar(fila["dz"], i, xerr=[[fila["dz"] - fila["dz_ic_lo"]], [fila["dz_ic_hi"] - fila["dz"]]],
                    fmt="o", color=colores[i], ecolor=colores[i], capsize=3, markersize=6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(MDE_95, color=PALETTE[5], linestyle=":", linewidth=1.4, label=f"Umbral HE1/HE2 ({MDE_95:.2f})")
    ax.axvline(MDE_95_SUB, color=PALETTE[6], linestyle=":", linewidth=1.4, label=f"Umbral HE1a-e ({MDE_95_SUB:.2f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{v} ({'IC 99' if v in TTCT_DIMS else 'IC 95'})" for v in efectos["variable"]])
    ax.invert_yaxis()
    ax.set_xlabel(r"$d_z$ de Cohen")
    ax.set_title("Tamanos de efecto: 5 dimensiones TTCT + PSSM")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    ax2 = axes[1]
    wf = comp.pivot_table(index="subject_id", columns="condicion", values="fluidez")
    base_f = comp.groupby("subject_id")["fluidez_base"].first().loc[wf.index]
    for cond, lbl in [("A", "A_MPI-DUAE"), ("B", "B_Comparacion")]:
        sns.regplot(x=base_f.values, y=wf[cond].values, ax=ax2, color=COND_PALETTE[lbl],
                    scatter_kws={"alpha": 0.5, "s": 22}, line_kws={"linewidth": 2}, label=lbl, ci=95)
    ax2.set_xlabel(r"Fluidez, linea base O1 ($z$)")
    ax2.set_ylabel(r"Fluidez tras el periodo ($z$)")
    ax2.set_title("Linea base como covariable: regresion con banda IC 95 %")
    ax2.legend(fontsize=8)
    fig.savefig(FIG_DIR / "c_forest_y_regresion_prepost.png", bbox_inches="tight")
    plt.close(fig)

    # (d) Violin + boxplot + puntos
    df_lp = comp_plot.melt(id_vars=["subject_id", "condicion_lbl"], value_vars=VARIABLES,
                           var_name="variable", value_name="puntaje")
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    sns.violinplot(data=df_lp, x="variable", y="puntaje", hue="condicion_lbl", split=True, inner=None,
                   palette=COND_PALETTE, linewidth=1, ax=ax, cut=0)
    sns.boxplot(data=df_lp, x="variable", y="puntaje", hue="condicion_lbl", width=0.15, showcaps=True,
                boxprops={"zorder": 3, "facecolor": "white"}, showfliers=False,
                whiskerprops={"zorder": 3}, ax=ax, dodge=True, legend=False)
    sns.stripplot(data=df_lp, x="variable", y="puntaje", hue="condicion_lbl", dodge=True, alpha=0.25,
                  size=2.5, palette=COND_PALETTE, ax=ax, legend=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Condicion", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(r"Puntaje estandarizado ($z$)")
    ax.set_title("Comparacion por condicion: TTCT Figural (5 dimensiones) y PSSM")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIG_DIR / "d_violin_boxplot_puntos.png", bbox_inches="tight")
    plt.close(fig)

    # (e) Mapa de calor rho x d con n = 95
    pivot = tabla_sens.pivot(index="d", columns="rho", values="potencia")
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    sns.heatmap(pivot.sort_index(ascending=False), cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7.5}, cbar_kws={"label": "Potencia empirica"}, ax=ax)
    filas_idx = list(pivot.sort_index(ascending=False).index)
    ax.add_patch(plt.Rectangle((list(pivot.columns).index(0.5), filas_idx.index(D_UMBRAL)), 1, 1,
                               fill=False, edgecolor="#c44e52", linewidth=2.5))
    ax.set_xlabel(r"$\rho$ (correlacion entre mediciones posteriores)")
    ax.set_ylabel(r"$d$ verdadero")
    ax.set_title(f"Sensibilidad de la potencia (n = {N_META} completos, alfa = {ALPHA})")
    fig.savefig(FIG_DIR / "e_sensibilidad_rho_d_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # (f) Tabla 12: diferencia minima detectable exacta vs. impresa
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    etiquetas = {"rho50_a05": r"$\rho$=.50, $\alpha$=.05", "rho50_a01": r"$\rho$=.50, $\alpha$=.01",
                 "rho30_a05": r"$\rho$=.30, $\alpha$=.05", "rho70_a05": r"$\rho$=.70, $\alpha$=.05"}
    n_fino = np.arange(50, 151, 1)
    for (clave, (rho, alpha)), color in zip(TABLA12_ESCENARIOS.items(), [PALETTE[1], PALETTE[4], PALETTE[6], PALETTE[2]]):
        ax.plot(n_fino, [d_minimo(int(n), rho, alpha) for n in n_fino], color=color, linewidth=1.8,
                label=etiquetas[clave] + " (exacto)")
        sub = tabla12[tabla12["escenario"] == clave]
        ax.scatter(sub["n"], sub["d_cap3"], color=color, edgecolor="black", zorder=3, s=36)
        malas = sub[~sub["coincide"]]
        ax.scatter(malas["n"], malas["d_cap3"], facecolor="none", edgecolor="#c44e52", s=160, linewidth=2, zorder=4)
    ax.axvline(N_META, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Casos completos (n)")
    ax.set_ylabel("Diferencia minima detectable (d)")
    ax.set_title("Tabla 12 del Cap. III: puntos impresos vs. curva exacta (t no central)")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "f_tabla12_diferencia_minima.png", bbox_inches="tight")
    plt.close(fig)

    # (g) Heterogeneidad por seccion (forest)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True, sharey=True)
    for ax, etiqueta in zip(axes, ["TTCT (compuesto z)", "PSSM"]):
        sub = heterog[heterog["variable"] == etiqueta].reset_index(drop=True)
        for i, fila in sub.iterrows():
            ax.errorbar(fila["dif_A_B"], i, xerr=[[fila["dif_A_B"] - fila["ic95_lo"]], [fila["ic95_hi"] - fila["dif_A_B"]]],
                        fmt="s", color=PALETTE[2], capsize=3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(D_UMBRAL, color="#c44e52", linestyle=":", linewidth=1.2, label=f"d verdadero = {D_UMBRAL}")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(sub["seccion"], sub["n"])])
        ax.set_title(etiqueta)
        ax.set_xlabel("Diferencia A - B (z) e IC 95 %")
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Efecto de la condicion en cada seccion (Tabla 20: dependencia del par de docentes)", fontsize=11)
    fig.savefig(FIG_DIR / "g_heterogeneidad_por_seccion.png", bbox_inches="tight")
    plt.close(fig)

    # (h) HE3: distribucion del numero de retiros
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ks = np.array(sorted(he3_mc["histograma_eventos"]))
    vs = np.array([he3_mc["histograma_eventos"][k] for k in ks]) / B_HE3
    ax.bar(ks, vs, color=["#c44e52" if k < 10 else PALETTE[2] for k in ks], edgecolor="white")
    ax.axvline(9.5, color="#c44e52", linestyle="--", linewidth=1.4, label="Regla: < 10 retiros, solo descriptivo")
    ax.set_xlabel(f"Retiros formales entre {N_INGRESANTES} ingresantes")
    ax.set_ylabel("Proporcion de cohortes simuladas")
    ax.set_title(f"HE3: P(< 10 retiros) = {he3_mc['prob_menos_de_10_eventos']:.2f} (B = {B_HE3})")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "h_HE3_eventos_retiro.png", bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # J. Datos del dashboard y resumen ejecutivo
    # -----------------------------------------------------------------------
    def registros(df):
        return json.loads(df.to_json(orient="records", double_precision=4))

    dashboard = {
        "parametros": {
            "semilla": SEED, "capacidad": N_CAPACIDAD, "ingresantes": N_INGRESANTES, "meta": N_META,
            "perdida_admitida": PERDIDA_ADMITIDA, "rho": RHO, "alpha": ALPHA, "alpha_sub": ALPHA_SUB,
            "potencia_objetivo": POWER_TARGET, "d_umbral": D_UMBRAL, "mde_95": MDE_95,
            "mde_95_sub": MDE_95_SUB, "B_iter": B_ITER, "B_sweep": B_SWEEP, "B_he3": B_HE3,
            "B_mixto": B_MIXTO, "B_wald": B_WALD, "tasa_retiro": TASA_RETIRO, "p_inasistencia": P_INASISTENCIA,
            "efecto_periodo": PERIOD_EFFECT, "efecto_forma": FORM_EFFECT,
        },
        "principal": principal,
        "tabla12": registros(tabla12),
        "potencia_n": registros(tabla_potencia_n),
        "sensibilidad": registros(tabla_sens),
        "robustez": registros(tabla_robustez),
        "he3_mc": he3_mc,
        "equivalencia": registros(resumen_eq),
        "error_tipo_I_HE1": registros(tipo1),
        "representativo": {
            "n_completos": n_rep,
            "n_ingresantes": int(len(est)),
            "n_retiros": n_retiros,
            "rho_O1_O2": rho_est,
            "rho_O2_O3": rho_post,
            "wald": {k: wald[k] for k in ("chi2", "gl", "p_chi2", "F", "gl2", "p_F")},
            "wald_sin_ocasion": {k: wald_sin[k] for k in ("chi2", "gl", "p_chi2", "F", "gl2", "p_F")},
            "hotelling": hot,
            "omnibus_intrasujeto": {"F": rapido_rep["F_omni"], "gl1": rapido_rep["gl1_omni"],
                                    "gl2": rapido_rep["gl2_omni"], "p": rapido_rep["p_omni"]},
            "he1_significativa": bool(he1_sig),
            "estructura_aleatoria": {v: modelos[v]["estructura"] for v in VARIABLES},
            "regla_a_cci": regla_a,
            "regla_b_correlacion": bool(regla_b),
            "r_corregida_max": r_corr_max,
            "par_r_max": [TTCT_DIMS[par_max[0]], TTCT_DIMS[par_max[1]]],
            "contaminacion_BA_p1": cont_ba_p1,
            "contaminacion_AB_p2": cont_ab_p2,
            "kappa_rfi": rfi_kappa,
        },
        "flujo": registros(flujo),
        "descriptivos": registros(desc),
        "supuestos": registros(supuestos),
        "efectos": registros(efectos),
        "periodo_secuencia": registros(periodo_sec),
        "he3": registros(he3),
        "he3_descriptivo": registros(he3_desc),
        "psicometria": registros(psico),
        "equivalencia_base": registros(equiv_base),
        "correlaciones": registros(correl),
        "rfi": registros(rfi),
        "heterogeneidad": registros(heterog),
        "sensibilidad_t20": registros(sens_t20),
    }
    datos_json = json.dumps(dashboard, ensure_ascii=False,
                            default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (RES_DIR / "dashboard_data.json").write_text(datos_json, encoding="utf-8")
    if actualizar_dashboard(datos_json):
        print("Dashboard actualizado:", DASHBOARD_HTML.name)

    m2 = modelos[DV2]
    with open(RES_DIR / "00_RESUMEN.md", "w", encoding="utf-8") as f:
        f.write(
            f"""# Resumen de la simulacion Monte Carlo -- MPI-DUAE (v2, Capitulo III)

**Semilla:** {SEED} | **ingreso estimado:** {N_INGRESANTES} (capacidad {N_CAPACIDAD}) |
**meta:** {N_META} casos completos | **rho:** {RHO} | **d verdadero simulado:** {D_UMBRAL}
(umbral de deteccion, no efecto esperado) | **alpha:** {ALPHA} (HE1a-e: {ALPHA_SUB}) |
**potencia objetivo:** {POWER_TARGET} | **replicas:** {B_ITER} (principal), {B_SWEEP} (barridos)

## 1. Verificacion del calculo de potencia del Capitulo III (3.3.3)

- Diferencia minima detectable exacta con n = {N_META}, rho = {RHO}: **d = {MDE_95:.3f}** (alfa .05)
  y **d = {MDE_95_SUB:.3f}** (alfa .01). El Cap. III informa 0,29 y 0,36; la seccion 1.7 del
  trabajo final informa 0,35 para las subhipotesis.
- Tabla 12: {int(tabla12['coincide'].sum())} de {len(tabla12)} celdas coinciden con el calculo exacto.
{''.join(f"  - Difiere n = {r['n']}, {r['escenario']}: impreso {r['d_cap3']:.2f}, exacto {r['d_exacto']:.4f}{chr(10)}" for _, r in difieren.iterrows())}
## 2. Escenario principal (110 ingresantes, perdida realista, B = {B_ITER})

- Casos completos: media {principal['n_completos_media']:.1f} (P5-P95: {principal['n_completos_p05']:.0f}-{principal['n_completos_p95']:.0f});
  P(n < 95) = {principal['prob_n_menor_95']:.2f}.
- Potencia HE2 (PSSM): **{principal['potencia_HE2']:.3f}**; con 95 casos exactos: {pot95:.3f}.
- Potencia HE1 omnibus (5 dimensiones): **{principal['potencia_HE1_omnibus']:.3f}**.
- Potencia de cada subhipotesis HE1a-e dentro de la puerta cerrada (alfa .01):
  {', '.join(f"{k} {v:.2f}" for k, v in principal['potencia_HE1_sub_puerta'].items())}.

## 3. Cohorte representativa ({len(est)} ingresantes, {n_rep} casos completos)

- HE1 (Wald 5 gl, modelo apilado): chi2(5) = {wald['chi2']:.2f}, p = {wald['p_chi2']:.4f};
  T2 de Hotelling (robustez): F({hot['gl1']}, {hot['gl2']}) = {hot['F']:.2f}, p = {hot['p']:.4f}.
- HE2 (PSSM): tau = {m2['tau']:.3f} [{m2['ic95'][0]:.3f}, {m2['ic95'][1]:.3f}], p = {m2['p']:.4f}.
- HE3 (Firth): {n_retiros} retiros; OR por DE del PSSM = {he3.loc[1, 'OR']:.2f}
  [{he3.loc[1, 'IC95_lo']:.2f}, {he3.loc[1, 'IC95_hi']:.2f}]{' -- menos de 10 retiros: solo reporte descriptivo' if n_retiros < 10 else ''}.
- Reglas del TTCT (3.6.1.5): CCI >= .75 en todas las dimensiones: {'si' if regla_a else 'no'};
  correlacion corregida maxima {r_corr_max:.2f} ({'<=' if regla_b else '>'} .85).

## 4. Robustez (B = {B_SWEEP} por escenario)

{tabla_robustez.to_markdown(index=False, floatfmt='.3f')}

## 5. HE3 (B = {B_HE3})

- Retiros esperados: {he3_mc['eventos_media']:.1f} (P5-P95: {he3_mc['eventos_p05']:.0f}-{he3_mc['eventos_p95']:.0f});
  P(< 10 retiros) = {he3_mc['prob_menos_de_10_eventos']:.2f} (binomial: {he3_mc['prob_menos_de_10_binomial']:.2f}).
- Con OR supuesto = {OR_PSSM_PERMANENCIA} por DE, el IC 95 % excluye 1 en {he3_mc['prob_IC_excluye_1']:.2f} de las cohortes.

## 6. Equivalencia modelo mixto vs. estimador rapido (B = {B_MIXTO})

{resumen_eq.to_markdown(index=False, floatfmt='.3f')}

## 7. Error tipo I de la prueba omnibus HE1 bajo H0 (B = {B_WALD})

{tipo1.to_markdown(index=False, floatfmt='.3f')}

Recomendacion para 3.8.3: en el modelo apilado, incluir el efecto aleatorio
estudiante x medicion (en lme4: `(1 | estudiante:periodo)`), o usar una covarianza
residual no estructurada entre dimensiones; sin ese termino la Wald conjunta ignora
que las cinco dimensiones se miden el mismo dia y rechaza H0 con mas frecuencia que alfa.

## 8. Archivos

- `01_descriptivos.csv`, `02_pruebas_supuestos.csv`, `03_tamanos_efecto.csv`, `04_prueba_omnibus_HE1.txt`,
  `05_modelo_mixto.txt`, `06_HE3_firth_OR.csv`, `07_verificacion_potencia_apriori.txt`,
  `08_potencia_vs_N.csv`, `09_sensibilidad_rho_d.csv`, `10_robustez_sesgos.csv`, `11_tabla12_replica.csv`,
  `12_flujo_participantes.csv`, `13_psicometria_reglas.csv`, `13b_correlaciones_linea_base_TTCT.csv`,
  `14_equivalencia_linea_base.csv`, `15_correlaciones_pssm_ttct.csv`, `16_sensibilidad_tabla20.csv`,
  `17_heterogeneidad_seccion.csv`, `18_rfi_fidelidad.csv`, `19_equivalencia_mixto_vs_rapido.csv`,
  `20_error_tipo_I_HE1.csv`, `dataset_simulado_representativo.csv`, `dashboard_data.json`.
- Figuras a-h en `../figuras/`.

## 9. Limitaciones de la simulacion (declaradas)

- Las correlaciones entre dimensiones, los efectos de periodo y forma, el OR de la permanencia y la
  confiabilidad de los calificadores son supuestos, no estimaciones empiricas.
- El PSSM se simula como puntaje global continuo; no se simulan los items ni la regla factorial
  M1/M2/M3 (3.6.2.2), la originalidad contextualizada (3.6.1.4) ni la imputacion multiple (3.8.8),
  que se ejecutaran en R (lavaan, mice) con los datos reales.
- statsmodels no ofrece Kenward-Roger: los grados de libertad del modelo mixto se aproximan con
  los de la regresion intrasujeto (n - 3).
- La escala bruta del dataset representativo es ilustrativa.
"""
        )

    print("Simulacion completa. Resultados en:", RES_DIR)
    print("Figuras en:", FIG_DIR)


if __name__ == "__main__":
    main()
