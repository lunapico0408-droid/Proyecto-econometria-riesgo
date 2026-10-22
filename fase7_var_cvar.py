# %% [markdown]
# # Fase 7: Valor en riesgo (VaR) y Expected Shortfall (CVaR)
# Libro guía: Moscote Flórez, *Introducción al riesgo financiero*, cap. 5.
#
# Métodos del capítulo:
#   5.2-5.4  Paramétrico (varianza-covarianza / delta normal)
#   5.5.1    Simulación histórica
#   5.5.2    Simulación bootstrap (con intervalo de confianza)
#   5.5.3    Simulación Montecarlo (rendimientos normales con la correlación observada)
#
# Se aplican a los 9 activos y a los portafolios construidos en las Fases 3, 3b y 5.
# NOTA: los portafolios de la Fase 5 rebalancean en el tiempo; aquí se usa su PESO
# PROMEDIO como portafolio estático representativo (el VaR de un portafolio con pesos fijos).

# %%
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

NIVELES = [0.95, 0.99]       # niveles de confianza (1 - alpha)
S = 100_000                  # monto invertido (USD), para expresar VaR en dinero
M_BOOT = 10_000              # réplicas bootstrap
N_MC = 100_000               # simulaciones Montecarlo
USAR_MEDIA = True            # fórmula del libro con media distinta de cero
SEED = 42
rng = np.random.default_rng(SEED)

rend_log = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rend_log.columns.tolist()
n = len(activos)

# VaR y CVaR se calculan sobre rendimientos SIMPLES (pérdida real de un portafolio)
rs = np.expm1(rend_log)
T = len(rs)
print(f"Observaciones: {T} días ({rs.index[0].date()} a {rs.index[-1].date()})")

# %% [markdown]
# ## 7.1 Portafolios a evaluar

# %%
def pesos_fase3(rend_log, rf=0.04):
    """Replica la Fase 3: mínima varianza y máximo Sharpe (largo, suma 1)."""
    mu = (rend_log.mean() * 252).values
    cov = (rend_log.cov() * 252).values
    w0 = np.repeat(1 / n, n)
    lim = [(0, 1)] * n
    rest = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    wmv = minimize(lambda w: np.sqrt(w @ cov @ w), w0, method="SLSQP", bounds=lim, constraints=rest).x
    wms = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), w0,
                   method="SLSQP", bounds=lim, constraints=rest).x
    return wmv, wms

portafolios = {}
wmv, wms = pesos_fase3(rend_log)
portafolios["Mín. varianza (Fase 3)"] = wmv
portafolios["Máx. Sharpe (Fase 3)"] = wms

if os.path.exists("pesos_portafolios_garch.csv"):
    pg = pd.read_csv("pesos_portafolios_garch.csv", index_col=0).reindex(activos)
    portafolios["Mín. varianza GARCH (Fase 3b)"] = pg["Min_Varianza_GARCH"].values
    portafolios["Máx. Sharpe GARCH (Fase 3b)"] = pg["Max_Sharpe_GARCH"].values
else:
    print("Nota: falta pesos_portafolios_garch.csv (Fase 3b); se omiten esos portafolios.")

if os.path.exists("fase5_pesos_portafolios.csv"):
    p5 = pd.read_csv("fase5_pesos_portafolios.csv", index_col=0).reindex(activos)
    for col in ["VAR(3)", "RandomForest", "XGBoost", "LSTM", "Pesos iguales (1/N)"]:
        if col in p5.columns:
            portafolios[f"{col} (Fase 5)" if col != "Pesos iguales (1/N)" else "Pesos iguales (1/N)"] = p5[col].values
else:
    print("Nota: falta fase5_pesos_portafolios.csv; se omiten los portafolios de la Fase 5.")

portafolios = {k: np.asarray(v, float) / np.sum(v) for k, v in portafolios.items()}   # suman exactamente 1

pesos_tabla = pd.DataFrame(portafolios, index=activos)
print("\nPesos de los portafolios evaluados:")
print(pesos_tabla.round(3).to_string())

# Cada portafolio se trata como una serie de rendimientos (pesos fijos sobre el histórico)
series = {a: rs[a].values for a in activos}
tipo = {a: "Activo" for a in activos}
for nombre, w in portafolios.items():
    series[nombre] = rs.values @ w
    tipo[nombre] = "Portafolio"

# %% [markdown]
# ## 7.2 Supuesto de normalidad (libro 5.2.2: las colas pesadas subestiman el VaR)

# %%
filas = []
for nombre, r in series.items():
    jb, p = stats.jarque_bera(r)
    filas.append({"Serie": nombre, "Tipo": tipo[nombre], "Asimetría": stats.skew(r),
                  "Curtosis en exceso": stats.kurtosis(r), "Jarque-Bera": jb,
                  "p-valor": p, "Rechaza normalidad (5%)": p < 0.05})
normalidad = pd.DataFrame(filas).set_index("Serie")
print("\n=== Prueba de normalidad (Jarque-Bera) ===")
print(normalidad.round(4).to_string())

# %% [markdown]
# ## 7.3 Los cuatro métodos
# VaR y CVaR se expresan como PÉRDIDA positiva, en % del monto invertido.

# %%
def var_es_normal(r, nivel):
    """Paramétrico (libro 5.2-5.4): VaR = (sigma*z - mu); CVaR = (sigma*phi(z)/alpha - mu)."""
    a = 1 - nivel
    mu = r.mean() if USAR_MEDIA else 0.0
    sd = r.std(ddof=1)
    z = stats.norm.ppf(nivel)
    return sd * z - mu, sd * stats.norm.pdf(z) / a - mu

def var_es_hist(r, nivel):
    """Simulación histórica (5.5.1): percentil empírico; CVaR = promedio de las peores pérdidas."""
    a = 1 - nivel
    m = max(1, int(np.floor(a * len(r))))
    return -np.quantile(r, a), -np.sort(r)[:m].mean()

def var_es_bootstrap(r):
    """Bootstrap (5.5.2): M submuestras con reemplazo; devuelve promedio e IC 95% del VaR."""
    Tn = len(r)
    muestras = np.sort(r[rng.integers(0, Tn, size=(M_BOOT, Tn))], axis=1)
    out = {}
    for nivel in NIVELES:
        a = 1 - nivel
        m = max(1, int(np.floor(a * Tn)))
        var_b = -np.quantile(muestras, a, axis=1)
        es_b = -muestras[:, :m].mean(axis=1)
        out[nivel] = (var_b.mean(), es_b.mean(), *np.percentile(var_b, [2.5, 97.5]))
    return out

def var_es_mc(sim, nivel):
    """Montecarlo (5.5.3): percentil de los rendimientos simulados."""
    a = 1 - nivel
    m = max(1, int(np.floor(a * len(sim))))
    return -np.quantile(sim, a), -np.sort(sim)[:m].mean()

# Montecarlo conjunto: normal multivariada con la media y covarianza muestrales
Z = rng.multivariate_normal(rs.mean().values, rs.cov().values, size=N_MC)
sim = {a: Z[:, i] for i, a in enumerate(activos)}
for nombre, w in portafolios.items():
    sim[nombre] = Z @ w

filas = []
for nombre, r in series.items():
    boot = var_es_bootstrap(r)
    for nivel in NIVELES:
        resultados = {
            "Paramétrico (normal)": var_es_normal(r, nivel) + (np.nan, np.nan),
            "Simulación histórica": var_es_hist(r, nivel) + (np.nan, np.nan),
            "Bootstrap": boot[nivel],
            "Montecarlo": var_es_mc(sim[nombre], nivel) + (np.nan, np.nan),
        }
        for metodo, (v, e, lo, hi) in resultados.items():
            filas.append({"Serie": nombre, "Tipo": tipo[nombre], "Método": metodo, "Nivel": nivel,
                          "VaR_%": v * 100, "CVaR_%": e * 100,
                          "VaR_USD": v * S, "CVaR_USD": e * S,
                          "VaR_IC95_inf_%": lo * 100, "VaR_IC95_sup_%": hi * 100})
resultados_var = pd.DataFrame(filas)

for nivel in NIVELES:
    for medida in ["VaR_%", "CVaR_%"]:
        tabla = resultados_var[resultados_var["Nivel"] == nivel].pivot(
            index="Serie", columns="Método", values=medida)
        orden = list(series.keys())
        print(f"\n=== {medida.replace('_%', '')} a 1 día, confianza {nivel:.0%} (% del monto invertido) ===")
        print(tabla.reindex(orden).round(3).to_string())

# %% [markdown]
# ## 7.4 ¿Subestima el supuesto normal? (cociente histórico / normal al 99%)
# Si es mayor que 1, la distribución normal está subestimando las pérdidas reales.

# %%
p99 = resultados_var[resultados_var["Nivel"] == 0.99]
cociente = pd.DataFrame({
    "VaR hist / VaR normal": (p99[p99["Método"] == "Simulación histórica"].set_index("Serie")["VaR_%"]
                              / p99[p99["Método"] == "Paramétrico (normal)"].set_index("Serie")["VaR_%"]),
    "CVaR hist / CVaR normal": (p99[p99["Método"] == "Simulación histórica"].set_index("Serie")["CVaR_%"]
                                / p99[p99["Método"] == "Paramétrico (normal)"].set_index("Serie")["CVaR_%"]),
}).reindex(list(series.keys()))
print(cociente.round(3).to_string())

# %% [markdown]
# ## 7.5 Efecto de la diversificación (libro 5.4)
# VaR diversificado < suma de VaR individuales ponderados. Beneficio = 1 - VaR_p / sum(w_i * VaR_i).

# %%
var_ind = {nivel: np.array([var_es_normal(rs[a].values, nivel)[0] for a in activos]) for nivel in NIVELES}
filas = []
for nombre, w in portafolios.items():
    for nivel in NIVELES:
        var_div = var_es_normal(series[nombre], nivel)[0]
        var_suma = float(w @ var_ind[nivel])
        filas.append({"Portafolio": nombre, "Nivel": nivel, "VaR diversificado_%": var_div * 100,
                      "Suma VaR individuales_%": var_suma * 100,
                      "Beneficio de diversificación": 1 - var_div / var_suma})
diversificacion = pd.DataFrame(filas)
print(diversificacion[diversificacion["Nivel"] == 0.99].set_index("Portafolio").round(4).to_string())

# %% [markdown]
# ## 7.6 Horizonte de 10 días (fórmula del libro, escalamiento por raíz del tiempo)

# %%
filas = []
for nombre, r in series.items():
    for nivel in NIVELES:
        v, e = var_es_normal(r, nivel)
        filas.append({"Serie": nombre, "Nivel": nivel, "VaR 1 día_%": v * 100,
                      "VaR 10 días_%": v * np.sqrt(10) * 100, "CVaR 10 días_%": e * np.sqrt(10) * 100})
var_10d = pd.DataFrame(filas)

# %% [markdown]
# ## 7.7 Gráficos

# %%
ref = "Pesos iguales (1/N)" if "Pesos iguales (1/N)" in series else list(portafolios)[0]
r_ref = series[ref] * 100
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(r_ref, bins=50, density=True, alpha=0.6, label="Rendimientos diarios (%)")
xs = np.linspace(r_ref.min(), r_ref.max(), 400)
ax.plot(xs, stats.norm.pdf(xs, r_ref.mean(), r_ref.std(ddof=1)), "k--", label="Normal ajustada")
colores = {"Paramétrico (normal)": "tab:red", "Simulación histórica": "tab:green",
           "Bootstrap": "tab:orange", "Montecarlo": "tab:purple"}
for metodo, c in colores.items():
    v = resultados_var[(resultados_var["Serie"] == ref) & (resultados_var["Nivel"] == 0.99)
                       & (resultados_var["Método"] == metodo)]["VaR_%"].iloc[0]
    ax.axvline(-v, color=c, lw=2, label=f"VaR 99% {metodo}: {v:.2f}%")
for metodo, c in [("Paramétrico (normal)", "tab:red"), ("Simulación histórica", "tab:green")]:
    e = resultados_var[(resultados_var["Serie"] == ref) & (resultados_var["Nivel"] == 0.99)
                       & (resultados_var["Método"] == metodo)]["CVaR_%"].iloc[0]
    ax.axvline(-e, color=c, lw=2, ls="--", label=f"CVaR 99% {metodo}: {e:.2f}%")
ax.set_title(f"Distribución de rendimientos, VaR y CVaR al 99% - {ref}")
ax.set_xlabel("Rendimiento diario (%)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fase7_distribucion_var.png", dpi=150)

tabla_g = p99.pivot(index="Serie", columns="Método", values="VaR_%").reindex(list(portafolios))
tabla_g.plot(kind="bar", figsize=(11, 5.5), width=0.8)
plt.ylabel("VaR 99% a 1 día (% del monto)"); plt.title("VaR 99% por portafolio y método")
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig("fase7_var_por_portafolio.png", dpi=150)

# %% [markdown]
# ## 7.8 Guardar resultados

# %%
resultados_var.to_csv("fase7_resultados_var_cvar.csv", index=False)
normalidad.to_csv("fase7_normalidad.csv")
cociente.to_csv("fase7_cociente_historico_normal.csv")
diversificacion.to_csv("fase7_diversificacion.csv", index=False)
var_10d.to_csv("fase7_var_10_dias.csv", index=False)
pesos_tabla.to_csv("fase7_pesos_portafolios_evaluados.csv")
print("\nGuardado: fase7_resultados_var_cvar.csv, fase7_normalidad.csv, fase7_cociente_historico_normal.csv, "
      "fase7_diversificacion.csv, fase7_var_10_dias.csv, fase7_pesos_portafolios_evaluados.csv")

# %%
