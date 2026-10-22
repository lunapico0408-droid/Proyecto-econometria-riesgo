# %% [markdown]
# # Fase 5: Portafolios basados en predicciones (econometría vs ML)
# Backtest EX-ANTE: ninguna decisión usa información posterior al momento en que se toma.
#
# En cada fecha de rebalanceo t:
#   - retorno esperado de cada modelo = media de sus últimas `ventana_mu` predicciones
#     (todas hechas con información previa, vía el walk-forward de las Fases 4 y 4b);
#   - retorno esperado "clásico" = media histórica de los rendimientos hasta t-1;
#   - covarianza = últimos 250 días previos a t;
#   - se optimiza el portafolio de máximo Sharpe (largo, sin apalancamiento) y se
#     mantiene hasta el siguiente rebalanceo, evaluándolo con rendimientos REALES.
# Referencia adicional: portafolio de pesos iguales (1/N).
#
# NOTA METODOLÓGICA: una versión anterior de esta fase promediaba las predicciones de
# TODO el periodo de prueba para fijar pesos que luego se evaluaban en ese mismo periodo.
# Eso filtra información futura y sobrestimaba a RF y XGBoost; este diseño lo corrige.

# %%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASA_LIBRE_RIESGO_ANUAL = 0.04          # misma que Fase 3
RF_DIARIO = (1 + TASA_LIBRE_RIESGO_ANUAL) ** (1 / 252) - 1
VENTANA_COV = 250

# (rebalanceo cada k días, ventana de suavizado de predicciones). El PRIMERO es el caso base,
# fijado por criterio (rotación razonable), no por resultados. Los demás son sensibilidad.
DISENOS = [(5, 5), (1, 1), (5, 20), (21, 20)]

rendimientos = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rendimientos.columns.tolist()
n = len(activos)

predicciones = {
    "VAR(3)": pd.read_csv("predicciones_var.csv", index_col=0, parse_dates=True),
    "RandomForest": pd.read_csv("predicciones_rf.csv", index_col=0, parse_dates=True),
    "XGBoost": pd.read_csv("predicciones_xgb.csv", index_col=0, parse_dates=True),
    "LSTM": pd.read_csv("predicciones_lstm.csv", index_col=0, parse_dates=True),
}
fechas_test = predicciones["VAR(3)"].index
estrategias = ["Clásico (media histórica)", "VAR(3)", "RandomForest", "XGBoost", "LSTM", "Pesos iguales (1/N)"]

# %% [markdown]
# ## 5.1 Optimización
# Máximo Sharpe; si ningún activo supera la tasa libre de riesgo, mínima varianza.

# %%
LIMITES = [(0, 1)] * n
RESTR = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

def min_varianza(cov, w0):
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=LIMITES, constraints=RESTR)
    return res.x if res.success else w0

def max_sharpe(mu, cov, w0):
    if mu.max() <= RF_DIARIO:
        return min_varianza(cov, w0)
    f = lambda w: -(w @ mu - RF_DIARIO) / np.sqrt(w @ cov @ w)
    res = minimize(f, w0, method="SLSQP", bounds=LIMITES, constraints=RESTR)
    return res.x if res.success else w0

# %% [markdown]
# ## 5.2 Backtest ex-ante

# %%
def backtest(rebal_cada, ventana_mu):
    pesos = {e: [] for e in estrategias}
    w_act = {e: np.repeat(1 / n, n) for e in estrategias}
    for i, t in enumerate(fechas_test):
        if i % rebal_cada == 0:
            hist = rendimientos.iloc[:rendimientos.index.get_loc(t)]   # solo hasta t-1
            cov = hist.iloc[-VENTANA_COV:].cov().values
            for e in estrategias:
                if e == "Pesos iguales (1/N)":
                    w_act[e] = np.repeat(1 / n, n)
                elif e == "Clásico (media histórica)":
                    w_act[e] = max_sharpe(hist.mean().values, cov, w_act[e])
                else:
                    mu = predicciones[e].loc[:t].iloc[-ventana_mu:].mean().values
                    w_act[e] = max_sharpe(mu, cov, w_act[e])
        for e in estrategias:
            pesos[e].append(w_act[e])
    pesos_df = {e: pd.DataFrame(pesos[e], index=fechas_test, columns=activos) for e in estrategias}
    r_simple = np.expm1(rendimientos.loc[fechas_test])
    retornos = pd.DataFrame({e: (pesos_df[e] * r_simple).sum(axis=1) for e in estrategias})
    return retornos, pesos_df

def metricas(retornos, pesos_df):
    filas = {}
    for e in estrategias:
        r = retornos[e]
        filas[e] = {
            "Retorno anualizado": r.mean() * 252,
            "Volatilidad anualizada": r.std() * np.sqrt(252),
            "Sharpe anualizado": (r.mean() - RF_DIARIO) / r.std() * np.sqrt(252),
            "Retorno acumulado (test)": (1 + r).prod() - 1,
            "Rotación diaria promedio": pesos_df[e].diff().abs().sum(axis=1).mean(),
        }
    return pd.DataFrame(filas).T

# %% [markdown]
# ## 5.3 Caso base

# %%
rebal_base, vmu_base = DISENOS[0]
retornos_base, pesos_base = backtest(rebal_base, vmu_base)
resultados_base = metricas(retornos_base, pesos_base)
print(f"=== CASO BASE: rebalanceo cada {rebal_base} días, media de las últimas {vmu_base} predicciones ===")
print(resultados_base.round(4).to_string())

pesos_promedio = pd.DataFrame({e: pesos_base[e].mean() for e in estrategias})
print("\n=== Peso promedio por activo (caso base) ===")
print(pesos_promedio.round(3).to_string())

# %% [markdown]
# ## 5.4 Sensibilidad al diseño del backtest
# Si el orden de las estrategias cambia con el diseño, la conclusión no es robusta.

# %%
sens = {}
for rebal, vmu in DISENOS:
    ret, pes = backtest(rebal, vmu)
    sens[f"rebal {rebal}d / media {vmu} pred."] = metricas(ret, pes)["Sharpe anualizado"]
sensibilidad = pd.DataFrame(sens)
print("\n=== Sharpe anualizado según diseño del backtest ===")
print(sensibilidad.round(3).to_string())

# Referencia de incertidumbre: con T años de datos, el error estándar aproximado del Sharpe
# anualizado es sqrt((1 + 0.5*SR^2) / T). Con 150 días (T ~ 0.6) y SR ~ 1, ronda 1.6:
T_anios = len(fechas_test) / 252
print(f"\nError estándar aproximado de un Sharpe de 1.0 con {len(fechas_test)} días: "
      f"{np.sqrt((1 + 0.5) / T_anios):.2f}")

# %%
plt.figure(figsize=(9, 5))
for e in estrategias:
    plt.plot((1 + retornos_base[e]).cumprod(), label=e)
plt.legend()
plt.ylabel("Valor del portafolio (base 1)")
plt.title("Crecimiento acumulado fuera de muestra (backtest ex-ante, caso base)")
plt.tight_layout()
plt.savefig("fase5_crecimiento_portafolios.png", dpi=150)

resultados_base.to_csv("fase5_resultados_portafolios.csv")
sensibilidad.to_csv("fase5_sensibilidad_portafolios.csv")
pesos_promedio.to_csv("fase5_pesos_portafolios.csv")
retornos_base.to_csv("fase5_retornos_portafolios.csv")
print("\nGuardado: fase5_resultados_portafolios.csv, fase5_sensibilidad_portafolios.csv, "
      "fase5_pesos_portafolios.csv, fase5_retornos_portafolios.csv")

# %%
