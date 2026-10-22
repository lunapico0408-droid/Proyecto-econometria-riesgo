# %% [markdown]
# # Fase 6: Volatilidad, riesgo sistemático y CAPM
# Libro guía: Moscote Flórez, *Introducción al riesgo financiero*.
#   Cap. 3  Rendimiento: hechos estilizados (3.3) y volatilidad (3.5):
#           histórica, ventanas móviles, suavizamiento exponencial EWMA (RiskMetrics), GARCH.
#   Cap. 4  Riesgo sistemático y no sistemático: beta, beta de un portafolio,
#           descomposición del riesgo y modelo CAPM.
# Se usan log-rendimientos (como en el cap. 3 del libro). El mercado es el SPY
# (columna SP500_Benchmark).

# %%
import os
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS   # (rutas directas: evitan statsmodels.api, bloqueado en Windows)
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.optimize import minimize
from arch import arch_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

TASA_LIBRE_RIESGO_ANUAL = 0.04      # misma que Fases 3 y 5
LAMBDA = 0.94                       # RiskMetrics (libro 3.5.3: entre 0.90 y 0.94)
VENTANA = 20                        # ventana móvil (libro, ejemplo 3.2)
DIAS_PRUEBA = 250                   # mismo esquema del libro para backtesting (cap. 6)
MERCADO = "SP500_Benchmark"

rend = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rend.columns.tolist()
n = len(activos)
T = len(rend)
T_EST = T - DIAS_PRUEBA
print(f"{T} observaciones; estimación: primeras {T_EST}; prueba: últimas {DIAS_PRUEBA}")

# %% [markdown]
# ## 6.1 Hechos estilizados (libro 3.3)
# Exceso de curtosis, agrupamiento de volatilidad (autocorrelación de r^2) y efecto leverage
# (rendimiento negativo hoy, mayor volatilidad mañana: correlación entre r_t y r_{t+1}^2 < 0).

# %%
filas = []
for a in activos:
    r = rend[a]
    r2 = r ** 2
    lb = acorr_ljungbox(r2, lags=[10], return_df=True)
    filas.append({
        "Activo": a,
        "Curtosis en exceso": r.kurt(),
        "Autocorr. de r² (rezago 1)": r2.autocorr(1),
        "Ljung-Box sobre r² (10 rezagos), p-valor": lb["lb_pvalue"].iloc[0],
        "Agrupamiento de volatilidad (5%)": lb["lb_pvalue"].iloc[0] < 0.05,
        "Corr(r_t, r²_t+1) (leverage)": np.corrcoef(r.values[:-1], r2.values[1:])[0, 1],
    })
hechos = pd.DataFrame(filas).set_index("Activo")
print(hechos.round(4).to_string())

# %% [markdown]
# ## 6.2 Métodos de volatilidad (libro 3.5)
# Pronóstico de varianza para el día t con información hasta t-1:
#   Histórica (expansiva):  varianza de todos los rendimientos previos
#   Ventana móvil:          varianza de los últimos 20 días
#   EWMA (RiskMetrics):     s2_t = lambda * s2_{t-1} + (1 - lambda) * r_{t-1}^2
#   GARCH(1,1) t-Student:   parámetros estimados con las primeras T_EST observaciones
#                           y filtrados hacia adelante (sin reestimar) en el periodo de prueba

# %%
def var_historica(r):
    return r.expanding(min_periods=30).var().shift(1)

def var_movil(r, m=VENTANA):
    return r.rolling(m).var().shift(1)

def var_ewma(r, lam=LAMBDA):
    f = np.full(len(r), np.nan)
    f[0] = r.iloc[:30].var()                      # varianza inicial
    for t in range(1, len(r)):
        f[t] = lam * f[t - 1] + (1 - lam) * r.iloc[t - 1] ** 2
    return pd.Series(f, index=r.index)

def var_garch_fuera_de_muestra(r, t_est):
    """Ajusta con r[:t_est] y filtra hacia adelante con parámetros fijos. Devuelve varianza (decimal)."""
    rp = r * 100
    res = arch_model(rp.iloc[:t_est], vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    mu, om, al, be = (res.params["mu"], res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"])
    s2 = res.conditional_volatility.iloc[-1] ** 2
    e_prev = rp.iloc[t_est - 1] - mu
    out = []
    for t in range(t_est, len(rp)):
        s2 = om + al * e_prev ** 2 + be * s2      # pronóstico para el día t
        out.append(s2)
        e_prev = rp.iloc[t] - mu
    return pd.Series(np.array(out) / 1e4, index=r.index[t_est:]), res

pron = {m: {} for m in ["Histórica", "Ventana móvil (20)", "EWMA (0.94)", "GARCH(1,1)-t"]}
garch_fit = {}
for a in activos:
    r = rend[a]
    pron["Histórica"][a] = var_historica(r).iloc[T_EST:]
    pron["Ventana móvil (20)"][a] = var_movil(r).iloc[T_EST:]
    pron["EWMA (0.94)"][a] = var_ewma(r).iloc[T_EST:]
    pron["GARCH(1,1)-t"][a], garch_fit[a] = var_garch_fuera_de_muestra(r, T_EST)
pron = {m: pd.DataFrame(d) for m, d in pron.items()}

# %% [markdown]
# ## 6.3 ¿Qué método pronostica mejor la varianza? (últimos 250 días)
# Pérdida QLIKE = log(s2) + r^2 / s2 (menor es mejor; penaliza pronósticos de varianza
# demasiado bajos, que son los que subestiman el riesgo).

# %%
r_prueba = rend.iloc[T_EST:]
qlike = pd.DataFrame({m: (np.log(pron[m]) + r_prueba ** 2 / pron[m]).mean() for m in pron})
print("QLIKE promedio por activo y método (menor = mejor):")
print(qlike.round(4).to_string())
resumen_vol = pd.DataFrame({
    "QLIKE promedio (9 activos)": qlike.mean(),
    "Activos donde es el mejor": qlike.idxmin(axis=1).value_counts().reindex(qlike.columns).fillna(0).astype(int),
}).sort_values("QLIKE promedio (9 activos)")
print("\nResumen por método:")
print(resumen_vol.to_string())

# %% [markdown]
# ## 6.4 Volatilidad actual (anualizada) según cada método

# %%
filas = {}
for a in activos:
    r = rend[a]
    ew = LAMBDA * var_ewma(r).iloc[-1] + (1 - LAMBDA) * r.iloc[-1] ** 2
    res_full = arch_model(r * 100, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    g = res_full.forecast(horizon=1).variance.iloc[-1, 0] / 1e4
    filas[a] = {
        "Histórica (toda la muestra)": r.std(ddof=1) * np.sqrt(252),
        "Ventana móvil (20)": np.sqrt(r.iloc[-VENTANA:].var(ddof=1)) * np.sqrt(252),
        "EWMA (0.94)": np.sqrt(ew) * np.sqrt(252),
        "GARCH(1,1)-t": np.sqrt(g) * np.sqrt(252),
    }
vol_actual = pd.DataFrame(filas).T
print(vol_actual.round(4).to_string())

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, a in zip(axes, ["WTI_Oil", "Apple", MERCADO]):
    r = rend[a]
    ax.plot(np.sqrt(var_movil(r)) * np.sqrt(252), lw=1, label="Ventana móvil (20)")
    ax.plot(np.sqrt(var_ewma(r)) * np.sqrt(252), lw=1.3, label="EWMA (0.94)")
    rg = arch_model(r * 100, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    ax.plot(rg.conditional_volatility / 100 * np.sqrt(252), lw=1.3, label="GARCH(1,1)-t")
    ax.axhline(r.std(ddof=1) * np.sqrt(252), color="k", ls="--", lw=1, label="Histórica")
    ax.set_title(a); ax.set_ylabel("Volatilidad anualizada")
axes[0].legend(fontsize=8, ncol=4)
plt.tight_layout(); plt.savefig("fase6_volatilidad_metodos.png", dpi=150)

# %% [markdown]
# ## 6.5 Beta y descomposición del riesgo (libro cap. 4)
# Modelo de mercado: r_i = alpha + beta * r_m + e_i  =>  var_i = beta^2 var_m + var_e.
# R^2 = proporción de riesgo sistemático; 1 - R^2 = riesgo no sistemático (diversificable).

# %%
mercado = rend[MERCADO]
var_m = mercado.var(ddof=1)
filas = []
for a in activos:
    if a == MERCADO:
        continue
    res = OLS(rend[a], add_constant(mercado)).fit()
    b = res.params[MERCADO]
    var_i = rend[a].var(ddof=1)
    filas.append({
        "Activo": a, "Beta": b, "Error estándar beta": res.bse[MERCADO],
        "p-valor beta": res.pvalues[MERCADO], "Alfa diario": res.params["const"],
        "R² (riesgo sistemático)": res.rsquared, "Riesgo no sistemático (1-R²)": 1 - res.rsquared,
        "Vol. total anual": np.sqrt(var_i * 252),
        "Vol. sistemática anual": abs(b) * np.sqrt(var_m * 252),
        "Vol. específica anual": np.sqrt(res.mse_resid * 252),
        "Clasificación": ("Beta no distinto de cero (5%)" if res.pvalues[MERCADO] >= 0.05
                          else "Agresivo (beta > 1)" if b > 1
                          else "Defensivo (0 < beta < 1)" if b > 0
                          else "Beta negativo"),
    })
betas = pd.DataFrame(filas).set_index("Activo")
print(betas.round(4).to_string())

# %% [markdown]
# ## 6.6 Beta de un portafolio (libro 4.3): beta_p = sum(w_i * beta_i)
# Se evalúan los mismos portafolios de la Fase 7 (el SPY entra como un activo más, con beta 1).

# %%
def pesos_fase3(rf=TASA_LIBRE_RIESGO_ANUAL):
    mu = (rend.mean() * 252).values
    cov = (rend.cov() * 252).values
    w0 = np.repeat(1 / n, n)
    lim = [(0, 1)] * n
    rest = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    wmv = minimize(lambda w: np.sqrt(w @ cov @ w), w0, method="SLSQP", bounds=lim, constraints=rest).x
    wms = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), w0, method="SLSQP", bounds=lim, constraints=rest).x
    return wmv, wms

portafolios = {}
portafolios["Mín. varianza (Fase 3)"], portafolios["Máx. Sharpe (Fase 3)"] = pesos_fase3()
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
            portafolios[col if col == "Pesos iguales (1/N)" else f"{col} (Fase 5)"] = p5[col].values
else:
    print("Nota: falta fase5_pesos_portafolios.csv; se omiten los portafolios de la Fase 5.")
portafolios = {k: np.asarray(v, float) / np.sum(v) for k, v in portafolios.items()}

beta_activo = pd.Series({a: (1.0 if a == MERCADO else betas.loc[a, "Beta"]) for a in activos})
filas = []
for nombre, w in portafolios.items():
    r_p = rend.values @ w
    res = OLS(r_p, add_constant(mercado.values)).fit()
    var_p = np.var(r_p, ddof=1)
    filas.append({
        "Portafolio": nombre, "Beta (suma ponderada)": float(w @ beta_activo.values),
        "Beta (regresión directa)": res.params[1], "R² (riesgo sistemático)": res.rsquared,
        "Riesgo no sistemático (1-R²)": 1 - res.rsquared,
        "Vol. total anual": np.sqrt(var_p * 252),
        "Peso en SPY": w[activos.index(MERCADO)],
    })
beta_port = pd.DataFrame(filas).set_index("Portafolio")
print(beta_port.round(4).to_string())

# %% [markdown]
# ## 6.7 Modelo CAPM (libro 4.5): E[r_i] = r_f + beta_i * (E[r_m] - r_f)
# Alfa de Jensen = rendimiento histórico - rendimiento CAPM. Con solo 3 años, el
# rendimiento medio del mercado es una estimación muy ruidosa: interpretar con cautela.

# %%
mu_m = mercado.mean() * 252
filas = []
for a in activos:
    if a == MERCADO:
        continue
    b = betas.loc[a, "Beta"]
    capm = TASA_LIBRE_RIESGO_ANUAL + b * (mu_m - TASA_LIBRE_RIESGO_ANUAL)
    hist = rend[a].mean() * 252
    filas.append({"Activo": a, "Beta": b, "Rend. histórico anual": hist,
                  "Rend. esperado CAPM anual": capm, "Alfa de Jensen anual": hist - capm})
capm_tab = pd.DataFrame(filas).set_index("Activo")
print(f"Rendimiento anual del mercado ({MERCADO}): {mu_m:.4f}; tasa libre de riesgo: {TASA_LIBRE_RIESGO_ANUAL:.2%}")
print(capm_tab.round(4).to_string())

# %%
plt.figure(figsize=(8, 6))
bs = np.linspace(0, max(1.3, capm_tab["Beta"].max() + 0.1), 50)
plt.plot(bs, TASA_LIBRE_RIESGO_ANUAL + bs * (mu_m - TASA_LIBRE_RIESGO_ANUAL), "k--", label="Línea del mercado (CAPM)")
plt.scatter(capm_tab["Beta"], capm_tab["Rend. histórico anual"], s=60)
for a, fila in capm_tab.iterrows():
    plt.annotate(a, (fila["Beta"], fila["Rend. histórico anual"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
plt.scatter([1], [mu_m], marker="*", s=200, color="red", label=MERCADO)
plt.xlabel("Beta"); plt.ylabel("Rendimiento anual histórico")
plt.title("Rendimiento histórico vs. CAPM"); plt.legend(); plt.tight_layout()
plt.savefig("fase6_capm_linea_mercado.png", dpi=150)

# %% [markdown]
# ## 6.8 Guardar resultados

# %%
hechos.to_csv("fase6_hechos_estilizados.csv")
qlike.to_csv("fase6_evaluacion_volatilidad_qlike.csv")
resumen_vol.to_csv("fase6_resumen_metodos_volatilidad.csv")
vol_actual.to_csv("fase6_volatilidad_actual.csv")
betas.to_csv("fase6_beta_activos.csv")
beta_port.to_csv("fase6_beta_portafolios.csv")
capm_tab.to_csv("fase6_capm.csv")
for m, df in pron.items():
    nombre = {"Histórica": "historica", "Ventana móvil (20)": "movil20", "EWMA (0.94)": "ewma", "GARCH(1,1)-t": "garch"}[m]
    df.to_csv(f"fase6_varianza_pronosticada_{nombre}.csv")
print("\nGuardado: fase6_*.csv y fase6_*.png")

# %%
