
# %% [markdown]
# # Fase 1: Construcción y exploración de la base de datos
# Portafolio diversificado datos diarios, últimos 3 años
#
# Requisitos: pip install yfinance pandas numpy matplotlib seaborn statsmodels

# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from statsmodels.tsa.stattools import adfuller, kpss

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_style("whitegrid")

# %% [markdown]
# ## 1.1 Definición del portafolio y descarga de precios

# %%
tickers = {
    "CL=F": "WTI_Oil",
    "AAPL": "Apple",
    "JPM": "JPMorgan",
    "KO": "CocaCola",
    "JNJ": "JohnsonJohnson",
    "CAT": "Caterpillar",
    "NEE": "NextEraEnergy",
    "GLD": "Oro",
    "SPY": "SP500_Benchmark",
}

periodo_anios = 3
end_date = pd.Timestamp("2026-10-06")
start_date = end_date - pd.DateOffset(years=periodo_anios)

raw = yf.download(
    list(tickers.keys()),
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

precios = raw["Close"].rename(columns=tickers)
precios = precios.dropna(how="all")
# Rellenar pequeños huecos (feriados no coincidentes entre mercados de commodities y equities)
precios = precios.ffill().dropna()

print(precios.shape)
precios.tail()

# %% [markdown]
# ## 1.2 Transformación a rendimientos logarítmicos

# %%
log_precios = np.log(precios)
rendimientos = log_precios.diff().dropna()

rendimientos.describe().T

# %% [markdown]
# ## 1.3 Medidas de rentabilidad y riesgo (anualizadas, 252 días hábiles)

# %%
resumen = pd.DataFrame({
    "Rend_medio_diario": rendimientos.mean(),
    "Rend_anualizado": rendimientos.mean() * 252,
    "Vol_anualizada": rendimientos.std() * np.sqrt(252),
    "Skewness": rendimientos.skew(),
    "Curtosis": rendimientos.kurt(),
})
resumen["Sharpe_aprox"] = resumen["Rend_anualizado"] / resumen["Vol_anualizada"]
resumen.sort_values("Sharpe_aprox", ascending=False)

# %% [markdown]
# ## 1.4 Matrices de correlación y covarianza

# %%
corr = rendimientos.corr()
cov = rendimientos.cov() * 252  # covarianza anualizada

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación de rendimientos")
plt.tight_layout()
plt.savefig("correlacion_rendimientos.png", dpi=150)
plt.show()

# %% [markdown]
# ## 1.5 Pruebas de estacionariedad (ADF y KPSS)
# Se prueban precios en logaritmo (se espera I(1): no estacionarios)
# y rendimientos (se espera I(0): estacionarios).

# %%
def prueba_estacionariedad(serie, nombre):
    adf_stat, adf_p, *_ = adfuller(serie.dropna(), autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(serie.dropna(), regression="c", nlags="auto")
    return {
        "Serie": nombre,
        "ADF_stat": adf_stat, "ADF_p": adf_p,
        "ADF_concl": "No estacionaria" if adf_p > 0.05 else "Estacionaria",
        "KPSS_stat": kpss_stat, "KPSS_p": kpss_p,
        "KPSS_concl": "Estacionaria" if kpss_p > 0.05 else "No estacionaria",
    }

resultados_precios = pd.DataFrame([
    prueba_estacionariedad(log_precios[col], col) for col in log_precios.columns
])
resultados_rend = pd.DataFrame([
    prueba_estacionariedad(rendimientos[col], col) for col in rendimientos.columns
])

print("=== Precios en log ===")
display(resultados_precios)
print("\n=== Rendimientos ===")
display(resultados_rend)

# %% [markdown]
# ## 1.6 Guardar datasets para las siguientes fases

# %%
precios.to_csv("precios.csv")
rendimientos.to_csv("rendimientos.csv")
resumen.to_csv("resumen_rentabilidad_riesgo.csv")
corr.to_csv("matriz_correlacion.csv")
cov.to_csv("matriz_covarianza.csv")

print("Archivos guardados: precios.csv, rendimientos.csv, resumen_rentabilidad_riesgo.csv, matriz_correlacion.csv, matriz_covarianza.csv")
# %%
