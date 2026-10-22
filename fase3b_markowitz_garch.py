# %% [markdown]
# # Fase 3b: Markowitz con volatilidad GARCH como insumo de riesgo
# Extension de la Fase 3: en vez de usar la covarianza historica simple
# (rendimientos.cov()), se construye la matriz de covarianza combinando
# la volatilidad condicional GARCH (mas reciente, prospectiva) de cada
# activo con la matriz de correlacion historica -> enfoque CCC-GARCH
# (Constant Conditional Correlation): Sigma = D * R * D
# donde D es diagonal de volatilidades y R es la correlacion.
#
# Para los activos SIN efectos ARCH significativos (ej. JohnsonJohnson en
# este portafolio), se usa su volatilidad historica simple, porque no hay
# evidencia de heterocedasticidad condicional que justifique un GARCH ahi.

# %%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

rendimientos = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
vol_condicional_garch = pd.read_csv("volatilidad_condicional_garch.csv", index_col=0, parse_dates=True)
# vol_condicional_garch esta en escala porcentual (rendimientos*100), se vuelve a escala decimal
vol_condicional_garch = vol_condicional_garch / 100

activos = rendimientos.columns.tolist()
n = len(activos)
tasa_libre_riesgo = 0.04  # misma tasa usada en Fase 3 y Fase 5

# %% [markdown]
# ## 3b.1 Construccion de la matriz de covarianza "GARCH" (CCC-GARCH)

# %%
# Volatilidad diaria por activo: GARCH (ultimo valor condicional) si tiene
# efectos ARCH significativos, historica simple si no.
vol_diaria = pd.Series(index=activos, dtype=float)
for activo in activos:
    if activo in vol_condicional_garch.columns:
        vol_diaria[activo] = vol_condicional_garch[activo].iloc[-1]
    else:
        vol_diaria[activo] = rendimientos[activo].std()

vol_anual = vol_diaria * np.sqrt(252)

# Matriz de correlacion historica (se asume constante, supuesto CCC)
R = rendimientos.corr()

# Sigma = D * R * D
D = np.diag(vol_anual.values)
cov_garch = pd.DataFrame(D @ R.values @ D, index=activos, columns=activos)

mu = rendimientos.mean() * 252  # mismo rendimiento esperado que en Fase 3 (media historica)

print("Volatilidad anualizada usada (GARCH donde aplica, historica si no):")
vol_anual.sort_values(ascending=False)

# %% [markdown]
# ## 3b.2 Funciones auxiliares (identicas a Fase 3)

# %%
def rendimiento_portafolio(w, mu):
    return np.dot(w, mu)

def riesgo_portafolio(w, cov_df):
    return np.sqrt(w.T @ cov_df.values @ w)

def sharpe_negativo(w, mu, cov_df, rf):
    r = rendimiento_portafolio(w, mu)
    s = riesgo_portafolio(w, cov_df)
    return -(r - rf) / s

restricciones = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
limites = tuple((0, 1) for _ in range(n))
w0 = np.repeat(1 / n, n)

# %% [markdown]
# ## 3b.3 Portafolios optimos con covarianza GARCH

# %%
resultado_minvar_g = minimize(
    lambda w: riesgo_portafolio(w, cov_garch),
    w0, method="SLSQP", bounds=limites, constraints=restricciones,
)
w_minvar_g = resultado_minvar_g.x

resultado_maxsharpe_g = minimize(
    sharpe_negativo, w0, args=(mu, cov_garch, tasa_libre_riesgo),
    method="SLSQP", bounds=limites, constraints=restricciones,
)
w_maxsharpe_g = resultado_maxsharpe_g.x

r_minvar_g = rendimiento_portafolio(w_minvar_g, mu)
s_minvar_g = riesgo_portafolio(w_minvar_g, cov_garch)
r_ms_g = rendimiento_portafolio(w_maxsharpe_g, mu)
s_ms_g = riesgo_portafolio(w_maxsharpe_g, cov_garch)

resumen_garch_port = pd.DataFrame({
    "Min_Varianza_GARCH": {
        "Rendimiento_esperado": r_minvar_g,
        "Riesgo_volatilidad": s_minvar_g,
        "Sharpe": (r_minvar_g - tasa_libre_riesgo) / s_minvar_g,
    },
    "Max_Sharpe_GARCH": {
        "Rendimiento_esperado": r_ms_g,
        "Riesgo_volatilidad": s_ms_g,
        "Sharpe": (r_ms_g - tasa_libre_riesgo) / s_ms_g,
    },
})
print(resumen_garch_port.round(4))

# %% [markdown]
# ## 3b.4 Comparacion directa: covarianza historica (Fase 3) vs covarianza GARCH (Fase 3b)

# %%
try:
    resumen_historico = pd.read_csv("resumen_portafolios_optimos.csv", index_col=0)
    comparacion_metodos = pd.DataFrame({
        "MinVar_Historico": resumen_historico["Min_Varianza"],
        "MinVar_GARCH": resumen_garch_port["Min_Varianza_GARCH"],
        "MaxSharpe_Historico": resumen_historico["Max_Sharpe"],
        "MaxSharpe_GARCH": resumen_garch_port["Max_Sharpe_GARCH"],
    })
    print("\n=== Comparacion Historico vs GARCH ===")
    print(comparacion_metodos.round(4))
    comparacion_metodos.to_csv("comparacion_markowitz_historico_vs_garch.csv")
except FileNotFoundError:
    print("Nota: corre primero Fase 3 (resumen_portafolios_optimos.csv) para ver la comparacion directa.")

# %% [markdown]
# ## 3b.5 Pesos y guardado de resultados

# %%
pesos_garch = pd.DataFrame({
    "Min_Varianza_GARCH": pd.Series(w_minvar_g, index=activos),
    "Max_Sharpe_GARCH": pd.Series(w_maxsharpe_g, index=activos),
}).round(4)

pesos_garch.to_csv("pesos_portafolios_garch.csv")
resumen_garch_port.to_csv("resumen_portafolios_garch.csv")
cov_garch.to_csv("matriz_covarianza_garch.csv")

print("\nGuardado: pesos_portafolios_garch.csv, resumen_portafolios_garch.csv, "
      "matriz_covarianza_garch.csv, comparacion_markowitz_historico_vs_garch.csv")

# %%
