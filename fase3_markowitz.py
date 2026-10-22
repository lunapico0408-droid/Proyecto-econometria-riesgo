# %% [markdown]
# # Fase 3: Construcción de portafolio (Markowitz)

# %%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

rendimientos = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rendimientos.columns.tolist()
n = len(activos)

# Insumos anualizados (252 días hábiles)
mu = rendimientos.mean() * 252            # rendimiento esperado por activo
cov = rendimientos.cov() * 252            # matriz de covarianza anualizada

tasa_libre_riesgo = 0.04  # ajusta según el T-Bill/CDT que uses como referencia

mu

# %% [markdown]
# ## Funciones auxiliares: rendimiento, riesgo y Sharpe de un portafolio

# %%
def rendimiento_portafolio(w, mu):
    return np.dot(w, mu)

def riesgo_portafolio(w, cov):
    return np.sqrt(w.T @ cov.values @ w)

def sharpe_negativo(w, mu, cov, rf):
    r = rendimiento_portafolio(w, mu)
    s = riesgo_portafolio(w, cov)
    return -(r - rf) / s

restricciones = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
limites = tuple((0, 1) for _ in range(n))  # long-only
w0 = np.repeat(1 / n, n)  # punto inicial: pesos iguales

# %% [markdown]
# ## 3.1 Portafolio de mínima varianza

# %%
resultado_minvar = minimize(
    lambda w: riesgo_portafolio(w, cov),
    w0, method="SLSQP", bounds=limites, constraints=restricciones,
)
w_minvar = resultado_minvar.x

pesos_minvar = pd.Series(w_minvar, index=activos).sort_values(ascending=False)
print(f"Rendimiento esperado: {rendimiento_portafolio(w_minvar, mu):.4f}")
print(f"Riesgo (volatilidad): {riesgo_portafolio(w_minvar, cov):.4f}")
pesos_minvar

# %% [markdown]
# ## 3.2 Portafolio de máximo Sharpe (tangente)

# %%
resultado_maxsharpe = minimize(
    sharpe_negativo, w0, args=(mu, cov, tasa_libre_riesgo),
    method="SLSQP", bounds=limites, constraints=restricciones,
)
w_maxsharpe = resultado_maxsharpe.x

pesos_maxsharpe = pd.Series(w_maxsharpe, index=activos).sort_values(ascending=False)
r_ms = rendimiento_portafolio(w_maxsharpe, mu)
s_ms = riesgo_portafolio(w_maxsharpe, cov)
print(f"Rendimiento esperado: {r_ms:.4f}")
print(f"Riesgo (volatilidad): {s_ms:.4f}")
print(f"Sharpe ratio: {(r_ms - tasa_libre_riesgo) / s_ms:.4f}")
pesos_maxsharpe

# %% [markdown]
# ## 3.3 Frontera eficiente

# %%
objetivos_retorno = np.linspace(mu.min(), mu.max(), 50)
riesgos_frontera = []

for objetivo in objetivos_retorno:
    restricciones_frontera = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, obj=objetivo: rendimiento_portafolio(w, mu) - obj},
    ]
    res = minimize(
        lambda w: riesgo_portafolio(w, cov),
        w0, method="SLSQP", bounds=limites, constraints=restricciones_frontera,
    )
    riesgos_frontera.append(res.fun if res.success else np.nan)

frontera = pd.DataFrame({"Riesgo": riesgos_frontera, "Retorno": objetivos_retorno}).dropna()

# %% [markdown]
# ## 3.4 Gráfico: frontera eficiente + activos individuales + portafolios óptimos

# %%
plt.figure(figsize=(10, 7))
plt.plot(frontera["Riesgo"], frontera["Retorno"], "b-", label="Frontera eficiente", linewidth=2)

# Activos individuales
for activo in activos:
    plt.scatter(np.sqrt(cov.loc[activo, activo]), mu[activo], marker="o", s=60)
    plt.annotate(activo, (np.sqrt(cov.loc[activo, activo]), mu[activo]), fontsize=8, xytext=(5, 5), textcoords="offset points")

# Portafolios óptimos
plt.scatter(riesgo_portafolio(w_minvar, cov), rendimiento_portafolio(w_minvar, mu),
            color="green", marker="*", s=300, label="Mínima varianza", zorder=5)
plt.scatter(s_ms, r_ms, color="red", marker="*", s=300, label="Máximo Sharpe", zorder=5)

plt.xlabel("Riesgo (volatilidad anualizada)")
plt.ylabel("Rendimiento esperado (anualizado)")
plt.title("Frontera eficiente de Markowitz")
plt.legend()
plt.tight_layout()
plt.savefig("frontera_eficiente.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3.5 Comparación de pesos entre los dos portafolios óptimos

# %%
comparacion = pd.DataFrame({
    "Min_Varianza": pesos_minvar,
    "Max_Sharpe": pesos_maxsharpe,
}).round(4)
comparacion

# %%
comparacion.to_csv("pesos_portafolios_optimos.csv")
frontera.to_csv("frontera_eficiente.csv", index=False)

resumen_portafolios = pd.DataFrame({
    "Min_Varianza": {
        "Rendimiento_esperado": rendimiento_portafolio(w_minvar, mu),
        "Riesgo_volatilidad": riesgo_portafolio(w_minvar, cov),
        "Sharpe": (rendimiento_portafolio(w_minvar, mu) - tasa_libre_riesgo) / riesgo_portafolio(w_minvar, cov),
    },
    "Max_Sharpe": {
        "Rendimiento_esperado": r_ms,
        "Riesgo_volatilidad": s_ms,
        "Sharpe": (r_ms - tasa_libre_riesgo) / s_ms,
    },
})
resumen_portafolios.loc["Tasa_libre_riesgo_anual"] = tasa_libre_riesgo
resumen_portafolios.to_csv("resumen_portafolios_optimos.csv")

print("Guardado: pesos_portafolios_optimos.csv, frontera_eficiente.csv, resumen_portafolios_optimos.csv")

# %%
