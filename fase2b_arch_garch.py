# %% [markdown]
# # Fase 2.2: Efectos ARCH y modelos GARCH
# Requiere haber corrido la Fase 1 (usa rendimientos.csv)

# %%
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

rendimientos = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
# arch_model funciona mejor con rendimientos en porcentaje (escala numérica más estable)
rendimientos_pct = rendimientos * 100

rendimientos.tail()

# %% [markdown]
# ## Prueba de efectos ARCH (univariada, por activo)
# H0: no hay efectos ARCH (varianza constante / homocedasticidad condicional)

# %%
def prueba_arch(serie, nlags=12):
    stat, pval, fstat, fpval = het_arch(serie, nlags=nlags)
    return {
        "LM_stat": stat, "LM_pval": pval,
        "F_stat": fstat, "F_pval": fpval,
        "Conclusion": "Hay efectos ARCH" if pval < 0.05 else "No hay efectos ARCH",
    }

resultados_arch = pd.DataFrame(
    {col: prueba_arch(rendimientos[col]) for col in rendimientos.columns}
).T
resultados_arch

# %% [markdown]
# ## Ajuste de GARCH(1,1) para los activos con efectos ARCH significativos
# (Ajusta la lista `activos_garch` según el resultado de la celda anterior)

# %%
activos_garch = resultados_arch[resultados_arch["LM_pval"] < 0.05].index.tolist()
print("Activos con efectos ARCH significativos:", activos_garch)

modelos_garch = {}
for activo in activos_garch:
    modelo = arch_model(rendimientos_pct[activo], vol="GARCH", p=1, q=1, dist="t")
    modelos_garch[activo] = modelo.fit(disp="off")

# %%
for activo, resultado in modelos_garch.items():
    print(f"\n===== GARCH(1,1) - {activo} =====")
    print(resultado.summary())

# %% [markdown]
# ## Resumen de parámetros GARCH (persistencia de volatilidad)
# alpha + beta cercano a 1 => volatilidad muy persistente (shocks tardan en disiparse)

# %%
resumen_garch = pd.DataFrame({
    activo: {
        "omega": res.params.get("omega"),
        "alpha[1]": res.params.get("alpha[1]"),
        "beta[1]": res.params.get("beta[1]"),
        "persistencia (a+b)": res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0),
        "nu (t-dist)": res.params.get("nu"),
    }
    for activo, res in modelos_garch.items()
}).T
resumen_garch

# %% [markdown]
# ## Volatilidad condicional estimada (para usar como insumo de riesgo en Fase 3/Markowitz)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(modelos_garch), 1, figsize=(10, 3 * len(modelos_garch)), sharex=True)
if len(modelos_garch) == 1:
    axes = [axes]

for ax, (activo, resultado) in zip(axes, modelos_garch.items()):
    ax.plot(resultado.conditional_volatility, label=f"Volatilidad condicional - {activo}")
    ax.set_title(activo)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("volatilidad_condicional_garch.png", dpi=150)
plt.show()

# %%
vol_condicional = pd.DataFrame({
    activo: resultado.conditional_volatility for activo, resultado in modelos_garch.items()
})
vol_condicional.to_csv("volatilidad_condicional_garch.csv")
resumen_garch.to_csv("resumen_garch.csv")
resultados_arch.to_csv("resultados_test_arch.csv")
print("Guardado: volatilidad_condicional_garch.csv, resumen_garch.csv, resultados_test_arch.csv")

# %%
