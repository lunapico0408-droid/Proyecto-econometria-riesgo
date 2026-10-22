# %% [markdown]
# # Fase 2.1: Cointegración (Johansen) y modelo VAR/VEC

# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.vector_ar.var_model import VAR   # (ruta directa: evita statsmodels.tsa.api, bloqueado en Windows)
from statsmodels.stats.diagnostic import acorr_ljungbox

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

precios = pd.read_csv("precios.csv", index_col=0, parse_dates=True)
log_precios = np.log(precios)
rendimientos = log_precios.diff().dropna()

log_precios.tail()

# %% [markdown]
# ## Prueba de cointegración de Johansen (sobre las 9 series en log)
# det_order=0: constante dentro de la relación de cointegración (caso estándar)
# k_ar_diff: rezagos en diferencias -> se prueban unos pocos valores razonables

# %%
def resumen_johansen(log_df, k_ar_diff=1):
    resultado = coint_johansen(log_df, det_order=0, k_ar_diff=k_ar_diff)
    n = log_df.shape[1]
    tabla = pd.DataFrame({
        "r_hipotesis": [f"r <= {i}" for i in range(n)],
        "Traza_stat": resultado.lr1,
        "CV_90%": resultado.cvt[:, 0],
        "CV_95%": resultado.cvt[:, 1],
        "CV_99%": resultado.cvt[:, 2],
    })
    tabla["Rechaza_H0_5%"] = tabla["Traza_stat"] > tabla["CV_95%"]
    return tabla

tabla_johansen = resumen_johansen(log_precios, k_ar_diff=1)
tabla_johansen

# %% [markdown]
# **Cómo leer esta tabla:** empieza en la fila `r <= 0`. Si `Rechaza_H0_5%` es True,
# hay al menos 1 relación de cointegración y sigues a la fila `r <= 1`, y así
# sucesivamente hasta la primera fila donde sea False. El número de filas True
# antes de la primera False es el rango de cointegración `r`.
#
# - Si **r = 0** → no hay cointegración → seguir con **VAR en rendimientos** (sección A)
# - Si **r > 0** → hay cointegración → seguir con **VEC** (sección B)

# %% [markdown]
# ## Sección A: VAR en rendimientos (si r = 0)

# %%
modelo_var_sel = VAR(rendimientos)
seleccion = modelo_var_sel.select_order(maxlags=10)
print(seleccion.summary())

# %%
p_optimo = seleccion.aic
print(f"Rezago óptimo por AIC: {p_optimo}")

p_manual = 3  # fijamos manualmente porque AIC con 9 variables da 0
var_fit = modelo_var_sel.fit(p_manual)
print(var_fit.summary())

# %% [markdown]
# ### Diagnósticos del VAR: autocorrelación residual y estabilidad

# %%
# Estabilidad (statsmodels ya aplica la convención correcta internamente)
estable = var_fit.is_stable(verbose=True)
print("Modelo estable:", estable)

# %%
# Autocorrelación residual (Portmanteau / Ljung-Box multivariado vía statsmodels)
resultado_whiteness = var_fit.test_whiteness(nlags=12)
print(resultado_whiteness.summary())

# %% [markdown]
# ## Sección B: VEC — NO APLICA
# El test de Johansen (tabla arriba) dio r = 0 (Rechaza_H0_5% = False en la
# fila `r <= 0`): no hay cointegración entre los 9 activos. Un VECM requiere
# r > 0 para que exista el término de corrección de error; con r = 0 el modelo
# correcto es el VAR en rendimientos de la Sección A (ya ajustado arriba).
# Se deja el código comentado solo como referencia metodológica, NO se ejecuta.

# %%
# r_manual = 1  # NO APLICA: Johansen dio r = 0 para este portafolio
# vec_fit = VECM(log_precios, k_ar_diff=1, coint_rank=r_manual, deterministic="ci")
# vec_resultado = vec_fit.fit()
# print(vec_resultado.summary())

# %% [markdown]
# ## Causalidad de Granger (bidireccional, par a par)
# Usa el VAR en rendimientos (ajusta según A o B según lo que aplique)

# %%
def matriz_granger(var_resultado, variables):
    n = len(variables)
    pvals = pd.DataFrame(np.ones((n, n)), index=variables, columns=variables)
    for causa in variables:
        for efecto in variables:
            if causa == efecto:
                continue
            test = var_resultado.test_causality(efecto, [causa], kind="f")
            pvals.loc[causa, efecto] = test.pvalue
    return pvals

pvals_granger = matriz_granger(var_fit, rendimientos.columns.tolist())
print("P-valores de causalidad de Granger (fila causa -> columna efecto)")
pvals_granger.round(4)

# %%
significativos = pvals_granger < 0.05
print("Relaciones de causalidad significativas al 5% (True = causa Granger):")
significativos

# %% [markdown]
# ## Guardar resultados de la Fase 2

# %%
tabla_johansen.to_csv("resultados_johansen.csv", index=False)
pvals_granger.to_csv("pvalores_granger.csv")
significativos.to_csv("granger_significativos.csv")

with open("resumen_var.txt", "w") as f:
    f.write(f"Rezago óptimo (AIC, forzado manualmente): {p_manual}\n")
    f.write(f"Modelo estable: {estable}\n\n")
    f.write(str(var_fit.summary()))

print("Guardado: resultados_johansen.csv, pvalores_granger.csv, granger_significativos.csv, resumen_var.txt")

# %%
