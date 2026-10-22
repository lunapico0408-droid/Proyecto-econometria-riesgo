# %% [markdown]
# # Fase 9: Pruebas de estrés (stress testing)
# Libro guía: Moscote Flórez, *Introducción al riesgo financiero*, cap. 6.1.
#
# El VaR dice cuánto se pierde "en condiciones normales" pero no cuánto se pierde en los días
# restantes. La prueba de estrés estudia el efecto de grandes movimientos de los factores de
# riesgo mediante escenarios (el libro los llama subjetivos), incluyendo situaciones que los
# datos históricos no captan. Procedimiento del libro: (1) factores de mercado que afectan el
# portafolio, (2) cuáles ocurren a la vez, (3) cambio razonable en cada uno, (4) evaluar las posiciones.
#
# Factores presentes en este portafolio (solo acciones, oro y petróleo; no hay tasas ni divisas):
# nivel del mercado (SPY), precio del oro, precio del petróleo, volatilidad y correlaciones.
#
# A. Escenarios deterministas: pérdida (%) de cada portafolio con pesos fijos.
#    Históricos (eventos reales de la muestra): E1, E2, E3 + peores ventanas observadas.
#    Hipotéticos (parámetros editables abajo): H1, H2, H3.
# B. Escenarios de parámetros: VaR 99% a 1 día con volatilidad mayor y/o correlaciones iguales a 1
#    (el libro menciona "cambios bruscos en las volatilidades históricas").

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

MERCADO = "SP500_Benchmark"
TASA_LIBRE_RIESGO_ANUAL = 0.04
S = 100_000                       # monto invertido (USD)
NIVEL = 0.99

# ---- Parámetros de los escenarios (editables) ----
CAIDA_MERCADO = -0.20             # H1 y H2
H3_PETROLEO, H3_ORO, H3_MERCADO = -0.30, -0.15, -0.05
K_VOLATILIDAD = [2, 3]            # multiplicadores de volatilidad
PERCENTIL_ESTRES = 0.05           # "días de estrés" = 5% peores días del mercado (para betas de estrés)

rend_log = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rend_log.columns.tolist()
n = len(activos)
rs = np.expm1(rend_log)
rm = rs[MERCADO]

# %% [markdown]
# ## 9.1 Portafolios (los mismos de las Fases 6 a 8, pesos fijos)

# %%
def pesos_fase3(rf=TASA_LIBRE_RIESGO_ANUAL):
    mu = (rend_log.mean() * 252).values
    cov = (rend_log.cov() * 252).values
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
series = {p: pd.Series(rs.values @ w, index=rs.index) for p, w in portafolios.items()}

# %% [markdown]
# ## 9.2 Betas normales y betas de estrés
# Beta normal: pendiente del modelo de mercado con toda la muestra (como en la Fase 6).
# Beta de estrés: pendiente (por el origen) usando solo los días en que el mercado cayó más.
# En las crisis las correlaciones suben, por lo que el beta de estrés suele superar al normal.

# %%
beta_normal = rs.apply(lambda c: c.cov(rm) / rm.var())
dias_estres = rm <= rm.quantile(PERCENTIL_ESTRES)
beta_estres = rs[dias_estres].apply(lambda c: (c * rm[dias_estres]).sum() / (rm[dias_estres] ** 2).sum())
betas = pd.DataFrame({"Beta normal": beta_normal, "Beta de estrés": beta_estres})
print(f"Días de estrés (mercado <= percentil {PERCENTIL_ESTRES:.0%}): {int(dias_estres.sum())}")
print(betas.round(3).to_string())

# %% [markdown]
# ## 9.3 Escenarios deterministas: choque por activo

# %%
def acumulado(desde, hasta):
    """Rendimiento simple acumulado de cada activo entre dos fechas (inclusive)."""
    return np.expm1(rend_log.loc[desde:hasta].sum())

def un_dia(fecha):
    return np.expm1(rend_log.loc[fecha])

E1 = "E1 Aranceles abr-2025 (3 a 8 abr)"
E2 = "E2 Corrección del oro (30 ene 2026)"
E3 = "E3 Colapso del petróleo (8 abr 2026)"
H1 = f"H1 Mercado {CAIDA_MERCADO:.0%} (betas normales)"
H2 = f"H2 Mercado {CAIDA_MERCADO:.0%} (betas de estrés)"
H3 = f"H3 Petróleo {H3_PETROLEO:.0%}, oro {H3_ORO:.0%}, mercado {H3_MERCADO:.0%}"

shocks = pd.DataFrame(index=activos)
shocks[E1] = acumulado("2025-04-03", "2025-04-08")
shocks[E2] = un_dia("2026-01-30")
shocks[E3] = un_dia("2026-04-08")
shocks[H1] = beta_normal * CAIDA_MERCADO
shocks[H2] = beta_estres * CAIDA_MERCADO
h3 = beta_normal * H3_MERCADO
h3["WTI_Oil"], h3["Oro"] = H3_PETROLEO, H3_ORO
shocks[H3] = h3
print("\nChoque aplicado a cada activo en cada escenario (rendimiento simple):")
print(shocks.round(4).to_string())

# %% [markdown]
# ## 9.4 Resultado de cada escenario sobre cada portafolio (pérdidas en negativo)

# %%
filas = {}
for p, w in portafolios.items():
    fila = {esc: float(shocks[esc].values @ w) for esc in shocks.columns}
    r = series[p]
    fila["Peor día de la muestra"] = r.min()
    fila["Peor semana (5 días)"] = np.expm1(np.log1p(r).rolling(5).sum()).min()
    fila["Peor ventana de 10 días"] = np.expm1(np.log1p(r).rolling(10).sum()).min()
    filas[p] = fila
resultado = pd.DataFrame(filas).T
print("\n=== Resultado del portafolio por escenario (%; negativo = pérdida) ===")
print((resultado * 100).round(2).to_string())

# Cuántas veces el VaR 99% a 1 día (normal, media cero) es la pérdida de un escenario de 1 día
z = stats.norm.ppf(NIVEL)
sd_p = pd.Series({p: r.std(ddof=1) for p, r in series.items()})
var_base = z * sd_p
# (E3 se omite: ese día el mercado subió y los portafolios no pierden, el múltiplo no tiene sentido)
multiplos = pd.DataFrame({
    "VaR 99% normal 1 día (%)": var_base * 100,
    f"{E2}: pérdida / VaR": -resultado[E2] / var_base,
    "Peor día: pérdida / VaR": -resultado["Peor día de la muestra"] / var_base,
    "Fecha del peor día": pd.Series({p: r.idxmin().date() for p, r in series.items()}),
})
print("\n=== Pérdida de escenarios de 1 día en múltiplos del VaR 99% ===")
print(multiplos.round(2).to_string())

# Qué habría predicho el beta normal para la caída real de abril de 2025 (escenario E1)
caida_mercado_e1 = float(shocks.loc[MERCADO, E1])
beta_port = pd.Series({p: float(w @ beta_normal.values) for p, w in portafolios.items()})
comparacion_e1 = pd.DataFrame({
    "Beta del portafolio": beta_port,
    "Pérdida predicha por beta (%)": beta_port * caida_mercado_e1 * 100,
    "Pérdida real en E1 (%)": resultado[E1] * 100,
})
comparacion_e1["Real / predicha"] = comparacion_e1["Pérdida real en E1 (%)"] / comparacion_e1["Pérdida predicha por beta (%)"]
print(f"\n=== Abril 2025: el mercado cayó {caida_mercado_e1:.1%}. ¿Qué predecía el beta? ===")
print(comparacion_e1.round(2).to_string())

# %% [markdown]
# ## 9.5 Escenarios de parámetros: VaR 99% a 1 día con volatilidad y correlaciones estresadas
# Base: normal sin media. Correlación = 1: no hay diversificación (riesgo = suma ponderada de volatilidades).

# %%
vol_act = rs.std(ddof=1)
filas = {}
for p, w in portafolios.items():
    s_base = sd_p[p]
    s_corr1 = float(w @ vol_act.values)
    fila = {"Base": z * s_base}
    for k in K_VOLATILIDAD:
        fila[f"Volatilidad x{k}"] = z * k * s_base
    fila["Correlación = 1"] = z * s_corr1
    fila[f"Volatilidad x{K_VOLATILIDAD[0]} y correlación = 1"] = z * K_VOLATILIDAD[0] * s_corr1
    filas[p] = fila
var_estres = pd.DataFrame(filas).T * 100
print("\n=== VaR 99% a 1 día (% del monto) bajo parámetros estresados ===")
print(var_estres.round(2).to_string())

cvar_estres = var_estres.copy() * (stats.norm.pdf(z) / (1 - NIVEL)) / z       # CVaR normal = VaR * phi(z)/(alpha*z)
print("\n=== CVaR 99% a 1 día (%) bajo parámetros estresados ===")
print(cvar_estres.round(2).to_string())

# %% [markdown]
# ## 9.6 Gráficos

# %%
datos = (resultado * 100)
import textwrap
from matplotlib.colors import TwoSlopeNorm
fig, ax = plt.subplots(figsize=(15, 6.2))
lim = np.nanmax(np.abs(datos.values))
norma = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=max(2.0, float(np.nanmax(datos.values)) + 0.5))
im = ax.imshow(datos.values, cmap="RdYlGn", norm=norma, aspect="auto")
ax.set_xticks(range(datos.shape[1]))
ax.set_xticklabels([textwrap.fill(c, 17) for c in datos.columns], fontsize=7.5)
ax.set_yticks(range(datos.shape[0])); ax.set_yticklabels(datos.index, fontsize=8)
for i in range(datos.shape[0]):
    for j in range(datos.shape[1]):
        ax.text(j, i, f"{datos.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_title("Resultado de cada portafolio por escenario de estrés (%)")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig("fase9_mapa_escenarios.png", dpi=150)

ax = var_estres.plot(kind="bar", figsize=(12, 5.5), width=0.85)
ax.set_ylabel("VaR 99% a 1 día (% del monto)")
ax.set_title("VaR 99% bajo volatilidad y correlaciones estresadas")
ax.legend(fontsize=8); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("fase9_var_estresado.png", dpi=150)

# %% [markdown]
# ## 9.7 Guardar resultados

# %%
shocks.to_csv("fase9_definicion_escenarios.csv")
resultado_usd = resultado * S
resultado.to_csv("fase9_resultados_escenarios_pct.csv")
resultado_usd.to_csv("fase9_resultados_escenarios_usd.csv")
multiplos.to_csv("fase9_multiplos_del_var.csv")
comparacion_e1.to_csv("fase9_beta_vs_realidad_abril2025.csv")
var_estres.to_csv("fase9_var_estresado_pct.csv")
cvar_estres.to_csv("fase9_cvar_estresado_pct.csv")
betas.to_csv("fase9_betas_normal_vs_estres.csv")
print("\nGuardado: fase9_*.csv y fase9_*.png")

# %%
