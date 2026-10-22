# %% [markdown]
# # Fase 8: Backtesting del VaR (prueba de respaldo)
# Libro guía: Moscote Flórez, *Introducción al riesgo financiero*, cap. 6.2.
#
# Esquema del ejemplo 6.1 del libro: las primeras T_EST observaciones sirven para
# estimar y las ÚLTIMAS 250 se usan para probar. Cada día t del periodo de prueba:
#   VaR_t se calcula con información hasta t-1  ->  se cuenta una EXCEPCIÓN si la
#   pérdida del día t supera el VaR_t.
#
# Pruebas del libro:
#   6.2     Número de excepciones frente al esperado (binomial)
#   6.2.1   Marco regulatorio: zonas de Basilea (tabla 6.2) y factor de la
#           Superfinanciera (tabla 6.3), definidos para VaR 99% y 250 días
#   6.2.2   Prueba de proporciones
#   6.2.3   Prueba de Kupiec (razón de verosimilitud, chi-cuadrado con 1 g.l.)
#
# Métodos de VaR comparados (1 día, media cero en los paramétricos):
#   1. Normal con volatilidad histórica ESTÁTICA (como en el ejemplo 6.1 del libro)
#   2. Normal con volatilidad histórica EXPANSIVA (se actualiza cada día)
#   3. Normal con volatilidad EWMA (RiskMetrics, lambda = 0.94)
#   4. GARCH(1,1) con innovaciones t-Student (cuantil t estandarizado)
#   5. Simulación histórica (ventana expansiva)

# %%
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy
from scipy.optimize import minimize
from arch import arch_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

NIVELES = [0.95, 0.99]
DIAS_PRUEBA = 250
LAMBDA = 0.94
TASA_LIBRE_RIESGO_ANUAL = 0.04

rend_log = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rend_log.columns.tolist()
n = len(activos)
rs = np.expm1(rend_log)                      # rendimientos simples
T = len(rs)
T_EST = T - DIAS_PRUEBA
fechas_prueba = rs.index[T_EST:]
print(f"{T} observaciones. Estimación: {rs.index[0].date()} a {rs.index[T_EST-1].date()} ({T_EST} días). "
      f"Prueba: {fechas_prueba[0].date()} a {fechas_prueba[-1].date()} ({DIAS_PRUEBA} días).")

# %% [markdown]
# ## 8.1 Portafolios (los mismos de las Fases 6 y 7, con pesos fijos)

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
series = {nombre: rs.values @ w for nombre, w in portafolios.items()}

# %% [markdown]
# ## 8.2 VaR diario fuera de muestra por método

# %%
METODOS = ["Normal, vol. histórica estática", "Normal, vol. histórica expansiva", "Normal, vol. EWMA (0.94)",
           "GARCH(1,1)-t", "Simulación histórica"]

def sigma_ewma(r, lam=LAMBDA):
    f = np.empty(len(r))
    f[0] = np.var(r[:30], ddof=1)
    for t in range(1, len(r)):
        f[t] = lam * f[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(f)

def sigma_garch(r, t_est):
    """GARCH(1,1)-t con parámetros de las primeras t_est obs., filtrado hacia adelante sin reestimar."""
    rp = r * 100
    res = arch_model(rp[:t_est], vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    mu, om, al, be, nu = (res.params["mu"], res.params["omega"], res.params["alpha[1]"],
                          res.params["beta[1]"], res.params["nu"])
    s2 = float(np.asarray(res.conditional_volatility)[-1]) ** 2
    e_prev = rp[t_est - 1] - mu
    out = []
    for t in range(t_est, len(rp)):
        s2 = om + al * e_prev ** 2 + be * s2
        out.append(np.sqrt(s2) / 100)
        e_prev = rp[t] - mu
    return np.array(out), nu

def var_diario(r, nivel):
    """Devuelve dict método -> arreglo de VaR (positivo, % del monto) para los días de prueba."""
    a = 1 - nivel
    z = stats.norm.ppf(nivel)
    idx = np.arange(T_EST, T)
    out = {}
    out["Normal, vol. histórica estática"] = np.full(DIAS_PRUEBA, z * np.std(r[:T_EST], ddof=1))
    out["Normal, vol. histórica expansiva"] = np.array([z * np.std(r[:t], ddof=1) for t in idx])
    out["Normal, vol. EWMA (0.94)"] = z * sigma_ewma(r)[T_EST:]
    sg, nu = sigma_garch(r, T_EST)
    q_t = stats.t.ppf(nivel, nu) * np.sqrt((nu - 2) / nu)        # cuantil t estandarizado (varianza 1)
    out["GARCH(1,1)-t"] = q_t * sg
    out["Simulación histórica"] = np.array([-np.quantile(r[:t], a) for t in idx])
    return out

VAR = {(p, nivel): var_diario(r, nivel) for p, r in series.items() for nivel in NIVELES}

# %% [markdown]
# ## 8.3 Pruebas de respaldo
# Zonas de Basilea (tabla 6.2) y factor de la Superfinanciera (tabla 6.3): solo VaR 99% a 250 días.

# %%
def zona_basilea(N):
    if N <= 4: return "Verde", 1.00
    mult = {5: 1.13, 6: 1.17, 7: 1.22, 8: 1.25, 9: 1.28}
    return ("Amarilla", mult[N]) if N <= 9 else ("Roja", 1.33)

def incremento_sfc(N):
    inc = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
    return 0.0 if N <= 4 else (1.0 if N >= 10 else inc[N])

def kupiec(N, Tn, p):
    """LR_POF = -2 ln[(1-p)^(T-N) p^N] + 2 ln[(1-N/T)^(T-N) (N/T)^N]  ~ chi2(1)."""
    ll0 = (Tn - N) * np.log(1 - p) + N * np.log(p)
    ll1 = xlogy(Tn - N, 1 - N / Tn) + xlogy(N, N / Tn)
    lr = -2 * (ll0 - ll1)
    return lr, stats.chi2.sf(lr, 1)

def prueba_proporciones(N, Tn, p):
    z = (N - p * Tn) / np.sqrt(p * (1 - p) * Tn)
    return z, 2 * stats.norm.sf(abs(z))

r_prueba = {p: r[T_EST:] for p, r in series.items()}
filas = []
for (p, nivel), metodos in VAR.items():
    a = 1 - nivel
    for m in METODOS:
        v = metodos[m]
        exc = r_prueba[p] < -v
        N = int(exc.sum())
        esperado = a * DIAS_PRUEBA
        z, p_z = prueba_proporciones(N, DIAS_PRUEBA, a)
        lr, p_k = kupiec(N, DIAS_PRUEBA, a)
        if p_k >= 0.05:
            decision = "No se rechaza"
        else:
            decision = "Se rechaza: subestima el riesgo" if N > esperado else "Se rechaza: sobreestima el riesgo"
        fila = {"Portafolio": p, "Nivel": nivel, "Método": m, "Excepciones": N, "Esperadas": esperado,
                "Proporción de fallas": N / DIAS_PRUEBA, "z (proporciones)": z, "p-valor (proporciones)": p_z,
                "LR Kupiec": lr, "p-valor Kupiec": p_k, "Decisión (5%)": decision,
                "VaR promedio (%)": v.mean() * 100}
        if nivel == 0.99:
            zona, mult = zona_basilea(N)
            fila.update({"Zona Basilea": zona, "Multiplicador Basilea": mult,
                         "Incremento factor SFC": incremento_sfc(N)})
        filas.append(fila)
resultados = pd.DataFrame(filas)

# %% [markdown]
# ## 8.4 Resultados

# %%
for nivel in NIVELES:
    sub = resultados[resultados["Nivel"] == nivel]
    print(f"\n=== Excepciones en {DIAS_PRUEBA} días, VaR {nivel:.0%} (esperadas: {(1-nivel)*DIAS_PRUEBA:.1f}) ===")
    print(sub.pivot(index="Portafolio", columns="Método", values="Excepciones")
             .reindex(list(portafolios))[METODOS].to_string())
    print(f"\n--- p-valor de Kupiec, VaR {nivel:.0%} (< 0.05 = se rechaza) ---")
    print(sub.pivot(index="Portafolio", columns="Método", values="p-valor Kupiec")
             .reindex(list(portafolios))[METODOS].round(3).to_string())

# Resumen por método (los 9 portafolios son parecidos entre sí: no son 9 pruebas independientes)
res_met = []
for nivel in NIVELES:
    for m in METODOS:
        s = resultados[(resultados["Nivel"] == nivel) & (resultados["Método"] == m)]
        res_met.append({"Nivel": nivel, "Método": m,
                        "Excepciones promedio": s["Excepciones"].mean(),
                        "Esperadas": (1 - nivel) * DIAS_PRUEBA,
                        "Portafolios donde Kupiec rechaza": int((s["p-valor Kupiec"] < 0.05).sum()),
                        "de": len(s), "VaR promedio (%)": s["VaR promedio (%)"].mean()})
resumen_metodos = pd.DataFrame(res_met)
print("\n=== Resumen por método ===")
print(resumen_metodos.round(3).to_string(index=False))

s99 = resultados[resultados["Nivel"] == 0.99]
print("\n=== Zonas de Basilea (VaR 99%): número de portafolios en cada zona, por método ===")
print(s99.pivot_table(index="Método", columns="Zona Basilea", values="Portafolio", aggfunc="count", fill_value=0)
        .reindex(METODOS).to_string())

# %% [markdown]
# ## 8.5 Gráficos (estilo del gráfico 6.1 del libro)

# %%
ref = "Pesos iguales (1/N)" if "Pesos iguales (1/N)" in series else list(portafolios)[0]
r_ref = r_prueba[ref] * 100
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(fechas_prueba, r_ref, color="gray", lw=0.8, label="Rendimiento diario (%)")
col = dict(zip(METODOS, ["tab:blue", "tab:cyan", "tab:orange", "tab:green", "tab:purple"]))
for m in METODOS:
    ax.plot(fechas_prueba, -VAR[(ref, 0.99)][m] * 100, color=col[m], lw=1.3, label=f"-VaR 99%: {m}")
exc = r_prueba[ref] < -VAR[(ref, 0.99)]["Normal, vol. EWMA (0.94)"]
ax.scatter(fechas_prueba[exc], r_ref[exc], color="red", zorder=5, label="Excepciones (EWMA)")
ax.set_title(f"Backtesting VaR 99% - {ref} (últimos {DIAS_PRUEBA} días)")
ax.set_ylabel("%"); ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("fase8_backtesting_serie.png", dpi=150)

tab = s99.pivot(index="Portafolio", columns="Método", values="Excepciones").reindex(list(portafolios))[METODOS]
ax = tab.plot(kind="bar", figsize=(12, 5.5), width=0.85)
ax.axhline(2.5, color="k", ls="--", lw=1, label="Esperadas (2.5)")
ax.axhline(4, color="green", ls=":", lw=1.5, label="Límite zona verde (4)")
ax.axhline(9, color="red", ls=":", lw=1.5, label="Límite zona amarilla (9)")
ax.set_ylabel("Excepciones en 250 días"); ax.set_title("Excepciones del VaR 99% por portafolio y método")
ax.legend(fontsize=7, ncol=2); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("fase8_excepciones_var99.png", dpi=150)

# %% [markdown]
# ## 8.6 Guardar resultados

# %%
resultados.to_csv("fase8_resultados_backtesting.csv", index=False)
resumen_metodos.to_csv("fase8_resumen_por_metodo.csv", index=False)
pd.DataFrame({f"{p} | VaR{int(n_*100)} | {m}": VAR[(p, n_)][m] * 100
              for (p, n_) in VAR for m in METODOS}, index=fechas_prueba).to_csv("fase8_var_diario_pct.csv")
print("\nGuardado: fase8_resultados_backtesting.csv, fase8_resumen_por_metodo.csv, fase8_var_diario_pct.csv")

# %%
