# %% [markdown]
# # Fase 4: Comparación econometría (VAR) vs Machine Learning (RF, XGBoost)

# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR   # (ruta directa: evita statsmodels.tsa.api, bloqueado en Windows)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

rendimientos = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rendimientos.columns.tolist()
n_activos = len(activos)

LAGS = 3            # mismo rezago que el VAR validado en la Fase 2
TEST_SIZE = 150      # últimas ~150 observaciones como prueba (~7 meses de datos diarios)
REFIT_EVERY = 10     # re-entrenar RF/XGBoost cada 10 pasos (balance costo computacional / fidelidad)

print(f"Total observaciones: {len(rendimientos)}, Test: {TEST_SIZE}, Train inicial: {len(rendimientos) - TEST_SIZE}")

# %% [markdown]
# ## 4.1 Construcción de features rezagadas (para RF y XGBoost)
# Usamos los mismos 3 rezagos de las 9 series como predictores, igual que el VAR(3)

# %%
def construir_features(df, lags):
    X = pd.concat([df.shift(i).add_suffix(f"_L{i}") for i in range(1, lags + 1)], axis=1)
    return X.dropna()

X_full = construir_features(rendimientos, LAGS)
y_full = rendimientos.loc[X_full.index]

split_idx = len(X_full) - TEST_SIZE
fechas_test = X_full.index[split_idx:]

print(f"Features: {X_full.shape[1]}, Observaciones utilizables: {len(X_full)}")

# %% [markdown]
# ## 4.2 Validación walk-forward (ventana expandible, un paso adelante)
# Para cada punto de prueba: se reentrena con todo lo disponible hasta ese momento
# y se predice el siguiente día. VAR se reentrena cada paso; RF/XGB cada REFIT_EVERY
# pasos (por costo computacional) manteniendo el modelo entre reentrenamientos.

# %%
predicciones_var = pd.DataFrame(index=fechas_test, columns=activos, dtype=float)
predicciones_rf = pd.DataFrame(index=fechas_test, columns=activos, dtype=float)
predicciones_xgb = pd.DataFrame(index=fechas_test, columns=activos, dtype=float)
predicciones_naive = pd.DataFrame(index=fechas_test, columns=activos, dtype=float)  # random walk: predice 0

modelos_rf = {activo: None for activo in activos}
modelos_xgb = {activo: None for activo in activos}

for paso, fecha in enumerate(fechas_test):
    pos = split_idx + paso

    # --- VAR: se reentrena cada paso con ventana expandible ---
    train_var = rendimientos.iloc[:pos]
    try:
        var_res = VAR(train_var).fit(LAGS)
        pred_var = var_res.forecast(train_var_values[-LAGS:], steps=1)[0]
        predicciones_var.loc[fecha] = pred_var
    except Exception as e:
        print(fecha, e)
        predicciones_var.loc[fecha] = np.nan

    # --- RF y XGBoost: se reentrenan cada REFIT_EVERY pasos ---
    if paso % REFIT_EVERY == 0:
        X_train = X_full.iloc[:pos]
        y_train = y_full.iloc[:pos]
        for activo in activos:
            modelos_rf[activo] = RandomForestRegressor(
                n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
            ).fit(X_train, y_train[activo])
            modelos_xgb[activo] = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=42, verbosity=0
            ).fit(X_train, y_train[activo])

    x_actual = X_full.loc[[fecha]]
    for activo in activos:
        predicciones_rf.loc[fecha, activo] = modelos_rf[activo].predict(x_actual)[0]
        predicciones_xgb.loc[fecha, activo] = modelos_xgb[activo].predict(x_actual)[0]

    predicciones_naive.loc[fecha] = 0.0  # hipótesis de martingala: mejor predicción = 0

    if paso % 25 == 0:
        print(f"Paso {paso}/{len(fechas_test)} ({fecha.date()})")

print("Walk-forward completado.")

# %% [markdown]
# ## 4.3 Métricas de error por activo y modelo (RMSE, MAE)

# %%
y_test = y_full.loc[fechas_test]

def calcular_metricas(y_true, predicciones_dict):
    filas = []
    for nombre_modelo, pred_df in predicciones_dict.items():
        for activo in activos:
            rmse = np.sqrt(mean_squared_error(y_true[activo], pred_df[activo]))
            mae = mean_absolute_error(y_true[activo], pred_df[activo])
            filas.append({"Modelo": nombre_modelo, "Activo": activo, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(filas)

predicciones_dict = {
    "VAR(3)": predicciones_var,
    "RandomForest": predicciones_rf,
    "XGBoost": predicciones_xgb,
    "Naive (RW)": predicciones_naive,
}

tabla_metricas = calcular_metricas(y_test, predicciones_dict)
tabla_pivote = tabla_metricas.pivot(index="Activo", columns="Modelo", values="RMSE")
tabla_pivote

# %% [markdown]
# ## 4.4 Precisión direccional (% de veces que acierta el signo del rendimiento)
# Métrica financiera clave: importa más acertar la dirección que el valor exacto

# %%
def precision_direccional(y_true, y_pred):
    return (np.sign(y_true) == np.sign(y_pred)).mean()

precision_dir = pd.DataFrame({
    nombre: {activo: precision_direccional(y_test[activo], pred_df[activo]) for activo in activos}
    for nombre, pred_df in predicciones_dict.items()
})
precision_dir

# %% [markdown]
# ## 4.5 Comparación visual: RMSE promedio por modelo

# %%
rmse_promedio = tabla_metricas.groupby("Modelo")["RMSE"].mean().sort_values()

plt.figure(figsize=(8, 5))
rmse_promedio.plot(kind="bar", color=["#2c7fb8", "#e34a33", "#31a354", "#756bb1"])
plt.ylabel("RMSE promedio (todos los activos)")
plt.title("Comparación de desempeño: menor RMSE = mejor")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("comparacion_rmse_modelos.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4.6 Exportar resultados
# Se guardan también las matrices de predicciones crudas (no solo las métricas),
# porque la Fase 5 las usa como "rendimientos esperados" para construir portafolios.

# %%
tabla_metricas.to_csv("metricas_comparacion_modelos.csv", index=False)
precision_dir.to_csv("precision_direccional_modelos.csv")

predicciones_var.to_csv("predicciones_var.csv")
predicciones_rf.to_csv("predicciones_rf.csv")
predicciones_xgb.to_csv("predicciones_xgb.csv")
y_test.to_csv("rendimientos_reales_test.csv")

print("Guardado: metricas_comparacion_modelos.csv, precision_direccional_modelos.csv, "
      "predicciones_{var,rf,xgb}.csv, rendimientos_reales_test.csv")

# %%
