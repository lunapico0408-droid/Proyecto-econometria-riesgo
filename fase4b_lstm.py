# %% [markdown]
# # Fase 4b: LSTM como comparacion adicional (econometria vs ML vs deep learning)
# Extiende la Fase 4: mismo split train/test, mismo TEST_SIZE, misma logica
# de walk-forward, para que las metricas sean directamente comparables con
# VAR(3), Random Forest, XGBoost y el benchmark naive (random walk).
#
# A diferencia de RF/XGBoost (un modelo por activo, features tabulares con
# rezagos aplanados), aqui se usa UN SOLO modelo LSTM multivariado que recibe
# la secuencia de los ultimos LAGS dias de los 9 activos y predice los 9
# rendimientos del dia siguiente en un solo paso (mas natural para series
# de tiempo secuenciales y comparable en espiritu al VAR).

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import keras
keras.utils.set_random_seed(42)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


tf.random.set_seed(42)
np.random.seed(42)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

rendimientos = pd.read_csv("rendimientos.csv", index_col=0, parse_dates=True)
activos = rendimientos.columns.tolist()
n_activos = len(activos)

LAGS = 3
TEST_SIZE = 150
REFIT_EVERY = 10   # igual que en Fase 4, por costo computacional

datos = rendimientos.values
fechas = rendimientos.index

# %% [markdown]
# ## 4b.1 Construccion de secuencias (samples, LAGS, n_activos)

# %%
def construir_secuencias(datos, lags):
    X, y, fechas_y = [], [], []
    for t in range(lags, len(datos)):
        X.append(datos[t - lags:t])
        y.append(datos[t])
        fechas_y.append(fechas[t])
    return np.array(X), np.array(y), pd.DatetimeIndex(fechas_y)

X_full, y_full, fechas_seq = construir_secuencias(datos, LAGS)
split_idx = len(X_full) - TEST_SIZE
fechas_test = fechas_seq[split_idx:]

print(f"Secuencias totales: {len(X_full)}, Test: {TEST_SIZE}")

# %% [markdown]
# ## 4b.2 Arquitectura del modelo

# %%
def construir_modelo():
    modelo = keras.Sequential([
        keras.layers.Input(shape=(LAGS, n_activos)),
        keras.layers.LSTM(32, activation="tanh"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(n_activos),
    ])
    modelo.compile(optimizer="adam", loss="mse")
    return modelo

# %% [markdown]
# ## 4b.3 Validacion walk-forward (ventana expandible, refit cada REFIT_EVERY pasos)

# %%
predicciones_lstm = pd.DataFrame(index=fechas_test, columns=activos, dtype=float)
modelo_actual = None

for paso, fecha in enumerate(fechas_test):
    pos = split_idx + paso

    if paso % REFIT_EVERY == 0:
        X_train, y_train = X_full[:pos], y_full[:pos]
        modelo_actual = construir_modelo()
        modelo_actual.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

    x_actual = X_full[pos:pos + 1]
    pred = modelo_actual.predict(x_actual, verbose=0)[0]
    predicciones_lstm.loc[fecha] = pred

    if paso % 25 == 0:
        print(f"Paso {paso}/{len(fechas_test)} ({fecha.date()})")

print("Walk-forward LSTM completado.")

# %% [markdown]
# ## 4b.4 Metricas: RMSE, MAE y precision direccional, comparables con Fase 4

# %%
y_test = pd.DataFrame(y_full[split_idx:], index=fechas_test, columns=activos)

filas = []
for activo in activos:
    rmse = np.sqrt(mean_squared_error(y_test[activo], predicciones_lstm[activo]))
    mae = mean_absolute_error(y_test[activo], predicciones_lstm[activo])
    filas.append({"Modelo": "LSTM", "Activo": activo, "RMSE": rmse, "MAE": mae})
tabla_metricas_lstm = pd.DataFrame(filas)

precision_lstm = pd.Series({
    activo: (np.sign(y_test[activo]) == np.sign(predicciones_lstm[activo])).mean()
    for activo in activos
}, name="LSTM")

print("RMSE promedio LSTM:", tabla_metricas_lstm["RMSE"].mean())
print("Precision direccional promedio LSTM:", precision_lstm.mean())

# %% [markdown]
# ## 4b.5 Comparacion final: Naive vs VAR vs RF vs XGBoost vs LSTM

# %%
try:
    metricas_fase4 = pd.read_csv("metricas_comparacion_modelos.csv")
    metricas_todas = pd.concat([metricas_fase4, tabla_metricas_lstm], ignore_index=True)
    rmse_promedio_todos = metricas_todas.groupby("Modelo")["RMSE"].mean().sort_values()
    print("\n=== RMSE promedio, los 5 modelos ===")
    print(rmse_promedio_todos)

    plt.figure(figsize=(8, 5))
    rmse_promedio_todos.plot(kind="bar", color=["#2c7fb8", "#e34a33", "#31a354", "#756bb1", "#fdae61"])
    plt.ylabel("RMSE promedio (todos los activos)")
    plt.title("Comparacion final: econometria vs ML vs deep learning")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("comparacion_rmse_modelos_con_lstm.png", dpi=150)

    metricas_todas.to_csv("metricas_comparacion_modelos_con_lstm.csv", index=False)
except FileNotFoundError:
    print("Nota: corre Fase 4 primero para la comparacion conjunta con VAR/RF/XGBoost.")

# %% [markdown]
# ## 4b.6 Guardar predicciones (para usarlas en Fase 5 si se quiere agregar
# un portafolio basado en LSTM, igual que se hizo con VAR/RF/XGBoost)

# %%
predicciones_lstm.to_csv("predicciones_lstm.csv")
tabla_metricas_lstm.to_csv("metricas_lstm.csv", index=False)
precision_lstm.to_csv("precision_direccional_lstm.csv")

print("\nGuardado: predicciones_lstm.csv, metricas_lstm.csv, precision_direccional_lstm.csv, "
      "metricas_comparacion_modelos_con_lstm.csv, comparacion_rmse_modelos_con_lstm.png")

# %%
