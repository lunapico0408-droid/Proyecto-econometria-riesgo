# Análisis y predicción de un portafolio de activos financieros mediante econometría, ciencia de datos e inteligencia artificial

Proyecto de Estadística (Universidad Santo Tomás) integrado con la clase de Teoría del Riesgo.

## Pregunta central
¿En qué medida los modelos de inteligencia artificial y aprendizaje automático mejoran la predicción de los rendimientos y la gestión del riesgo de un portafolio de activos financieros frente a los modelos econométricos tradicionales?

## Portafolio
9 series diarias de los últimos 3 años (Yahoo Finance): WTI (CL=F), Apple, JPMorgan, Coca-Cola, Johnson & Johnson, Caterpillar, NextEra Energy, Oro (GLD) y S&P 500 (SPY, benchmark).

## Fases y orden de ejecución
Ejecutar en este orden, desde la carpeta del proyecto (cada fase usa los archivos que generan las anteriores):

| Fase | Script | Contenido |
|---|---|---|
| 1 | `fase1_datos_exploracion.py` | Descarga de datos, rendimientos logarítmicos, ADF/KPSS, correlaciones |
| 2 | `fase2_cointegracion_var.py` | Johansen, VAR(3), causalidad de Granger |
| 2b | `fase2b_arch_garch.py` | Efectos ARCH y GARCH(1,1) con distribución t |
| 3 | `fase3_markowitz.py` | Markowitz: mínima varianza, máximo Sharpe, frontera eficiente |
| 3b | `fase3b_markowitz_garch.py` | Markowitz con covarianza CCC-GARCH |
| 4 | `fase4_comparacion_ml.py` | VAR vs Random Forest vs XGBoost vs caminata aleatoria (walk-forward) |
| 4b | `fase4b_lstm.py` | LSTM (requiere TensorFlow) |
| 5 | `fase5_portafolios_predictivos.py` | Portafolios con predicciones, backtest ex-ante y sensibilidad |
| 6 | `fase6_volatilidad_beta_capm.py` | Volatilidad (EWMA, GARCH), betas, CAPM y alfa de Jensen |
| 7 | `fase7_var_cvar.py` | VaR y CVaR (paramétrico, histórico, bootstrap, Monte Carlo) |
| 8 | `fase8_backtesting_var.py` | Backtesting del VaR: Kupiec, zonas de Basilea |
| 9 | `fase9_stress_testing.py` | Pruebas de estrés: escenarios históricos e hipotéticos |

## Instalación
```
python -m pip install -r requirements.txt
```
Los datos (`precios.csv`, `rendimientos.csv`) se regeneran con la Fase 1 al descargarlos de Yahoo Finance, por lo que las cifras pueden variar ligeramente según la fecha de ejecución.

## Resultados
Cada fase guarda sus tablas (CSV) y gráficos (PNG) en la misma carpeta.

## Limitaciones
Muestra de 3 años, ventanas de prueba cortas (150 y 250 días), sin costos de transacción y pesos estáticos en las fases 6 a 9.
