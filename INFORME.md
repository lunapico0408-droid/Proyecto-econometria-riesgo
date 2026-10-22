# Informe: análisis y predicción de un portafolio de activos financieros mediante econometría, ciencia de datos e inteligencia artificial

Universidad Santo Tomás · Estadística · Proyecto integrado con Teoría del Riesgo

> Todas las cifras de este informe provienen de los archivos CSV y PNG de este repositorio, generados por los scripts `fase1` a `fase9`. Datos: 9 series diarias de Yahoo Finance, del 6 de octubre de 2023 al 5 de octubre de 2026 (752 rendimientos logarítmicos).

---

## 1. Pregunta de investigación

> ¿En qué medida los modelos de inteligencia artificial y aprendizaje automático mejoran la predicción de los rendimientos y la gestión del riesgo de un portafolio de activos financieros frente a los modelos econométricos tradicionales?

## 2. Respuesta corta

**En esta muestra, la evidencia no respalda que la IA mejore ni la predicción ni la construcción del portafolio.**

- **Predicción.** Ningún modelo supera a la caminata aleatoria (predecir un rendimiento de cero). El LSTM la iguala (error relativo 1.0006) y los demás quedan entre 3 % y 10 % por encima. La precisión direccional de todos los modelos ronda el 50 %.
- **Portafolios.** En el caso base, el VAR(3) obtiene el mayor Sharpe (1.31) y el portafolio de pesos iguales 1/N obtiene 1.10. Random Forest, XGBoost y LSTM quedan en −0.12, 0.08 y −0.18. La ventana de prueba es de solo 150 días y el error estándar del Sharpe es de 1.30, así que ninguna de estas diferencias es estadísticamente distinguible de cero.
- **Gestión del riesgo.** Las medidas de riesgo (volatilidad, VaR, CVaR, backtesting, estrés) se calcularon con métodos estadísticos y econométricos. La IA solo influyó de forma indirecta, a través de los pesos de los portafolios. Entre esos métodos, EWMA y GARCH resultaron los más útiles, y los portafolios construidos con IA no mostraron mejor resistencia a escenarios de estrés que 1/N.

## 3. Datos y portafolio

Nueve series: petróleo WTI, Apple, JPMorgan, Coca-Cola, Johnson & Johnson, Caterpillar, NextEra Energy, oro (GLD) y el S&P 500 (SPY, usado como benchmark y factor de mercado). Los activos cubren sectores y clases distintas (materias primas, tecnología, finanzas, consumo, salud, industria, servicios públicos).

Del resumen de la Fase 1 (cifras anualizadas):

| Activo | Rendimiento | Volatilidad | Curtosis en exceso |
|---|---|---|---|
| S&P 500 | 21.0 % | 15.2 % | 18.8 |
| Caterpillar | 40.3 % | 32.3 % | 3.7 |
| Oro | 27.0 % | 21.7 % | 7.1 |
| JPMorgan | 29.7 % | 23.1 % | 7.8 |
| WTI | 2.6 % | 41.3 % | 5.8 |

La muestra incluye un período muy favorable para renta variable y oro. Eso condiciona todas las medidas de rentabilidad.

## 4. Fases 1 y 2: estadística y econometría

**Estacionariedad.** Las pruebas ADF y KPSS (Fase 1) se aplican a precios en logaritmo y a rendimientos; se trabaja con rendimientos.

**Cointegración.** La prueba de Johansen no rechaza la hipótesis de ausencia de cointegración (r = 0): el estadístico de traza de 172.7 es menor que el valor crítico al 5 % de 197.4. Por lo tanto **no existe un modelo VEC aplicable**, y el modelo multivariado correcto es un VAR en rendimientos. Esto significa que no es posible la comparación VAR contra VEC que se planteó al inicio del proyecto: los datos no la permiten.

**VAR(3).** El criterio de información AIC selecciona cero rezagos con nueve variables, por lo que el rezago 3 se fijó manualmente. El modelo es estable. Con 72 pares posibles, la prueba de causalidad de Granger al 5 % encuentra solo 5 relaciones significativas:

- Apple → Caterpillar (p = 0.019)
- Caterpillar → Johnson & Johnson (p = 0.016)
- JPMorgan → Apple (p = 0.002)
- JPMorgan → S&P 500 (p = 0.017)
- S&P 500 → Coca-Cola (p = 0.010)

Con 72 pruebas, se esperarían unas 3.6 significativas por azar. La evidencia de predictibilidad cruzada es débil, lo que anticipa los resultados de la Fase 4.

**Efectos ARCH y GARCH.** La prueba ARCH-LM detecta heterocedasticidad condicional en 7 de 9 series; Johnson & Johnson (p = 0.060) y NextEra (p = 0.625) no la presentan. Para las siete series con efectos ARCH se ajustó un GARCH(1,1) con distribución t. La persistencia (α + β) va de 0.77 (Apple) a 0.98 (WTI y Coca-Cola). Los grados de libertad de la t, entre 3.4 y 8.3, confirman colas pesadas.

## 5. Fase 3: construcción de portafolios (Markowitz)

Portafolios de largo plazo (sin ventas en corto) con una tasa libre de riesgo de 4 % anual:

| Portafolio | Rendimiento | Volatilidad | Sharpe |
|---|---|---|---|
| Mínima varianza (histórico) | 19.6 % | 9.4 % | 1.67 |
| Máximo Sharpe (histórico) | 24.9 % | 10.9 % | 1.92 |
| Mínima varianza (CCC-GARCH) | 19.6 % | 8.3 % | 1.89 |
| Máximo Sharpe (CCC-GARCH) | 23.0 % | 9.3 % | 2.04 |

Usar la matriz de covarianza GARCH reduce la volatilidad estimada (de 9.4 % a 8.3 % en mínima varianza) y concentra más el peso en el S&P 500 (de 34 % a 50 %).

**Advertencia importante.** Estos Sharpe están calculados dentro de la muestra: los pesos se optimizan con los mismos datos con los que se evalúan. Por eso son optimistas y **no son comparables** con los resultados de la Fase 5, que son fuera de muestra.

## 6. Fase 4: predicción de rendimientos (IA frente a econometría)

Validación walk-forward con una ventana de prueba de 150 días (3 de marzo al 5 de octubre de 2026), un paso adelante. Los modelos Random Forest, XGBoost y LSTM se reajustan cada 10 pasos. El punto de comparación es la caminata aleatoria (pronóstico de rendimiento cero).

| Modelo | RMSE promedio (9 activos) | RMSE relativo a la caminata aleatoria | Precisión direccional promedio |
|---|---|---|---|
| Caminata aleatoria | 0.01803 | 1.000 | no aplica |
| LSTM | 0.01804 | 1.001 | 50.7 % |
| VAR(3) | 0.01852 | 1.031 | 51.0 % |
| Random Forest | 0.01862 | 1.029 | 49.1 % |
| XGBoost | 0.01974 | 1.096 | 50.5 % |

La caminata aleatoria no tiene precisión direccional (siempre predice cero), por lo que no se compara en esa columna.

**Lectura.** Ningún modelo supera de forma consistente a la caminata aleatoria. Por activo, el menor error lo logran el LSTM en 4 de 9, la caminata aleatoria en 3, y el VAR y el Random Forest en uno cada uno, con diferencias mínimas. Con 150 observaciones el error estándar de una precisión direccional es de unos 4 puntos porcentuales, así que valores entre 49 % y 51 % no se distinguen de lanzar una moneda. Esto es coherente con la hipótesis de mercados eficientes en su forma débil para rendimientos diarios.

## 7. Fase 5: del pronóstico al portafolio

Cada estrategia construye un portafolio de máximo Sharpe (largo plazo) con las predicciones de su modelo y una matriz de covarianza de los últimos 250 días. Se rebalancea y se evalúa de forma **ex ante**: las decisiones usan solo información disponible hasta ese día. Esto corrige un error de una versión anterior del análisis, que fijaba los pesos con las predicciones de todo el período de prueba y luego los evaluaba en ese mismo período (sesgo de anticipación), inflando artificialmente los resultados.

Caso base (rebalanceo cada 5 días con el promedio de las últimas 5 predicciones; 150 días de prueba):

| Estrategia | Rend. anual | Volatilidad | Sharpe | Rotación diaria |
|---|---|---|---|---|
| VAR(3) | 24.1 % | 15.3 % | 1.31 | 26.9 % |
| Pesos iguales (1/N) | 15.0 % | 10.0 % | 1.10 | 0 % |
| XGBoost | 5.1 % | 14.7 % | 0.08 | 23.2 % |
| Random Forest | 2.2 % | 14.0 % | −0.12 | 18.1 % |
| LSTM | 1.7 % | 12.3 % | −0.18 | 14.5 % |
| Clásico (media histórica) | 1.8 % | 10.8 % | −0.20 | 1.9 % |

**Sensibilidad al diseño** (Sharpe según la frecuencia de rebalanceo y el promedio de predicciones):

| Estrategia | 5 d / 5 pred. | 1 d / 1 pred. | 5 d / 20 pred. | 21 d / 20 pred. |
|---|---|---|---|---|
| VAR(3) | 1.31 | 2.20 | 2.47 | 2.58 |
| Random Forest | −0.12 | 0.91 | 0.65 | 0.91 |
| XGBoost | 0.08 | −0.48 | −0.26 | 0.99 |
| LSTM | −0.18 | 0.91 | 0.86 | 0.56 |
| Clásico | −0.20 | −0.19 | −0.20 | −0.31 |
| 1/N | 1.10 | 1.10 | 1.10 | 1.10 |

**Lectura.** El VAR(3) queda por encima de 1/N en las cuatro variantes, y los tres modelos de IA cambian de posición según el diseño (XGBoost va de −0.48 a 0.99). Sin embargo:
- El error estándar del Sharpe anualizado con 150 días es de aproximadamente 1.30. Una diferencia de 0.2 entre VAR(3) y 1/N es muy pequeña frente a ese margen.
- Una prueba t sobre los rendimientos diarios de cada estrategia menos los de 1/N da valores entre −1.4 y 0.6, todos dentro de lo esperable por azar.
- La rotación diaria del VAR(3) (27 %) implica costos de transacción que **no se modelaron**; con costos realistas su ventaja se reduciría.

Es decir, la ventaja aparente del VAR(3) es **sugerente, no concluyente**. No hay evidencia de que Random Forest, XGBoost o LSTM mejoren la construcción del portafolio frente a 1/N.

## 8. Fase 6: volatilidad, beta y CAPM

**Hechos estilizados.** Los nueve activos tienen curtosis en exceso positiva (entre 2.3 y 18.8) y ocho de nueve muestran agrupamiento de volatilidad (autocorrelación significativa de los rendimientos al cuadrado; la excepción es NextEra). El efecto apalancamiento (correlación entre el rendimiento y la varianza futura) es débil (valor absoluto de 0.15 como máximo).

**Pronóstico de volatilidad** (pérdida QLIKE sobre los últimos 250 días; menor es mejor):

| Método | QLIKE promedio | Activos donde es el mejor |
|---|---|---|
| EWMA (λ = 0.94) | −7.29 | 5 |
| GARCH(1,1)-t | −7.27 | 2 |
| Ventana móvil de 20 días | −7.20 | 0 |
| Histórica | −7.14 | 2 |

Los métodos que se adaptan (EWMA y GARCH) superan a la volatilidad histórica constante, pero la diferencia entre EWMA y GARCH es pequeña.

**Betas frente al S&P 500.** Caterpillar (1.29) y Apple (1.05) son agresivos; JPMorgan (0.87), NextEra (0.33) y el oro (0.25) son defensivos. Para WTI (−0.15), Johnson & Johnson (0.003) y Coca-Cola (0.012), la beta no es estadísticamente distinta de cero. Las betas de los portafolios van de 0.38 a 0.56. Para activos con beta cercana a cero, el riesgo sistemático explica casi nada de la varianza (R² ≈ 0), por lo que su riesgo es casi enteramente específico.

**CAPM y alfa de Jensen.** El alfa anual es positivo en 7 de 8 activos (la excepción es Apple): oro 18.7 %, Coca-Cola 15.0 %, Johnson & Johnson 14.6 %, Caterpillar 14.3 %. No se probó su significancia estadística, y la muestra de tres años favorece a estos activos, por lo que **no debe leerse como evidencia de habilidad ni de ineficiencia del mercado**. Es un resultado descriptivo de este período.

## 9. Fase 7: VaR y CVaR

VaR y CVaR al 95 % y 99 % con cuatro métodos (paramétrico normal, simulación histórica, bootstrap con 10 000 réplicas y Montecarlo con 100 000), para los 9 activos y 9 portafolios, sobre un capital de referencia de USD 100 000.

- **La normalidad se rechaza en las 18 series** (Jarque-Bera, p ≈ 0). Hay colas pesadas.
- **El VaR histórico y el normal son parecidos en los portafolios** (cociente entre 0.94 y 1.10). Pero **el CVaR histórico es entre 29 % y 53 % mayor que el normal** en los portafolios (entre 15 % y 57 % en los activos). Es decir, el supuesto de normalidad subestima la pérdida esperada en la cola aunque el VaR parezca razonable.
- **Beneficio de diversificación:** entre 47 % y 55 % del VaR (suma de VaR individuales frente a VaR del portafolio), con diferencias pequeñas entre portafolios.
- **Horizonte de 10 días** (escalado por raíz de 10): el VaR al 99 % de los portafolios va de 4.1 % a 5.1 %. Este escalamiento supone independencia de los rendimientos y debe tomarse como aproximación.

## 10. Fase 8: backtesting del VaR

Se estimó con las primeras 504 observaciones y se probó en los últimos 250 días (7 de octubre de 2025 al 5 de octubre de 2026). Se evaluaron 5 métodos en 9 portafolios con la prueba de proporciones, la prueba de Kupiec y las zonas de Basilea.

| Nivel | Método | Excepciones promedio (esperadas) | Portafolios donde Kupiec rechaza (de 9) |
|---|---|---|---|
| 99 % | Normal, volatilidad EWMA | 2.3 (2.5) | 0 |
| 99 % | GARCH(1,1)-t | 0.9 | 5 |
| 99 % | Simulación histórica | 0.4 | 6 |
| 99 % | Normal, volatilidad estática | 0.7 | 7 |
| 95 % | Normal, volatilidad EWMA | 9.3 (12.5) | 0 |
| 95 % | GARCH(1,1)-t | 11.4 | 0 |
| 95 % | Simulación histórica | 13.1 | 1 |
| 95 % | Normal, volatilidad estática | 6.6 | 7 |

**Lectura.** Los modelos de volatilidad estática y la simulación histórica generan VaR demasiado conservadores (menos excepciones de las esperadas); al 99 %, GARCH también. EWMA es el único método que Kupiec nunca rechaza en ningún nivel. Según las zonas de Basilea al 99 %, 44 de 45 combinaciones portafolio-método quedan en zona verde y una (volatilidad estática) en zona amarilla.

**Cautela.** Con 250 días, al 99 % se esperan solo 2.5 excepciones; la prueba de Kupiec tiene poca potencia para detectar un VaR demasiado conservador, y los períodos de muy baja volatilidad dentro de la ventana de prueba pueden explicar parte del resultado. Por eso es más seguro afirmar que EWMA "no fue rechazado" que afirmar que es el mejor.

## 11. Fase 9: pruebas de estrés

Escenarios históricos: aranceles (3 al 8 de abril de 2025), corrección del oro (30 de enero de 2026) y caída del petróleo (8 de abril de 2026). Escenarios hipotéticos: mercado −20 % con betas normales y con betas de estrés, y petróleo −30 % con oro −15 % y mercado −5 %.

- **Aranceles de abril de 2025.** Las pérdidas de los portafolios van de 7.8 % (mínima varianza, la mejor) a 11.4 % (1/N). Entre los portafolios de IA, las pérdidas van de 9.5 % (Random Forest) a 10.0 % (XGBoost); el VAR(3) pierde 10.6 %.
- **La beta subestimó la pérdida real** en este episodio: la pérdida observada fue entre 1.3 y 1.8 veces la predicha por la beta del portafolio. En una crisis, las correlaciones y betas aumentan.
- **Peor día de la muestra: 4 de abril de 2025.** La pérdida fue entre 2.9 y 3.6 veces el VaR normal al 99 %.
- **Estrés de volatilidad y correlación.** Duplicar la volatilidad duplica el VaR; fijar las correlaciones en 1 lo multiplica por aproximadamente 2 (de 1.4 % a 2.8 % en mínima varianza); ambos efectos juntos lo multiplican por 4.
- **Los portafolios de IA no resisten mejor.** Ante el escenario de abril de 2025 y los escenarios hipotéticos, los portafolios de RF, XGBoost, LSTM y VAR(3) tienen pérdidas del mismo orden que 1/N y no mejores que los de Markowitz clásico.

## 12. Conclusiones

1. **Predicción.** Los rendimientos diarios de este portafolio son prácticamente impredecibles con la información usada. Ni los modelos de IA ni el VAR baten a la caminata aleatoria.
2. **Portafolio.** Convertir pronósticos débiles en pesos no generó una ventaja robusta de la IA. El VAR(3) es la estrategia con mejor desempeño aparente, pero la ventana de prueba es demasiado corta para afirmarlo. El portafolio ingenuo 1/N es un competidor difícil de superar.
3. **Riesgo.** Las herramientas que sí aportaron valor son estadísticas: modelos de volatilidad que se adaptan (EWMA, GARCH), medidas que reconocen las colas pesadas (CVaR histórico) y pruebas de estrés. La normalidad subestima la cola, y las betas históricas subestiman las pérdidas en crisis.
4. **Respuesta a la pregunta.** Con estos datos, los modelos de IA no mejoran de forma medible la predicción ni la gestión del riesgo frente a los métodos tradicionales. Esto no prueba que la IA no sirva en general: prueba que, con rendimientos diarios, nueve activos y tres años, no se encontró evidencia de ventaja.

## 13. Limitaciones

- **Muestra corta:** tres años, y un período con fuerte alza de renta variable y oro. Los resultados dependen de ese régimen.
- **Ventanas de prueba cortas** (150 días en las fases 4 y 5, 250 en la 8): poca potencia estadística. El error estándar del Sharpe (1.30) es mayor que las diferencias entre estrategias.
- **Sin costos de transacción**, a pesar de la alta rotación de las estrategias predictivas.
- **Pesos estáticos** en las fases 6 a 9: se usan los pesos de Fase 3 y los pesos promedio de Fase 5, no los pesos dinámicos de la estrategia.
- **Pesos de la Fase 3 optimizados en la muestra.**
- **Selección de los activos:** se eligieron con conocimiento del período, lo que puede sesgar los resultados.
- **IA aplicada solo al rendimiento esperado.** No se usó aprendizaje automático para pronosticar volatilidad o riesgo, por lo que la parte de la pregunta sobre "gestión del riesgo" se responde solo de forma indirecta.
- **Hiperparámetros** de RF, XGBoost y LSTM fijados sin una búsqueda exhaustiva.
- **Cifras que dependen de la fecha de descarga:** al regenerar los datos con la Fase 1 los resultados cambian ligeramente.

## 14. Trabajo futuro

- Más historia, más activos y frecuencia semanal para reducir el ruido.
- Incorporar indicadores técnicos (RSI, momentum, promedios móviles) como variables de entrada de los modelos de IA.
- Modelar costos de transacción y restricciones de rotación.
- Usar aprendizaje automático para pronosticar volatilidad y compararlo con EWMA y GARCH.
- Probar la significancia estadística de las diferencias entre estrategias (por ejemplo con bootstrap de bloques).

## 15. Mapa del repositorio

| Fase | Script | Resultados principales |
|---|---|---|
| 1 | `fase1_datos_exploracion.py` | `resumen_rentabilidad_riesgo.csv`, `matriz_correlacion.csv` |
| 2 | `fase2_cointegracion_var.py` | `resultados_johansen.csv`, `pvalores_granger.csv`, `resumen_var.txt` |
| 2b | `fase2b_arch_garch.py` | `resultados_test_arch.csv`, `resumen_garch.csv` |
| 3 | `fase3_markowitz.py` | `pesos_portafolios_optimos.csv`, `frontera_eficiente.png` |
| 3b | `fase3b_markowitz_garch.py` | `comparacion_markowitz_historico_vs_garch.csv` |
| 4 | `fase4_comparacion_ml.py` | `metricas_comparacion_modelos.csv` |
| 4b | `fase4b_lstm.py` | `metricas_comparacion_modelos_con_lstm.csv` |
| 5 | `fase5_portafolios_predictivos.py` | `fase5_resultados_portafolios.csv`, `fase5_sensibilidad_portafolios.csv` |
| 6 | `fase6_volatilidad_beta_capm.py` | `fase6_*.csv` |
| 7 | `fase7_var_cvar.py` | `fase7_*.csv` |
| 8 | `fase8_backtesting_var.py` | `fase8_*.csv` |
| 9 | `fase9_stress_testing.py` | `fase9_*.csv` |
