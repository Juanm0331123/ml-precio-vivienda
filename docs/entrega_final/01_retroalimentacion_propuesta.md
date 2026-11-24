# Retroalimentación de la propuesta y ajustes para la entrega final

**Tarjeta:** T16 · **Responsable:** Juan Miguel León Gómez (Project Manager) · **Participan:** todos
**Exposición:** sábado 7 de noviembre de 2026 (sesión 6). Duró cerca de 13 minutos, dentro del límite de 15.
**Entrega:** UAO Virtual, 7 de noviembre, 9:21 a. m. Versión del repositorio: etiqueta `e1-propuesta`.

---

## 1. Lo que dijo el docente

| # | Comentario o pregunta | Qué respondimos en la exposición |
|---|---|---|
| R1 | La propuesta está muy completa y adelantada. | — |
| R2 | ¿La explicabilidad busca explicar **de dónde se deriva el precio** o determinar **qué variables impactan el precio**? Son dos cosas distintas. | Lo segundo: medimos cuánto pesa cada variable en la predicción del modelo. No afirmamos causalidad ni describimos cómo se forma el precio en el mercado. |
| R3 | Con métodos como SHAP conviene inclinarse desde el inicio por modelos fáciles de interpretar, como los basados en árboles (por ejemplo XGBoost). | De acuerdo: Random Forest y XGBoost ya estaban en el diseño; pasan a ser el foco del análisis de interpretabilidad. |

## 2. Ajustes que salen de la retroalimentación

| Ajuste | Qué cambia | Tarjeta | Responsable |
|---|---|---|---|
| A1 · Prioridad a los modelos de árboles | Random Forest y XGBoost reciben la búsqueda de hiperparámetros más amplia y son los candidatos principales a modelo final. La elección sigue saliendo de la validación cruzada (H1), no se decide de antemano. | T21 | Brandon |
| A2 · SHAP sobre árboles | La interpretabilidad se hace con `TreeExplainer` sobre el mejor modelo de árboles: resumen global, dependencia y explicación por vivienda. Se contrasta con los coeficientes del modelo hedónico. | T24 | Brandon |
| A3 · Redacción del objetivo de interpretabilidad | En el paper se escribe que la explicabilidad **mide el impacto de cada variable en la predicción** del modelo. Se evita cualquier frase que sugiera causalidad o formación del precio. La pregunta de investigación pasa de "qué atributos explican sus predicciones" a "qué atributos tienen mayor impacto en sus predicciones". | T27, T28 | Juan Miguel |
| A4 · Demo con explicación por predicción | La aplicación muestra, para cada vivienda consultada, el aporte SHAP de cada atributo al precio estimado. | T26 | Mauricio |
| A5 · Tiempo de la presentación final | La exposición de la propuesta quedó en 13 de 15 minutos. Para la final (20 min con demo) se mantiene el margen: ensayos en 19:30 o menos. | T30 | Juan Miguel |

Cada ajuste quedó como ítem en el checklist "Ajustes por retroalimentación" de su tarjeta en Trello.

## 3. Un matiz que hay que tener presente

El piloto de viabilidad (T12, solo con entrenamiento y sin ajuste fino) dejó a los modelos lineales un poco por delante de los árboles:

| Preprocesamiento | Modelo | RMSE de validación cruzada (media ± desv.) |
|---|---|---|
| P2 | Ridge | 1.077.050 ± 130.909 |
| P2 | OLS hedónico | 1.077.789 ± 131.977 |
| P1 | XGBoost | 1.099.481 ± 126.230 |
| P1 | Random Forest | 1.122.366 ± 135.210 |

Fuente: `reports/tables/piloto_resultados.csv`.

Las diferencias son menores que una desviación estándar, así que el piloto no permite decir cuál es mejor. Esto es coherente con la literatura sobre estos mismos datos de Windsor, donde una especificación paramétrica bien elegida es competitiva (Haupt et al., 2010).

Por eso el ajuste A1 se lee así: los árboles tienen prioridad en el ajuste y en la interpretabilidad, pero **el modelo final lo decide la validación cruzada anidada con su prueba estadística (T22)**. Si el modelo hedónico resulta igual de bueno, se reporta tal cual y se usa SHAP sobre el mejor modelo de árboles para comparar qué variables pesan en cada uno.

## 4. Alcance y cronograma

El alcance de la entrega final no cambia: paper IEEE en inglés, tres preprocesamientos, al menos tres modelos con análisis de complejidad, archivos para reproducir el experimento y presentación práctica de 20 minutos.

La fecha sigue siendo la **estimada**: sábado 5 de diciembre de 2026. Cuando el docente la confirme se ajustan las fechas del tablero y la tabla de [`README.md`](README.md).
