# Entrega final · Paper IEEE, código reproducible y demo

**Fecha estimada:** sábado 5 de diciembre de 2026 (por confirmar con el docente).
**Presentación:** práctica, 20 minutos, con demostración del sistema y de los resultados.

## Qué pide la guía y qué tarjeta lo cubre

| Requisito de la guía (paper IEEE, en inglés) | Tarjetas | Responsable |
|---|---|---|
| 1. Fenómeno o proceso modelado y problema a resolver | T27 | Juan Miguel |
| 2. Proceso de obtención del dataset | T27 (a partir de T05–T08) | Juan Miguel |
| 3. Problema de aprendizaje automático: tipo de problema y tarea | T27 | Juan Miguel |
| 4. Preprocesamiento con al menos tres metodologías | T17, T18 → T28 | Mauricio |
| 5. Entrenamiento y evaluación de al menos tres modelos, con análisis de complejidad | T19–T25 → T28 | Brandon |
| Archivos necesarios para correr el experimento | T29 | Juan Miguel |
| Presentación práctica de 20 min con demo | T26, T30 | Mauricio · Juan Miguel |
| Reporte de la herramienta de gestión (20 %) y coevaluación (10 %) | T31 | Juan Miguel |

## Cronograma

| Tarjeta | Tarea | Responsable | Depende de | Fecha |
|---|---|---|---|---|
| T16 | Retroalimentación de la propuesta | Juan Miguel | — | mié 11 nov |
| T17 | Pipelines de preprocesamiento P1, P2 y P3 | Mauricio | T06, T12 | sáb 14 nov |
| T18 | Ingeniería y selección de variables, ablación de `furnishingstatus` | Mauricio | T17 | lun 16 nov |
| T19 | Modelos lineales: OLS hedónico, Ridge, Lasso y ElasticNet | Brandon | T17 | mié 18 nov |
| T20 | KNN y SVR | Brandon | T17 | jue 19 nov |
| T21 | Ensambles: Random Forest y XGBoost | Brandon | T17 | sáb 21 nov |
| T22 | Validación cruzada anidada y comparación estadística | Brandon | T19–T21 | lun 23 nov |
| T23 | Análisis de complejidad | Brandon | T22 | mié 25 nov |
| T24 | Análisis de errores e interpretabilidad (SHAP) | Brandon | T22 | jue 26 nov |
| T25 | Evaluación final en el conjunto de prueba sellado | Brandon | T22–T24 | vie 27 nov |
| T27 | Paper: introducción, trabajos relacionados y datos | Juan Miguel | — | sáb 28 nov |
| T26 | Demo: aplicación de predicción | Mauricio | T25 | dom 29 nov |
| T28 | Paper: metodología, resultados, discusión y conclusiones | Juan Miguel | T22–T25 | mié 2 dic |
| T29 | Reproducibilidad y limpieza del repositorio | Juan Miguel | todas | mié 2 dic |
| T30 | Presentación final y ensayo de la demo | Juan Miguel | T26, T28 | vie 4 dic |
| T31 | Entrega final, reporte de gestión y coevaluación | Juan Miguel | T27–T30 | sáb 5 dic |

La ruta crítica es T17 → T19–T21 → T22 → T25 → T28. T27 y la base de T29 no dependen de nadie y se adelantan desde ya.

## Archivos de esta carpeta

| Archivo | Tarjeta | Contenido |
|---|---|---|
| [`01_retroalimentacion_propuesta.md`](01_retroalimentacion_propuesta.md) | T16 | Comentarios del docente y ajustes que generan |
| `Reporte_Gestion_Final.pdf` | T31 | Se genera el día de la entrega |

El paper vive en [`../paper/`](../paper/).
