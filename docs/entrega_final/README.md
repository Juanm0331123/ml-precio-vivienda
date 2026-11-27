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

Replan del lunes 24 de noviembre: T19 a T22 estaban vencidas y se movieron las fechas que dependen de ellas. Todas las fechas son a las 9 p. m. (hora de Colombia), salvo la entrega.

| Tarjeta | Tarea | Responsable | Depende de | Fecha | Estado al 26 nov |
|---|---|---|---|---|---|
| T16 | Retroalimentación de la propuesta | Juan Miguel | — | mié 11 nov | Hecho |
| T17 | Pipelines de preprocesamiento P1, P2 y P3 | Mauricio | T06, T12 | vie 20 nov | Hecho |
| T18 | Ingeniería y selección de variables, ablación de `furnishingstatus` | Mauricio | T17 | vie 20 nov | Hecho (P3 alineado en #29) |
| T27 | Paper: introducción, trabajos relacionados y datos | Juan Miguel | — | sáb 28 nov | Hecho |
| T19 | Modelos lineales: OLS hedónico, Ridge, Lasso y ElasticNet | Brandon | T17 | jue 26 nov | Hecho (#26) |
| T20 | KNN y SVR | Brandon | T17 | jue 26 nov | En revisión (#27), falta re-correr P3 |
| T21 | Ensambles: Random Forest y XGBoost | Brandon | T17 | jue 26 nov | En revisión (#28), falta re-correr P3 |
| T22 | Validación cruzada anidada y comparación estadística | Brandon | T19–T21 | vie 27 nov | En progreso (dentro del #30, sin PR propio) |
| T23 | Análisis de complejidad | Brandon | T22 | dom 29 nov | En revisión (#30), cambios pedidos |
| T24 | Análisis de errores e interpretabilidad (SHAP) | Brandon | T22 | dom 29 nov | En revisión (#31), cambios pedidos |
| T25 | Evaluación final en el conjunto de prueba sellado | Brandon | T22–T24 | dom 29 nov | En revisión (#32), cambios pedidos |
| T26 | Demo: aplicación de predicción | Mauricio | T25 | mar 1 dic | Por empezar |
| T28 | Paper: metodología, resultados, discusión y conclusiones | Juan Miguel | T22–T25 | mié 2 dic | En progreso |
| T29 | Reproducibilidad y limpieza del repositorio | Juan Miguel | todas | mié 2 dic | En revisión (#20, base) |
| T30 | Presentación final y ensayo de la demo | Juan Miguel | T26, T28 | jue 3 dic | En revisión (#21, storyboard) |
| T31 | Entrega final, reporte de gestión y coevaluación | Juan Miguel | T27–T30 | sáb 5 dic | En revisión (#22, checklist) |

La ruta crítica es T19–T21 → T22 → T25 → T26 y T28. El piloto (T12) ya entrena los seis modelos con P1, P2 y P3 en `src/evaluacion.py` y `src/modelos.py`, así que T19 a T21 parten de ahí y agregan la búsqueda de hiperparámetros.

## Archivos de esta carpeta

| Archivo | Tarjeta | Contenido |
|---|---|---|
| [`01_retroalimentacion_propuesta.md`](01_retroalimentacion_propuesta.md) | T16 | Comentarios del docente y ajustes que generan |
| `02_storyboard_presentacion_final.md` | T30 | Storyboard, reparto de tiempos y guion de la presentación de 20 minutos |
| [`03_checklist_entrega_final.md`](03_checklist_entrega_final.md) | T31 | Lista de verificación de la entrega: archivos, paper, reproducibilidad, reporte y coevaluación |
| [`Coevaluacion_Final.xlsx`](Coevaluacion_Final.xlsx) | T31 | Plantilla de coevaluación (cada integrante califica a sus dos compañeros de 1 a 5) |
| [`Reporte_Gestion_Final.pdf`](Reporte_Gestion_Final.pdf) | T31 | Reporte de Trello y GitHub al corte del 26 de noviembre |
| [`Paper_ML_Precio_Vivienda.pdf`](Paper_ML_Precio_Vivienda.pdf) | T27 · T28 | Copia del paper IEEE (`docs/paper/main.pdf`) para subir a UAO Virtual |
| [`LEEME.txt`](LEEME.txt) | T31 | Enlaces al repositorio y al tablero, cómo correr el experimento y lista de archivos de la entrega |

## Qué se sube a UAO Virtual (Entrega 2, cierra el viernes 27 de noviembre a las 6:29 p. m.)

Máximo 5 archivos:

1. `Paper_ML_Precio_Vivienda.pdf`.
2. `Presentacion_ML_Precio_Vivienda.pdf`, descargado de Canva (Compartir → Descargar → PDF estándar).
3. `Codigo_ML_Precio_Vivienda.zip`: el ZIP de `main` desde GitHub (Code → Download ZIP) cuando estén fusionados los PR, con `LEEME.txt` adentro.
4. `Reporte_Gestion_Final.pdf`.
5. `Coevaluacion_Final.xlsx` con las seis calificaciones.

El paper vive en [`../paper/`](../paper/).
