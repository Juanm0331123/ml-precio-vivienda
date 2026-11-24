# Storyboard y guion de la presentación final (20 min)

**Tarjeta:** T30 · **Responsable:** Juan Miguel León Gómez (Reporting Lead) · **Participan:** Mauricio y Brandon · **Revisa:** Brandon
**Guía del curso:** presentación práctica de 20 minutos que demuestre cómo funciona el sistema y cuáles fueron los resultados.
**Meta:** 19:30 con la demo incluida, para tener 30 s de margen. En la propuesta usamos 13 de los 15 minutos.
**Versión maestra:** la presentación vive en Canva; en el repositorio se guarda solo la copia exportada, igual que en el Entregable 1.

Las diapositivas que dependen de resultados todavía no están listas (T22 a T26). Aquí aparecen marcadas como **Pendiente**, con lo que tienen que mostrar y la tarjeta de donde sale cada cifra. El resto ya tiene su guion completo.

## 1. Reparto y tiempos

| Bloque | Diapositivas | Presenta | Tiempo | Duración |
|---|---|---|---|---|
| Apertura y problema | 1–3 | Juan Miguel | 0:00–1:40 | 1:40 |
| Datos y preprocesamiento | 4–6 | Mauricio | 1:40–5:30 | 3:50 |
| Modelos, resultados, complejidad e interpretabilidad | 7–12 | Brandon | 5:30–13:15 | 7:45 |
| Demo en vivo | 13 | Mauricio | 13:15–16:45 | 3:30 |
| Discusión, reproducibilidad y cierre | 14–16 | Juan Miguel | 16:45–19:30 | 2:45 |
| **Total** | | | | **19:30** |

Las diapositivas 17 (referencias) y 18 a 20 (respaldo para preguntas) no se presentan.

## 2. Secuencia

| # | Título (mensaje) | Contenido y visual | Sale de | Presenta | Tiempo |
|---|---|---|---|---|---|
| 1 | ¿Supera el aprendizaje automático al modelo hedónico? | Portada: equipo, programa y fecha | — | Juan Miguel | 0:00–0:15 |
| 2 | Cinco bloques y una demo | Agenda con responsable y minuto de inicio | — | Juan Miguel | 0:15–0:40 |
| 3 | Valorar viviendas: un problema de alto impacto | 3 cifras de contexto, pregunta de investigación y H1–H3 en una línea cada una | T04, T16 | Juan Miguel | 0:40–1:40 |
| 4 | 545 ventas de Windsor, verificadas contra el original | Cadena Kaggle → AER → Anglin y Gençay (1996): factor 70, 1 duplicado, 8 variables renombradas, 1 añadida | T05, T06, T27 | Mauricio | 1:40–3:00 |
| 5 | Tres preprocesamientos acumulativos | P1 → P2 → P3 en tres columnas, con lo que agrega cada uno y la regla "todo se ajusta dentro del pliegue" | T17 | Mauricio | 3:00–4:15 |
| 6 | Seleccionar variables no mejoró el error | Tabla de RMSE por selector y figura de la ablación de `furnishingstatus` | T18 | Mauricio | 4:15–5:30 |
| 7 | 18 pipelines con los mismos pliegues | 3 × 6, CV 5×3 estratificada, prueba sellada y prueba t corregida | T11, T22 | Brandon | 5:30–6:45 |
| 8 | **Pendiente.** Resultados de la validación cruzada | Gráfico de RMSE (media e IC) de los 18 pipelines | T22 | Brandon | 6:45–8:30 |
| 9 | **Pendiente.** Respuesta a H1 y H2 | Mejor ensamble vs hedónico (H1) y P1 vs P2 en MAPE (H2), con p-valor | T22 | Brandon | 8:30–9:30 |
| 10 | **Pendiente.** Precisión frente a costo | Frente de Pareto: RMSE vs tiempo de entrenamiento, y tabla O(·) | T23 | Brandon | 9:30–10:45 |
| 11 | **Pendiente.** Qué atributos pesan en la predicción | SHAP del mejor modelo de árboles junto a los coeficientes del hedónico | T24 | Brandon | 10:45–12:15 |
| 12 | **Pendiente.** Una sola evaluación en la prueba sellada | RMSE, MAE y R² en prueba con IC por *bootstrap*; real vs predicho | T25 | Brandon | 12:15–13:15 |
| 13 | **Pendiente.** Demo: el precio de una vivienda, explicado | App en vivo con una vivienda típica y una cara; plan B en video | T26 | Mauricio | 13:15–16:45 |
| 14 | Lo que estos resultados permiten decir | Alcance y limitaciones (1987, una ciudad, n = 545, variable de origen desconocido) | T28 | Juan Miguel | 16:45–17:45 |
| 15 | Todo se reproduce con un comando | `python reproducir.py`, pruebas automáticas, Trello y GitHub | T29, T31 | Juan Miguel | 17:45–18:30 |
| 16 | **Pendiente (cifras).** Conclusiones | Tres mensajes de cierre con las cifras finales | T28 | Juan Miguel | 18:30–19:30 |
| 17 | Referencias | Anexo; no se presenta | T27, T28 | — | — |
| 18 | Respaldo: procedencia del dataset | Tabla de variables renombradas | T05 | — | — |
| 19 | Respaldo: hiperparámetros | Espacios de búsqueda de cada modelo | T19–T21 | — | — |
| 20 | Respaldo: ablación por pliegue | Diagramas de caja de ΔRMSE | T18 | — | — |

## 3. Guion por diapositiva

Lectura a unos 140 palabras por minuto. Los precios van en dólares canadienses de 1987, como en el paper.

### Diapositiva 1 · Juan Miguel · 0:00–0:15

Buenos días. Somos Juan Miguel, Mauricio y Brandon. Hace un mes les propusimos comparar el aprendizaje automático con el modelo hedónico para predecir el precio de una vivienda. Hoy les mostramos qué encontramos y cómo funciona.

### Diapositiva 2 · Juan Miguel · 0:15–0:40

Vamos en cinco bloques. Yo recuerdo el problema. Mauricio presenta los datos y el preprocesamiento. Brandon muestra los modelos, los resultados, el costo de cada uno y qué atributos pesan en la predicción. Mauricio hace la demo en vivo y yo cierro con lo que esto permite concluir.

### Diapositiva 3 · Juan Miguel · 0:40–1:40

Recordemos por qué importa. La vivienda es cerca de la mitad de la riqueza de los hogares en la OCDE, y en Colombia solo el 5,68 % del territorio tenía el catastro al día en 2019. Valorar bien y a escala es un problema real.

Nuestra pregunta es en qué medida los modelos de aprendizaje automático predicen mejor que el modelo hedónico lineal y qué atributos tienen más impacto en sus predicciones. Como nos sugirió el profesor, la explicabilidad la entendemos así: medir cuánto pesa cada atributo en la predicción del modelo, no explicar cómo se forma el precio en el mercado.

Pusimos a prueba tres hipótesis: que un ensamble de árboles le gana al hedónico, que el logaritmo del precio reduce el error relativo y que la variable de amoblado no aporta.

### Diapositiva 4 · Mauricio · 1:40–3:00

Trabajamos con 545 ventas de viviendas en Windsor, Canadá, de 1987. Lo primero fue verificar de dónde venían los datos de Kaggle, porque no traían documentación.

Los comparamos fila por fila con el archivo original del estudio de Anglin y Gençay. El precio de Kaggle es el precio original en dólares canadienses multiplicado exactamente por 70. Faltaba una fila porque el original tiene un registro duplicado. Ocho variables tenían otro nombre, y algunas cambiaban de sentido: `area` es el tamaño del lote, no el área construida. Y `furnishingstatus` no existe en el original: la agregó Kaggle sin explicar cómo.

Antes de mirar cualquier relación con el precio separamos un 20 % de prueba, 109 ventas, que no se toca hasta la evaluación final.

### Diapositiva 5 · Mauricio · 3:00–4:15

Construimos tres preprocesamientos que se acumulan. P1 es la base: codifica las binarias, pasa el amoblado a variables indicadoras y estandariza los numéricos para los modelos que lo necesitan. P2 agrega el logaritmo del precio y una transformación Yeo-Johnson al tamaño del lote, que es la variable más sesgada. P3 agrega variables derivadas, recorta los extremos y selecciona variables.

La regla clave es que todo lo que aprende de los datos, como el escalador, los percentiles o la selección, se ajusta solo con la parte de entrenamiento de cada pliegue. Así no se filtra información de validación. Tenemos pruebas automáticas que lo verifican.

### Diapositiva 6 · Mauricio · 4:15–5:30

Dos resultados ya están listos. Primero, comparamos tres métodos de selección de variables dentro de los 15 pliegues. Seleccionar no mejoró el error: Lasso se queda con casi todo y empata con usar todas las variables, mientras que quedarse con 10 variables por información mutua o por eliminación recursiva sube el error entre 3 y 7 %.

Segundo, la ablación de `furnishingstatus`. Al quitarla, el error sube un poco en cinco de los seis modelos, como máximo 1,6 %, pero ninguna diferencia es significativa y todos los intervalos incluyen el cero. Esto es consistente con la tercera hipótesis: la variable que agregó Kaggle no mejora la predicción de forma significativa.

### Diapositiva 7 · Brandon · 5:30–6:45

Con esto armamos 18 pipelines: tres preprocesamientos por seis modelos, desde la regresión lineal hasta los ensambles de árboles. Todos se evalúan con los mismos 15 pliegues de validación cruzada, así que las comparaciones son pareadas.

Como los pliegues comparten datos, una prueba t normal subestima la varianza. Por eso usamos la prueba t corregida de Nadeau y Bengio. Y los hiperparámetros se ajustan en un ciclo interno dentro de cada pliegue, con más búsqueda para Random Forest y XGBoost, como nos recomendó el profesor.

### Diapositivas 8 a 13 · Pendientes

Se escriben cuando estén los resultados. Lo que tiene que quedar claro en cada una:

| # | Mensaje | Cifras que lleva | Tarjeta |
|---|---|---|---|
| 8 | Cuál pipeline tiene el menor RMSE y qué tan lejos queda el hedónico | RMSE medio ± desv. de los 18 pipelines | T22 |
| 9 | Si H1 y H2 se sostienen | Diferencia de RMSE (H1) y de MAPE (H2) con su p-valor corregido | T22 |
| 10 | Cuánto cuesta cada punto de precisión | Tiempos de entrenamiento y predicción, tabla O(·) y frente de Pareto | T23 |
| 11 | Qué atributos pesan más y si coinciden con el hedónico | Top 5 SHAP del mejor modelo de árboles vs coeficientes del OLS | T24 |
| 12 | Qué tan bien generaliza el modelo final | RMSE, MAE y R² en las 109 ventas de prueba con IC 95 % | T25 |
| 13 | Cómo se usa el sistema | Precio estimado con su intervalo y el aporte SHAP de cada atributo | T26 |

Para la demo: una vivienda típica (3 habitaciones, 1 baño, 2 pisos, lote de unos 4.600 pies²) y una cara (barrio preferido, aire acondicionado, 2 baños). El contraste muestra cómo cambian los aportes de cada atributo.

### Diapositiva 14 · Juan Miguel · 16:45–17:45

Es importante decir hasta dónde llegan estos resultados. Los datos son de una sola ciudad y de 1987, así que comparamos métodos, no producimos precios para hoy. Con 545 ventas, las diferencias entre modelos tienen mucha variabilidad, y por eso reportamos intervalos y pruebas corregidas en lugar de un solo número. La importancia de un atributo es predictiva, no causal. Y una de las variables, el amoblado, no tiene origen documentado; por eso hicimos la ablación.

Lo que sí es transferible es la metodología: el mismo pipeline, la misma validación y el mismo análisis de impacto se pueden aplicar a avalúos catastrales u ofertas inmobiliarias de Cali.

### Diapositiva 15 · Juan Miguel · 17:45–18:30

Todo el experimento se reproduce con un solo comando, `python reproducir.py`, que corre las pruebas y los notebooks en orden. Cada cambio entró por un Pull Request revisado por otro integrante, enlazado a su tarjeta de Trello, y las pruebas corren solas en GitHub. El repositorio y el tablero son públicos.

### Diapositiva 16 · Juan Miguel · 18:30–19:30

**Pendiente de cifras.** Estructura del cierre, en tres mensajes:

1. Respuesta a la pregunta: si el mejor modelo de aprendizaje automático supera o no al hedónico, por cuánto y con qué significancia.
2. Qué atributos pesan más en la predicción y si coinciden con lo que dice el hedónico.
3. A qué costo computacional, y cuál modelo recomendamos.

Cierre fijo: "Gracias. Quedamos atentos a sus preguntas."

## 4. Banco de preguntas (lo que ya se puede responder)

Formato: quién responde y dos o tres frases cortas.

**¿Por qué usar un dataset de 1987? · Juan Miguel**
Profesor, porque es un benchmark clásico con resultados publicados, y eso nos permite comparar métodos contra una referencia conocida. No buscamos precios actuales: la metodología es la que se transfiere a datos locales.

**¿Cómo saben que no hay fuga de información? · Mauricio**
Profesor, todo lo que aprende de los datos está dentro de un Pipeline y se ajusta solo con la parte de entrenamiento de cada pliegue. La prueba sellada no se usó hasta el final, y tenemos pruebas automáticas que lo verifican.

**¿Por qué no eliminaron `furnishingstatus` si no aporta? · Mauricio**
Profesor, porque quitarla subió un poco el error en cinco de seis modelos, aunque no de forma significativa. Preferimos dejarla y reportar la ablación como prueba de robustez.

**¿Por qué la prueba t corregida y no una normal? · Brandon**
Profesor, porque los pliegues de la validación cruzada repetida comparten datos, y la prueba normal subestima la varianza. La corrección de Nadeau y Bengio infla la varianza para no declarar diferencias que no existen.

**¿Por qué no usaron redes neuronales? · Brandon**
Profesor, con 545 observaciones tabulares los árboles y los modelos regularizados son el estado del arte, como muestra Grinsztajn y coautores. Una red profunda tendría más riesgo de sobreajuste sin una ganancia esperada.

**¿SHAP explica cómo se forma el precio? · Brandon**
No, profesor. SHAP mide cuánto aporta cada atributo a la predicción del modelo. Es una lectura predictiva, no causal.

## 5. Ensayos cronometrados

| Ensayo | Fecha | Duración total | Con demo | Ajustes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |

Meta: dos ensayos de 19:30 o menos, con la demo incluida y el plan B probado.
