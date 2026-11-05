# Guion, ensayo y banco de preguntas (Entregable 1)

**Tarjeta:** T14 · **Coordina:** Juan Miguel León Gómez · **Participan:** Mauricio José Mozo Terán y Brandon Eduardo Tobar Sánchez
**Rúbrica:** Relevancia y viabilidad (15 %, se califica con las respuestas a las preguntas) y Comunicación oral (15 %).
**Límite:** 15:00 de exposición (−0,1 por cada minuto extra). Meta: 14:30. Luego, 5 min de preguntas.

El texto completo de cada intervención va en las **notas del orador de Canva**. Aquí queda lo que tiene que quedar claro en cada diapositiva, las transiciones, el banco de preguntas y el registro de los ensayos.

## 1. Reparto y tiempos

| Bloque | Diapositivas | Presenta | Tiempo | Duración |
|---|---|---|---|---|
| Problema y motivación | 1–5 | Juan Miguel | 0:00–3:10 | 3:10 |
| Revisión de literatura | 6–7 | Juan Miguel | 3:10–4:30 | 1:20 |
| Solución y metodología | 8–9 | Brandon | 4:30–7:30 | 3:00 |
| Viabilidad: piloto | 10 | Brandon | 7:30–9:00 | 1:30 |
| Dataset y EDA | 11–14 | Mauricio | 9:00–13:30 | 4:30 |
| Plan y cierre | 15–16 | Juan Miguel | 13:30–14:30 | 1:00 |
| **Total** | | | | **14:30** |

## 2. Guion por diapositiva

| # | Presenta | Tiempo | Lo que tiene que quedar claro |
|---|---|---|---|
| 1 | Juan Miguel | 0:00–0:15 | La pregunta del proyecto en una frase. |
| 2 | Juan Miguel | 0:15–0:40 | Seguimos las cuatro secciones de la guía y quién presenta cada una. |
| 3 | Juan Miguel | 0:40–1:30 | Valorar viviendas importa: riqueza de los hogares, catastro desactualizado y avalúos automáticos que ya compiten con el manual. |
| 4 | Juan Miguel | 1:30–2:15 | Es regresión supervisada: 12 atributos predicen el precio; también modelamos log(precio). |
| 5 | Juan Miguel | 2:15–3:10 | Pregunta, cuatro objetivos y tres hipótesis que vamos a contrastar. |
| 6 | Juan Miguel | 3:10–3:50 | La búsqueda fue sistemática y reproducible: de 192 registros a 26 estudios. |
| 7 | Juan Miguel | 3:50–4:30 | Los ensambles suelen ganar, pero en Windsor el hedónico es competitivo: es una línea base fuerte. |
| 8 | Brandon | 4:30–6:00 | Cómo fluye el proceso de punta a punta y en qué punto se evita la fuga de datos. *(Guion: Brandon)* |
| 9 | Brandon | 6:00–7:30 | Qué se compara (3 preprocesamientos × 6 modelos), con qué validación y qué métricas. *(Guion: Brandon)* |
| 10 | Brandon | 7:30–9:00 | Que la propuesta es viable: el pipeline corre y los primeros resultados son razonables. *(Guion: Brandon)* |
| 11 | Mauricio | 9:00–10:15 | De dónde vienen los datos y qué significa de verdad cada variable. *(Guion: Mauricio)* |
| 12 | Mauricio | 10:15–11:15 | Los datos son de buena calidad y se prepararon sin fuga; la partición de prueba queda sellada. *(Guion: Mauricio)* |
| 13 | Mauricio | 11:15–12:25 | El precio es asimétrico y por eso modelamos log(precio). *(Guion: Mauricio)* |
| 14 | Mauricio | 12:25–13:30 | Qué atributos se asocian más con el precio y qué implica para los modelos. *(Guion: Mauricio)* |
| 15 | Juan Miguel | 13:30–14:10 | Cuatro fases hasta el 5 de diciembre, roles claros y trabajo trazable en Trello y GitHub. |
| 16 | Juan Miguel | 14:10–14:30 | Tres mensajes de cierre y paso a preguntas. |

### Transiciones

- **Juan Miguel → Brandon (4:30):** "Con esa base, Brandon presenta la solución que proponemos y cómo la vamos a evaluar."
- **Brandon → Mauricio (9:00):** "Todo esto descansa en los datos. Mauricio explica de dónde vienen y cómo se comportan."
- **Mauricio → Juan Miguel (13:30):** "Con los datos claros, Juan Miguel cierra con el plan de trabajo."

## 3. Banco de preguntas

Cada uno revisa y ajusta las respuestas de su parte antes del ensayo del sábado.

### Problema, literatura y plan · responde Juan Miguel

**1. ¿Esto sirve para Colombia?**
No directamente: los datos son de Windsor (Canadá) en 1987, y un modelo entrenado ahí no se aplica a otro mercado. Lo que sí se transfiere es el método: la comparación rigurosa, el pipeline y la forma de explicar las predicciones. Con datos catastrales o de transacciones locales se puede reentrenar.

**2. ¿Por qué usar un dataset de 1987 si el mercado de hoy es otro?**
Porque no buscamos predecir precios actuales, sino comparar métodos. Es un dataset de referencia con procedencia verificada y con estudios publicados sobre los mismos datos, así que podemos contrastar nuestros resultados con evidencia previa.

**3. ¿Qué aporta el proyecto frente a lo que ya existe?**
El aporte está en el diseño de la comparación: el hedónico como línea base fuerte, validación anidada y pruebas estadísticas, sobre un dataset con estudios previos para contrastar. Además medimos costo e interpretabilidad, y revisamos si la variable que agregó Kaggle aporta o no.

**4. ¿Cómo se usaría en la práctica?**
Como un modelo automático de valoración de apoyo: da una estimación del precio y explica qué atributos la empujan. Sirve para avalúos masivos o para priorizar la revisión del perito, no para reemplazarlo.

**5. ¿Cómo garantizan que la revisión de literatura es sistemática?**
Seguimos PRISMA 2020: 11 fuentes, 6 cadenas en inglés y español, criterios de inclusión y exclusión explícitos, y un registro de búsquedas con tamizaje y scripts en el repositorio. Para el paper replicamos las cadenas booleanas en Scopus y Web of Science con el acceso de la biblioteca de la UAO.

**6. ¿Y si el aprendizaje automático no supera al hedónico?**
También es un resultado válido. La literatura sobre estos mismos datos anticipa que un modelo paramétrico bien especificado es competitivo, y por eso las hipótesis se contrastan con pruebas estadísticas y no se dan por ciertas.

### Solución y metodología · responde Brandon

**7. ¿Por qué no deep learning?**
Son 545 observaciones tabulares. En datos tabulares de tamaño pequeño o medio, los modelos de árboles siguen siendo el estado del arte (Grinsztajn et al., 2022), con menos costo y más interpretabilidad. Una red profunda tendría alto riesgo de sobreajuste.

**8. ¿Cómo evitan la fuga de datos?**
La partición de prueba (20 %) se sella al inicio y no se toca hasta la evaluación final. Todo escalado, transformación y selección de variables va dentro de un `Pipeline` que se ajusta solo con los datos de entrenamiento de cada pliegue.

**9. ¿Por qué validación cruzada repetida y anidada?**
La anidada separa la elección de hiperparámetros, en los pliegues internos, de la estimación del error, en los externos, y así el error no sale optimista. Repetirla con distintas particiones da una medida de la variabilidad, que es la que necesitan las pruebas estadísticas.

**10. ¿Cómo sabrán si un modelo es mejor de verdad y no por azar?**
Comparamos los modelos sobre los mismos pliegues y usamos una prueba t corregida para validación cruzada repetida (α = 0,05). Una diferencia pequeña en el RMSE promedio no basta.

**11. ¿Por qué modelar el log del precio?**
El precio es asimétrico, con una cola de valores altos. Con el logaritmo se estabiliza la varianza y el error pasa a ser relativo, lo que ayuda sobre todo a los modelos lineales y de distancia. La hipótesis H2 lo contrasta comparando el mismo modelo con y sin la transformación.

**12. ¿Cómo medirán la complejidad?**
En dos frentes: la teórica, con el orden de costo de entrenamiento y predicción de cada modelo, y la empírica, con los tiempos de entrenamiento y predicción y el número de parámetros. Así se ve cuánto se gana en error por cada unidad de costo.

**13. ¿Cómo explican lo que predice un ensamble?**
Con importancia por permutación para el efecto global de cada atributo y con SHAP para explicar predicciones individuales. Así comparamos lo que aprende el ensamble con los coeficientes del modelo hedónico.

**14. ¿545 observaciones alcanzan?**
Es un tamaño pequeño, y por eso el diseño es conservador: validación cruzada repetida, modelos regularizados y curvas de aprendizaje en el piloto para ver si más datos cambiarían el resultado.

### Dataset · responde Mauricio

**15. ¿De dónde salen los datos y cómo saben que son confiables?**
Son las 546 ventas de Windsor, Ontario, entre julio y septiembre de 1987, publicadas por Anglin y Gençay (1996) en el *Journal of Applied Econometrics* y disponibles en el paquete AER de R. La versión de Kaggle tiene 545 filas: comprobamos que coincide fila por fila con el original y que la fila que falta era un duplicado.

**16. ¿Por qué este dataset no viola la regla de no usar datasets de scikit-learn?**
scikit-learn trae, por ejemplo, California Housing y antes traía Boston. El nuestro es el de Windsor, que viene del archivo del *Journal of Applied Econometrics* y del paquete AER de R, y lo usamos en su versión publicada en Kaggle.

**17. ¿Por qué el precio está multiplicado por 70?**
Kaggle no lo documenta, pero al dividir entre 70 las 545 filas coinciden exactamente con el precio original en dólares canadienses de 1987. Para el modelo no cambia nada, porque es una escala; para interpretar los errores en dinero se divide entre 70, y el MAPE no depende de la escala.

**18. ¿Qué pasa con `furnishingstatus`, si fue añadida en Kaggle?**
No existe en el original y su origen no está documentado, así que la tratamos con cautela. La hipótesis H3 la evalúa con una ablación: entrenamos con y sin la variable y comparamos el RMSE.

**19. ¿Qué hacen con los atípicos?**
Hay 15 precios atípicos altos según la regla de 1,5 × IQR. Son ventas reales, así que los conservamos. Para que no dominen el ajuste modelamos log(precio) y, en uno de los preprocesamientos, aplicamos winsorización ajustada dentro de cada pliegue.

**20. ¿Qué significan realmente las variables?**
Varios nombres de Kaggle engañan:
- `area` es el tamaño del lote, no el área construida;
- `mainroad` indica si hay entrada vehicular;
- `guestroom` indica si hay sala de recreación;
- `parking` cuenta las plazas de garaje;
- `prefarea` indica si la casa está en el barrio preferido.

El diccionario de datos documenta el significado real de cada una.

### Cómo respondemos

- Contesta quien domina el tema, según este banco. Si la pregunta es general, la recibe Juan Miguel y la pasa a quien corresponda.
- Treinta segundos o menos: el dato, su fuente y qué implica para el proyecto.
- Si algo aún no está medido, lo decimos y explicamos en qué parte del proyecto se resuelve.

## 4. Ensayos cronometrados

Meta: los dos ensayos en 14:30 o menos. Los tiempos también se registran como comentario en la tarjeta T14.

| Bloque | Plan | Ensayo 1 | Ensayo 2 | Ajuste |
|---|---|---|---|---|
| Problema y motivación (Juan Miguel) | 3:10 | | | |
| Revisión de literatura (Juan Miguel) | 1:20 | | | |
| Solución y metodología (Brandon) | 3:00 | | | |
| Viabilidad: piloto (Brandon) | 1:30 | | | |
| Dataset y EDA (Mauricio) | 4:30 | | | |
| Plan y cierre (Juan Miguel) | 1:00 | | | |
| **Total** | **14:30** | | | |
