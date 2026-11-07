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

Texto que dice cada uno, tal como está en las notas del orador de Canva. Las transiciones ya van al final de las diapositivas 7, 10 y 14.

El guion leído completo dura unos **11:20** a 140 palabras por minuto, dentro del límite de 15:00; con pausas y cambios de diapositiva queda entre 13 y 14 minutos. Por persona: Juan Miguel 4:01, Brandon 3:42, Mauricio 3:36. Las diapositivas 17 y 18 son de respaldo para preguntas y no se presentan.

### Diapositiva 1 · Juan Miguel · 0:00–0:15

Buenos días. Somos Juan Miguel, Mauricio y Brandon. Nuestra propuesta responde una pregunta concreta: ¿el aprendizaje automático predice mejor el precio de una vivienda que el modelo hedónico clásico que se usa en avalúos?

*34 palabras, unos 0:15 leído a 140 palabras por minuto.*

### Diapositiva 2 · Juan Miguel · 0:15–0:40

Seguimos las cuatro secciones que pide la guía. Yo presento el problema y la revisión de literatura; Brandon, la solución, la metodología y el piloto que muestra que es viable; Mauricio, el dataset y su análisis exploratorio. Cierro con el plan de trabajo.

*43 palabras, unos 0:18 leído a 140 palabras por minuto.*

### Diapositiva 3 · Juan Miguel · 0:40–1:30

¿Por qué importa? Primero, la vivienda es el principal activo de los hogares: en la OCDE es cerca de la mitad de su riqueza. Segundo, en Colombia el catastro estaba casi todo desactualizado: solo el 5,68 % del territorio tenía información al día en 2019, y actualizar avalúos a esa escala exige valoración masiva. Tercero, los modelos automáticos de valoración ya logran errores comparables a un avalúo manual. La pregunta es cuál modelo usar y con qué evidencia.

*78 palabras, unos 0:33 leído a 140 palabras por minuto.*

### Diapositiva 4 · Juan Miguel · 1:30–2:15

Formalmente es un problema de regresión supervisada. Cada vivienda se describe con doce atributos: cinco numéricos, seis binarios y uno categórico. El modelo aprende de ventas observadas y predice el precio; también modelamos el logaritmo del precio porque su distribución es asimétrica. Son 545 ventas sin faltantes, y no es un dataset de scikit-learn, como pide la guía. Mauricio explica luego su origen.

*63 palabras, unos 0:27 leído a 140 palabras por minuto.*

### Diapositiva 5 · Juan Miguel · 2:15–3:10

Nuestra pregunta: ¿en qué medida el aprendizaje automático mejora la predicción frente al modelo hedónico, y qué atributos explican esas predicciones? Para responderla tenemos cuatro objetivos: caracterizar los datos, comparar tres estrategias de preprocesamiento, comparar al menos cinco modelos con validación cruzada anidada, y medir su costo e interpretabilidad. Planteamos tres hipótesis que vamos a contrastar, no a dar por ciertas.

*61 palabras, unos 0:26 leído a 140 palabras por minuto.*

### Diapositiva 6 · Juan Miguel · 3:10–3:50

La revisión fue sistemática: seis cadenas de búsqueda en inglés y español, veinte búsquedas registradas y once fuentes académicas, que cubrimos con un buscador web filtrado por el dominio de cada base. Tras quitar duplicados tamizamos 116 registros con criterios explícitos; verificamos el resumen y los datos de cada artículo en la página de la editorial y, sumando la búsqueda por citas, quedaron 26 estudios. La lectura completa de cada uno va en la matriz de literatura. Todo queda en un registro reproducible y en el diagrama PRISMA.

*88 palabras, unos 0:38 leído a 140 palabras por minuto.*

### Diapositiva 7 · Juan Miguel · 3:50–4:30

Tres ideas de la literatura. Los ensambles de árboles suelen ganar en exactitud, tanto frente al hedónico como frente a otros algoritmos. Pero en nuestros mismos datos de Windsor, la literatura econométrica muestra que un modelo paramétrico bien elegido es competitivo: el hedónico no es un rival débil. Y en datos tabulares de este tamaño los árboles siguen siendo el estado del arte. La matriz de 26 estudios lo confirma con matices: en Alemania XGBoost gana pero OLS queda cerca, y en Colombia Ridge queda casi igual a Random Forest. Por eso comparamos con pruebas estadísticas. Con esa base, Brandon presenta la solución que proponemos.

*105 palabras, unos 0:45 leído a 140 palabras por minuto.*

### Diapositiva 8 · Brandon · 4:30–6:00

Esta es la solución completa, de punta a punta. Partimos del dataset de Windsor, con 545 viviendas, que Mauricio ya limpió y dejó con una partición fija 80/20: 436 filas para entrenar y 109 selladas. Lo importante es lo que está en rojo: ese conjunto sellado no entra en la validación, ni en el ajuste, ni en la elección de modelos. Solo se abre una vez, al final del proyecto. Con las 436 filas de entrenamiento armamos un pipeline de scikit-learn con tres partes: un preprocesamiento, un modelo y una validación cruzada. Todo lo que se aprende de los datos, como el escalado, las transformaciones o la selección de variables, se ajusta únicamente con el pliegue de entrenamiento de cada vuelta. Así evitamos la fuga de información. Combinamos tres preprocesamientos con seis modelos: son 18 pipelines, y todos se evalúan con los mismos pliegues. Al final medimos error, complejidad e interpretabilidad. La idea de fondo: el error que reportemos será una estimación honesta.

*163 palabras, unos 1:10 leído a 140 palabras por minuto.*

### Diapositiva 9 · Brandon · 6:00–7:30

Ahora, qué comparamos. Tres preprocesamientos, que se van sumando. El P1 es la base: codificar las categorías y estandarizar solo donde el modelo lo necesita. El P2 agrega transformaciones: el logaritmo del precio, que es asimétrico, Yeo-Johnson y un escalado robusto. El P3 agrega variables derivadas, interacciones, recorte de extremos y selección de variables. Los seis modelos van de lo simple a lo flexible: la regresión lineal hedónica como línea base, Ridge, vecinos cercanos, SVR con kernel, Random Forest y XGBoost. Los evaluamos con validación cruzada repetida 5 por 3 en el piloto, y anidada en el experimento final, que separa la elección de hiperparámetros del cálculo del error. La métrica principal es el RMSE; también reportamos MAE, MAPE y R². Para saber si una diferencia es real, usamos una prueba t corregida sobre los mismos pliegues. Además medimos la complejidad, con tiempos y memoria, y la interpretabilidad, con coeficientes, importancia por permutación, SHAP y gráficos de dependencia parcial. Y siempre dejamos claro que explicar un modelo no es probar causalidad.

*171 palabras, unos 1:13 leído a 140 palabras por minuto.*

### Diapositiva 10 · Brandon · 7:30–9:00

Para comprobar que la propuesta es viable, corrimos un piloto: los 18 pipelines, tres preprocesamientos por seis modelos, con validación cruzada repetida 5 por 3, y solo con las 436 filas de entrenamiento. El test sellado no se tocó. En esta gráfica cada grupo de barras es un modelo y cada color un preprocesamiento. Lo primero: todo funciona de punta a punta, los 270 ajustes corrieron sin fallos y en menos de un minuto de ajuste. Lo segundo, el primer resultado. El menor error lo obtuvo Ridge con transformaciones, con un RMSE de 1,08 millones de unidades de Kaggle, unos 15 mil dólares canadienses de 1987; el R² es 0,66. XGBoost queda apenas un 2 % por encima, y la desviación entre pliegues es mayor que esa diferencia. Es decir: todavía no hay una ventaja clara de los ensambles sobre el modelo lineal, y eso lo contrastamos con pruebas estadísticas en la siguiente fase. Es un piloto con hiperparámetros fijos, así que lo leemos como viabilidad, no como un ganador. Todo esto descansa en los datos: Mauricio explica de dónde vienen y cómo se comportan.

*185 palabras, unos 1:19 leído a 140 palabras por minuto.*

### Diapositiva 11 · Mauricio · 9:00–10:15

Gracias, Brandon. Ahora el dataset. Son ventas reales de 1987 en Windsor, Canadá. Anglin y Gençay las publicaron en 1996 en el Journal of Applied Econometrics: 546 viviendas, 12 variables y precios en dólares canadienses. En el año 2000 quedaron estandarizadas en R, en el paquete AER, con el nombre HousePrices. La versión de Kaggle que usamos tiene 545 filas: es la misma base sin un registro duplicado, con el precio multiplicado por 70 y las columnas renombradas; lo verificamos fila a fila por código. A la derecha está la estructura: el precio es la variable objetivo y hay doce predictoras, entre numéricas, binarias y una categórica. Dos hallazgos importan para modelar: el área es la del lote, no la construida, y la variable de amoblado no existía en el estudio original; la agregó Kaggle, así que la tratamos con cuidado.

*141 palabras, unos 1:00 leído a 140 palabras por minuto.*

### Diapositiva 12 · Mauricio · 10:15–11:15

Antes de modelar auditamos la calidad. Los datos están completos: cero nulos en las 13 variables y ningún duplicado exacto. Sí hay ruido irreducible: seis casas con los mismos atributos y distinto precio, un error que ningún modelo puede eliminar. Hay 15 precios atípicos por encima de 9,2 millones; los conservamos porque son ventas reales y quitarlos sesgaría el modelo en las casas caras. Y el calentador de gas está desbalanceado: solo 25 casos. Un hallazgo clave: el archivo crudo viene ordenado por precio, de mayor a menor. Si se parte sin barajar, lo caro queda en entrenamiento y lo barato en prueba. Por eso barajamos con semilla fija y partimos 80/20, estratificado por quintiles de precio: 436 filas para entrenar y 109 selladas, que no se tocan hasta la evaluación final.

*132 palabras, unos 0:57 leído a 140 palabras por minuto.*

### Diapositiva 13 · Mauricio · 11:15–12:25

Ahora el análisis exploratorio. A la izquierda, el precio original: tiene sesgo a la derecha, con asimetría de 1,21. La media, 4,77 millones, queda por encima de la mediana, 4,34, y el diagrama de caja muestra los 15 atípicos de la cola alta. A la derecha, el mismo precio en logaritmo: la asimetría baja a 0,14, la media y la mediana casi coinciden y la distribución se acerca a una normal. Esto respalda lo que explicó Brandon: en el preprocesamiento P2 modelamos el logaritmo del precio, para que los errores no queden dominados por las casas más caras. Las predicciones se devuelven a la escala original antes de calcular el RMSE, así comparamos todos los modelos en la misma unidad.

*120 palabras, unos 0:51 leído a 140 palabras por minuto.*

### Diapositiva 14 · Mauricio · 12:25–13:30

¿Qué se relaciona con el precio? A la izquierda comparamos tres medidas de asociación: Pearson, Spearman e información mutua. El área del lote es lo más asociado, seguida por baños, aire acondicionado, pisos y garaje. Que las tres medidas no ordenen igual las variables sugiere relaciones no lineales, que es justo donde los modelos de árboles pueden aportar. A la derecha, las comodidades: las casas con aire acondicionado tienen una mediana de precio un 44 % mayor y las de zona preferencial un 36 % mayor; el amoblado muestra un gradiente más leve. Esto es asociación, no causalidad. Con esto le devuelvo la palabra a Juan Miguel para el plan de trabajo.

*112 palabras, unos 0:48 leído a 140 palabras por minuto.*

### Diapositiva 15 · Juan Miguel · 13:30–14:10

Así vamos a ejecutar el proyecto: cuatro fases hasta el 5 de diciembre. Cada uno tiene un rol claro y todo el trabajo está en Trello, en 31 tarjetas con responsable y fecha. Cada tarjeta se convierte en una rama y un Pull Request en GitHub, y los notebooks corren en Colab.

*52 palabras, unos 0:22 leído a 140 palabras por minuto.*

### Diapositiva 16 · Juan Miguel · 14:10–14:30

En resumen: una comparación rigurosa entre el modelo hedónico y el aprendizaje automático, con datos de origen verificado, una validación que demuestra la mejora y atención al costo y a la explicabilidad. Gracias; quedamos atentos a sus preguntas.

*38 palabras, unos 0:16 leído a 140 palabras por minuto.*

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

**5. ¿Cómo garantizan que la revisión de literatura es sistemática? ¿Consultaron directamente las bases?**
Seguimos PRISMA 2020: 6 cadenas en inglés y español, criterios de inclusión y exclusión explícitos, y un registro de búsquedas con tamizaje y scripts en el repositorio. Las 11 fuentes no se consultaron desde su propio buscador: usamos un buscador web filtrado por el dominio de cada base, y así lo reportamos, porque los conteos dependen de ese buscador. Para el paper replicamos las cadenas booleanas en Scopus y Web of Science con el acceso de la biblioteca de la UAO.

**6. ¿Leyeron completos los 26 estudios?**
No todavía, y lo decimos en el documento. La elegibilidad se evaluó con el resumen y los metadatos verificados en la página de la editorial. La lectura completa se hace al construir la matriz de literatura; si algún estudio no cumple al leerlo completo, se excluye y se actualizan los conteos del PRISMA.

**7. ¿Qué tan confiable es el 9 % de error de los avalúos automáticos?**
Es la cifra que reportan Kok et al. (2017) para activos multifamiliares en Estados Unidos, y dos de los autores trabajan en una empresa que vende ese tipo de modelos. La usamos como referencia del orden de magnitud, no como un resultado que se transfiera a nuestro caso.

**8. ¿Y si el aprendizaje automático no supera al hedónico?**
También es un resultado válido. La literatura sobre estos mismos datos anticipa que un modelo paramétrico bien especificado es competitivo, y por eso las hipótesis se contrastan con pruebas estadísticas y no se dan por ciertas.

### Solución y metodología · responde Brandon

**9. ¿Por qué no deep learning?**
Son 545 observaciones tabulares. En datos tabulares de tamaño pequeño o medio, los modelos de árboles siguen siendo el estado del arte (Grinsztajn et al., 2022), con menos costo y más interpretabilidad. Una red profunda tendría alto riesgo de sobreajuste.

**10. ¿Cómo evitan la fuga de datos?**
La partición de prueba (20 %) se sella al inicio y no se toca hasta la evaluación final. Todo escalado, transformación y selección de variables va dentro de un `Pipeline` que se ajusta solo con los datos de entrenamiento de cada pliegue.

**11. ¿Por qué validación cruzada repetida y anidada?**
La anidada separa la elección de hiperparámetros, en los pliegues internos, de la estimación del error, en los externos, y así el error no sale optimista. Repetirla con distintas particiones da una medida de la variabilidad, que es la que necesitan las pruebas estadísticas.

**12. ¿Cómo sabrán si un modelo es mejor de verdad y no por azar?**
Comparamos los modelos sobre los mismos pliegues y usamos una prueba t corregida para validación cruzada repetida (α = 0,05). Una diferencia pequeña en el RMSE promedio no basta.

**13. ¿Por qué modelar el log del precio?**
El precio es asimétrico, con una cola de valores altos. Con el logaritmo se estabiliza la varianza y el error pasa a ser relativo, lo que ayuda sobre todo a los modelos lineales y de distancia. La hipótesis H2 lo contrasta comparando el mismo modelo con y sin la transformación.

**14. ¿Cómo medirán la complejidad?**
En dos frentes: la teórica, con el orden de costo de entrenamiento y predicción de cada modelo, y la empírica, con los tiempos de entrenamiento y predicción y el número de parámetros. Así se ve cuánto se gana en error por cada unidad de costo.

**15. ¿Cómo explican lo que predice un ensamble?**
Con importancia por permutación para el efecto global de cada atributo y con SHAP para explicar predicciones individuales. Así comparamos lo que aprende el ensamble con los coeficientes del modelo hedónico.

**16. ¿545 observaciones alcanzan?**
Es un tamaño pequeño, y por eso el diseño es conservador: validación cruzada repetida, modelos regularizados y curvas de aprendizaje en el piloto para ver si más datos cambiarían el resultado.

### Dataset · responde Mauricio

**17. ¿De dónde salen los datos y cómo saben que son confiables?**
Son las 546 ventas de Windsor, Ontario, entre julio y septiembre de 1987, publicadas por Anglin y Gençay (1996) en el *Journal of Applied Econometrics* y disponibles en el paquete AER de R. La versión de Kaggle tiene 545 filas: comprobamos que coincide fila por fila con el original y que la fila que falta era un duplicado.

**18. ¿Por qué este dataset no viola la regla de no usar datasets de scikit-learn?**
scikit-learn trae, por ejemplo, California Housing y antes traía Boston. El nuestro es el de Windsor, que viene del archivo del *Journal of Applied Econometrics* y del paquete AER de R, y lo usamos en su versión publicada en Kaggle.

**19. ¿Por qué el precio está multiplicado por 70?**
Kaggle no lo documenta, pero al dividir entre 70 las 545 filas coinciden exactamente con el precio original en dólares canadienses de 1987. Para el modelo no cambia nada, porque es una escala; para interpretar los errores en dinero se divide entre 70, y el MAPE no depende de la escala.

**20. ¿Qué pasa con `furnishingstatus`, si fue añadida en Kaggle?**
No existe en el original y su origen no está documentado, así que la tratamos con cautela. La hipótesis H3 la evalúa con una ablación: entrenamos con y sin la variable y comparamos el RMSE.

**21. ¿Qué hacen con los atípicos?**
Hay 15 precios atípicos altos según la regla de 1,5 × IQR. Son ventas reales, así que los conservamos. Para que no dominen el ajuste modelamos log(precio) y, en uno de los preprocesamientos, aplicamos winsorización ajustada dentro de cada pliegue.

**22. ¿Qué significan realmente las variables?**
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

| Bloque | Plan | Guion leído | Ensayo 1 | Ensayo 2 |
|---|---|---|---|---|
| Problema y motivación (Juan Miguel) | 3:10 | 2:00 | | |
| Revisión de literatura (Juan Miguel) | 1:20 | 1:23 | | |
| Solución y metodología (Brandon) | 3:00 | 2:23 | | |
| Viabilidad: piloto (Brandon) | 1:30 | 1:19 | | |
| Dataset y EDA (Mauricio) | 4:30 | 3:36 | | |
| Plan y cierre (Juan Miguel) | 1:00 | 0:39 | | |
| **Total** | **14:30** | **11:20** | | |
