# Diapositivas 8 y 9 · Contenido para Canva (T11)

**Tarjeta:** T11 · **Prepara y presenta:** Brandon Eduardo Tobar Sánchez · **Revisa:** Juan Miguel León Gómez
**Tiempo:** 90 s por diapositiva (4:30–6:00 y 6:00–7:30 del guion de T14).
**Fuente:** [`03_metodologia.md`](03_metodologia.md) y [`../diagramas/pipeline.png`](../diagramas/pipeline.png). No hay resultados experimentales: el piloto es T12 y la diapositiva 10.
**Diseño (storyboard de T13):** título como mensaje de una línea, etiqueta de sección en mayúsculas, IBM Plex Serif en títulos y IBM Plex Sans en el texto (14 pt o más), azul marino `#0F2A47`, azul `#2B6CC4`, fondo `#F5F6F9`, ladrillo `#B5532F` como acento y píldora lavanda para Brandon. Una idea por diapositiva. Este archivo no edita Canva ni el PPTX del repositorio.

---

## Diapositiva 8 · Solución: pipeline de extremo a extremo

**1. Título (39 caracteres):** `Solución: pipeline de extremo a extremo`
**Etiqueta de sección:** SOLUCIÓN PROPUESTA

**2. Contenido visible**

- **Objetivo (una frase):** comparar el modelo hedónico con modelos de aprendizaje automático con validación rigurosa y sin fuga de información.
- **Flujo (en el diagrama):** dataset → calidad y preparación → partición 80/20 → entrenamiento → preprocesamiento → 6 modelos → validación cruzada → métricas → interpretabilidad → resultados.
- **Tres llamadas junto a la figura:**
  1. **436 filas** entrenan; **109 quedan selladas** hasta la evaluación final.
  2. **Todo se ajusta dentro de cada pliegue** (escalado, transformaciones, selección).
  3. **18 pipelines** (3 × 6) evaluados con **los mismos pliegues**.
- **Mensaje académico principal (recuadro azul marino):** *El conjunto de prueba no participa en ninguna decisión, así que el error final que reportemos será una estimación independiente.*

**3. Figura sugerida:** `docs/diagramas/pipeline.png` (3840 × 2160, 16:9) a ancho completo o al 70 % con las tres llamadas a la derecha. El test sellado está en rojo con borde discontinuo y no se conecta a la validación.

**4. Guion del orador (≈ 90 s)**

> "Esta es la solución completa, de punta a punta. Partimos del dataset de Windsor, con 545 viviendas, que Mauricio ya limpió y dejó con una partición fija 80/20: 436 filas para entrenar y 109 selladas.
>
> Lo importante es lo que está en rojo: ese conjunto sellado no entra en la validación, ni en el ajuste, ni en la elección de modelos. Solo se abre una vez, al final del proyecto.
>
> Con las 436 filas de entrenamiento armamos un pipeline de scikit-learn con tres partes: un preprocesamiento, un modelo y una validación cruzada. Todo lo que se aprende de los datos, como el escalado, las transformaciones o la selección de variables, se ajusta únicamente con el pliegue de entrenamiento de cada vuelta. Así evitamos la fuga de información.
>
> Combinamos tres preprocesamientos con seis modelos: son 18 pipelines, y todos se evalúan con los mismos pliegues. Al final medimos error, complejidad e interpretabilidad. La idea de fondo: el error que reportemos será una estimación honesta."

**5. Tiempo objetivo:** 90 s. El guion tiene ≈ 165 palabras (75–90 s según el ritmo); cronometrar en los ensayos de T14 y ajustar.

**6. Preguntas probables y respuestas defendibles**

| Pregunta | Respuesta |
|---|---|
| ¿Cómo evitan la fuga de datos? | Cada transformación aprendida (escalado, Yeo-Johnson, winsorización, selección) es un paso del `Pipeline` y se ajusta solo con el pliegue de entrenamiento. Además, el test está sellado desde T06 y los notebooks de modelado cargan solo las filas de entrenamiento. |
| ¿Por qué 80/20 y por qué sellado? | Con 545 filas, 436 alcanzan para validar con repetición y 109 dan una estimación final independiente. Se sella para que ninguna decisión se tome mirando ese error. |
| ¿Quién garantiza que no se mira el test? | Es una regla de proceso: el cargador filtra `entrenamiento` y verifica que no hay ids de prueba; además, la revisión en cada PR. No es una barrera técnica, y lo declaramos como tal. |
| ¿Por qué 18 pipelines y no solo el mejor? | Para atribuir las diferencias de error al preprocesamiento o al modelo, con pares comparables sobre los mismos pliegues. |
| ¿Esto sirve para predecir precios en Colombia? | No. Los datos son de Windsor, 1987. Lo que se transfiere es el método. |

---

## Diapositiva 9 · Metodología: 3 preprocesamientos × 6 modelos

**1. Título (44 caracteres):** `Metodología: 3 preprocesamientos × 6 modelos`
**Etiqueta de sección:** METODOLOGÍA

**2. Contenido visible** (una tabla y una franja; sin párrafos)

| Preprocesamiento | Qué cambia |
|---|---|
| **P1 · Base** | Codificación de categorías + estandarización donde el modelo la necesita |
| **P2 · Transformaciones** | log(precio) · Yeo-Johnson · escalado robusto |
| **P3 · Ingeniería y selección** | Variables derivadas · interacciones · winsorización · selección |

**Modelos (6):** OLS (hedónico) · Ridge · KNN · SVR (RBF) · Random Forest · XGBoost.

**Franja inferior, en cuatro bloques:**

| Validación | Métricas | Complejidad | Interpretabilidad |
|---|---|---|---|
| Piloto: repetida 5 × 3. Final: anidada. Prueba t corregida | **RMSE** (principal) · MAE · MAPE · R² | Teórica y empírica: tiempo, memoria | Coeficientes · permutación · SHAP · PDP |

**Mensaje académico principal (recuadro azul marino):** *Un diseño factorial controlado: cada diferencia de error se puede asociar a un preprocesamiento o a un modelo, con los mismos pliegues.*

**3. Figura o tabla sugerida:** la tabla P1/P2/P3 a la izquierda, los seis modelos en seis fichas a la derecha (colores por familia: lineales, distancia, kernel, árboles) y la franja de cuatro bloques abajo. Opcional: miniatura de `pipeline.png` en la esquina como referencia.

**4. Guion del orador (≈ 90 s)**

> "Ahora, qué comparamos. Tres preprocesamientos, que se van sumando. El P1 es la base: codificar las categorías y estandarizar solo donde el modelo lo necesita. El P2 agrega transformaciones: el logaritmo del precio, que es asimétrico, Yeo-Johnson y un escalado robusto. El P3 agrega variables derivadas, interacciones, recorte de extremos y selección de variables.
>
> Los seis modelos van de lo simple a lo flexible: la regresión lineal hedónica como línea base, Ridge, vecinos cercanos, SVR con kernel, Random Forest y XGBoost.
>
> Los evaluamos con validación cruzada repetida 5 por 3 en el piloto, y anidada en el experimento final, que separa la elección de hiperparámetros del cálculo del error. La métrica principal es el RMSE; también reportamos MAE, MAPE y R². Para saber si una diferencia es real, usamos una prueba t corregida sobre los mismos pliegues.
>
> Además medimos la complejidad, con tiempos y memoria, y la interpretabilidad, con coeficientes, importancia por permutación, SHAP y gráficos de dependencia parcial. Y siempre dejamos claro que explicar un modelo no es probar causalidad."

**5. Tiempo objetivo:** 90 s. El guion tiene ≈ 170 palabras (75–90 s según el ritmo); cronometrar en los ensayos de T14 y ajustar.

**6. Preguntas probables y respuestas defendibles**

| Pregunta | Respuesta |
|---|---|
| ¿Por qué estos seis modelos? | Cubren cuatro familias: lineal (OLS, Ridge), basada en instancias (KNN), de kernel (SVR) y ensambles de árboles (Random Forest, XGBoost). T04 las fija en la pregunta y el objetivo 3, y los ensambles son los que contrasta H1. |
| ¿Por qué no deep learning? | Son 545 observaciones tabulares. En datos tabulares de tamaño medio los modelos de árboles siguen siendo el estado del arte según Grinsztajn et al. (2022), y una red profunda tendría alto riesgo de sobreajuste. |
| ¿Por qué el logaritmo del precio? | El precio es asimétrico a la derecha. El logaritmo estabiliza la varianza y vuelve el error relativo. La hipótesis H2 lo contrasta con los mismos pliegues; no damos por hecho que mejore. Las métricas se calculan siempre en la escala original, y para el modelo lineal es la forma semilogarítmica hedónica clásica. |
| ¿Por qué P3 es acumulativo y no independiente? | Para aislar efectos: P1→P2 mide transformar y P2→P3 mide diseñar variables. Las variables derivadas y las interacciones tienen una justificación hedónica y se evalúan por ablación; no se mantienen si no ayudan. |
| ¿Qué diferencia hay entre validación repetida y anidada? | La repetida mide la variabilidad del error repitiendo las particiones. La anidada, además, separa la elección de hiperparámetros (bucle interno) de la estimación del error (bucle externo); sin eso el error sale optimista. En el piloto no ajustamos hiperparámetros, por eso la repetida basta. |
| ¿Por qué RMSE como métrica principal? | H1 se formula sobre el RMSE y penaliza los errores grandes, que son los costosos al valorar una vivienda. Pero es sensible a extremos, por eso se complementa con MAE, MAPE y R². |
| ¿Cómo saben que una diferencia no es azar? | Con la prueba t corregida de Nadeau y Bengio sobre diferencias pareadas por pliegue, con corrección de Holm en comparaciones múltiples. Una diferencia pequeña de RMSE promedio no basta. |
| ¿Y si el modelo lineal gana o queda cerca? | También es un resultado válido y publicable. La literatura sobre estos datos anticipa que un modelo paramétrico bien especificado puede ser competitivo; no suponemos el resultado. |
| ¿SHAP dice qué causa el precio? | No. SHAP y la permutación describen qué usa el modelo, no causas. Con atributos correlacionados, además, la importancia se reparte entre ellos. |

---

## Notas para Juan Miguel (integración en Canva)

- La figura es la misma de la diapositiva 8; el archivo editable está en `docs/diagramas/pipeline.drawio` por si hay que ajustar colores.
- Las diapositivas 8 y 9 no incluyen cifras de desempeño; los resultados llegan con T12 (diapositiva 10).
- Las referencias que se pueden citar al pie: Rosen (1974), Grinsztajn et al. (2022), Nadeau y Bengio (2003), con la numeración de `03_metodologia.md`.
- Si hay que recortar: en la diapositiva 9, conservar la tabla P1–P3, la lista de modelos y el mensaje; las cuatro columnas de la franja pasan a las notas del orador.
