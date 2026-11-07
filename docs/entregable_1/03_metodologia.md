# 3. Metodología propuesta

**Tarjeta:** T11 · **Responsable:** Brandon Eduardo Tobar Sánchez (Model Developer · Evaluation Lead) · **Revisa:** Juan Miguel León Gómez
**Uso:** sección 3 de la propuesta ("Solución y proceso propuesto"), diapositivas 8 y 9, y especificación que implementa T12 (piloto) y, después, T17–T25.
**Diagrama:** [`../diagramas/pipeline.png`](../diagramas/pipeline.png) (editable: [`pipeline.drawio`](../diagramas/pipeline.drawio)).
**Estado:** este documento **no contiene resultados experimentales**. Todo lo que dice sobre el desempeño de los modelos es una hipótesis (H1–H3) que se contrastará con datos de entrenamiento. El conjunto de prueba sellado no se usó para escribirlo.

---

## 3.1 Descripción del enfoque

**Pregunta que responde** (T04, 1.4): ¿en qué medida los modelos de aprendizaje automático mejoran la predicción del precio de venta de una vivienda frente al modelo hedónico lineal, y qué atributos explican sus predicciones?

**Enfoque.** Es un experimento comparativo de regresión supervisada con un diseño factorial de **3 estrategias de preprocesamiento × 6 modelos = 18 pipelines**, evaluados con **los mismos pliegues** de validación cruzada sobre el 80 % de entrenamiento. El 20 % de prueba queda sellado hasta la evaluación final (T25).

| Decisión de diseño | Razón | Se vincula con |
|---|---|---|
| El modelo hedónico lineal (OLS) es la línea base | Es el estándar en economía de la vivienda [1] y, sobre los datos de Windsor, una especificación paramétrica bien elegida puede ser competitiva [4] | OE3, H1 |
| Seis modelos de cuatro familias (lineales, distancia, kernel, ensambles de árboles) | T04 fija esas familias en la pregunta y en OE3; con n = 545 se priorizan modelos tabulares y regularizados sobre redes profundas [26] | OE3, H1 |
| Tres preprocesamientos acumulativos | Permite aislar el efecto de transformar (P1→P2) y de diseñar variables (P2→P3) sobre los mismos modelos y folds | OE2, H2 |
| Validación cruzada repetida (piloto) y anidada (final) | Con 436 filas de entrenamiento la varianza de una sola partición es alta; la repetición mide esa variabilidad que necesitan las pruebas estadísticas | H1, H2 |
| Todo ajuste dentro del `Pipeline` | Evita la fuga de información (3.8) | OE1 |
| Complejidad e interpretabilidad como criterios de comparación | La literatura reconoce que el aprendizaje automático gana exactitud con costo en transparencia [12] | OE4 |

**Lo que el proyecto no afirma.** No predice precios actuales ni colombianos, y sus conclusiones son predictivas, no causales (T04, 1.7). Su aporte es comparar **métodos** con rigor sobre un *benchmark* con estudios previos [2], [4].

**Fuentes de la metodología.** T04 (pregunta, objetivos, hipótesis), T05 y T06 (datos, calidad y partición), T09 (protocolo y estudios incluidos), el storyboard (T13, "3 preprocesamientos × 6 modelos") y el banco de preguntas (T14). La tarjeta de Trello no era accesible al redactar; la lista de familias de modelos de T04 (OLS, regularizados, KNN, SVR y ensambles) coincide con la selección de abajo, y la concreción en seis modelos es una decisión de T11 que Juan Miguel puede ajustar en la revisión.

---

## 3.2 Datos y variable objetivo

| Elemento | Definición | Fuente |
|---|---|---|
| Archivo de trabajo | `data/raw/Housing.csv` (solo lectura), 545 filas × 13 columnas | T05 |
| Archivos de modelado | `data/processed/housing_limpio.csv` (545 filas, columna `id`) y `data/processed/particion.csv` (`id`, `conjunto`) | T06 |
| Variable objetivo | `price`, entera, positiva; mínimo 1.750.000 y máximo 13.300.000 en unidades de Kaggle | T05, T06 |
| Escala del precio | Unidades de Kaggle = dólares canadienses de 1987 × 70, factor verificado fila por fila en T05 | T05 |
| Predictoras (12) | Numéricas: `area` (pies²), `bedrooms`, `bathrooms`, `stories`, `parking`. Binarias: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`. Categórica: `furnishingstatus` (3 niveles) | T05 |
| Partición | 436 filas `entrenamiento` y 109 `prueba` (80/20), estratificada por quintiles de precio, semilla 5007 | T06 |

**Cómo se identifica el conjunto de entrenamiento y el sellado.** `particion.csv` tiene una fila por `id` con `conjunto ∈ {entrenamiento, prueba}`. El modelado une `housing_limpio.csv` con `particion.csv` por `id` y **conserva solo** `conjunto == "entrenamiento"`. Las filas `prueba` no se usan para ajustar, comparar ni elegir preprocesamientos o modelos (T12 a T24); solo T25 las evalúa. Las asimetrías citadas en este documento se calcularon solo con entrenamiento.

**Estado de los datos que importan al diseño** (T06): completitud del 100 % (no hay valores faltantes que imputar); 15 precios atípicos altos por la regla 1,5 × IQR, que son ventas reales y se conservan; `hotwaterheating` es una clase muy poco frecuente (4,59 % de "sí"); `furnishingstatus` fue agregada en Kaggle sin documentar su procedimiento (T05).

**Un detalle de T06 que condiciona P1.** `housing_limpio.csv` guarda `furnishingstatus` como entero ordinal (0 = unfurnished, 1 = semi-furnished, 2 = furnished). Como el orden de esa variable no está documentado, los preprocesamientos la tratan como **nominal** y la expanden con *one-hot* a partir de esos tres códigos. No se modifica el archivo de T06.

---

## 3.3 Preparación de datos

La preparación se reparte en dos capas, que es lo que evita la fuga de información:

| Capa | Qué hace | Dónde vive | ¿Usa estadísticos del conjunto? |
|---|---|---|---|
| **Determinista (T06)** | Codifica binarias a 0/1, codifica `furnishingstatus`, asigna `id`, verifica dominios y crea la partición | `src/datos.py` → `housing_limpio.csv` | No: son reglas fijas, iguales para toda fila |
| **Ajustable (T11 → T12)** | Escalado, transformaciones de variables, derivadas, winsorización, selección | `Pipeline` de scikit-learn, ajustado **dentro de cada pliegue** | Sí: solo con las filas de entrenamiento del pliegue |

**Valores faltantes.** T06 mostró 0 nulos, de modo que los pipelines **no incluyen imputación**. Se agrega una verificación de integridad al cargar los datos (`assert df.isna().sum().sum() == 0`): si algún día aparecen nulos, el experimento se detiene y se decide cómo tratarlos, en vez de imputar sin querer.

**Atípicos del precio.** No se eliminan filas: son ventas reales y borrarlas sesgaría la cola de la distribución que el modelo debe predecir. Su efecto se controla con el logaritmo de `price` (P2) y con modelos robustos (árboles).

---

## 3.4 Estrategias de preprocesamiento P1, P2 y P3

Los tres preprocesamientos son **acumulativos**: P2 = P1 + transformaciones, P3 = P2 + ingeniería y selección. Así, P1 → P2 aísla el efecto de transformar (H2) y P2 → P3 el de diseñar variables.

### P1 · Base (punto de comparación)

| Componente | Decisión | Justificación |
|---|---|---|
| Binarias (6) | Pasan sin cambio (ya son 0/1 desde T06) | No necesitan más codificación |
| `furnishingstatus` | `OneHotEncoder(drop="first")` sobre los códigos 0/1/2 | Orden no documentado (3.2); `drop="first"` evita la colinealidad perfecta en OLS y Ridge |
| Numéricas (5) | `StandardScaler` **solo** para los modelos sensibles a la escala (Ridge, KNN, SVR); sin escalar para OLS, Random Forest y XGBoost | OLS y los árboles son invariantes a la escala; escalar donde no hace falta solo agrega pasos |
| Objetivo `price` | Escala original. Para SVR se estandariza el objetivo dentro del pliegue (`TransformedTargetRegressor` con `StandardScaler`), porque `price` es del orden de 10⁶ y los parámetros de SVR no están pensados para esa magnitud | Necesidad numérica, no una estrategia de mejora |
| Faltantes | Ninguno (3.3) | T06 |

### P2 · Transformaciones

| Componente | Decisión | Justificación |
|---|---|---|
| Objetivo | `log(price)` con `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)` | El precio es asimétrico a la derecha (asimetría del conjunto de entrenamiento ≈ 1,2); el logaritmo estabiliza la varianza y vuelve el error relativo. Para OLS equivale a la forma semilogarítmica hedónica clásica |
| `area` | Yeo-Johnson (`PowerTransformer(method="yeo-johnson")`) | `area` también tiene cola derecha (asimetría de entrenamiento ≈ 1,3). Yeo-Johnson acepta cualquier valor y estima el exponente solo con el pliegue de entrenamiento [8] |
| `bedrooms`, `bathrooms`, `stories`, `parking` | No se transforman; se escalan con `RobustScaler` (mediana y rango intercuartílico) | Son enteros pequeños (1–6, 1–4, 1–4, 0–3). Yeo-Johnson no tiene sentido estadístico sobre conteos tan acotados |
| Escalado | `RobustScaler` en lugar de `StandardScaler`, solo para Ridge, KNN y SVR | Menos sensible a los valores extremos de `area` |

**Inversión del objetivo y métricas comparables.** Las métricas se calculan **siempre en la escala original del precio**. `TransformedTargetRegressor` aplica `np.exp` a las predicciones antes de devolverlas, de modo que `RMSE`, `MAE` y `MAPE` de P1 y P2 se comparan sobre la misma variable. Hay un sesgo conocido: `exp(E[log y])` estima la mediana condicional y no la media, así que subestima en promedio. Se reporta tal cual en el piloto; como análisis de sensibilidad se puede aplicar la corrección de *smearing* de Duan [9] y comprobar si cambia el ranking. Como diagnóstico secundario (no como sustituto) se puede reportar el error en escala logarítmica.

**Sin fuga.** `log` no tiene parámetros que aprender; Yeo-Johnson y los escaladores sí, y se ajustan en cada pliegue (3.8).

### P3 · Ingeniería y selección de variables

| Componente | Decisión | Justificación |
|---|---|---|
| Base | Todo P2 | Diseño acumulativo |
| Variables derivadas | `area_por_dormitorio = area / bedrooms` (`bedrooms ≥ 1`, T06); `n_comodidades` = suma de `guestroom`, `basement`, `hotwaterheating`, `airconditioning` | Resumen del espacio por habitación y del nivel de equipamiento, conceptos centrales de la teoría hedónica de atributos [1]. Ninguna usa estadísticos de la muestra |
| Interacciones | `area × prefarea` y `area × airconditioning` | Hipótesis de diseño: el valor del lote puede depender de la localización preferida y del equipamiento. No se asume que ayuden; se miden por ablación |
| Atípicos | Winsorización (recorte) de `area` y de las variables derivadas a los percentiles 1 y 99, **aprendidos en el pliegue de entrenamiento** | Limita la influencia de los extremos de `area` sin borrar filas |
| Selección | `SelectKBest(mutual_info_regression, k)` con `k ∈ {8, 10, 12, todas}` elegido en el bucle interno | La información mutua captura relaciones no lineales; el EDA (T07) usa el mismo criterio. Se reporta con qué frecuencia cada variable sobrevive a los pliegues |
| Reducción de dimensionalidad | **No se incluye PCA** | Con 12 variables y 436 filas no hay un problema de alta dimensionalidad que lo justifique; PCA además dificulta la interpretación del experimento. Solo se reconsiderará si el EDA muestra colinealidad severa (VIF alto) |
| Variables originales | **Ninguna se elimina a mano.** Solo la selección dentro del pliegue puede excluirlas, y queda registrado cuáles | Evita decisiones arbitrarias y fuga |

**Ajuste de H3.** La ablación de `furnishingstatus` (T18) es un experimento aparte: se repite el mismo pipeline sin esa variable y se compara el RMSE con los mismos pliegues.

**Implementación.** Los tres preprocesamientos se construyen con `ColumnTransformer` y `Pipeline`. La winsorización es un transformador propio (`fit` aprende los percentiles; `transform` recorta). T12 los implementa en `src/preprocesamiento.py`; esta rama no incluye el experimento.

---

## 3.5 Modelos de regresión

Se usan seis modelos de cuatro familias. Los hiperparámetros se fijan en T12 (valores por defecto razonables) y se ajustan en el experimento final (T19–T22) con el bucle interno de la validación anidada.

| Id | Modelo | Familia y relación que representa | Ventajas | Limitaciones | Sensibilidad al preprocesamiento | Hiperparámetros principales (búsqueda final) |
|---|---|---|---|---|---|---|
| **M1** | Regresión lineal OLS | Lineal; efectos aditivos y constantes por atributo. Es el **modelo hedónico** | Coeficientes interpretables, sin hiperparámetros, costo mínimo | Sin no linealidad ni interacciones; sensible a colinealidad y a extremos | Alta al logaritmo del objetivo; invariante a la escala de las predictoras | Ninguno |
| **M2** | Ridge | Lineal regularizada (penalización L2) | Estabiliza los coeficientes con variables correlacionadas | Sigue siendo lineal; sesga los coeficientes | Requiere predictoras en una misma escala | `alpha` (rejilla logarítmica, 10⁻³ a 10³) |
| **M3** | KNN regresor | Basada en instancias; promedia vecinos cercanos | Sin supuesto de forma funcional; capta relaciones locales | Sufre con muchas dimensiones; predicción costosa; depende de la distancia | **Muy alta** a la escala y a las variables irrelevantes | `n_neighbors` {3, 5, 7, 11, 15, 21}, `weights` {uniform, distance}, `p` {1, 2} |
| **M4** | SVR kernel RBF | Kernel; regresión no lineal con margen ε | Flexible con pocos datos; controlada por regularización | Costo cuadrático o peor en n; muchos hiperparámetros; poco interpretable | **Muy alta** a la escala de predictoras y del objetivo | `C` {0,1, 1, 10, 100}, `epsilon` {0,01, 0,1, 0,2}, `gamma` {scale, 0,01, 0,1} |
| **M5** | Random Forest | Ensamble por *bagging* de árboles; interacciones y no linealidad automáticas | Robusto a atípicos y a la escala; superó a otros métodos comparados en valoración masiva [6] | No extrapola; menos interpretable; puede sobreajustar con ruido | Baja a la escala; media a las variables derivadas | `max_depth` {None, 6, 10}, `min_samples_leaf` {1, 3, 5}, `max_features` {0,33, 0,6, 1,0}; `n_estimators` = 500 fijo |
| **M6** | XGBoost | Ensamble por *boosting* de gradiente [7] | Suele ser el más exacto en datos tabulares [20], [26]; regularización integrada | Más hiperparámetros; riesgo de sobreajuste con n pequeño | Baja a la escala | `n_estimators` {200, 400}, `learning_rate` {0,03, 0,1}, `max_depth` {2, 3, 4}, `subsample` {0,7, 1,0}, `colsample_bytree` {0,7, 1,0}, `reg_lambda` {1, 5} |

**Papel en el proyecto.** M1 es la línea base hedónica; M2 prueba si la regularización basta con una relación lineal; M3 y M4 prueban si hay estructura local o de kernel; M5 y M6 representan los ensambles que H1 compara con el hedónico (T04 los nombra: Random Forest o *gradient boosting*). La variante de *gradient boosting* elegida es XGBoost.

**Cómo se comparan los lineales y los no lineales.** Con las mismas particiones, el mismo preprocesamiento y las mismas métricas. Las diferencias se estiman **por pares** (mismo pliegue) y se contrastan con la prueba t corregida (3.6); no se decide por el promedio solo.

**Sobre lo que se espera.** La literatura revisada en T09 reporta que los ensambles suelen ser más exactos [20], [26], pero también que una especificación paramétrica bien elegida puede ser competitiva sobre estos datos [4]. **Este documento no afirma que ningún modelo sea mejor.** Lo decide el experimento.

---

## 3.6 Validación experimental

Tres niveles, que no deben confundirse:

| Nivel | Qué es | Datos | Para qué | Cuándo |
|---|---|---|---|---|
| **A. Piloto (T12)** | Validación cruzada **repetida** 5 × 3 (5 pliegues, 3 repeticiones = 15 evaluaciones) | Solo las 436 filas de entrenamiento | Estimar el error de los 18 pipelines con hiperparámetros **fijos** y medir viabilidad | Antes del 7 de noviembre |
| **B. Experimento final (T19–T22)** | Validación cruzada **anidada**: bucle externo repetido 5 × 3; bucle interno de 3 pliegues para elegir hiperparámetros (y `k` de la selección en P3) | Solo las 436 filas de entrenamiento | Estimar el error **sin sesgo optimista** cuando se ajustan hiperparámetros | Hasta el 5 de diciembre |
| **C. Prueba sellada (T25)** | Un único ajuste final con la configuración elegida por A y B, evaluado **una vez** en las 109 filas de `prueba` | Conjunto sellado | Estimación final e independiente del error | Al final del proyecto |

**Por qué no es lo mismo.** La validación repetida mide la variabilidad del error sobre repeticiones, pero si en ella se eligen hiperparámetros, el mejor resultado queda optimista. La validación **anidada** separa la elección de hiperparámetros (bucle interno) de la estimación del error (bucle externo). Por eso el piloto usa hiperparámetros fijos: con ellos, la repetida es suficiente y barata.

**Detalles de implementación.**

- **Semilla:** 5007 en `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=5007)`, en cada estimador con aleatoriedad y en `mutual_info_regression`.
- **Estratificación:** los pliegues se estratifican por quintiles del precio de **entrenamiento**, como en T06, para que cada pliegue tenga un rango de precios representativo. Las etiquetas de estrato solo salen de `price` del entrenamiento.
- **Mismos pliegues para todos:** se genera una lista fija de índices (`list(cv.split(X, estratos))`) y se pasa a todos los pipelines. Así cada diferencia entre modelos o preprocesamientos es **pareada**.
- **Prueba estadística.** Para cada contraste planificado (H1: ensamble vs OLS; H2: P2 vs P1) se calcula la diferencia por pliegue y se aplica la **prueba t corregida para validación cruzada repetida** de Nadeau y Bengio [5]: con $J$ evaluaciones, $t = \bar d / \sqrt{(1/J + n_{prueba}/n_{entren})\, s_d^2}$ y $J-1$ grados de libertad. Con 5 pliegues, $n_{prueba}/n_{entren} = 1/4$. Con varias comparaciones se aplica la corrección de Holm. α = 0,05, como en T04.

---

## 3.7 Métricas de evaluación

Se calculan **por pliegue** sobre las filas de validación, en la **escala original del precio**, y se resumen con media y desviación estándar. Sea $y_i$ el precio real, $\hat y_i$ la predicción y $n$ el número de filas del pliegue.

| Métrica | Fórmula | Unidades | Interpretación | Ventajas | Limitaciones |
|---|---|---|---|---|---|
| **RMSE** (principal) | $\sqrt{\tfrac{1}{n}\sum (y_i-\hat y_i)^2}$ | Las del precio | Error típico, con más peso para los errores grandes | Penaliza los fallos grandes, que son los costosos en valoración; coincide con el criterio de H1 | Sensible a valores extremos: una sola vivienda de precio muy alto mal predicha puede dominarlo |
| **MAE** | $\tfrac{1}{n}\sum \lvert y_i-\hat y_i\rvert$ | Las del precio | Error absoluto promedio: cuánto se equivoca el modelo en una vivienda típica | Robusto a extremos; fácil de explicar | No distingue un error grande de varios pequeños |
| **MAPE** | $\tfrac{100}{n}\sum \lvert (y_i-\hat y_i)/y_i\rvert$ | Porcentaje | Error relativo promedio, comparable entre escalas | No depende del factor ×70 de Kaggle; lo usa H2 | Explota si $y_i \approx 0$ (aquí el mínimo es 1.750.000, sin ese riesgo); castiga más la sobreestimación que la subestimación; no se acota por arriba |
| **R²** | $1-\tfrac{\sum (y_i-\hat y_i)^2}{\sum (y_i-\bar y)^2}$ | Sin unidades | Fracción de varianza explicada respecto de predecir la media | Escala libre; fácil de comparar con T04 (R² dentro de muestra, 1.8) | **Puede ser negativo** fuera de muestra si el modelo predice peor que la media; con $\bar y$ calculada en el pliegue |

**Por qué RMSE como principal.** H1 se plantea sobre el RMSE de validación cruzada y es coherente con T04. MAE, MAPE y R² complementan: ninguna métrica sola cuenta toda la historia, y la literatura de AVM recomienda reportar varias porque la métrica elegida puede cambiar el modelo ganador [16].

**Cómo se informan las unidades.** El precio de Kaggle es dólares canadienses de 1987 × 70 (T05). Se reporta el RMSE y el MAE en esas unidades y, para dar sentido, también divididos entre 70 (dólares canadienses de 1987). **No se interpretan** como precios actuales ni como pesos colombianos. MAPE y R² no dependen de la escala. No se incluyen cifras en este documento.

---

## 3.8 Prevención de fuga de información

| # | Riesgo de fuga | Mecanismo de control |
|---|---|---|
| 1 | Usar el test en cualquier decisión | Las filas `prueba` no se cargan en T12 ni en EDA: el cargador filtra `conjunto == "entrenamiento"` y verifica con `assert` que no hay ids de prueba. T25 es el único que abre el test |
| 2 | Escaladores y Yeo-Johnson aprendidos con todo el conjunto | Están **dentro** del `Pipeline`: `fit` solo con el pliegue de entrenamiento; `transform` aplicado al de validación |
| 3 | Selección de variables con el objetivo de todo el conjunto | `SelectKBest` es un paso del `Pipeline`, ajustado por pliegue |
| 4 | Percentiles de winsorización de todo el conjunto | Transformador propio con `fit` por pliegue |
| 5 | Ingeniería de variables con estadísticos globales | Las variables derivadas son fórmulas fijas por fila; las que requieren estadísticos se ajustan por pliegue |
| 6 | Transformación del objetivo | `log` no tiene parámetros; cualquier transformación con parámetros del objetivo se hace con `TransformedTargetRegressor`, que se ajusta en el pliegue |
| 7 | Hiperparámetros elegidos en los mismos datos que estiman el error | Validación anidada en el experimento final (3.6). En el piloto no se ajustan |
| 8 | Estratos de la partición | T06 los calculó con el precio de todo el conjunto antes de partir, solo para asignar filas; no los usa ningún modelo |
| 9 | Duplicados casi idénticos entre pliegues | T06 audita unicidad y reporta ruido irreducible por vectores repetidos; se mantiene la partición por fila y se declara como limitación |

**Cómo se arman las piezas** (resumen del esquema que implementa T12):

- `ColumnTransformer` aplica a cada grupo de columnas (numéricas, binarias, `furnishingstatus`) la transformación que le toca.
- `Pipeline` encadena preprocesamiento, selección (P3) y modelo.
- `TransformedTargetRegressor` envuelve el pipeline para transformar el objetivo (log o estandarización) y revertirlo en la predicción.
- La misma lista de pliegues se reutiliza en `cross_validate`, con `n_jobs` fijado y todas las semillas en 5007.

**Esquema de referencia.** Los 18 pipelines (P1–P3 × M1–M6) descritos en 3.4 y 3.5 se construyeron con scikit-learn y se comprobó que ajustan y predicen en un pliegue de entrenamiento (18 de 18). No se calcularon métricas ni se usó el conjunto de prueba; la evaluación es trabajo de T12. Fragmento ilustrativo:

```python
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
import numpy as np

# P2 + M2: transformaciones (Yeo-Johnson en area, RobustScaler en el resto) + Ridge sobre log(price)
prepro_p2 = ColumnTransformer([
    ("area", PowerTransformer(method="yeo-johnson"), ["area"]),
    ("conteos", RobustScaler(), ["bedrooms", "bathrooms", "stories", "parking"]),
    ("amoblado", OneHotEncoder(drop="first"), ["furnishingstatus"]),
], remainder="passthrough")      # binarias ya codificadas en T06

modelo = TransformedTargetRegressor(
    regressor=Pipeline([("prepro", prepro_p2), ("ridge", Ridge(alpha=1.0))]),
    func=np.log, inverse_func=np.exp)   # las métricas se calculan en escala original
```

---

## 3.9 Complejidad computacional

Se documenta en dos dimensiones. No existe una fórmula asintótica universal: depende de la implementación (scikit-learn, XGBoost), de los hiperparámetros y de la estructura de datos usada.

### Complejidad teórica aproximada

$n$ = filas de entrenamiento, $p$ = variables, $T$ = árboles, $d$ = profundidad, $m$ = variables por división, $n_{sv}$ = vectores de soporte.

| Modelo | Costo de entrenamiento | Costo de predicción por observación | Memoria | Depende principalmente de |
|---|---|---|---|---|
| OLS | $O(np^2 + p^3)$ | $O(p)$ | $O(p^2)$ | $n$ y $p$ |
| Ridge | $O(np^2 + p^3)$ (solución cerrada) | $O(p)$ | $O(p^2)$ | $n$, $p$; `alpha` casi no cambia el costo |
| KNN | Casi nulo (almacena $n$ filas; con árboles $O(n\log n)$) | $O(np)$ por búsqueda exhaustiva; menor con índices | $O(np)$ | $n$, $p$, `n_neighbors` |
| SVR RBF | Entre $O(n^2 p)$ y $O(n^3)$ (libsvm) | $O(n_{sv}\,p)$ | $O(n_{sv})$ más el caché del kernel | $n$ (superlineal), `C`, `epsilon`, `gamma` |
| Random Forest | $O(T\, m\, n \log n)$ aproximado | $O(T\, d)$ | $O(T \cdot n_{nodos})$ | $T$, $n$, $m$, `max_depth` |
| XGBoost | $O(K\, d\, n\, p)$ aproximado con histogramas ($K$ = árboles) | $O(K\, d)$ | $O(K \cdot n_{nodos})$ | $K$, `max_depth`, `learning_rate` indirectamente |

Estas cifras ordenan el costo esperado entre familias; no predicen tiempos reales.

### Complejidad empírica (se mide en T12 y T23)

| Medición | Cómo | Nota |
|---|---|---|
| Tiempo de entrenamiento | `time.perf_counter` alrededor de `fit`, por pliegue; mediana de las 15 evaluaciones | Se registran CPU, núcleos y versiones de librerías |
| Tiempo de inferencia | Por observación y por lote de 109 filas | Mismo hardware |
| Memoria | Pico de memoria durante `fit` cuando sea viable (`tracemalloc` o `psutil`) | No siempre medible para XGBoost |
| Tamaño del modelo | Coeficientes, $n_{sv}$, número de árboles y nodos | Complementa el tiempo |
| Escalabilidad | Curva de aprendizaje: tiempo y RMSE frente al número de filas de entrenamiento | Mostrar si más datos cambiarían el resultado (n pequeño) |

El piloto T12 aporta las primeras mediciones reales. Este documento **no incluye tiempos**.

---

## 3.10 Interpretabilidad

Se distinguen dos tipos:

- **Intrínseca:** el modelo es interpretable por construcción (coeficientes de OLS y Ridge).
- **Post hoc:** se explica un modelo ya entrenado (permutación, SHAP, PDP), con supuestos propios.

| Técnica | Tipo | Qué explica | Cómo se aplica | Cuándo |
|---|---|---|---|---|
| **Coeficientes** (M1, M2) | Intrínseca | Cambio esperado del objetivo por unidad de cada atributo, con los demás fijos | Coeficientes estandarizados. En P2, con `log(price)`, $100(e^{\beta}-1)$ aproxima el cambio porcentual en el precio | Piloto |
| **Importancia por permutación** | Post hoc, global | Cuánto empeora el error al desordenar un atributo | Sobre las filas de **validación** del pliegue, con RMSE como métrica; para grupos de variables correlacionadas, permutar en bloque | Piloto y final |
| **SHAP** [10] | Post hoc, local y global | Contribución de cada atributo a una predicción concreta | `TreeExplainer` para M5 y M6; en la escala del objetivo del modelo (en P2, log-precio) | Final (T24) |
| **PDP / ICE** | Post hoc, global | Efecto marginal medio de un atributo en la predicción | Para los mejores modelos; con curvas ICE para ver heterogeneidad | Final (T24) |

**Riesgos que se declaran:**

1. **Correlación entre variables** (por ejemplo `area`, `bedrooms`, `bathrooms`): reparte la importancia entre ellas, y la permutación puede evaluar combinaciones irreales.
2. **Importancias inestables** con n pequeño: se reportan con su variabilidad entre pliegues.
3. **Interpretación causal:** una importancia alta o un valor SHAP alto **no** significa que el atributo cause el precio. Son asociaciones aprendidas por el modelo (T04, 1.7).
4. **Extrapolación:** PDP y SHAP son fiables solo en el rango de los datos; los árboles no extrapolan.
5. **Comparabilidad:** se contrasta lo aprendido por el ensamble con los coeficientes del modelo hedónico, que es el objetivo OE4.

---

## 3.11 Riesgos y limitaciones

| Limitación del dataset o del diseño | Consecuencia | Decisión metodológica asociada |
|---|---|---|
| Muestra pequeña (n = 545; 436 de entrenamiento) | Alta varianza de las estimaciones; riesgo de sobreajuste | Validación repetida y anidada; modelos regularizados; curvas de aprendizaje; no se usa *deep learning* [26] |
| Origen histórico: Windsor (Canadá), julio a septiembre de 1987 | Un modelo entrenado aquí no se aplica a mercados actuales | Se comparan métodos, no se predicen precios vigentes; no se afirma capacidad predictiva para Colombia |
| Precio de Kaggle × 70 | Las unidades no son dólares ni pesos de hoy | Se reportan RMSE/MAE en unidades de Kaggle y en CAD de 1987; MAPE y R² no dependen de la escala |
| `furnishingstatus` añadida sin documentación | Su origen y orden son inciertos | Tratada como nominal; ablación de H3 (T18) |
| `hotwaterheating` con solo 4,59 % de "sí" | Aparece en pocas filas de cada pliegue; coeficientes inestables | Estratificación por precio; regularización; se reporta su estabilidad |
| Duplicados y ruido irreducible (T06) | Puede inflar ligeramente la validación | Se declara; se mantiene la partición por fila |
| Sin interpretación causal | Las explicaciones describen asociaciones | Se evita lenguaje causal (3.10) |
| Costo computacional de la validación anidada | Tiempo de cómputo con 18 pipelines × 15 pliegues externos × bucle interno | Búsquedas acotadas; `RandomizedSearchCV` para M5 y M6; paralelismo; presupuesto estimado en T12 |
| Comparaciones múltiples | Riesgo de falsos positivos al comparar 18 pipelines | Contrastes planificados (H1–H3) y corrección de Holm |
| Búsqueda de literatura | La búsqueda de T09 usó un motor web con filtro por dominio y evaluó por resumen; las citas de este documento se apoyan en ese material | Se replicará en Scopus o Web of Science para el paper final |

---

## 3.12 Plan de ejecución y relación con T12

| Paso | Tarjeta | Entregable | Criterio de hecho |
|---|---|---|---|
| Metodología y diagrama | **T11** | `03_metodologia.md`, `docs/diagramas/pipeline.drawio` y `pipeline.png`, `resumen_T11_canva.md` | Revisión de Juan Miguel |
| Piloto de viabilidad | **T12** | `notebooks/03_piloto_viabilidad.ipynb`; `src/preprocesamiento.py` (P1–P3), `src/modelos.py` (M1–M6), `src/evaluacion.py` (pliegues, métricas, prueba t); `reports/tables/piloto_resultados.csv`; figuras de RMSE por pipeline y curva de aprendizaje | *Restart & Run All* sin errores; solo filas de entrenamiento; semilla 5007; hiperparámetros fijos; mismos pliegues; sin tocar `prueba` |
| Preprocesamiento y ablación | T17–T18 | Efecto de P1, P2, P3 y ablación de `furnishingstatus` (H2, H3) | Prueba t corregida con los mismos pliegues |
| Modelos y validación anidada | T19–T22 | Ajuste de hiperparámetros y comparación final (H1) | CV anidada repetida; Holm |
| Complejidad e interpretabilidad | T23–T24 | Tiempos, memoria, permutación, SHAP, PDP | Mediciones en el mismo hardware |
| Evaluación final | T25 | Un único ajuste y evaluación sobre `prueba` | Una sola vez |

**Qué debe respetar T12.** Implementar exactamente P1–P3 y M1–M6 de este documento; usar solo el conjunto de entrenamiento; no ajustar hiperparámetros (usar los valores por defecto razonables); registrar tiempo de ajuste y de inferencia; y entregar la lista de pliegues para reutilizarla en T17–T22. Si T12 detecta un problema de diseño (por ejemplo, que Ridge requiere otra escala), lo documenta y propone el cambio en su PR, sin alterar el test.

**Para las diapositivas 8 y 9** (T13): el contenido está en [`resumen_T11_canva.md`](resumen_T11_canva.md).

---

## Referencias

Las referencias [1], [2], [4], [6], [12], [16], [20] y [26] siguen la numeración de la revisión de literatura de T09 ([`02_protocolo_busqueda.md`](02_protocolo_busqueda.md)). Las demás son específicas de este documento. Los estudios de T09 se evaluaron por resumen y metadatos; su lectura completa corresponde a T10.

[1] S. Rosen, "Hedonic prices and implicit markets: Product differentiation in pure competition," *Journal of Political Economy*, vol. 82, no. 1, pp. 34–55, 1974, doi: 10.1086/260169.

[2] P. M. Anglin and R. Gençay, "Semiparametric estimation of a hedonic price function," *Journal of Applied Econometrics*, vol. 11, no. 6, pp. 633–648, 1996, doi: 10.1002/(SICI)1099-1255(199611)11:6<633::AID-JAE414>3.0.CO;2-T.

[4] H. Haupt, J. Schnurbus, and R. Tschernig, "On nonparametric estimation of a hedonic price function," *Journal of Applied Econometrics*, vol. 25, no. 5, pp. 894–901, 2010, doi: 10.1002/jae.1186.

[5] C. Nadeau and Y. Bengio, "Inference for the generalization error," *Machine Learning*, vol. 52, no. 3, pp. 239–281, 2003, doi: 10.1023/A:1024068626366.

[6] E. A. Antipov and E. B. Pokryshevskaya, "Mass appraisal of residential apartments: An application of Random forest for valuation and a CART-based approach for model diagnostics," *Expert Systems with Applications*, vol. 39, no. 2, pp. 1772–1778, 2012, doi: 10.1016/j.eswa.2011.08.077.

[7] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[8] I.-K. Yeo and R. A. Johnson, "A new family of power transformations to improve normality or symmetry," *Biometrika*, vol. 87, no. 4, pp. 954–959, 2000, doi: 10.1093/biomet/87.4.954.

[9] N. Duan, "Smearing estimate: A nonparametric retransformation method," *Journal of the American Statistical Association*, vol. 78, no. 383, pp. 605–610, 1983, doi: 10.1080/01621459.1983.10478017.

[10] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017. [Online]. Available: https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html.

[12] A. Valier, "Who performs better? AVMs vs hedonic models," *Journal of Property Investment & Finance*, vol. 38, no. 3, pp. 213–225, 2020, doi: 10.1108/JPIF-12-2019-0157.

[16] M. Steurer, R. J. Hill, and N. Pfeifer, "Metrics for evaluating the performance of machine learning based automated valuation models," *Journal of Property Research*, vol. 38, no. 2, pp. 99–129, 2021, doi: 10.1080/09599916.2020.1858937.

[20] M. Stang, B. Krämer, C. Nagl, and W. Schäfers, "From human business to machine learning—methods for automating real estate appraisals and their practical implications," *Zeitschrift für Immobilienökonomie*, vol. 9, no. 2, pp. 81–108, 2022, doi: 10.1365/s41056-022-00063-1.

[26] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track*, 2022. [Online]. Available: https://papers.nips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf.
