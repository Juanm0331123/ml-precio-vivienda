# 1. Problema, motivación, pregunta de investigación y objetivos

**Tarjeta:** T04 · **Responsable:** Juan Miguel León Gómez (Reporting Lead) · **Revisa:** Brandon Eduardo Tobar Sánchez
**Uso:** texto base de la sección 1 de la propuesta ("Introducción y motivación") y de las diapositivas 2–4.

---

## 1.1 Contexto

El precio de una vivienda resume el valor que el mercado asigna a un conjunto de atributos: tamaño del lote, número de habitaciones y baños, pisos, garaje, comodidades y ubicación. La **teoría de precios hedónicos** formaliza esta idea: el precio de un bien diferenciado es una función de sus características, y cada característica tiene un precio implícito [1]. Durante décadas, esa función se ha estimado con **regresión hedónica**, en general lineal o log-lineal.

Estimar bien el precio de una vivienda no es un ejercicio académico. Es un insumo de decisiones con consecuencias económicas directas:

- **Hogares:** para la mayoría de los hogares, la vivienda es el activo más importante. En promedio, en los países de la OCDE la vivienda propia representa cerca de **la mitad de la riqueza bruta de los hogares**, y **más del 60 %** en los hogares de clase media [3].
- **Crédito, impuestos y mercado:** el valor estimado del inmueble determina el monto de un crédito hipotecario, la base del impuesto predial y la referencia de precio para compradores y vendedores.
- **Escala del problema:**
  - El avalúo manual es lento, costoso y depende del criterio del perito.
  - Por eso existen los **modelos automáticos de valoración (AVM)**: modelos matemáticos que estiman el valor de mercado a partir de las características del inmueble, su ubicación y las condiciones del mercado. Están normalizados para la valoración masiva por la Asociación Internacional de Funcionarios de Avalúo (IAAO) [4].
  - Un AVM basado en aprendizaje automático puede alcanzar un error absoluto cercano al 9 %, comparable con el de los avalúos tradicionales [5].
- **Relevancia en Colombia:** según el diagnóstico del CONPES 3958 de 2019, solo el **5,68 %** del territorio nacional tenía información catastral actualizada. El **66 %** estaba desactualizado y el **28,32 %** no tenía formación catastral [6]. Actualizar los avalúos catastrales a esa escala requiere métodos de **valoración masiva** confiables, reproducibles y auditables. Ese es justamente el tipo de modelo que este proyecto compara.

## 1.2 Problema

> **¿Cómo estimar con precisión, y de forma reproducible y explicable, el precio de venta de una vivienda a partir de sus atributos estructurales y de localización?**

El modelo hedónico lineal es interpretable, pero impone una forma funcional rígida: efectos aditivos y sin interacciones. Los modelos de aprendizaje automático capturan no linealidades e interacciones, pero pueden sobreajustar cuando hay pocos datos, son más costosos y menos transparentes. El problema práctico es **decidir, con evidencia, cuál ofrece el mejor balance entre error de predicción, costo computacional e interpretabilidad**.

## 1.3 Formulación como problema de aprendizaje automático

| Elemento | Definición |
|---|---|
| Tipo de aprendizaje | **Supervisado** |
| Tarea | **Regresión**: predecir una variable continua |
| Entrada `x` | 12 atributos por vivienda: 5 numéricos discretos o continuos (tamaño del lote, habitaciones, baños, pisos, plazas de garaje), 6 binarios (entrada vehicular, sala de recreación, sótano terminado, calentador a gas, aire acondicionado central, barrio preferido) y 1 categórico de 3 niveles (estado de amoblado) |
| Salida `y` | Precio de venta (`price`); también se modela `log(price)` por la asimetría de la distribución |
| Datos | 545 ventas de viviendas, tabulares y sin faltantes ([Housing Prices Dataset, Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) [7]). Provienen de los datos clásicos de Windsor, Canadá, 1987 [2]. No es un dataset incluido en scikit-learn, como exige la guía |
| Función de pérdida | Error cuadrático (entrenamiento); evaluación con RMSE, MAE, MAPE y R² |
| Generalización | Partición de prueba sellada (20 %) y validación cruzada repetida en entrenamiento, porque n es pequeño |

**Ventaja del dataset para la comparación:** es un *benchmark* clásico de la econometría hedónica [2]. Por eso los resultados de ML se pueden contrastar con modelos hedónicos publicados sobre los mismos datos.

## 1.4 Pregunta de investigación

> **¿En qué medida los modelos de aprendizaje automático mejoran la predicción del precio de venta de una vivienda frente al modelo hedónico lineal, usando solo atributos estructurales y de localización, y qué atributos explican sus predicciones?**
>
> Se consideran modelos regularizados, basados en instancias, de kernel y ensambles de árboles.

## 1.5 Objetivos

**Objetivo general.** Desarrollar y evaluar comparativamente modelos de aprendizaje supervisado para predecir el precio de venta de viviendas a partir de sus atributos estructurales y de localización. La comparación con el modelo hedónico lineal se hace en error de predicción, costo computacional e interpretabilidad.

**Objetivos específicos.**

1. **Caracterizar el conjunto de datos** (procedencia, calidad y distribución de las variables) y fijar una partición de prueba sellada que evite fuga de información. *(T05–T07)*
2. **Diseñar y comparar tres estrategias de preprocesamiento** y medir su efecto en el desempeño de cada modelo *(T17–T18)*:
   - codificación y escalado;
   - transformación de distribuciones;
   - ingeniería y selección de variables.
3. **Entrenar, ajustar y comparar al menos cinco modelos** con validación cruzada anidada *(T19–T22)*:
   - modelos: OLS hedónico, regularizados, KNN, SVR y ensambles de árboles;
   - métricas: RMSE, MAE, MAPE y R²;
   - pruebas estadísticas de diferencia entre modelos.
4. **Analizar la complejidad y la interpretabilidad** del mejor modelo frente a la línea base *(T23–T24)*:
   - complejidad teórica y empírica: tiempos de entrenamiento y predicción, número de parámetros;
   - interpretabilidad: importancia por permutación y SHAP.

## 1.6 Hipótesis

| | Hipótesis | Cómo se contrasta |
|---|---|---|
| **H1** | Al menos un ensamble de árboles (Random Forest o *gradient boosting*) obtiene un RMSE de validación cruzada **menor** que el modelo hedónico lineal. | Prueba t corregida para CV repetida (α = 0,05) |
| **H2** | Modelar `log(price)` **reduce el error relativo (MAPE)** de los modelos lineales y basados en distancia, frente a modelar el precio en escala original. | Comparación P1 vs P2 con los mismos *folds* |
| **H3** | La variable `furnishingstatus`, agregada en la versión de Kaggle y ausente en los datos originales, **no reduce significativamente** el RMSE una vez incluidas las 11 variables originales. | Ablación con y sin la variable (T18) |

Las hipótesis se plantean para contrastarse y no se dan por ciertas. Cualquier resultado, incluido que el modelo lineal sea competitivo, es informativo.

## 1.7 Alcance y supuestos

- **Las conclusiones son predictivas, no causales.** Un atributo importante para el modelo no implica que causa el precio.
- **Datos de 1987 y de una sola ciudad (Windsor, Canadá):**
  - El objetivo es comparar **métodos** sobre un *benchmark* documentado, no producir precios vigentes para Colombia.
  - La metodología (pipeline, validación, complejidad, explicabilidad) sí es transferible a datos locales, por ejemplo avalúos catastrales u ofertas inmobiliarias de Cali.
- **Muestra pequeña (n = 545):** se priorizan la validación cruzada repetida, la regularización y los modelos tabulares, no el *deep learning*.
- **Entregable final:** paper IEEE en inglés, código reproducible y una demo que estime el precio de una vivienda con su explicación.

## 1.8 Conexión con el trabajo previo

En el curso de Inferencia Estadística se analizó este mismo conjunto de datos con métodos inferenciales [8]: pruebas de hipótesis, ANOVA, regresión múltiple y log-lineal. Los modelos alcanzaron un R² ajustado de 67,4 % (lineal) y 69,2 % (log-lineal) **dentro de la muestra**. Ese trabajo respondió **qué atributos se asocian con el precio**.

Este proyecto cambia la pregunta a **qué tan bien se predice el precio de viviendas no vistas**. Para eso mide la generalización fuera de muestra y compara familias de modelos más flexibles.

---

## 1.9 Texto sugerido para las diapositivas (T13)

**Diapositiva 2: El problema.** *Título:* "Estimar el precio de una vivienda: un problema de alto impacto"
- La vivienda es el principal activo de los hogares: ≈ 50 % de su riqueza en la OCDE [3].
- Créditos, impuestos y compraventas dependen del valor estimado.
- El avalúo manual es lento, costoso y subjetivo; los AVM lo automatizan [4], [5].
- Colombia: solo 5,68 % del territorio con catastro actualizado (CONPES 3958) [6].

**Diapositiva 3: Formulación.** *Título:* "Regresión supervisada: 12 atributos → precio"
- Entrada: tamaño del lote, habitaciones, baños, pisos, garaje, 6 comodidades binarias y estado de amoblado.
- Salida: precio de venta, también `log(price)`.
- 545 ventas (Windsor, Canadá, 1987; vía Kaggle). No es un dataset de scikit-learn.

**Diapositiva 4: Pregunta, objetivos e hipótesis.** *Título:* "¿Supera el aprendizaje automático al modelo hedónico?"
- Pregunta de investigación (1.4).
- Cuatro objetivos específicos (1.5), en una línea cada uno.
- H1–H3 (1.6).

---

## Referencias

[1] S. Rosen, "Hedonic prices and implicit markets: Product differentiation in pure competition," *J. Political Economy*, vol. 82, no. 1, pp. 34–55, 1974.

[2] P. M. Anglin and R. Gençay, "Semiparametric estimation of a hedonic price function," *J. Applied Econometrics*, vol. 11, no. 6, pp. 633–648, 1996.

[3] OECD, *Housing Taxation in OECD Countries*, OECD Tax Policy Studies, no. 29. Paris, France: OECD Publishing, 2022, doi: 10.1787/03dfe007-en.

[4] International Association of Assessing Officers (IAAO), *Standard on Automated Valuation Models (AVMs)*. Kansas City, MO, USA: IAAO, 2018. [Online]. Available: https://www.iaao.org/wp-content/uploads/Standard_on_Automated_Valuation_Models.pdf

[5] N. Kok, E.-L. Koponen, and C. A. Martínez-Barbosa, "Big data in real estate? From manual appraisal to automated valuation," *J. Portfolio Management*, vol. 43, no. 6, pp. 202–211, 2017.

[6] Departamento Nacional de Planeación (DNP), "Estrategia para la implementación de la política pública de catastro multipropósito," Documento CONPES 3958, Bogotá, Colombia, mar. 2019.

[7] Yasser H., "Housing Prices Dataset," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

[8] O. Arias Gómez, M. J. Mozo Terán, and B. E. Tobar Sánchez, "Trabajo final de Inferencia Estadística: análisis inferencial y modelación del precio de viviendas," Universidad Autónoma de Occidente, Cali, Colombia, 2026.
