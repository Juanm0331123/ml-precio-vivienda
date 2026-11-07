# Diapositiva 10 · Viabilidad: resultados del experimento piloto (T12)

**Tarjeta:** T12 · **Prepara y presenta:** Brandon Eduardo Tobar Sánchez · **Revisa:** Juan Miguel León Gómez
**Tiempo:** 90 s (7:30–9:00 del guion de T14).
**Fuente de todas las cifras:** `reports/tables/piloto_resultados.csv` (270 ajustes: 18 pipelines × 15 pliegues), generado por `notebooks/03_piloto_viabilidad.ipynb`. Las figuras están en `reports/figures/` y salen del mismo experimento.
**Alcance:** validación cruzada repetida 5 × 3 **solo sobre las 436 filas de entrenamiento**, con hiperparámetros fijos. El test sellado (109 filas) no se usó. No hay pruebas estadísticas todavía (T17–T22).
**Diseño (storyboard de T13):** título como mensaje, etiqueta de sección en mayúsculas, IBM Plex Serif/Sans (14 pt o más), azul marino `#0F2A47`, azul `#2B6CC4`, fondo `#F5F6F9`, ladrillo `#B5532F` como acento y píldora lavanda para Brandon. Este archivo no edita Canva ni el PPTX del repositorio.

---

## A. Título (46 caracteres)

`Viabilidad: resultados del experimento piloto`
**Etiqueta de sección:** VIABILIDAD

## B. Mensaje principal

> **El pipeline corre de punta a punta y da una primera lectura: con 436 filas, un lineal regularizado y XGBoost quedan a ≈ 2 % entre sí, así que no hay una ventaja clara de los ensambles todavía.** La comparación formal queda para las fases siguientes.

(Un mensaje honesto: el piloto demuestra **viabilidad**, no un ganador.)

## C. Mejor configuración observada

| | Configuración | RMSE medio de validación cruzada |
|---|---|---|
| **Menor RMSE observado** | **M2 · Ridge con P2** | **1.077.050** (± 130.909 entre pliegues) |
| Mejor de árboles | M6 · XGBoost con P1 | 1.099.481 (± 126.230); 2,1 % mayor |
| Línea base hedónica | M1 · OLS con P1 | 1.109.629 (± 141.567) |

## D. RMSE de validación cruzada (cifras exactas del CSV)

- **Mejor (M2 · P2):** RMSE = **1.077.050** unidades de Kaggle ≈ **15.386 CAD de 1987**; MAE = 775.008; MAPE = **16,66 %**; R² = **0,663**.
- **Unidades:** las de Kaggle = dólares canadienses de 1987 × 70 (T05). **No** son precios actuales ni pesos colombianos. MAPE y R² no dependen de la escala.
- **Peor:** M4 · SVR con P2: RMSE = 1.286.418.
- **Variabilidad:** la desviación entre pliegues (≈ 125.000–170.000) es mayor que la diferencia entre los mejores (22.431 entre Ridge·P2 y XGBoost·P1).

## E. Comparación visual principal

**Figura:** `reports/figures/fig_piloto_rmse.png` (2782 × 1562, 200 dpi, 16:9). Barras agrupadas por modelo, con los tres preprocesamientos en tonos de azul, el RMSE medio escrito en la base de cada barra y barras de error de ± 1 desviación estándar entre pliegues.

Tabla compacta (RMSE medio, unidades de Kaggle), para el pie o las notas:

| Modelo | P1 | P2 | P3 |
|---|---|---|---|
| M1 · OLS (hedónico) | 1.109.629 | 1.077.789 | 1.117.120 |
| M2 · Ridge | 1.108.595 | **1.077.050** | 1.117.234 |
| M3 · KNN | 1.173.656 | 1.220.872 | 1.278.230 |
| M4 · SVR (RBF) | 1.248.158 | 1.286.418 | 1.282.701 |
| M5 · Random Forest | 1.122.366 | 1.138.406 | 1.168.221 |
| M6 · XGBoost | 1.099.481 | 1.117.359 | 1.146.690 |

## F. Curva de aprendizaje

**Figura:** `reports/figures/fig_curva_aprendizaje.png` (2782 × 1559, 200 dpi). Dos paneles: la mejor configuración observada (M2 · Ridge con P2) y la línea base hedónica (M1 · OLS con P1). Regla de elección declarada antes de verla: mejor configuración observada + línea base.

Lectura (M2 · P2): con 348 filas de ajuste, el error de validación es 1.076.966 y el de ajuste 1.031.869 (brecha de ≈ 4 %: **sin sobreajuste marcado**). El error de validación baja de 1.202.537 con 69 filas a 1.076.966 con 348 y **sigue descendiendo lentamente**: más datos probablemente ayudarían un poco.

## G. Limitaciones (para el pie o las notas)

- **Piloto con hiperparámetros fijos:** el orden podría cambiar al ajustarlos (T19–T22, validación anidada).
- **Sin pruebas estadísticas:** las medias no prueban diferencias; los pliegues repetidos no son independientes.
- **Muestra pequeña** (436 filas) y datos de Windsor, 1987, con precio escalado ×70.
- **Test sellado sin usar:** esto no es rendimiento final.
- **Tiempos:** un solo hilo, en un equipo; sirven para comparar modelos entre sí.

**Costo computacional (promedio por pliegue):** OLS y Ridge ≈ 0,01 a 0,09 s según el preprocesamiento; Random Forest ≈ 0,62 a 0,74 s (el más costoso); XGBoost ≈ 0,08 a 0,15 s. Los 270 ajustes sumaron 44,4 s: la validación anidada es viable.

## H. Guion de exposición (≈ 90 s)

> "Para comprobar que la propuesta es viable, corrimos un piloto: los 18 pipelines, tres preprocesamientos por seis modelos, con validación cruzada repetida 5 por 3, y solo con las 436 filas de entrenamiento. El test sellado no se tocó.
>
> En esta gráfica cada grupo de barras es un modelo y cada color un preprocesamiento. Lo primero: todo funciona de punta a punta, los 270 ajustes corrieron sin fallos y en menos de un minuto de ajuste.
>
> Lo segundo, el primer resultado. El menor error lo obtuvo Ridge con transformaciones, con un RMSE de 1,08 millones de unidades de Kaggle, unos 15 mil dólares canadienses de 1987; el R² es 0,66. XGBoost queda apenas un 2 % por encima, y la desviación entre pliegues es mayor que esa diferencia. Es decir: todavía no hay una ventaja clara de los ensambles sobre el modelo lineal, y eso lo contrastamos con pruebas estadísticas en la siguiente fase.
>
> La curva de aprendizaje muestra que el error de validación sigue bajando con más datos, y que no hay una brecha grande con el error de ajuste.
>
> Es un piloto con hiperparámetros fijos, así que lo leemos como viabilidad, no como un ganador."

**Tiempo objetivo:** 90 s. El guion tiene ≈ 195 palabras (80 a 95 s según el ritmo); si pasa de 90 s, recortar el párrafo de la curva. Cronometrar en los ensayos de T14.

---

## Preguntas probables y respuestas defendibles

**1. ¿Por qué no usaron el conjunto de prueba?**
Porque se usaría para decidir, y la prueba sirve para estimar el error final sin sesgo. Está sellado desde T06 y solo se abre una vez, en T25. En el piloto, el código carga solo las 436 filas de entrenamiento: los valores de las 109 de prueba nunca entran a memoria, y las huellas de los archivos coinciden al inicio y al final.

**2. ¿Cuáles son las tres estrategias de preprocesamiento y por qué?**
P1 es la base: codificar categorías y estandarizar solo donde el modelo lo necesita. P2 agrega transformaciones: el logaritmo del precio, Yeo-Johnson en el área y escalado robusto; atiende la asimetría del precio y contrasta H2. P3 agrega variables derivadas, dos interacciones, winsorización y selección por información mutua. Son acumulativas, para aislar el efecto de transformar y el de diseñar variables.

**3. ¿Por qué estos seis modelos?**
Cubren cuatro familias: lineal (OLS y Ridge), basada en instancias (KNN), de kernel (SVR) y ensambles de árboles (Random Forest y XGBoost). OLS es el modelo hedónico, la línea base; los ensambles son los que contrasta H1. No hay redes profundas porque con 545 observaciones tabulares los árboles siguen siendo el estado del arte según la literatura revisada y las redes sobreajustarían.

**4. ¿Por qué P3 no mejoró el error?**
En el piloto, P3 solo mejoró a uno de los seis modelos respecto de P2. Sus variables, las interacciones y k = 10 se fijaron a priori por razones hedónicas, no por búsqueda. Su aporte real se mide por ablación en T17–T18. Además, corregimos un defecto de escala: las interacciones tienen más del 75 % de ceros y RobustScaler no las escalaba, lo que arruinaba a KNN y SVR (en la primera ejecución, SVR con P3 daba R² ≈ 0,18). Eso está documentado en el notebook.

**5. ¿Y si el lineal es el mejor?**
También es un resultado válido. La literatura sobre estos mismos datos anticipa que un modelo paramétrico bien especificado puede ser competitivo; por eso las hipótesis se contrastan con pruebas estadísticas y no se dan por ciertas.

**6. ¿Pueden afirmar que Ridge con P2 es el mejor modelo?**
No. Es la configuración con menor RMSE medio en este piloto. La diferencia con las siguientes es menor que la desviación entre pliegues y no se hizo ninguna prueba estadística. Tampoco garantiza el mismo orden en el test sellado.

---

## Notas para Juan Miguel (integración en Canva)

- La diapositiva debe mostrar la figura 1 a ancho principal y la figura 2 en un recuadro lateral; el mensaje va en el recuadro azul marino. Si hay poco espacio, la curva pasa a la diapositiva de respaldo.
- Las cifras de este archivo son las del CSV de esta rama; si alguien reejecuta el notebook, las métricas se repiten idénticas (los tiempos cambian ligeramente).
- Sin interpretación causal ni extrapolación a Colombia: el dataset es de Windsor, 1987.
