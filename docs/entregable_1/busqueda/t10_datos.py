"""T10 · Extracción y verificación de los 26 estudios incluidos (E01–E26).

Cada registro resume SOLO lo que se leyó en la fuente consultada el 5 y 6 de noviembre de 2026.
NR = "No reportado en lo leído": el dato no apareció en el texto al que se tuvo acceso (no significa
que el estudio no lo reporte). Los metadatos bibliográficos (autores, título, revista, DOI) salen de
`incluidos.py` (T09) y no se duplican aquí.

Niveles de verificación (campo `nivel`):
  TC   texto completo leído (en E20 mediante un extracto estructurado de la página de PMC)
  RES  solo resumen verificado (más metadatos)
  META solo metadatos verificados (título, año, revista); el resumen no se pudo leer
"""

NR = "No reportado en lo leído"

NIVELES = {"TC": "Texto completo", "RES": "Solo resumen", "META": "Solo metadatos"}

# id -> dict(datos, modelos, prep, val, met, res, hall, lim, rel, nivel, fuente, obs)
V = {
 "E01": dict(
  datos="Teórico (sin datos)", modelos="Teoría de precios hedónicos (equilibrio en el espacio de atributos)",
  prep=NR, val=NR, met=NR, res="No aplica (artículo teórico)",
  hall="Base teórica: el precio de un bien diferenciado es función de sus atributos y cada atributo tiene un precio implícito.",
  lim="Solo se verificó la cita y una descripción; no se leyó el artículo.",
  rel="Fundamenta el modelo hedónico que usamos como línea base.",
  nivel="META", fuente="EconPapers / RePEc (cita)", obs="Metadatos coinciden. DOI no abierto en el editor."),
 "E02": dict(
  datos="Resumen no menciona dataset ni n (Windsor, 546 ventas y jul–sep 1987 vienen de T09 y no se reverificaron aquí)",
  modelos="Modelo paramétrico de referencia (Box-Cox) vs modelo semiparamétrico",
  prep=NR, val="NR; el resumen menciona intervalos de predicción", met=NR,
  res="El semiparamétrico da predicciones medias más precisas que el paramétrico de referencia (sin cifras leídas)",
  hall="El enfoque semiparamétrico mejora la predicción media frente al paramétrico estándar, que sí pasa las pruebas de especificación usuales.",
  lim="Resumen solamente; sin cifras.", rel="Origen del dataset de Windsor y primer antecedente de comparación paramétrico vs flexible sobre estos datos.",
  nivel="RES", fuente="IDEAS / RePEc", obs="Metadatos coinciden."),
 "E03": dict(
  datos="Réplica de un modelo parcialmente lineal; el resumen no nombra Windsor", modelos="Regresión kernel no paramétrica con suavizado de variables categóricas",
  prep=NR, val=NR, met=NR, res=NR,
  hall="Replica el modelo parcialmente lineal previo con un enfoque no paramétrico (según el resumen).",
  lim="Resumen vía resultado de búsqueda; sin cifras.", rel="Réplica sobre el mismo conjunto de referencia: contexto de comparación paramétrico vs no paramétrico.",
  nivel="RES", fuente="RePEc (vía buscador)", obs="Metadatos coinciden. Fuente secundaria."),
 "E04": dict(
  datos="Datos de vivienda canadienses (resumen); n NR", modelos="Especificación paramétrica previa vs métodos no paramétricos",
  prep=NR, val="Comparación de predicciones (detalle no leído)", met=NR,
  res="No se rechaza la especificación paramétrica previamente propuesta (sin cifras)",
  hall="Una especificación paramétrica bien elegida no tiene por qué rechazarse; los métodos no paramétricos aportan en cada etapa del modelado.",
  lim="Resumen solamente; el resumen no nombra Windsor.", rel="Anticipa que un modelo lineal bien especificado puede ser competitivo en nuestro dataset.",
  nivel="RES", fuente="EconPapers / RePEc", obs="Metadatos coinciden."),
 "E05": dict(
  datos="Turquía, Encuesta de Presupuesto de los Hogares 2004 (resumen); n NR", modelos="Regresión hedónica vs red neuronal artificial (ANN)",
  prep=NR, val=NR, met=NR,
  res="La ANN resulta la mejor alternativa predictiva frente a la hedónica (sin cifras leídas)",
  hall="Compara hedónico y ANN; la ANN predice mejor. La lista de atributos clave de T09 no se verificó.",
  lim="Resumen vía buscador; página del editor no abierta.", rel="Antecedente directo de la pregunta hedónico vs modelo flexible.",
  nivel="RES", fuente="Buscador (datalearner, researchr)", obs="Metadatos coinciden; ScienceDirect no abierto."),
 "E06": dict(
  datos="El resumen no confirma país ni n (T09: apartamentos de Rusia)", modelos="Random Forest vs CHAID, CART, KNN, regresión múltiple, ANN (MLP y RBF), boosted trees; CART para diagnóstico",
  prep=NR, val=NR, met=NR, res="Random Forest superó a todos los métodos comparados (sin cifras leídas)",
  hall="Primera aplicación declarada de Random Forest a valoración masiva; usa segmentación para detectar dónde el modelo subestima o sobreestima.",
  lim="Se leyó la versión preprint (MPRA); no la versión del editor.", rel="Apoya incluir Random Forest y un análisis de errores por segmentos.",
  nivel="RES", fuente="EconPapers (preprint MPRA)", obs="Metadatos coinciden."),
 "E07": dict(
  datos="5.359 casas adosadas (MLS), Fairfax County, Virginia, EE. UU.", modelos="C4.5, RIPPER, Naïve Bayes, AdaBoost (clasificación)",
  prep=NR, val=NR, met="Exactitud de clasificación", res="RIPPER supera consistentemente a los demás por exactitud (sin cifras leídas)",
  hall="Problema planteado como clasificación del precio, no como regresión.",
  lim="Resumen vía buscador; la tarea es clasificación.", rel="Contraste: su formulación no es comparable con nuestra regresión.",
  nivel="RES", fuente="Buscador; DOI no abierto", obs="Metadatos coinciden."),
 "E08": dict(
  datos="EE. UU., activos multifamiliares; n NR", modelos="AVM basado en aprendizaje automático; algoritmos NR",
  prep="NR (solo se menciona información de ubicación hiperlocal)", val=NR, met="Error absoluto del 9 % (sin precisar la métrica)",
  res="Error absoluto cercano al 9 %, comparable con la exactitud de los avalúos tradicionales",
  hall="Un AVM con ML alcanza una exactitud comparable a la del avalúo manual.",
  lim="Ficha de repositorio; sin detalles metodológicos.", rel="Sustenta la relevancia práctica de los AVM (motivación).",
  nivel="RES", fuente="Maastricht CRIS + Crossref", obs="DOI según la fuente: 10.3905/jpm.2017.43.6.202. La fuente da \"Carmen Martínez-Barbosa\" (sin inicial intermedia)."),
 "E09": dict(
  datos=NR, modelos=NR, prep=NR, val=NR, met=NR, res=NR,
  hall="El ML resuelve problemas de predicción, no de estimación de parámetros (según T09).",
  lim="Texto completo inaccesible; el ejemplo de viviendas y la evaluación fuera de muestra no se verificaron.",
  rel="Marco conceptual: evaluar fuera de muestra y distinguir predicción de inferencia.",
  nivel="RES", fuente="Crossref + AEA", obs="Metadatos coinciden. Hallazgo tomado de T09, no reverificado."),
 "E10": dict(
  datos="Liubliana (Eslovenia): 7.407 transacciones de apartamentos, 2008–2013; variables estructurales, ambientales y de vecindario (GIS)",
  modelos="Random Forest vs regresión múltiple hedónica", prep=NR, val=NR, met="El resumen dice \"todas las métricas\"; no las enumera",
  res="Random Forest superó sustancialmente a la regresión en todas las métricas (sin cifras leídas)",
  hall="RF supera al hedónico en este mercado.", lim="Resumen solamente; MDPI devolvió 403.", rel="Evidencia a favor de H1 (ensamble vs hedónico).",
  nivel="RES", fuente="Crossref", obs="Metadatos coinciden."),
 "E11": dict(
  datos="Nicosia (Chipre), unidades residenciales; n y período NR", modelos="Algoritmos de ML/IA (no enumerados en el resumen) con análisis de sensibilidad",
  prep=NR, val=NR, met=NR, res=NR,
  hall="Propone ML para mejorar la exactitud y análisis de sensibilidad para dar transparencia a los AVM.",
  lim="Resumen solamente.", rel="Referencia de interpretabilidad (RQ4) mediante análisis de sensibilidad.",
  nivel="RES", fuente="Crossref + OpenAlex", obs="Metadatos coinciden."),
 "E12": dict(
  datos="Revisión sistemática (Scopus hasta jul. 2019): 381 resultados, 40 artículos comparables, ampliada con citas", modelos="Revisión: regresión vs redes neuronales, SVM, kNN y otros",
  prep=NR, val=NR, met="Conteo de victorias frente a la regresión",
  res="ML superó a la regresión en 57 comparaciones y la regresión a ML en 13; redes neuronales 37 vs 21, SVM 21 vs 5, kNN 5 vs 12; ningún algoritmo domina siempre",
  hall="El ML predice mejor pero es menos interpretable; los enfoques econométricos ofrecen mejor inferencia.",
  lim="Variables y definiciones heterogéneas entre los estudios comparados; solo precisión predictiva.", rel="Evidencia agregada sobre ML vs regresión y su costo en interpretabilidad.",
  nivel="RES", fuente="Emerald", obs="Metadatos coinciden; el autor es Agostino Valier. El resumen habla de 40 artículos; el conteo 53 informado por la lectura automática no coincide y no se usa."),
 "E13": dict(
  datos="China; ciudad NO confirmada en la fuente primaria (una fuente secundaria sugiere el dataset \"Housing Price in Beijing\"); n NR",
  modelos="Métodos de ML convencionales y avanzados con regresión; lista RF/XGBoost/LightGBM/stacking de T09 no verificada",
  prep=NR, val=NR, met=NR, res=NR,
  hall="Explora cómo influyen las variables en la predicción. La conclusión de sobreajuste de RF (T09) no se verificó.",
  lim="Texto completo inaccesible (403, incluido el PDF abierto).", rel="Pendiente de confirmar para uso en la síntesis.",
  nivel="RES", fuente="Crossref + OpenAlex", obs="Metadatos coinciden. Abrir manualmente el PDF para confirmar ciudad, modelos y resultados."),
 "E14": dict(
  datos="Hong Kong: ≈40.000 transacciones durante más de 18 años", modelos="SVM, Random Forest, Gradient Boosting Machine",
  prep=NR, val=NR, met="MSE, RMSE, MAPE", res="RF y GBM superan a SVM en las tres métricas (sin cifras leídas)",
  hall="RF y GBM predicen mejor; SVM es más rápido.", lim="Resumen solamente.", rel="Apoya comparar RF/GBM con regresión y reportar RMSE y MAPE.",
  nivel="RES", fuente="PolyU + Crossref/OpenAlex", obs="Año: 2020 en línea (Crossref) y 2021 en volumen (RePEc); T09 usa 2021."),
 "E15": dict(
  datos="El título confirma Alicante (España), precios de oferta", modelos="NR; el título indica ML con explicabilidad vs hedónico espacial",
  prep=NR, val=NR, met=NR, res="No verificado (el error < 2 % de T09 no se leyó)",
  hall="No verificado.", lim="No se pudo leer ni el resumen.", rel="No se usa como evidencia hasta verificarlo.",
  nivel="META", fuente="Semantic Scholar (título y año)", obs="Autores, volumen y artículo no verificados (Crossref devolvió 429)."),
 "E16": dict(
  datos="Resumen: aplicación empírica con cinco modelos AVM; dataset, país y n NR", modelos="Cinco modelos AVM comunes (no nombrados en el resumen)",
  prep=NR, val="El resumen indica consenso en usar validación cruzada para selección de modelos", met="48 métricas en 7 grupos; lista corta de 7",
  res="La elección de la métrica puede cambiar la elección del modelo",
  hall="Clasifica 48 métricas en 7 grupos según el aspecto de la distribución del error y propone una lista corta de 7.",
  lim="Graz (Austria) 2015–2020 de T09 no verificado.", rel="Respalda reportar varias métricas (RMSE, MAE, MAPE, R²).",
  nivel="RES", fuente="Crossref", obs="Metadatos coinciden."),
 "E17": dict(
  datos="Turquía: encuesta a expertos y ciudadanos en Ankara, Konya y Kayseri; tres regiones de Konya para implementación", modelos="Regresión múltiple para validar las variables seleccionadas (no es ML)",
  prep="Selección de variables: análisis de frecuencia (FRA), PCA, análisis factorial y AHP", val="Validación en condiciones reales de mercado en tres regiones de Konya", met=NR, res="Cifras no accesibles; no se extrajeron",
  hall="Aplica técnicas de selección de variables y validación para valoración masiva.",
  lim="Texto completo de pago; resultados cuantitativos no extraídos.", rel="Antecedente de selección de variables (RQ2); no es comparación de modelos.",
  nivel="RES", fuente="Crossref", obs="Metadatos coinciden."),
 "E18": dict(
  datos="Alicante (España): microdatos georreferenciados antes y después de la pandemia, más catastro e indicadores socioeconómicos", modelos="Gradient Boosting, XGBoost, LightGBM (boosting); Random Forest y Extra-Trees (bagging); regresión lineal como referencia",
  prep="Preparación de datos, ingeniería de variables y optimización de hiperparámetros (resumen)", val="NR; se mencionan fases de evaluación y selección", met=NR,
  res="ML supera a los modelos lineales; el boosting rinde mejor y sobreajusta menos; el bagging muestra sobreajuste. Los R² ≈ 0,91 y ≈ 0,90 de T09 no se verificaron",
  hall="El boosting generaliza mejor que el bagging en este estudio (según el resumen).",
  lim="Resumen solamente.", rel="Apoya incluir boosting y vigilar el sobreajuste de Random Forest.",
  nivel="RES", fuente="Crossref + Semantic Scholar", obs="DOI según Crossref: 10.3390/land11112100. El estudio es de Alicante, no de España en general."),
 "E19": dict(
  datos="El título confirma Gran Sídney (Australia), áreas estadísticas", modelos="NR; el título menciona algoritmos de ML",
  prep=NR, val=NR, met=NR, res="No verificado (la superioridad de GB y RF sobre el hedónico de T09 no se leyó)",
  hall="No verificado.", lim="No se pudo leer el resumen.", rel="No se usa como evidencia hasta verificarlo.",
  nivel="META", fuente="Crossref (metadatos)", obs="Metadatos coinciden."),
 "E20": dict(
  datos="Alemania: 1.212.546 viviendas, 2014–2020, 327 distritos NUTS-3; tasaciones de peritos de un grupo bancario", modelos="Comparables automatizados (EXF), OLS con selección hacia atrás, GAM (splines penalizados), XGBoost",
  prep="Limpieza de atípicos, errores y faltantes; georreferenciación; 36 variables independientes", val="Ventana expansiva: entrenamiento 2014–2019, prueba en los cuatro trimestres de 2020, reentrenando cada trimestre",
  met="MAPE, MdAPE, PE(10), PE(20), R²",
  res="XGBoost (2020): MAPE 0,1465, R² 0,8995. GAM: MAPE 0,1646, R² 0,8664. OLS: MAPE 0,1736, R² 0,8654. EXF: MAPE 0,2130, R² 0,7735. XGBoost fue el mejor en el 79,2 % de los distritos",
  hall="XGBoost es el más exacto, pero OLS y GAM quedan cerca y a veces ganan en zonas rurales con pocos datos.",
  lim="Datos de un solo banco; leído mediante un extracto estructurado de la página de PMC, no el texto literal.", rel="Evidencia cuantitativa de que un lineal/GAM bien ajustado queda cerca del ensamble.",
  nivel="TC", fuente="PMC (acceso abierto)", obs="Vol. 9, no. 2, pp. 81–108 no verificados (PMC muestra la versión en línea, pp. 1–28)."),
 "E21": dict(
  datos="1.982 datos de precedentes, 68 variables (Turquía inferido de la afiliación)", modelos="ANN, SVM, CHAID, CART y Random Forest",
  prep=NR, val="Partición entrenamiento/prueba (proporción NR)", met="Curva ROC y otros métodos estadísticos (detalle NR)", res="Cifras no accesibles; no se extrajeron",
  hall="Compara cinco técnicas de ML para valoración masiva.",
  lim="El resumen no incluye modelo hedónico: la etiqueta \"ML vs hedónico\" de T09 no corresponde. Texto completo de pago.", rel="Comparación solo entre algoritmos de ML.",
  nivel="RES", fuente="Crossref + Semantic Scholar", obs="Publicado en línea en 2021 y en papel en 2023 (T09 cita 2023)."),
 "E22": dict(
  datos="Isla de Jeju (Corea del Sur), 2012–2021; el n = 63.306 de T09 no se leyó", modelos="Ensamble apilado (stacking) con meta-modelo ridge; algoritmos base NR",
  prep=NR, val=NR, met="Varias métricas (no especificadas)", res="El ensamble supera a los algoritmos individuales con menores errores (sin cifras leídas)",
  hall="El stacking mejora a los modelos individuales.", lim="Resumen solamente (MDPI 403).", rel="Evidencia de ensambles apilados frente a modelos simples.",
  nivel="RES", fuente="Crossref", obs="DOI según Crossref: 10.3390/land13091436."),
 "E23": dict(
  datos="Hong Kong; n y período NR", modelos="Ensamble de tres niveles + IA explicable (XAI); método de XAI no leído",
  prep=NR, val=NR, met=NR, res="Ensamble de tres niveles propuesto para mejorar la exactitud (sin cifras)",
  hall="La mayoría de atributos tienen relación no lineal con el precio; la edad del inmueble y las coordenadas son lo más influyente; hay efectos umbral en la densidad edificada.",
  lim="Resumen solamente.", rel="Referencia de interpretabilidad (RQ4) y de no linealidad.",
  nivel="RES", fuente="Crossref", obs="Cita según Crossref: vol. 74, no. 1, art. 32 (2025). Las pp. 1–28 de T09 no se confirmaron."),
 "E24": dict(
  datos="Colombia: 15.177 inmuebles (apartamentos)", modelos="Regresión lineal, árboles de regresión, Random Forest, bagging",
  prep="Atributos derivados de texto (detalle NR)", val="Pruebas de estabilidad de atributos ante cambios de muestra; partición NR", met=NR,
  res="Random Forest y bagging tienen mayor poder predictivo (sin cifras leídas)", hall="Los ensambles superan a los modelos simples; atributos de texto se relacionan con el precio.",
  lim="Resumen solamente.", rel="Contexto colombiano de comparación lineal vs ensamble.",
  nivel="RES", fuente="Revista DYNA", obs="Metadatos coinciden."),
 "E25": dict(
  datos="Colombia: Fusagasugá, Manizales y Villavicencio; 2.397 anuncios de Fincaraiz (web scraping)", modelos="Regresión lineal, Ridge, Random Forest, Gradient Boosting",
  prep="Web scraping y limpieza automatizada (CRISP-DM)", val=NR, met="MAPE",
  res="MAPE ≈ 0,34–0,35 para RF y Ridge, diferencia mínima; el orden NO está confirmado (el extracto leído da RF 0,35 y Ridge 0,34, lo contrario de T09)",
  hall="Metodología de dos etapas (scraping + ML) con fuentes alternativas.", lim="Resumen solamente; el extracto puede ser impreciso.",
  rel="Muestra que un lineal regularizado rivaliza con los ensambles en datos pequeños.",
  nivel="RES", fuente="Revista Ingeniería (U. Distrital)", obs="Discrepancia con T09 en el orden de RF y Ridge: revisar el texto original."),
 "E26": dict(
  datos="45 datasets tabulares (OpenML), no específicos de vivienda; entrenamiento truncado a 10.000 muestras", modelos="Random Forest, XGBoost, GradientBoosting vs MLP, ResNet, FT-Transformer, SAINT",
  prep="Eliminación de columnas y filas con faltantes; categóricas codificadas; transformaciones", val="Búsqueda aleatoria de hiperparámetros con presupuestos crecientes; mejor modelo según conjunto de validación",
  met="Exactitud (clasificación) y R² (regresión), normalizados", res="Los árboles siguen siendo el estado del arte en datos de tamaño medio (~10.000 muestras), incluso sin contar su velocidad (sin cifras extraídas)",
  hall="Los modelos de árboles superan a las redes profundas en datos tabulares típicos.", lim="No es un estudio de vivienda.",
  rel="Respalda priorizar ensambles y modelos regularizados sobre redes profundas con 545 filas.",
  nivel="TC", fuente="PDF de NeurIPS + arXiv", obs="El PDF de NeurIPS trae \"typical\" en el título; arXiv no."),
}
