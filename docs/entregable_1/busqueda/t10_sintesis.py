"""T10 · Texto de la síntesis (sección 4.2) y de las brechas (sección 5) de Revision_Literatura.docx.

Cada afirmación cita estudios y respeta el nivel de verificación de `t10_datos.py`. E15 y E19 (solo
metadatos) y E13 (sin ciudad ni resultados confirmados) NO se usan como evidencia.
"""

# (RQ, título, [mensajes])
SINTESIS = [
 ("RQ1", "Modelos de regresión y de aprendizaje automático",
  ["Los resúmenes verificados reportan ventaja de los ensambles de árboles sobre la regresión: Random Forest supera a la regresión múltiple en Liubliana [10], a los demás métodos comparados en [6] y, con Random Forest y bagging, a los modelos simples en Colombia [24]; XGBoost es el más exacto en el estudio alemán leído completo [20]; el apilamiento supera a los modelos individuales [22] y la revisión de [12] cuenta 57 comparaciones favorables al ML frente a 13 favorables a la regresión.",
   "La ventaja no es uniforme. En [20], GAM (R² 0,8664) y OLS (R² 0,8654) quedan cerca de XGBoost (R² 0,8995) y a veces ganan en zonas rurales con pocos datos; en [12] ningún algoritmo domina siempre; en [25] Ridge y Random Forest quedan casi empatados en MAPE (≈ 0,34–0,35, con el orden sin confirmar); y sobre los datos canadienses de referencia, [4] indica que una especificación paramétrica bien elegida no tiene por qué rechazarse.",
   "En datos tabulares de tamaño medio los árboles superan a las redes profundas [26], y entre ensambles el boosting generaliza mejor que el bagging en [18]. Esto respalda comparar regresión hedónica, modelos regularizados y ensambles, y no redes profundas, con 545 viviendas.",
   "Límite: 21 de los 26 estudios se verificaron solo por resumen, sin cifras; [15] y [19] solo por metadatos, y [13] no confirmó ni su ciudad. La comparación formal con la literatura se hará en el artículo final."]),
 ("RQ2", "Estrategias de preprocesamiento",
  ["El preprocesamiento se reporta poco en lo leído. Lo verificado incluye limpieza de atípicos, errores y faltantes con georreferenciación y 36 variables [20]; eliminación de faltantes, codificación de categóricas y truncamiento a 10.000 muestras [26]; selección de variables con análisis de frecuencia, PCA, análisis factorial y AHP [17]; atributos derivados de texto [24]; y raspado web con limpieza automatizada [25].",
   "En los textos accedidos no aparece una comparación sistemática de varias estrategias de preprocesamiento sobre los mismos datos y modelos. Esto no prueba que no exista: depende del acceso al texto completo.",
   "Para el proyecto, las tres estrategias de preprocesamiento (P1–P3, que define T11) y la comparación P1 vs P2 con log(precio) (H2) se plantean como decisión de diseño, no como práctica ya establecida en la literatura revisada."]),
 ("RQ3", "Validación y métricas de evaluación",
  ["Los esquemas de validación verificados son pocos: ventana expansiva con prueba temporal en 2020 [20] y búsqueda aleatoria de hiperparámetros con selección en un conjunto de validación [26]; [16] indica consenso en el uso de validación cruzada para seleccionar modelos y [24] evalúa la estabilidad de atributos ante cambios de muestra. Para el resto, el resumen no describe la validación.",
   "Las métricas varían: RMSE, MSE y MAPE [14]; MAPE, MdAPE, PE(10), PE(20) y R² [20]; MAPE [25]; exactitud [7]; curva ROC [21]. [16] clasifica 48 métricas en siete grupos y advierte que la métrica elegida puede cambiar el modelo ganador.",
   "El proyecto reportará RMSE, MAE, MAPE y R², con validación cruzada repetida y anidada sobre la partición de entrenamiento (T04, H1). La prueba t corregida es una decisión metodológica del proyecto; no se verificó que los estudios revisados la usen."]),
 ("RQ4", "Interpretabilidad",
  ["La interpretabilidad aparece como contrapeso del ML: [12] resume que el ML predice mejor pero es menos interpretable, y que los enfoques econométricos ofrecen mejor inferencia. Para [15], el título anuncia un contraste entre ML explicable e hedónico espacial, pero solo se verificaron sus metadatos.",
   "Las técnicas verificadas son el análisis de sensibilidad [11], la IA explicable en un ensamble [23] (el método concreto no se leyó) y el diagnóstico por segmentos con CART [6]. En [23], la edad del inmueble y las coordenadas son los atributos más influyentes y la mayoría de relaciones es no lineal.",
   "Ningún resumen verificado nombra SHAP de forma explícita; la afirmación de T09 de que SHAP es de las herramientas más usadas queda sin verificar y se contrastará con el texto completo."]),
]

BRECHAS = [
 "Comparación sobre el mismo dataset: los estudios econométricos sobre los datos de referencia (T09 los ubica en Windsor) comparan enfoques paramétricos y no paramétricos [2]–[4], y en lo leído no aparecen ensambles de árboles. El proyecto compara regresión hedónica, modelos regularizados y ensambles sobre esos mismos datos.",
 "Reporte de validación y preprocesamiento: en los textos accedidos estos aspectos aparecen poco o nada (RQ2 y RQ3). El proyecto los documenta de forma explícita: tres preprocesamientos, validación cruzada repetida y anidada sobre la partición sellada, y pruebas estadísticas.",
 "Costo y transparencia: [12] y [20] muestran que el ML mejora la exactitud con un costo en interpretabilidad y datos. El proyecto incorpora análisis de complejidad e interpretabilidad en la comparación.",
 "Tamaño pequeño: entre los estudios con n verificado, el menor es de 2.397 anuncios [25]; el proyecto trabaja con 545 viviendas, por lo que sus conclusiones no se generalizan a otros mercados.",
]
BRECHAS_NOTA = ("Estas brechas se formulan a partir de lo leído (21 resúmenes, 2 textos completos y 3 solo metadatos). "
  "No se afirma novedad: se confirmará con las cadenas replicadas en Scopus o Web of Science para el artículo final.")
