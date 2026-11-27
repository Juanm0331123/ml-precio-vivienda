# Predicción del precio de vivienda con aprendizaje automático

Proyecto del curso **Aprendizaje Automático**: Maestría en Ciencia de Datos e Inteligencia Artificial, Universidad Autónoma de Occidente (UAO), periodo 2026-2.

El proyecto estima el precio de venta de una vivienda a partir de sus atributos estructurales y de localización. Es un problema de **regresión supervisada**: se compara el modelo hedónico lineal con modelos de aprendizaje automático, con validación rigurosa, análisis de complejidad e interpretabilidad.

## Estado del proyecto

| Fase | Estado | Detalle |
|---|---|---|
| **Entregable 1 · Propuesta** | Entregado el 7 nov 2026 | Tarjetas T01–T15 cerradas. Versión congelada en la etiqueta [`e1-propuesta`](https://github.com/Juanm0331123/ml-precio-vivienda/tree/e1-propuesta) |
| **Entrega final · Paper IEEE + demo** | En curso | Tarjetas T16–T31. Plan y cronograma en [`docs/entrega_final/README.md`](docs/entrega_final/README.md) |

Tras la exposición, el docente recomendó dar prioridad a los modelos basados en árboles (Random Forest, XGBoost) con SHAP y precisar que la explicabilidad mide el impacto de las variables en la predicción, no la formación del precio. Los ajustes están en [`docs/entrega_final/01_retroalimentacion_propuesta.md`](docs/entrega_final/01_retroalimentacion_propuesta.md).

## Equipo y roles

| Integrante | Rol(es) | GitHub |
|---|---|---|
| Juan Miguel León Gómez | Project Manager · Reporting Lead | [@Juanm0331123](https://github.com/Juanm0331123) |
| Mauricio José Mozo Terán | Data Engineer | [@Ironjaber26](https://github.com/Ironjaber26) |
| Brandon Eduardo Tobar Sánchez | Model Developer · Evaluation Lead | [@brandon5254](https://github.com/brandon5254) |

Las tareas se gestionan en el tablero de Trello [**"Proyecto ML · Predicción del precio de vivienda"**](https://trello.com/b/5UQEZmsC). Cada tarjeta tiene un ID (`T01`…`T31`) que se usa en ramas, commits y Pull Requests. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## Datos

| Archivo | Descripción |
|---|---|
| `data/raw/Housing.csv` | *Housing Prices Dataset* (Kaggle, Yasser H.): 545 viviendas × 13 variables. Archivo de trabajo principal, **no se modifica**. |
| `data/raw/HousePrices_Windsor_1987_AER.csv` | *HousePrices* (paquete R AER / Rdatasets, Anglin & Gençay 1996): 546 viviendas × 12 variables. Archivo histórico original de referencia para verificación de procedencia. |

La ficha técnica completa (procedencia, genealogía con el dataset original, diccionario de datos, calidad y licencia) se documenta en la tarjeta **T05** → `data/README.md`.

## Estructura del repositorio

```
ml-precio-vivienda/
├── app/                # aplicación demo interactiva en Streamlit (T26)
├── data/
│   ├── raw/            # datos originales (solo lectura)
│   └── processed/      # datos limpios y partición train/test (T06)
├── notebooks/          # notebooks numerados: 00_plantilla, 01_..., 02_...
├── src/                # código reutilizable (config, datos, preprocesamiento, modelos, evaluación)
├── tests/              # pruebas unitarias (unittest / pytest)
├── reports/
│   ├── figures/        # figuras exportadas por los notebooks (PNG ≥ 200 dpi)
│   └── tables/         # tablas de resultados (CSV)
├── docs/
│   ├── curso/          # guía del proyecto y rúbrica
│   ├── entregable_1/   # propuesta: diapositivas, revisión de literatura, descripción del dataset
│   ├── entrega_final/  # plan de la entrega final, retroalimentación y reporte de gestión
│   ├── paper/          # paper IEEE (LaTeX)
│   └── diagramas/      # diagramas (draw.io + PNG)
├── tests/              # pruebas (pytest)
├── .streamlit/         # configuración de la demo sin telemetría
├── .github/            # plantilla de PR, CODEOWNERS y pruebas automáticas
├── CONTRIBUTING.md     # flujo de trabajo Trello ↔ GitHub
├── reproducir.py       # regenera todo el experimento con un comando
├── LICENSE             # MIT (los datos conservan su licencia)
├── CITATION.cff        # cómo citar el repositorio y los datos
└── requirements.txt
```

## Cómo reproducir

Todo el experimento se regenera desde los datos de `data/raw/` con código. La semilla global es **5007** (`src/config.py`) y el conjunto de prueba sellado (T06) solo se usa en la evaluación final.

### Opción A: local, con un solo comando

```bash
git clone https://github.com/Juanm0331123/ml-precio-vivienda.git
cd ml-precio-vivienda
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python reproducir.py
```

`reproducir.py` hace, en orden:

1. Corre las pruebas (`pytest`).
2. Ejecuta los notebooks numerados (`01_`, `02_`, …) de principio a fin y guarda sus salidas. Cada notebook vuelve a escribir sus figuras en `reports/figures/` y sus tablas en `reports/tables/`.
3. Regenera las figuras del paper (`docs/paper/figures/`).

Si algo falla, se detiene en ese paso. Otras opciones:

```bash
python reproducir.py --sin-pruebas   # salta pytest
python reproducir.py --solo 04       # solo los notebooks que empiezan por 04
```

Para compilar el paper, ver [`docs/paper/README.md`](docs/paper/README.md).

### Opción B: Google Colab (sin instalar nada)

1. Entrar a https://colab.research.google.com → **File → Open notebook → GitHub**.
2. Escribir `Juanm0331123/ml-precio-vivienda` y abrir el notebook deseado.
3. Ejecutar las celdas. La primera clona el repositorio e instala las dependencias.

Los notebooks se corren en orden numérico, porque cada uno usa lo que deja el anterior.

### Pruebas automáticas

Cada push y cada Pull Request corre las pruebas en GitHub Actions ([`.github/workflows/pruebas.yml`](.github/workflows/pruebas.yml)).

### Demo interactiva (Streamlit, sin conexión)

Con las dependencias instaladas, desde la raíz del repositorio:

```bash
streamlit run app/app.py
```

Se abre en `http://localhost:8501`. Permite ingresar los 12 atributos de la vivienda (o cargar perfiles de ejemplo), ver el precio estimado con su intervalo de predicción (80 %, 90 % o 95 %) y la explicación SHAP de esa predicción. Mientras no exista `models/modelo_final.joblib` (T25), entrena el OLS hedónico P2 con las 436 filas de entrenamiento.

## Licencia y cita

El código y los documentos del equipo están bajo licencia MIT ([`LICENSE`](LICENSE)). Los datos conservan su licencia: `Housing.csv` es CC0 (Kaggle) y el archivo original de Windsor viene del paquete AER de R (GPL-2 | GPL-3); ver [`data/README.md`](data/README.md). Para citar el repositorio y los datos, ver [`CITATION.cff`](CITATION.cff).

## Entregables y fechas

| Entregable | Fecha (hora de Colombia) | Contenido |
|---|---|---|
| **Entregable 1 · Propuesta** (entregado) | sáb 7 nov 2026 | Diapositivas (15 min + 5 de preguntas), revisión de literatura con tabla de artículos, descripción del dataset con EDA y código comentado |
| **Entrega final** | sáb 5 dic 2026 (estimada) | Paper IEEE en inglés, archivos para reproducir el experimento, presentación práctica de 20 min |

Detalle de cada entregable y su tarjeta responsable: [`docs/entregable_1/README.md`](docs/entregable_1/README.md) y [`docs/entrega_final/README.md`](docs/entrega_final/README.md).
