# Predicción del precio de vivienda con aprendizaje automático

Proyecto del curso **Aprendizaje Automático**: Maestría en Ciencia de Datos e Inteligencia Artificial, Universidad Autónoma de Occidente (UAO), periodo 2026-2.

El proyecto estima el precio de venta de una vivienda a partir de sus atributos estructurales y de localización. Es un problema de **regresión supervisada**: se compara el modelo hedónico lineal con modelos de aprendizaje automático, con validación rigurosa, análisis de complejidad e interpretabilidad.

## Equipo y roles

| Integrante | Rol(es) | GitHub |
|---|---|---|
| Juan Miguel León Gómez | Project Manager · Reporting Lead | [@Juanm0331123](https://github.com/Juanm0331123) |
| Mauricio José Mozo Terán | Data Engineer | [@Ironjaber26](https://github.com/Ironjaber26) |
| Brandon Eduardo Tobar Sánchez | Model Developer · Evaluation Lead | [@brandon5254](https://github.com/brandon5254) |

Las tareas se gestionan en el tablero de Trello **"Proyecto ML · Predicción del precio de vivienda"** (privado; pedir acceso al PM). Cada tarjeta tiene un ID (`T01`…`T31`) que se usa en ramas, commits y Pull Requests. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## Datos

| Archivo | Descripción |
|---|---|
| `data/raw/Housing.csv` | *Housing Prices Dataset* (Kaggle, Yasser H.): 545 viviendas × 13 variables. Archivo original, **no se modifica**. |

La ficha técnica completa (procedencia, diccionario de datos, calidad y licencia) se documenta en la tarjeta **T05** → `data/README.md`.

## Estructura del repositorio

```
ml-precio-vivienda/
├── data/
│   ├── raw/            # datos originales (solo lectura)
│   └── processed/      # datos limpios y partición train/test (T06)
├── notebooks/          # notebooks numerados: 00_plantilla, 01_..., 02_...
├── src/                # código reutilizable (config, datos, preprocesamiento, modelos, evaluación)
├── reports/
│   ├── figures/        # figuras exportadas por los notebooks (PNG ≥ 200 dpi)
│   └── tables/         # tablas de resultados (CSV)
├── docs/
│   ├── curso/          # guía del proyecto y rúbrica
│   ├── entregable_1/   # propuesta: diapositivas, revisión de literatura, descripción del dataset
│   └── diagramas/      # diagramas (draw.io + PNG)
├── .github/            # plantilla de PR y CODEOWNERS
├── CONTRIBUTING.md     # flujo de trabajo Trello ↔ GitHub
└── requirements.txt
```

## Cómo ejecutar

### Opción A: Google Colab (sin instalar nada)

1. Entrar a https://colab.research.google.com → **File → Open notebook → GitHub**.
2. Marcar **Include private repos**, autorizar el acceso a GitHub y abrir el notebook deseado.
3. **Solo mientras el repositorio sea privado:** crear un token de GitHub de solo lectura y guardarlo en Colab como secreto `GITHUB_TOKEN` (ícono de llave en el panel izquierdo).
   - Ruta del token en GitHub: *Settings → Developer settings → Fine-grained tokens*.
   - Acceso: solo a este repositorio, con permiso *Contents: Read*.
4. Ejecutar las celdas. La primera clona el repositorio e instala las dependencias.

### Opción B: local

```bash
git clone https://github.com/Juanm0331123/ml-precio-vivienda.git
cd ml-precio-vivienda
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Semilla global del proyecto: **5007** (`src/config.py`).

## Entregables y fechas

| Entregable | Fecha (hora de Colombia) | Contenido |
|---|---|---|
| **Entregable 1 · Propuesta** | sáb 7 nov 2026, cierre 6:29 p. m. | Diapositivas (15 min + 5 de preguntas), revisión de literatura con tabla de artículos, descripción del dataset con EDA y código comentado |
| **Entrega final** | sáb 5 dic 2026 (estimada) | Paper IEEE en inglés, archivos para reproducir el experimento, presentación práctica de 20 min |

Detalle de cada entregable y su tarjeta responsable: [`docs/entregable_1/README.md`](docs/entregable_1/README.md).
