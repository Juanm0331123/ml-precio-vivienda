# Paper IEEE

Paper de la entrega final, en inglés y en formato IEEE de conferencia (`IEEEtran`).

| Archivo | Contenido |
|---|---|
| `main.tex` | Texto del paper |
| `references.bib` | Referencias (BibTeX, estilo `IEEEtran`) |
| `figures/make_figures.py` | Genera las figuras del paper, en inglés y solo con la partición de entrenamiento |
| `main.pdf` | Última versión compilada |
| `IEEEtran.cls`, `IEEEtran.bst`, `cite.sty` | Clase y estilos de CTAN, incluidos para compilar sin instalar nada |

## Estado

| Sección | Tarjeta | Estado |
|---|---|---|
| Abstract, Introduction, Related Work, Dataset, Problem Formulation | T27 | Redactadas |
| Methodology (diseño, CV anidada, preprocesamientos, modelos, espacios de búsqueda, métricas y pruebas) | T28 | Redactada |
| Experiments and Results: selección de variables, ablación (H3), comparación de los 18 pipelines (H1, H2) y prueba sellada | T28 | Redactadas con los resultados de T18 y T22 a T25 |
| Complexity Analysis: complejidad teórica y costo empírico con frente de Pareto | T28 | Redactada con T23 |
| Discussion: impacto de los atributos (SHAP contra el hedónico), residuos y limitaciones | T28 | Redactada con T24 |
| Conclusions | T28 | Redactadas |

Las filas de P3 de la CV anidada salen de correr `src/ejecutar_t22.py` (rama de T22) solo para P3 con el P3 que quedó en `main` después del #29. Cuando entre la corrida completa de T22, se regeneran las tablas y figuras con `python docs/paper/figures/make_result_figures.py` y se vuelve a compilar.

## Cómo compilar

Desde la raíz del repositorio:

```bash
python docs/paper/figures/make_figures.py
python docs/paper/figures/make_result_figures.py
cd docs/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

También se puede subir la carpeta completa a Overleaf y compilar `main.tex`.

## De dónde sale cada cifra

- Procedencia del dataset (factor 70, duplicado, variables renombradas, `furnishingstatus`): `notebooks/01_procedencia_y_calidad.ipynb` y `data/README.md` (T05, T06).
- Calidad y partición sellada: `reports/tables/calidad_datos.csv` y `data/processed/particion.csv` (T06).
- Asimetrías y correlaciones: las calcula `figures/make_figures.py` con las 436 filas de entrenamiento.
- Selección de variables y ablación de `furnishingstatus`: `reports/tables/t18_*.csv`, que genera `notebooks/04_preprocesamiento.ipynb` (T18). En el paper se pasan a CAD de 1987 dividiendo entre 70.
- Revisión de literatura (conteos PRISMA y hallazgos): `docs/entregable_1/02_protocolo_busqueda.md` y `docs/entregable_1/matriz_literatura.xlsx` (T09, T10). Solo se usan los hallazgos verificados en T10.
