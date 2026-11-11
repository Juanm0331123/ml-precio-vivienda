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
| Methodology, Experiments and Results, Complexity Analysis, Discussion, Conclusions | T28 | Pendientes de los resultados de T17–T25. En el PDF aparecen marcadas en gris como *Pending* |

El abstract tiene una frase pendiente con los resultados principales; se completa en T28.

## Cómo compilar

Desde la raíz del repositorio:

```bash
python docs/paper/figures/make_figures.py
cd docs/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

También se puede subir la carpeta completa a Overleaf y compilar `main.tex`.

## De dónde sale cada cifra

- Procedencia del dataset (factor 70, duplicado, variables renombradas, `furnishingstatus`): `notebooks/01_procedencia_y_calidad.ipynb` y `data/README.md` (T05, T06).
- Calidad y partición sellada: `reports/tables/calidad_datos.csv` y `data/processed/particion.csv` (T06).
- Asimetrías y correlaciones: las calcula `figures/make_figures.py` con las 436 filas de entrenamiento.
- Revisión de literatura (conteos PRISMA y hallazgos): `docs/entregable_1/02_protocolo_busqueda.md` y `docs/entregable_1/matriz_literatura.xlsx` (T09, T10). Solo se usan los hallazgos verificados en T10.
