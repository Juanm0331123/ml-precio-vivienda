# Cómo trabajamos: Trello ↔ GitHub

Cada tarea del proyecto es una **tarjeta de Trello** con un ID (`T01`…`T31`), un responsable, una fecha y sus criterios de aceptación. Todo cambio al repositorio sale de una tarjeta y llega a `main` por medio de un **Pull Request (PR)**.

## Flujo para cada tarjeta

1. **Tomar la tarjeta.** En Trello, mueve tu tarjeta de *Por hacer* a **En progreso** y haz *Join* para quedar como miembro.
2. **Crear la rama** desde `main` actualizado. El nombre empieza con el ID de la tarjeta:
   ```bash
   git checkout main
   git pull
   git checkout -b T07-eda-visualizaciones
   ```
3. **Hacer commits pequeños** con el ID al inicio del mensaje, verbo en presente y en español:
   ```bash
   git add notebooks/02_eda.ipynb
   git commit -m "T07: agrega boxplots de variables binarias"
   ```
4. **Subir la rama y abrir el PR** hacia `main`:
   ```bash
   git push -u origin T07-eda-visualizaciones
   ```
   - Título del PR: `T07 · EDA y visualizaciones` (ID + nombre de la tarjeta).
   - Completa la plantilla y pega el **enlace de la tarjeta de Trello**.
   - Si aún no está listo, ábrelo como *Draft*.
5. **Avisar en Trello.** Mueve la tarjeta a **En revisión (PR abierto)** y pega el enlace del PR como comentario, o adjúntalo con el Power-Up de GitHub.
6. **Revisión y merge.**
   - Juan Miguel (PM) revisa y hace **Squash and merge**. Los PR de Juan Miguel los revisa Mauricio o Brandon.
   - Si piden cambios, corrígelos **en la misma rama**: el PR se actualiza solo.
   - Con *Squash and merge*, todo el PR queda como **un solo commit** en `main` con el título del PR. Así el historial queda trazable a cada tarjeta.
7. **Cerrar la tarjeta.** Tras el merge, la rama se borra automáticamente. Mueve la tarjeta a **Hecho** y completa su checklist.

> ¿Sin experiencia con la terminal? **GitHub Desktop** o la pestaña *Source Control* de **VS Code** hacen los mismos pasos con botones: crear rama → commit → push → *Create Pull Request*.

## Reglas

- **Nadie hace push directo a `main`.**
- **1 tarjeta = 1 rama = 1 PR.** Si una tarea crece, se divide en tarjetas nuevas.
- Antes de abrir un PR **actualiza tu rama** con lo último de `main`: `git pull origin main`.
- No subir archivos > 25 MB, credenciales, entornos virtuales ni temporales de Office (`~$*.docx`).
- Los datos de `data/raw/` **no se modifican**. Toda transformación se hace por código y se guarda en `data/processed/`.

## Convenciones de código y notebooks

- **Notebooks numerados** en `notebooks/`: `01_procedencia_y_calidad.ipynb`, `02_eda.ipynb`, `03_piloto_viabilidad.ipynb`… Parten de `00_plantilla_notebook.ipynb`, que incluye encabezado, preparación para Colab y semilla.
- **Código debidamente comentado** (lo exige la rúbrica):
  - Celdas de texto que expliquen qué se hace y por qué.
  - Docstrings en todas las funciones de `src/`.
  - Comentarios en español en las líneas que no sean obvias.
- **Reproducibilidad:**
  - Semilla `5007` (`src/config.py`).
  - Rutas relativas desde la raíz del repositorio; nada de rutas como `C:\Users\...`.
  - Antes de cada PR, ejecutar *Kernel → Restart & Run All* sin errores.
- **Sin fuga de información.** Escalado, transformaciones y selección de variables se ajustan **solo** con los datos de entrenamiento, dentro de `Pipeline` de scikit-learn. El conjunto de prueba sellado (T06) no se usa hasta la evaluación final (T25).
- **Figuras:** numeradas, con título y ejes en español; exportadas en PNG ≥ 200 dpi a `reports/figures/`, con nombres descriptivos (`fig03_boxplot_binarias.png`).
- **Tablas de resultados:** CSV en `reports/tables/`.

## Etiquetas de GitHub

Los PR llevan la etiqueta del rol (`rol: data-engineer`, `rol: model-developer`, `rol: evaluation-lead`, `rol: project-manager`, `rol: reporting-lead`) y el milestone de la entrega (`Entregable 1 · Propuesta` o `Entrega final`).

## Cheat sheet de Git

| Necesito… | Comando |
|---|---|
| Ver en qué rama estoy y qué cambió | `git status` |
| Traer lo último de `main` | `git checkout main && git pull` |
| Crear mi rama | `git checkout -b T05-procedencia-diccionario` |
| Guardar cambios | `git add <archivos>` y luego `git commit -m "T05: describe el cambio"` |
| Subir mi rama | `git push -u origin T05-procedencia-diccionario` |
| Actualizar mi rama con `main` | `git pull origin main` |
| Descartar cambios de un archivo | `git restore <archivo>` |
