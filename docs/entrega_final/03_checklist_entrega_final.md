# Lista de verificación · Entrega final

**Tarjeta:** T31 · **Responsable:** Juan Miguel León Gómez (Project Manager)
**Entrega:** sábado 5 de diciembre de 2026. Paper IEEE en inglés, archivos para correr el experimento y presentación práctica de 20 minutos.
**Nota individual:** 70 % entregable final + 20 % reporte de la herramienta de gestión + 10 % coevaluación.

## 1. Lo que se sube a UAO Virtual

Solo lo que pide la guía, más un archivo de texto con los enlaces, como en el Entregable 1.

| Archivo | Tarjeta | Estado |
|---|---|---|
| Paper IEEE en PDF (`docs/paper/main.pdf`, versión final) | T27 · T28 | [ ] |
| Diapositivas finales en PDF, exportadas de Canva | T30 | [ ] |
| ZIP con los archivos para correr el experimento (código, datos, notebooks, `requirements.txt`, `reproducir.py` y README) | T29 | [ ] |
| `Reporte_Gestion_Final.pdf` | T31 | [ ] |
| `Coevaluacion_Final.xlsx` con las seis calificaciones | T31 | [ ] |
| `LEEME.txt` con el enlace al repositorio (etiqueta `v1.0-final`) y al tablero de Trello | T31 | [ ] |

El nombre del ZIP lleva los apellidos del equipo: `EntregaFinal_LeonGomez_MozoTeran_TobarSanchez.zip`.

## 2. Paper (guía del curso)

| Requisito | Sección del paper | Estado |
|---|---|---|
| 1. Fenómeno o proceso modelado y problema | Introduction | [x] |
| 2. Obtención del dataset | Dataset | [x] |
| 3. Tipo de problema y tarea de aprendizaje | Problem Formulation | [x] |
| 4. Preprocesamiento con al menos tres metodologías | Methodology (P1, P2, P3) y Results (selección y ablación) | [ ] resultados de los modelos pendientes |
| 5. Al menos tres modelos con análisis de complejidad | Results y Complexity Analysis | [ ] |
| Sin marcas *Pending* en el PDF | Todo el paper | [ ] |
| Abstract con las cifras finales | Abstract | [ ] |

## 3. Archivos para correr el experimento

- [ ] Todos los PR de T17 a T29 fusionados en `main`.
- [ ] `python reproducir.py` corre completo en un clon limpio, sin errores.
- [ ] `requirements.txt` con versiones fijadas (cierre de T29).
- [ ] Modelo final en `models/modelo_final.joblib` regenerable por código (T25).
- [ ] La demo corre sin conexión, con su plan B en video (T26).
- [ ] Etiqueta de la versión entregada:
  ```bash
  git checkout main && git pull
  git tag v1.0-final
  git push origin v1.0-final
  ```

## 4. Reporte de gestión (20 % de la nota individual)

Se arma el mismo día con el estado real del tablero y del repositorio, con el generador que usamos en el Entregable 1:

- [ ] Captura del tablero completo de Trello, nítida.
- [ ] Tarjetas planeadas vs completadas de T16 a T31, con su fecha.
- [ ] Avance por integrante: PR, revisiones, commits y movimientos en Trello.
- [ ] Pull Requests fusionados, con autor, revisor y tarjeta.
- [ ] Bitácora de check-ins y el replan del 24 de noviembre.
- [ ] Exportar a `docs/entrega_final/Reporte_Gestion_Final.pdf`.

## 5. Coevaluación (10 % de la nota individual)

- [ ] Cada integrante califica a sus dos compañeros en `Coevaluacion_Final.xlsx` (escala de 1 a 5, celdas amarillas).
- [ ] Revisar que estén las seis filas completas.

## 6. Presentación (20 minutos)

- [ ] Diapositivas finales en Canva, con las cifras de T22 a T25.
- [ ] Demo probada en el equipo de la presentación y plan B listo.
- [ ] Dos ensayos cronometrados de 19:30 o menos, registrados en `02_storyboard_presentacion_final.md`.

## 7. Cierre en Trello y GitHub

- [ ] Tarjetas T16 a T31 en **Hecho**, con checklist completo, fecha en verde y PR adjunto.
- [ ] Milestone "Entrega final · Paper IEEE + demo" cerrado.
- [ ] Hito de la entrega final marcado como completo.
