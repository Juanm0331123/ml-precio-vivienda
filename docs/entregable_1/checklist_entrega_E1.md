# Lista de verificación · Entrega del Entregable 1

**Tarjeta:** T15 · **Responsable:** Juan Miguel León Gómez (Project Manager)
**Cierre en UAO Virtual:** sábado 7 de noviembre de 2026, **6:29 p. m.** Meta interna: **5:00 p. m.** Cada hora o fracción de retraso resta 1 punto.

## 1. Archivos que se suben a UAO Virtual

| Archivo | Tarjeta | Estado |
|---|---|---|
| `docs/entregable_1/Propuesta_ML_Precio_Vivienda.pdf` (y `.pptx`), exportados de Canva | T13 | [x] versión final en main (PR #4) |
| `docs/entregable_1/Revision_Literatura.pdf`: protocolo, PRISMA, matriz de 26 estudios y síntesis | T09 · T10 | [x] en main |
| `docs/entregable_1/registro_busquedas.xlsx` y `docs/entregable_1/matriz_literatura.xlsx` | T09 · T10 | [x] en main |
| `docs/entregable_1/Descripcion_Dataset.pdf` (8 páginas) | T05–T08 | [x] en main |
| `notebooks/01_procedencia_y_calidad.html`, `notebooks/02_eda.html` y `notebooks/03_piloto_viabilidad.html` | T05–T07 · T12 | [x] en esta rama |
| Enlace al repositorio con la etiqueta `e1-propuesta` | T15 | [x] etiqueta creada |
| `Reporte_Gestion_E1.pdf` (si la plataforma lo pide en esta entrega) | T15 | [x] versión final |

## 2. Reporte de gestión (20 % de la nota individual)

Se arma el sábado 7 con el estado real del tablero y del repositorio:

- [x] Captura del tablero completo de Trello (todas las listas).
- [x] Tarjetas por integrante (filtro por miembro o por etiqueta de rol) con su estado.
- [x] Tabla de tarjetas planeadas vs completadas del Entregable 1, con fecha de cierre.
- [x] Bitácora de check-ins: comentarios de la tarjeta T03.
- [x] Pull Requests fusionados en GitHub, con autor, revisor y tarjeta asociada.
- [x] Historial de `main` (`git log --oneline`): un commit por tarjeta gracias a *Squash and merge*.
- [x] Exportar todo a `docs/entregable_1/Reporte_Gestion_E1.pdf`.

## 3. Coevaluación (10 % de la nota individual)

- [ ] Cada integrante llena sus filas en `Coevaluacion_E1.xlsx` (hoja *Coevaluacion*). Las celdas editables son las amarillas y la escala va de 1 a 5.
- [ ] Revisar la hoja *Resumen*: 2 evaluaciones recibidas por integrante.
- [ ] Entregarla en el formato que pida el docente (archivo o formulario).

## 4. Acceso del docente

- [ ] **GitHub:** volver público el repositorio o agregar al docente como colaborador con permiso de lectura.
- [ ] **Trello:** invitar al docente como observador o compartir el enlace del tablero.
- [ ] Verificar los enlaces en una ventana de incógnito.

## 5. Cierre en el repositorio

- [x] Todos los PR del Entregable 1 fusionados con *Squash and merge*.
- [x] Etiqueta de la versión entregada:
  ```bash
  git checkout main && git pull
  git tag e1-propuesta
  git push origin e1-propuesta
  ```
- [x] Tarjetas del Entregable 1 en **Hecho** y el hito marcado como completo.

## 6. Agenda del sábado 7 de noviembre

| Hora | Actividad | Tarjeta |
|---|---|---|
| 8:00 a. m. | Exposición en clase | T13 · T14 |
| 10:00 a. m. | Versión final de las diapositivas (ya cerrada) | T13 |
| 11:00 a. m. | Ensayo cronometrado 1 (meta ≤ 14:30) | T14 |
| 12:00 m. | Ajustes y ensayo cronometrado 2 | T14 |
| 3:00 p. m. | PDF finales, reporte de gestión y coevaluación | T15 |
| 5:00 p. m. | **Subida a UAO Virtual** | T15 |
