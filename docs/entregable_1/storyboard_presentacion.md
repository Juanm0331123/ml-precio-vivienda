# Storyboard de la presentación (Entregable 1)

**Tarjeta:** T13 · **Responsable:** Juan Miguel León Gómez (Reporting Lead) · **Revisa:** Mauricio José Mozo Terán
**Archivo:** [`Propuesta_ML_Precio_Vivienda.pptx`](Propuesta_ML_Precio_Vivienda.pptx) (borrador) y su PDF
**Duración objetivo:** 14:30 de exposición, con 30 s de margen frente al límite de 15:00. Luego, 5 min de preguntas.

## Secuencia, responsables y tiempos

| # | Título (mensaje) | Contenido y visual | Prepara | Presenta | Tiempo |
|---|---|---|---|---|---|
| 1 | ¿Supera el aprendizaje automático al modelo hedónico? | Portada: equipo, curso, fecha | T13 · Juan Miguel | Juan Miguel | 0:00–0:15 |
| 2 | Agenda y responsables | Línea de tiempo de 6 secciones con responsable y minuto de inicio | T13 · Juan Miguel | Juan Miguel | 0:15–0:40 |
| 3 | Valorar viviendas es un problema de alto impacto | 3 cifras: ≈50 % riqueza (OCDE), 5,68 % catastro actualizado (CONPES 3958), ≈9 % error AVM (Kok et al.) | T04 · Juan Miguel | Juan Miguel | 0:40–1:30 |
| 4 | Regresión supervisada: 12 atributos → precio | Diagrama entradas → modelo → precio; ficha del dataset | T04 · Juan Miguel | Juan Miguel | 1:30–2:15 |
| 5 | Pregunta, objetivos e hipótesis | Pregunta, OE1–OE4 y H1–H3 | T04 · Juan Miguel | Juan Miguel | 2:15–3:10 |
| 6 | Búsqueda sistemática: de 192 registros a 26 estudios | Protocolo en 4 ideas + diagrama PRISMA 2020 | T09 · Juan Miguel | Juan Miguel | 3:10–3:50 |
| 7 | Lo que dice la literatura | Tabla de 6–8 estudios clave + síntesis + brecha | **T10 · Brandon** | Juan Miguel | 3:50–4:30 |
| 8 | Solución propuesta: el pipeline de extremo a extremo | Diagrama esquemático (draw.io → PNG) | **T11 · Brandon** | Brandon | 4:30–6:00 |
| 9 | Metodología: 3 preprocesamientos × 6 modelos | P1–P3, modelos, validación, métricas, complejidad | **T11 · Brandon** | Brandon | 6:00–7:30 |
| 10 | Viabilidad: resultados del experimento piloto | Gráfico de RMSE por modelo y preprocesamiento + curva de aprendizaje | **T12 · Brandon** | Brandon | 7:30–9:00 |
| 11 | El dataset: origen y variables | Cadena de procedencia + diccionario | **T05 · Mauricio** | Mauricio | 9:00–10:15 |
| 12 | Calidad de los datos y preparación | Tabla de calidad + partición sellada | **T06 · Mauricio** | Mauricio | 10:15–11:15 |
| 13 | EDA: el precio es asimétrico | Histograma + boxplot + log(precio) | **T07 · Mauricio** | Mauricio | 11:15–12:25 |
| 14 | EDA: qué se relaciona con el precio | Correlaciones, información mutua y boxplots | **T07 · Mauricio** | Mauricio | 12:25–13:30 |
| 15 | Plan de trabajo, roles y herramientas | 4 fases, roles, Trello / GitHub / Colab | T13 · Juan Miguel | Juan Miguel | 13:30–14:10 |
| 16 | Una comparación rigurosa, reproducible y explicable | 3 mensajes de cierre + preguntas | T13 · Juan Miguel | Juan Miguel | 14:10–14:30 |
| 17 | Referencias | Anexo; no se presenta | T13 · Juan Miguel | — | — |

Las diapositivas 7 a 14 aparecen en el borrador como **PENDIENTE**, con la lista de lo que debe contener cada una.

## Cómo se completan las diapositivas pendientes

El `.pptx` es un archivo binario: si dos personas lo editan en paralelo, los cambios chocan al hacer merge. Por eso:

1. Cada responsable entrega el **contenido de su diapositiva en el PR de su tarjeta**:
   - figuras en PNG a ≥ 200 dpi, en `reports/figures/` o `docs/diagramas/`;
   - un bloque *"Para la diapositiva N"* con título, 3–5 viñetas cortas y el guion de lo que dirá (para T14).
2. Juan Miguel (Reporting Lead) integra ese contenido en el `.pptx` dentro de esta tarjeta (T13) y actualiza el PDF.
3. **Convenciones del deck:**
   - título como mensaje, a lo sumo dos líneas;
   - texto de 14 pt o más;
   - paleta del tema (azul pizarra `#17324D`, ladrillo `#C0502F`, acero `#2A6F97`);
   - fuentes Cambria (títulos) y Calibri (texto);
   - una idea por diapositiva.

## Fechas

- **Versión preliminar completa:** viernes 6 de noviembre, 9:00 p. m.
- **Versión final y PDF:** sábado 7 de noviembre, 10:00 a. m.
- **Ensayos cronometrados (T14):** sábado 7 de noviembre en la mañana.
