# Storyboard de la presentación (Entregable 1)

**Tarjeta:** T13 · **Responsable:** Juan Miguel León Gómez (Reporting Lead) · **Revisa:** Mauricio José Mozo Terán
**Versión maestra:** la presentación en Canva (https://www.canva.com/d/6FiXYq0edYonA3Z). En esta carpeta se guarda solo la copia más reciente exportada desde Canva: [`Propuesta_ML_Precio_Vivienda.pptx`](Propuesta_ML_Precio_Vivienda.pptx) y su [PDF](Propuesta_ML_Precio_Vivienda.pdf).
**Duración objetivo:** 14:30 de exposición, con 30 s de margen frente al límite de 15:00. Luego, 5 min de preguntas.

## Secuencia, responsables y tiempos

| # | Título (mensaje) | Contenido y visual | Prepara | Presenta | Tiempo |
|---|---|---|---|---|---|
| 1 | ¿Supera el aprendizaje automático al modelo hedónico? | Portada: equipo, programa y fecha | T13 · Juan Miguel | Juan Miguel | 0:00–0:15 |
| 2 | Seis bloques que siguen la guía del curso | Agenda con responsable y minuto de inicio, junto a las 4 secciones que pide la guía | T13 · Juan Miguel | Juan Miguel | 0:15–0:40 |
| 3 | Valorar viviendas es un problema de alto impacto | 3 cifras con su fuente: ≈ 50 % de la riqueza de los hogares (OCDE), 5,68 % del territorio con catastro al día (CONPES 3958), ≈ 9 % de error de un avalúo automático (Kok et al.) | T04 · Juan Miguel | Juan Miguel | 0:40–1:30 |
| 4 | Regresión supervisada: 12 atributos → precio | Entradas → modelo f(x) → precio y log(precio); ficha del dataset | T04 · Juan Miguel | Juan Miguel | 1:30–2:15 |
| 5 | Una pregunta, cuatro objetivos y tres hipótesis | Pregunta de investigación, OE1–OE4 y H1–H3 | T04 · Juan Miguel | Juan Miguel | 2:15–3:10 |
| 6 | De 192 registros a 26 estudios incluidos | Protocolo en 3 cifras y embudo PRISMA (182 → 116 → 36 → 17, más 9 por citas) | T09 · Juan Miguel | Juan Miguel | 3:10–3:50 |
| 7 | En Windsor, el hedónico compite con los ensambles | Hallazgos de la búsqueda (T09) y matriz de 6–8 estudios clave | T09 · Juan Miguel y **T10 · Brandon** | Juan Miguel | 3:50–4:30 |
| 8 | La solución: un pipeline de extremo a extremo | Diagrama esquemático del pipeline | **T11 · Brandon** | Brandon | 4:30–6:00 |
| 9 | Metodología: 3 preprocesamientos × 6 modelos | P1–P3, modelos, validación, métricas, complejidad e interpretabilidad | **T11 · Brandon** | Brandon | 6:00–7:30 |
| 10 | Viabilidad: resultados del experimento piloto | RMSE por modelo y preprocesamiento + curva de aprendizaje | **T12 · Brandon** | Brandon | 7:30–9:00 |
| 11 | El dataset: origen y variables | Cadena de procedencia + diccionario | **T05 · Mauricio** | Mauricio | 9:00–10:15 |
| 12 | Calidad de los datos y preparación | Tabla de calidad + partición sellada | **T06 · Mauricio** | Mauricio | 10:15–11:15 |
| 13 | EDA: la distribución del precio | Histograma, boxplot y efecto de log(precio) | **T07 · Mauricio** | Mauricio | 11:15–12:25 |
| 14 | EDA: qué se relaciona con el precio | Correlaciones, información mutua y comparaciones por grupo | **T07 · Mauricio** | Mauricio | 12:25–13:30 |
| 15 | Cuatro fases hasta la entrega del 5 de diciembre | Fases, roles y herramientas (Trello, GitHub, Colab) | T13 · Juan Miguel | Juan Miguel | 13:30–14:10 |
| 16 | Una comparación rigurosa, reproducible y explicable | 3 mensajes de cierre y preguntas | T13 · Juan Miguel | Juan Miguel | 14:10–14:30 |
| 17 | Referencias | Anexo; no se presenta | T13 · Juan Miguel | — | — |
| 18 | Diagrama de flujo PRISMA 2020 | Anexo para preguntas; no se presenta | T09 · Juan Miguel | — | — |

Las diapositivas 7 a 14 tienen un recuadro **Pendiente** con lo que debe contener cada una, quién la prepara y quién la presenta.

## Cómo se trabaja la presentación

1. **Todo cambio se hace en Canva.** Mauricio y Brandon tienen acceso de edición y arman directamente sus diapositivas, reemplazando el recuadro *Pendiente* y el espacio de la figura.
2. **Figuras y código siguen entrando por el PR de cada tarjeta:**
   - figuras en PNG a ≥ 200 dpi, en `reports/figures/` o `docs/diagramas/`;
   - desde ahí se insertan en Canva, para que cada figura de la presentación tenga su código en el repositorio.
3. **El guion va en las notas del orador** de cada diapositiva, con su tiempo (T14).
4. **En el repositorio solo se guarda la copia exportada desde Canva** (PPTX y PDF), en cada punto de control. El PPTX del repositorio no se edita.

## Convenciones del diseño

- Título como mensaje, en una sola línea (50 caracteres o menos).
- Etiqueta de sección en mayúsculas arriba del título.
- IBM Plex Serif en títulos e IBM Plex Sans en el texto; texto de apoyo de 14 pt o más y cifras destacadas en azul.
- Paleta:
  - azul marino `#0F2A47` para portada, cierre y bloques destacados;
  - azul `#2B6CC4` para cifras y etiquetas;
  - fondo `#F5F6F9` y tarjetas blancas con borde `#E1E5EC`;
  - ladrillo `#B5532F` como acento.
- Pie con el nombre del proyecto y la píldora de quién presenta: Juan Miguel en durazno, Mauricio en menta y Brandon en lavanda.
- Una idea por diapositiva.

## Fechas

- **Versión preliminar completa:** viernes 6 de noviembre, 9:00 p. m.
- **Versión final y PDF:** sábado 7 de noviembre, 10:00 a. m.
- **Ensayos cronometrados (T14):** sábado 7 de noviembre en la mañana.
