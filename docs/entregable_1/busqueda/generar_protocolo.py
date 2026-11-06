"""Genera docs/entregable_1/02_protocolo_busqueda.md a partir del registro y la lista de incluidos."""
import sys
from pathlib import Path
from collections import Counter
from openpyxl import load_workbook
from incluidos import INCLUIDOS
from decisiones import MOTIVOS

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
xlsx = REPO / "docs" / "entregable_1" / "registro_busquedas.xlsx"
salida = REPO / "docs" / "entregable_1" / "02_protocolo_busqueda.md"
wb = load_workbook(xlsx, data_only=True)
P = {r[0].strip(): r[1] for r in wb["PRISMA"].iter_rows(min_row=2, values_only=True)}
B = [r for r in wb["Busquedas"].iter_rows(min_row=2, values_only=True)]

def ref_ieee(e):
    i, aut, anio, tit, fuente, volpp, doi = e[:7]
    enlace = f"doi: {doi[4:]}" if doi.startswith("doi:") else f"[Online]. Available: {doi}"
    return f"{aut}, \"{tit},\" *{fuente}*, {volpp}, {anio}, {enlace}."

L = []
L += ["# 2. Revisión de literatura: protocolo y ejecución de la búsqueda sistemática", "",
      "**Tarjeta:** T09 · **Responsable:** Juan Miguel León Gómez (Reporting Lead) · **Revisa:** Brandon Eduardo Tobar Sánchez",
      "**Fecha de ejecución de la búsqueda:** 4 de noviembre de 2026",
      "**Registro completo:** [`registro_busquedas.xlsx`](registro_busquedas.xlsx). Hojas: protocolo, PRISMA, criterios, búsquedas, registros, tamizaje e incluidos.",
      "**Siguiente paso:** T10 (Brandon) hace la extracción completa y la síntesis sobre los estudios incluidos.", "", "---", "",
      "## 2.1 Objetivo y preguntas de revisión", "",
      "Identificar las soluciones publicadas al problema de estimar el precio de venta de viviendas, tanto con modelos hedónicos como de aprendizaje automático. El fin es fundamentar el diseño metodológico del proyecto.", "",
      "| Código | Pregunta de revisión |", "|---|---|",
      "| RQ1 | ¿Qué métodos de aprendizaje automático se usan para predecir precios de vivienda y cómo se comparan con la regresión hedónica? |",
      "| RQ2 | ¿Qué estrategias de preprocesamiento e ingeniería de variables se emplean? |",
      "| RQ3 | ¿Qué esquemas de validación y métricas de desempeño se reportan? |",
      "| RQ4 | ¿Qué técnicas de interpretabilidad se aplican a los modelos? |", "",
      "## 2.2 Fuentes de información", "",
      "La búsqueda cubrió **11 fuentes académicas**:",
      "- Editoriales con revisión por pares: IEEE Xplore, ScienceDirect (Elsevier), SpringerLink, MDPI, Taylor & Francis, Emerald y Wiley Online Library.",
      "- Repositorios: arXiv y RePEc (IDEAS/EconPapers).",
      "- Índices latinoamericanos: SciELO y Redalyc.",
      "",
      "**Cómo se accedió a las fuentes:** las bases no se consultaron desde su propio buscador ni desde Scopus o Web of Science. Cada búsqueda se hizo en un motor de búsqueda web restringido al dominio de la base (operador `site:`), así que los conteos son los resultados que devolvió el motor para ese dominio, no el total de registros de la base.",
      "",
      "Se complementó con **búsqueda por citas** (bola de nieve hacia atrás y hacia adelante) desde los estudios recuperados, en particular las réplicas del *Journal of Applied Econometrics* sobre los datos de Windsor.", "",
      "## 2.3 Estrategia de búsqueda", "",
      "**Procedimiento:**",
      "- Cada cadena se ejecutó en un motor de búsqueda web con filtro por dominio (operador `site:`), una vez por base. No se usaron los buscadores internos de las bases ni sus filtros (año, tipo de documento, área).",
      "- Se registraron los primeros resultados por relevancia (hasta 10 por búsqueda), con título, URL y posición.",
      "- Cada cadena tiene su equivalente booleano para replicarla en Scopus o Web of Science.", "",
      "| Cadena | Texto ejecutado | Equivalente booleano |", "|---|---|---|"]
vistas = {}
for b in B:
    vistas.setdefault(b[2], (b[3], b[4]))
for c, (t, bo) in vistas.items():
    L.append(f"| {c} | {t} | `{bo}` |")
L += ["", "**Matriz de búsquedas ejecutadas** (resultados registrados por búsqueda):", "",
      "| ID | Cadena | Base de datos | Resultados |", "|---|---|---|---|"]
for b in B:
    L.append(f"| {b[0]} | {b[2]} | {b[5]} | {b[7]} |")
L += ["", "## 2.4 Criterios de elegibilidad", "",
      "**Inclusión:**",
      "- **I1:** estudia la estimación o predicción del precio de inmuebles residenciales, o usa los datos de Windsor (Anglin y Gençay, 1996).",
      "- **I2:** aplica o compara modelos de aprendizaje automático y/o hedónicos, o aporta metodología directamente aplicable (métricas, selección de variables, interpretabilidad).",
      "- **I3:** es una publicación académica con metadatos verificables (autores, año, fuente, DOI o URL del editor).",
      "- **I4:** está en inglés o español.", "",
      "**Exclusión:**"]
for k, v in MOTIVOS.items():
    L.append(f"- **{k}:** {v}.")
L += ["", "## 2.5 Proceso de selección", "",
      "1. **Eliminación de duplicados:** por título normalizado (sin tildes, mayúsculas ni signos). Además se eliminó una publicación indexada en dos idiomas.",
      "2. **Tamizaje** por título y resumen con los criterios E1–E7.",
      "3. **Recuperación y verificación (evaluación por resumen):** de cada registro que pasó el tamizaje se verificaron los metadatos (autores, año, revista, volumen y DOI o URL del editor) en la página del editor y se leyó el resumen. La elegibilidad se decidió con el resumen y esos metadatos; en esta etapa no se leyó el texto completo. Los registros sin acceso al resumen o con metadatos no verificables se marcaron como *no recuperados*.",
      "4. **Búsqueda por citas** desde los estudios verificados, con el mismo procedimiento de verificación.",
      "5. **Lectura del texto completo:** la hace T10 al llenar la matriz. Si un estudio no cumple los criterios al leerlo completo, se excluye y se actualizan los conteos.", "",
      "## 2.6 Resultados (PRISMA 2020)", "",
      "![Diagrama PRISMA 2020](../diagramas/prisma_2020.png)", "",
      "| Etapa | n |", "|---|---|"]
for k in ["Registros identificados en bases de datos", "Registros identificados por búsqueda por citas",
          "Duplicados eliminados antes del tamizaje", "Registros tamizados (título y resumen)",
          "Registros excluidos en el tamizaje", "Reportes buscados para recuperación (bases de datos)",
          "Reportes no recuperados (bases de datos)", "Reportes evaluados para elegibilidad (bases de datos)",
          "Reportes no recuperados (búsqueda por citas)", "Reportes evaluados (búsqueda por citas)",
          "Total de estudios incluidos en la revisión"]:
    L.append(f"| {k} | **{P[k]}** |")
L += ["", "*Reportes evaluados para elegibilidad* quiere decir evaluados con el resumen y los metadatos verificados, no con el texto completo (ver 2.5)."]
cat = Counter(e[13] for e in INCLUIDOS)
anios = [e[2] for e in INCLUIDOS]
L += ["", f"**Composición de los {len(INCLUIDOS)} estudios incluidos** ({min(anios)}–{max(anios)}; {sum(a >= 2015 for a in anios)} publicados desde 2015):", "",
      "| Categoría | Estudios |", "|---|---|"]
for k, n in cat.most_common():
    ids = ", ".join(e[0] for e in INCLUIDOS if e[13] == k)
    L.append(f"| {k} | {n} ({ids}) |")
L += ["", "## 2.7 Estudios incluidos", "",
      "Hallazgo preliminar tomado del resumen y los metadatos. La matriz con todas las columnas (datos, métodos, validación, métricas, resultados y relevancia) está en la hoja *Incluidos* del registro y la completa T10.", "",
      "| ID | Estudio | Fuente | Datos | Hallazgo preliminar | RQ |", "|---|---|---|---|---|---|"]
for e in INCLUIDOS:
    autores = [a.strip() for a in e[1].replace(", and ", ", ").replace(" and ", ", ").split(",") if a.strip()]
    ap = lambda a: a.split()[-1] if "Taltavull" not in a else "Taltavull de La Paz"
    corto = ap(autores[0]) if len(autores) == 1 else (f"{ap(autores[0])} y {ap(autores[1])}" if len(autores) == 2 else f"{ap(autores[0])} et al.")
    L.append(f"| {e[0]} | {corto} ({e[2]}) | {e[4]} | {e[9]} | {e[11]} | {e[12]} |")
L += ["", "## 2.8 Primeras observaciones (insumo para T10)", "",
      "- **Tendencia dominante:** los ensambles de árboles (Random Forest, *gradient boosting*, XGBoost, *stacking*) logran la mayor exactitud predictiva, tanto frente al modelo hedónico (E06, E10, E12, E19, E20) como frente a otros algoritmos de ML (E13, E14, E18, E22, E24).",
      "- **Contrapeso:** sobre **nuestro mismo dataset de Windsor**, la literatura econométrica muestra que una especificación paramétrica bien elegida es competitiva (E02–E04). Esto justifica usar el modelo hedónico como línea base fuerte y comparar con pruebas estadísticas.",
      "- **Datos tabulares de tamaño pequeño o medio:** los modelos de árboles siguen siendo el estado del arte (E26). Esto respalda priorizar ensambles y modelos regularizados sobre redes profundas.",
      "- **Métricas:** no hay una métrica única. Conviene reportar varias (RMSE, MAE, MAPE, R²) según el propósito (E16).",
      "- **Interpretabilidad:** es una brecha reconocida de los AVM. SHAP y el análisis de sensibilidad son las herramientas más usadas (E11, E15, E23).",
      "- **Contexto local:** en Colombia, los estudios encontrados muestran la misma ventaja de los ensambles, con Ridge muy cerca (E24, E25).", "",
      "## 2.9 Limitaciones y siguientes pasos", "",
      "- **Alcance de la búsqueda:** las bases no se consultaron directamente; el motor devuelve los resultados más relevantes de cada dominio, no el universo completo. Para el paper final se replicarán las cadenas booleanas en Scopus o Web of Science con el acceso de la biblioteca UAO, y se actualizarán los conteos.",
      "- **Reportes no recuperados (19 + 1):** se intentarán recuperar con el acceso institucional. Si cumplen los criterios, se agregan en T10.",
      "- **Evaluación por resumen:** la elegibilidad y los hallazgos de la tabla 2.7 se basan en el resumen y los metadatos. T10 los verifica contra el texto completo y completa la matriz.", "",
      "## Referencias", ""]
for n, e in enumerate(INCLUIDOS, 1):
    L.append(f"[{n}] {ref_ieee(e)}")
    L.append("")
L.append(f"[{len(INCLUIDOS)+1}] M. J. Page et al., \"The PRISMA 2020 statement: An updated guideline for reporting systematic reviews,\" *BMJ*, vol. 372, art. n71, 2021, doi: 10.1136/bmj.n71.")
open(salida, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("Escrito", salida, len(L), "líneas")
