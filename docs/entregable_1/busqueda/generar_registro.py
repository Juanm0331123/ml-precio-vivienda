"""Genera docs/entregable_1/registro_busquedas.xlsx a partir del registro de búsquedas."""
import json, re, sys, unicodedata
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from decisiones import DECISIONES, MOTIVOS
from incluidos import INCLUIDOS, OTROS_NO_RECUPERADOS

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
SALIDA = REPO / "docs" / "entregable_1" / "registro_busquedas.xlsx"
FECHA = "2026-11-04"
CADENAS = {
 "C1": ("house price prediction machine learning random forest gradient boosting XGBoost",
        '("house price*" OR "housing price*") AND ("machine learning" OR "random forest" OR "gradient boosting" OR XGBoost)'),
 "C2": ("hedonic pricing model versus machine learning house prices comparison",
        'hedonic AND "machine learning" AND ("house price*" OR housing) AND (compar* OR versus)'),
 "C3": ("automated valuation model mass appraisal machine learning residential property",
        '("automated valuation model*" OR AVM OR "mass appraisal") AND "machine learning" AND residential'),
 "C4": ("explainable machine learning SHAP house price prediction interpretability",
        '(explainab* OR interpretab* OR SHAP) AND "machine learning" AND ("house price*" OR "property valuation")'),
 "C5": ("predicción precio vivienda aprendizaje automático modelos regresión",
        '("precio* de vivienda" OR avalúo) AND ("aprendizaje automático" OR "machine learning" OR regresión)'),
 "C6": ("Windsor house prices Anglin Gencay hedonic semiparametric nonparametric estimation replication",
        '(Windsor OR "Anglin and Gencay") AND hedonic AND (semiparametric OR nonparametric)'),
}

def norm(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode().lower()
    t = re.sub(r"\(.*?\)", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()

d = json.load(open(BASE / "resultados_busquedas.json", encoding="utf-8"))
busq, regs = d["busquedas"], d["registros"]

# --- estilos ---
F = "Arial"
H_FILL = PatternFill("solid", fgColor="1F3A5F")
H_FONT = Font(name=F, bold=True, color="FFFFFF", size=10)
B_FONT = Font(name=F, size=10)
T_FONT = Font(name=F, bold=True, size=13, color="1F3A5F")
WRAP = Alignment(wrap_text=True, vertical="top")
thin = Side(style="thin", color="D0D0D0")
BORDE = Border(left=thin, right=thin, top=thin, bottom=thin)

def hoja_tabla(ws, encabezados, filas, anchos):
    ws.append(encabezados)
    for c in ws[1]:
        c.fill, c.font, c.alignment, c.border = H_FILL, H_FONT, Alignment(wrap_text=True, vertical="center"), BORDE
    for f in filas:
        ws.append(list(f))
    for fila in ws.iter_rows(min_row=2):
        for c in fila:
            c.font, c.alignment, c.border = B_FONT, WRAP, BORDE
    for i, w in enumerate(anchos, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 30

wb = Workbook()

# ---------- Protocolo ----------
ws = wb.active; ws.title = "Protocolo"
filas = [
 ("Registro de la búsqueda sistemática de literatura", ""),
 ("Proyecto", "Predicción del precio de vivienda con aprendizaje automático (UAO · Aprendizaje Automático · 2026-2)"),
 ("Tarjeta / responsable", "T09 · Juan Miguel León Gómez (Reporting Lead)"),
 ("Fecha de ejecución", FECHA),
 ("Objetivo", "Identificar soluciones publicadas al problema de estimar el precio de venta de viviendas con modelos hedónicos y de aprendizaje automático, para fundamentar la metodología del proyecto."),
 ("RQ1", "¿Qué métodos de aprendizaje automático se usan para predecir precios de vivienda y cómo se comparan con la regresión hedónica?"),
 ("RQ2", "¿Qué estrategias de preprocesamiento e ingeniería de variables se emplean?"),
 ("RQ3", "¿Qué esquemas de validación y métricas de desempeño se reportan?"),
 ("RQ4", "¿Qué técnicas de interpretabilidad se aplican a los modelos?"),
 ("Fuentes", "IEEE Xplore, ScienceDirect, SpringerLink, MDPI, Taylor & Francis, Emerald, arXiv, SciELO, Redalyc, Wiley Online Library y RePEc, más búsqueda por citas (bola de nieve) desde los estudios recuperados."),
 ("Acceso a las fuentes", "Las bases no se consultaron desde su propio buscador ni desde Scopus o Web of Science. Cada búsqueda se hizo en un motor de búsqueda web restringido al dominio de la base (operador site:), así que los conteos son los resultados que devolvió el motor para ese dominio, no el total de registros de la base."),
 ("Procedimiento", "Cada cadena se ejecutó en un motor de búsqueda web con filtro por dominio (operador site:), una vez por base. Se registraron los primeros resultados por relevancia (hasta 10 por búsqueda), se eliminaron duplicados por título normalizado y se tamizaron por título y resumen."),
 ("Elegibilidad", "Se evaluó con el resumen y los metadatos verificados en la página del editor, sin leer el texto completo. La lectura completa se hace en T10 al llenar la matriz."),
 ("Periodo", "Sin límite inferior para estudios clásicos y del dataset de Windsor; foco en 2015–2026 para métodos de aprendizaje automático."),
 ("Idiomas", "Inglés y español"),
 ("Tipo de documento", "Artículos de revista, actas de conferencia con revisión por pares y capítulos académicos"),
 ("Limitación", "Las bases no se consultaron directamente: el motor devuelve los resultados más relevantes de cada dominio, no el universo completo. Para la entrega final se recomienda replicar las cadenas booleanas de la hoja 'Busquedas' en Scopus o Web of Science mediante la biblioteca UAO."),
]
for f in filas:
    ws.append(f)
ws["A1"].font = T_FONT
for fila in ws.iter_rows(min_row=2):
    fila[0].font = Font(name=F, bold=True, size=10); fila[1].font = B_FONT
    for c in fila: c.alignment = WRAP
ws.column_dimensions["A"].width = 24; ws.column_dimensions["B"].width = 110

# ---------- Criterios ----------
ws = wb.create_sheet("Criterios")
crit = [("I1","Inclusión","Estudia la estimación o predicción del precio de inmuebles residenciales, o usa los datos de Windsor (Anglin y Gençay, 1996)."),
        ("I2","Inclusión","Aplica o compara modelos de aprendizaje automático y/o hedónicos, o aporta metodología directamente aplicable (métricas, selección de variables, interpretabilidad)."),
        ("I3","Inclusión","Publicación académica con metadatos verificables (autores, año, fuente, DOI o URL del editor)."),
        ("I4","Inclusión","Idioma inglés o español.")]
crit += [(k, "Exclusión", v) for k, v in MOTIVOS.items()]
crit += [("NR","No recuperado","Pasó el tamizaje, pero no fue posible acceder al resumen o al texto completo, ni verificar sus metadatos.")]
hoja_tabla(ws, ["Código","Tipo","Descripción"], crit, [10, 16, 110])

# ---------- Busquedas ----------
ws = wb.create_sheet("Busquedas")
filas = []
for b in busq:
    txt, booleana = CADENAS[b["cadena"]]
    filas.append((b["id"], FECHA, b["cadena"], txt, booleana, b["base"], b["dominio"], None))
hoja_tabla(ws, ["ID búsqueda","Fecha","Cadena","Texto ejecutado","Equivalente booleano (para Scopus/WoS)","Base de datos","Dominio filtrado","Resultados registrados"],
           filas, [12, 12, 9, 48, 60, 22, 26, 14])
for r in range(2, ws.max_row + 1):
    ws.cell(r, 8).value = f'=COUNTIF(Registros!$B:$B,A{r})'
    ws.cell(r, 8).alignment = Alignment(horizontal="center", vertical="top")

# ---------- Registros (todos los resultados, con duplicados) ----------
ws = wb.create_sheet("Registros")
vistos, ids, filas = {}, {}, []
for n, r in enumerate(regs, 1):
    k = r["clave"]
    if k not in vistos:
        vistos[k] = f"U{len(vistos)+1:03d}"
        dup = "No"
    else:
        dup = "Sí"
    filas.append((n, r["busqueda"], r["pos"], r["titulo"], r["url"], dup, vistos[k]))
hoja_tabla(ws, ["N.º","ID búsqueda","Posición","Título","URL","Duplicado","ID registro único"], filas, [6, 12, 9, 80, 60, 11, 14])

# ---------- Tamizaje ----------
ws = wb.create_sheet("Tamizaje")
primero = {}
for r in regs:
    primero.setdefault(r["clave"], r)
def decision(clave):
    for p, v in DECISIONES.items():
        if clave.startswith(p): return v
    raise KeyError(clave)
inc_por_id = {e[0]: e for e in INCLUIDOS}
filas = []
for k, uid in vistos.items():
    r = primero[k]; dec, mot = decision(k)
    if dec == "DUP":
        filas.append((uid, r["titulo"], r["url"], r["busqueda"], "Bases de datos", "Duplicado", "", "Misma publicación que U-DYNA (versión en otro idioma)", "", "", ""))
    elif dec == "EXC":
        filas.append((uid, r["titulo"], r["url"], r["busqueda"], "Bases de datos", "Excluido", mot, MOTIVOS[mot], "", "", ""))
    elif dec == "NR":
        filas.append((uid, r["titulo"], r["url"], r["busqueda"], "Bases de datos", "Pasa", "NR", "Sin acceso al resumen o al texto, o metadatos no verificables", "No", "", ""))
    else:
        filas.append((uid, r["titulo"], r["url"], r["busqueda"], "Bases de datos", "Pasa", "", "", "Sí", "Incluido", mot))
# DYNA: referencia al ID de la versión en inglés
dyna_en = vistos[norm("Modeling of apartment prices in a Colombian context from a machine learning approach with stable-important attributes")]
filas = [tuple(x if x != "Misma publicación que U-DYNA (versión en otro idioma)" else f"Misma publicación que {dyna_en} (versión en otro idioma)" for x in f) for f in filas]
# Búsqueda por citas
n = 0
for e in INCLUIDOS:
    if e[7] == "Búsqueda por citas":
        n += 1
        filas.append((f"C{n:02d}", e[3], e[6], "Citas", "Búsqueda por citas", "Pasa", "", "", "Sí", "Incluido", e[0]))
for o in OTROS_NO_RECUPERADOS:
    n += 1
    filas.append((f"C{n:02d}", o[2], "", "Citas", "Búsqueda por citas", "Pasa", "NR", "Sin acceso al resumen o al texto completo", "No", "", ""))
hoja_tabla(ws, ["ID","Título","URL","Origen","Vía de identificación","Decisión tamizaje","Código","Motivo","Recuperado","Elegibilidad","ID estudio"],
           filas, [8, 70, 45, 9, 18, 14, 8, 50, 11, 12, 10])

# ---------- Incluidos ----------
ws = wb.create_sheet("Incluidos")
enc = ["ID","Autores","Año","Título","Revista / conferencia","Volumen y páginas","DOI o URL","Origen","Base / vía","Datos","Métodos","Hallazgo preliminar (resumen)","RQ","Categoría"]
hoja_tabla(ws, enc, INCLUIDOS, [6, 34, 7, 60, 30, 22, 40, 10, 18, 36, 40, 70, 10, 18])

# ---------- PRISMA (conteos con fórmulas) ----------
ws = wb.create_sheet("PRISMA")
T = "Tamizaje"
prisma = [
 ("Etapa PRISMA 2020", "Cantidad", "Fórmula / fuente"),
 ("Búsquedas ejecutadas en bases de datos", "=COUNTA(Busquedas!A:A)-1", "Hoja Busquedas"),
 ("Registros identificados en bases de datos", "=COUNTA(Registros!A:A)-1", "Hoja Registros"),
 ("Registros identificados por búsqueda por citas", f'=COUNTIF({T}!E:E,"Búsqueda por citas")', "Hoja Tamizaje"),
 ("Duplicados eliminados antes del tamizaje", f'=COUNTIF(Registros!F:F,"Sí")+COUNTIF({T}!F:F,"Duplicado")', "Duplicados exactos + misma publicación en otro idioma"),
 ("Registros tamizados (título y resumen)", f'=COUNTIFS({T}!E:E,"Bases de datos",{T}!F:F,"<>Duplicado")', ""),
 ("Registros excluidos en el tamizaje", f'=COUNTIFS({T}!E:E,"Bases de datos",{T}!F:F,"Excluido")', ""),
]
for k in MOTIVOS:
    prisma.append((f"   {k}: {MOTIVOS[k]}", f'=COUNTIFS({T}!E:E,"Bases de datos",{T}!G:G,"{k}")', ""))
prisma += [
 ("Reportes buscados para recuperación (bases de datos)", "=B6-B7", ""),
 ("Reportes no recuperados (bases de datos)", f'=COUNTIFS({T}!E:E,"Bases de datos",{T}!I:I,"No")', ""),
 ("Reportes evaluados para elegibilidad (bases de datos)", f'=COUNTIFS({T}!E:E,"Bases de datos",{T}!I:I,"Sí")', ""),
 ("Reportes excluidos en elegibilidad (bases de datos)", f'=B{len(prisma)+3}-COUNTIFS({T}!E:E,"Bases de datos",{T}!J:J,"Incluido")', ""),
 ("Reportes buscados (búsqueda por citas)", "=B4", ""),
 ("Reportes no recuperados (búsqueda por citas)", f'=COUNTIFS({T}!E:E,"Búsqueda por citas",{T}!I:I,"No")', ""),
 ("Reportes evaluados (búsqueda por citas)", f'=COUNTIFS({T}!E:E,"Búsqueda por citas",{T}!I:I,"Sí")', ""),
 ("Estudios incluidos desde bases de datos", f'=COUNTIFS({T}!E:E,"Bases de datos",{T}!J:J,"Incluido")', ""),
 ("Estudios incluidos desde búsqueda por citas", f'=COUNTIFS({T}!E:E,"Búsqueda por citas",{T}!J:J,"Incluido")', ""),
 ("Total de estudios incluidos en la revisión", None, ""),
 ("Control: filas en la hoja Incluidos", "=COUNTA(Incluidos!A:A)-1", "Debe coincidir con el total"),
]
for f in prisma: ws.append(f)
total_row = ws.max_row - 1
ws.cell(total_row, 2).value = f"=B{total_row-2}+B{total_row-1}"
for c in ws[1]: c.fill, c.font, c.border = H_FILL, H_FONT, BORDE
for fila in ws.iter_rows(min_row=2):
    for c in fila: c.font, c.border, c.alignment = B_FONT, BORDE, WRAP
    fila[1].alignment = Alignment(horizontal="center", vertical="top")
for c in ws[total_row]: c.font = Font(name=F, bold=True, size=10)
ws.column_dimensions["A"].width = 70; ws.column_dimensions["B"].width = 12; ws.column_dimensions["C"].width = 50
wb.move_sheet("PRISMA", offset=-(len(wb.sheetnames)-2))
wb.properties.creator = "Juan Miguel León Gómez"
wb.properties.title = "Registro de la búsqueda sistemática de literatura"
wb.save(SALIDA)
print("Escrito", SALIDA, wb.sheetnames)
