"""T10 · Genera la matriz de literatura y completa Revision_Literatura.docx.

Desde la raíz del repositorio:

    python docs/entregable_1/busqueda/t10_generar.py

Salidas (todas en docs/entregable_1/):
  - matriz_literatura.xlsx        matriz de 26 estudios (T10), con leyenda de niveles de verificación
  - registro_busquedas.xlsx       se AÑADE la hoja "Matriz_T10"; las hojas de T09 no se tocan
  - Revision_Literatura.docx      reemplaza los recuadros "Pendiente · T10" de las secciones 3, 4.2 y 5
El script es idempotente: si la hoja o las tablas de T10 ya existen, las reemplaza.
Después hay que abrir y guardar el Excel una vez para recalcular las fórmulas de PRISMA, y exportar el PDF.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import docx
import openpyxl
from docx.shared import Pt
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from incluidos import INCLUIDOS            # noqa: E402  (T09, solo lectura)
from t10_datos import NIVELES, NR, V       # noqa: E402
from t10_sintesis import BRECHAS, BRECHAS_NOTA, SINTESIS  # noqa: E402

CARPETA = AQUI.parent
XLSX_REGISTRO = CARPETA / "registro_busquedas.xlsx"
XLSX_MATRIZ = CARPETA / "matriz_literatura.xlsx"
DOCX = CARPETA / "Revision_Literatura.docx"

# Metadatos de T09 por identificador (id, autores, año, título, revista, vol, doi, origen, base, datos, métodos, hallazgo, RQ, categoría)
META = {r[0]: r for r in INCLUIDOS}
assert len(META) == 26 and set(META) == set(V), "E01–E26 deben coincidir entre T09 y T10"
IDS = sorted(META)


def doi_url(valor):
    """Devuelve una URL navegable a partir de 'doi:xxx' o de una URL."""
    return "https://doi.org/" + valor[4:] if valor.startswith("doi:") else valor


# ----------------------------------------------------------------------------- Excel
COLUMNAS = [
    ("ID", 6), ("Autores", 28), ("Año", 6), ("Título", 45), ("Revista / conferencia", 28), ("Volumen y páginas", 18),
    ("DOI o URL", 40), ("Base / vía de T09", 18), ("País y datos (n)", 40), ("Modelos", 38),
    ("Preprocesamiento", 34), ("Validación", 34), ("Métricas", 24), ("Mejor modelo y resultado verificable", 50),
    ("Hallazgo", 42), ("Limitaciones", 34), ("Relevancia para el proyecto", 38), ("RQ", 10),
    ("Nivel de verificación", 16), ("Fuente consultada (T10)", 26), ("Observaciones sobre metadatos", 42),
]


def filas():
    for i in IDS:
        m, v = META[i], V[i]
        yield [i, m[1], m[2], m[3], m[4], m[5], doi_url(m[6]), m[8], v["datos"], v["modelos"], v["prep"], v["val"],
               v["met"], v["res"], v["hall"], v["lim"], v["rel"], m[12], NIVELES[v["nivel"]], v["fuente"], v["obs"]]


def escribir_hoja(ws):
    """Escribe la matriz en la hoja dada con el formato del proyecto."""
    ws.append([c for c, _ in COLUMNAS])
    for f in filas():
        ws.append(f)
    for j, (_, ancho) in enumerate(COLUMNAS, 1):
        ws.column_dimensions[get_column_letter(j)].width = ancho
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="0F2A47")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    relleno = {"Texto completo": "DFF0E1", "Solo resumen": "FFF4D6", "Solo metadatos": "F9DCD8"}
    for fila in ws.iter_rows(min_row=2):
        for c in fila:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        fila[18].fill = PatternFill("solid", fgColor=relleno[fila[18].value])
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions


def lista(nivel):
    """Identificadores de los estudios con un nivel de verificación dado, para los textos de Word."""
    return ", ".join(i for i in IDS if V[i]["nivel"] == nivel)


def conteos():
    return Counter(NIVELES[v["nivel"]] for v in V.values())


def generar_excel():
    # 1) Libro independiente con matriz y leyenda
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Matriz"
    escribir_hoja(ws)
    ley = wb.create_sheet("Leyenda")
    ley.append(["Concepto", "Detalle"])
    ley.append(["Fuente de los metadatos", "Hoja Incluidos de registro_busquedas.xlsx (T09). T10 no los modifica; las diferencias se anotan en 'Observaciones'."])
    ley.append(["Fecha de verificación", "5 y 6 de noviembre de 2026"])
    ley.append(["No reportado en lo leído", "El dato no apareció en el texto al que se tuvo acceso; no significa que el estudio no lo reporte."])
    for nivel, n in conteos().most_common():
        ley.append([nivel, n])
    ley.append(["Texto completo", "Se leyó el artículo. En E20 mediante un extracto estructurado de la página de PMC."])
    ley.append(["Solo resumen", "Se verificó el resumen y los metadatos; el texto completo no fue accesible (403, pago o redirección)."])
    ley.append(["Solo metadatos", "Solo título, año o revista; el resumen no se pudo leer. No se usa como evidencia."])
    ley.column_dimensions["A"].width = 28
    ley.column_dimensions["B"].width = 110
    for c in ley[1]:
        c.font = Font(bold=True)
    wb.save(XLSX_MATRIZ)

    # 2) Hoja adicional en el registro de T09 (las demás hojas se conservan)
    wr = openpyxl.load_workbook(XLSX_REGISTRO)
    if "Matriz_T10" in wr.sheetnames:
        del wr["Matriz_T10"]
    escribir_hoja(wr.create_sheet("Matriz_T10"))
    wr.save(XLSX_REGISTRO)


# ----------------------------------------------------------------------------- Word
def celda(c, texto, ancho, negrita=False, encabezado=False, tam=15):
    """Da formato de T09 a una celda (fuente 7,5 pt; encabezado azul marino con texto blanco)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    c.width = ancho
    tcPr = c._tc.get_or_add_tcPr()
    if encabezado:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "0F2A47")
        tcPr.append(shd)
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(texto)
    r.font.size = Pt(tam / 2)
    r.bold = negrita or encabezado
    from docx.shared import RGBColor
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if encabezado else RGBColor(0x1F, 0x29, 0x33)


def tabla(d, encabezados, anchos_cm, filas_):
    """Crea una tabla con estilo Table Grid y encabezado repetido en cada página."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm
    t = d.add_table(rows=1, cols=len(encabezados))
    t.style = "Table Grid"
    t.autofit = False
    for j, h in enumerate(encabezados):
        celda(t.rows[0].cells[j], h, Cm(anchos_cm[j]), encabezado=True)
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for fila in filas_:
        cs = t.add_row().cells
        for j, texto in enumerate(fila):
            celda(cs[j], str(texto), Cm(anchos_cm[j]), negrita=(j == 0))
        cant = OxmlElement("w:cantSplit"); cant.set(qn("w:val"), "true")
        t.rows[-1]._tr.get_or_add_trPr().append(cant)
    return t


def nr(t):
    """Abrevia 'No reportado en lo leído' y el token NR como 'n. r.' para las tablas de Word."""
    return re.sub(r"\bNR\b", "n. r.", t.replace(NR, "n. r."))


def parrafo(d, texto, estilo=None, negrita_inicio=None, titulo_tabla=False):
    """Párrafo nuevo (al final del cuerpo; luego se mueve). Con titulo_tabla imita los rótulos 'Tabla n.' de T09."""
    from docx.shared import RGBColor
    p = d.add_paragraph(style=estilo)
    if titulo_tabla:
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(10)
        r = p.add_run(texto); r.bold = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x5A, 0x6B, 0x7E)
        return p
    if negrita_inicio:
        p.add_run(negrita_inicio).bold = True
    p.add_run(texto)
    return p


def generar_word():
    d = docx.Document(DOCX)
    cuerpo = d.element.body
    sectPr = cuerpo[-1]

    def textos():
        return [(i, "".join(x.text or "" for x in el.iter() if x.tag.endswith("}t"))) for i, el in enumerate(cuerpo)]

    # Idempotencia: si ya no hay recuadros pendientes, no se vuelve a aplicar
    if not any(t.startswith("Pendiente · T10") for _, t in textos()):
        print("Word: ya contiene T10; no se modifica.")
        return

    def indice(inicio):
        return next(i for i, t in textos() if t.startswith(inicio))

    def mover(elems, despues_de):
        ref = despues_de
        for e in elems:
            ref.addnext(e)
            ref = e
        return ref

    nuevos = []  # elementos creados al final del cuerpo; se recolocan abajo

    # ----- Sección 3: matriz -------------------------------------------------
    i_s3 = indice("3. Estudios incluidos")
    intro = cuerpo[i_s3 + 1]
    for r in intro.findall(".//{*}r")[1:]:
        intro.remove(r)
    intro.findall(".//{*}t")[0].text = (
        "La tabla 5 agrupa los 26 estudios incluidos por categoría. La tabla 6 resume cada uno con sus datos, modelos, hallazgo, "
        "resultado verificable y nivel de verificación; la tabla 7 reúne preprocesamiento, validación y métricas; y la tabla 8 "
        "documenta la verificación de los identificadores. La matriz completa, con todas las columnas, está en matriz_literatura.xlsx "
        "y en la hoja Matriz_T10 de registro_busquedas.xlsx. El identificador Exx corresponde a la referencia [xx].")

    # Eliminar recuadro pendiente, sus dos líneas y la tabla 6 antigua (con su rótulo)
    i_p = indice("Pendiente · T10")
    for e in list(cuerpo[i_p:i_p + 3]):
        cuerpo.remove(e)
    i_t6 = indice("Tabla 6. Estudios incluidos")
    tbl6_vieja = cuerpo[i_t6 + 1]
    assert tbl6_vieja.tag.endswith("}tbl")
    ancla = cuerpo[i_t6 - 1]
    cuerpo.remove(cuerpo[i_t6]); cuerpo.remove(tbl6_vieja)

    c = conteos()
    bloque = []
    bloque.append(parrafo(d, f"Nivel de verificación (T10, 5–6 nov. 2026): texto completo, {c['Texto completo']} estudios ({lista('TC')}); "
                          f"solo resumen, {c['Solo resumen']}; solo metadatos, {c['Solo metadatos']} ({lista('META')}). \"n. r.\" significa no reportado en lo leído: "
                          "el dato no apareció en el texto al que se tuvo acceso. Los hallazgos de T09 no verificados se señalan como tales.")._p)
    p6 = parrafo(d, "Tabla 6. Matriz de estudios incluidos", titulo_tabla=True)
    p6.paragraph_format.page_break_before = True  # evita que la tabla empiece con una sola fila al pie de la página
    bloque.append(p6._p)
    f6 = []
    for i in IDS:
        m, v = META[i], V[i]
        aut = m[1].split(" and ")[0].split(", ")[0].split(".")[-1].strip()
        if ", " in m[1]:
            et = f"{aut} et al."
        elif " and " in m[1]:
            et = f"{aut} y {m[1].split(' and ')[1].split('.')[-1].strip()}"
        else:
            et = aut
        res = v["res"] if v["res"] not in (NR, "No aplica (artículo teórico)") else ""
        f6.append([i, f"{et} ({m[2]})", nr(v["datos"]), nr(v["modelos"]),
                   nr(v["hall"] + (" Resultado: " + res + "." if res and res != "No verificado" else "")),
                   NIVELES[v["nivel"]], m[12]])
    bloque.append(tabla(d, ["ID", "Estudio", "País y datos (n)", "Modelos", "Hallazgo y resultado verificable", "Nivel", "RQ"],
                        [1.0, 2.4, 4.6, 4.4, 6.6, 2.2, 1.2], f6)._tbl)
    bloque.append(parrafo(d, "")._p)
    bloque.append(parrafo(d, "Tabla 7. Preprocesamiento, validación y métricas", titulo_tabla=True)._p)
    f7 = [[i, nr(V[i]["prep"]), nr(V[i]["val"]), nr(V[i]["met"])] for i in IDS]
    bloque.append(tabla(d, ["ID", "Preprocesamiento", "Validación", "Métricas"], [1.0, 7.4, 7.8, 6.2], f7)._tbl)
    bloque.append(parrafo(d, "")._p)
    bloque.append(parrafo(d, "Tabla 8. Verificación de identificadores y observaciones sobre los metadatos de T09", titulo_tabla=True)._p)
    f8 = [[i, doi_url(META[i][6]), V[i]["fuente"], V[i]["obs"]] for i in IDS]
    bloque.append(tabla(d, ["ID", "DOI o URL", "Fuente consultada", "Observaciones"], [1.0, 7.2, 4.4, 9.8], f8)._tbl)
    bloque.append(parrafo(d, "")._p)
    mover(bloque, ancla)

    # ----- Sección 4.2: síntesis --------------------------------------------
    i_p = indice("Pendiente · T10")
    ancla = cuerpo[i_p - 1]
    for e in list(cuerpo[i_p:i_p + 2]):
        cuerpo.remove(e)
    bloque = [parrafo(d, "Las afirmaciones citan el número de referencia de cada estudio y respetan el nivel de verificación de la tabla 6. "
                         "E13, E15 y E19 no se usan como evidencia.")._p]
    for rq, titulo, msgs in SINTESIS:
        bloque.append(parrafo(d, "", negrita_inicio=f"{rq} · {titulo}")._p)
        for m in msgs:
            bloque.append(parrafo(d, m, estilo="List Bullet")._p)
    mover(bloque, ancla)

    # ----- Sección 5: brechas -----------------------------------------------
    i_p = indice("Pendiente · T10")
    ancla = cuerpo[i_p - 1]
    for e in list(cuerpo[i_p:i_p + 2]):
        cuerpo.remove(e)
    bloque = [parrafo(d, "Brechas respecto de lo leído que justifican la propuesta: comparación rigurosa entre aprendizaje automático y modelo "
                         "hedónico sobre el dataset de Windsor, con validación cruzada repetida y anidada, pruebas estadísticas, análisis de "
                         "complejidad e interpretabilidad.")._p]
    bloque += [parrafo(d, b, estilo="List Bullet")._p for b in BRECHAS]
    bloque.append(parrafo(d, BRECHAS_NOTA)._p)
    mover(bloque, ancla)

    # ----- Sección 6: limitaciones propias de T10 (las de T09 se conservan) ---
    i_ref = indice("Referencias")
    ancla = cuerpo[i_ref - 1]
    lim = [
        ("Verificación de T10:", f" de los 26 estudios, {c['Texto completo']} se leyeron completos, {c['Solo resumen']} solo en resumen y {c['Solo metadatos']} solo en metadatos. "
         "Las páginas de ScienceDirect, MDPI, Taylor & Francis, Wiley y Springer bloquearon el acceso al texto completo. Los campos sin dato dicen \"n. r.\"."),
        ("Hallazgos de T09 sin confirmar:", " la inconsistencia interna de [25] entre el resumen (Ridge 0,34; RF 0,35) y la Figura 10 (RF con el menor MAPE), los R² de [18], el error < 2 % de [15], la ventaja de GB y RF en [19], "
         "la ciudad de [13] y la etiqueta \"ML vs hedónico\" de [21]."),
        ("Referencias:", " solo se corrigieron las de T09 con evidencia de Crossref: [7] (el DOI era el de la fe de erratas), [23] (art. 32 en lugar de pp. 1–28) y se agregó el DOI en [8], [18] y [22]. El detalle está en la tabla 8 y en t10_corregir_referencias.py."),
    ]
    mover([parrafo(d, t, estilo="List Bullet", negrita_inicio=b)._p for b, t in lim], ancla)

    # Mover el sectPr al final (add_paragraph/add_table lo respetan, pero se asegura)
    cuerpo.remove(sectPr); cuerpo.append(sectPr)
    # Los elementos agregados quedaron antes de sectPr al final; ya fueron movidos a su lugar
    d.save(DOCX)


if __name__ == "__main__":
    generar_excel()
    generar_word()
    print("Excel y Word generados.", dict(conteos()))
