"""Generador del documento formal 'Descripcion_Dataset.docx' y '.pdf' (Tarjeta T08).

Estructura calibrada y paginada profesionalmente en exactamente 9 páginas (rango sugerido: 6–10 páginas).
Proyecto: Predicción del precio de vivienda con aprendizaje automático (UAO · 2026-2)
Entregable: Entregable 1 (Propuesta) · Descripción del dataset (40 % de la nota)
Autores: Mauricio José Mozo Terán (Data Engineer), Brandon Eduardo Tobar Sánchez (Model Developer),
         Juan Miguel León Gómez (Project Manager & Reporting Lead).
"""

import os
import subprocess
import sys
from pathlib import Path

import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

# Constantes de rutas
RAIZ = Path(__file__).resolve().parents[1]
DIR_FIGURAS = RAIZ / "reports" / "figures"
DIR_TABLAS = RAIZ / "reports" / "tables"
DIR_SALIDA = RAIZ / "docs" / "entregable_1"
RUTA_DOCX = DIR_SALIDA / "Descripcion_Dataset.docx"
RUTA_PDF = DIR_SALIDA / "Descripcion_Dataset.pdf"

# Paleta de colores profesionales
COLOR_PRIMARIO_HEX = "1B365D"      # Azul marino profundo UAO
COLOR_SECUNDARIO_HEX = "2B6CB0"    # Azul pizarra medio
COLOR_GRIS_FONDO_HEX = "F8FAFC"    # Fondo filas alternas tabla
COLOR_BORDE_TABLA_HEX = "CBD5E0"   # Borde suave tabla
COLOR_TEXTO_OSCURO_HEX = "2D3748"  # Texto principal oscuro
COLOR_DESTACADO_BG = "EBF8FF"      # Fondo cuadro de nota / callout
COLOR_DESTACADO_BORDE = "3182CE"   # Borde izquierdo cuadro de nota

RGB_PRIMARIO = RGBColor(0x1B, 0x36, 0x5D)
RGB_SECUNDARIO = RGBColor(0x2B, 0x6C, 0xB0)
RGB_TEXTO_OSCURO = RGBColor(0x2D, 0x37, 0x48)
RGB_MUTED = RGBColor(0x71, 0x80, 0x96)


def set_cell_background(cell, hex_color: str):
    """Aplica color de fondo a una celda de tabla."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=40, bottom=40, left=80, right=80):
    """Establece padding interno compacto en una celda (en dxa = 1/20 de punto)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color=COLOR_BORDE_TABLA_HEX, sz="4", val="single"):
    """Configura bordes limpios y formales para una tabla."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def format_row(row, is_header=False, is_alternate=False, font_size=7.5):
    """Formatea una fila de tabla con tipografía y alineación optimizada."""
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    if is_header:
        trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

    for cell in row.cells:
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if is_header:
            set_cell_background(cell, COLOR_PRIMARIO_HEX)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.name = "Calibri"
                    run.font.size = Pt(font_size + 0.5)
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        else:
            if is_alternate:
                set_cell_background(cell, COLOR_GRIS_FONDO_HEX)
            else:
                set_cell_background(cell, "FFFFFF")
            set_cell_margins(cell, top=35, bottom=35, left=70, right=70)
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.name = "Calibri"
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = RGB_TEXTO_OSCURO


def add_heading_1(doc, text: str):
    """Encabezado de nivel 1 estilizado."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGB_PRIMARIO
    return p


def add_heading_2(doc, text: str):
    """Encabezado de nivel 2 estilizado."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.0)
    run.font.bold = True
    run.font.color.rgb = RGB_SECUNDARIO
    return p


def add_heading_3(doc, text: str):
    """Encabezado de nivel 3 (títulos de tabla/figura)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(8.5)
    run.font.bold = True
    run.font.color.rgb = RGB_TEXTO_OSCURO
    return p


def add_body_paragraph(doc, text: str, bold_prefix: str = "", space_after=1.5):
    """Párrafo de cuerpo de texto justificado y compacto."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.10
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(9.0)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGB_TEXTO_OSCURO

    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(9.0)
    r_text.font.color.rgb = RGB_TEXTO_OSCURO
    return p


def add_callout_box(doc, text: str, title: str = "NOTA METODOLÓGICA"):
    """Caja de llamada destacada."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(7.0)

    cell = tbl.cell(0, 0)
    set_cell_background(cell, COLOR_DESTACADO_BG)
    set_cell_margins(cell, top=60, bottom=60, left=100, right=100)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="{COLOR_DESTACADO_BORDE}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.10
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    r_tit = p.add_run(f"[{title}] ")
    r_tit.font.name = "Calibri"
    r_tit.font.size = Pt(8.5)
    r_tit.font.bold = True
    r_tit.font.color.rgb = RGB_SECUNDARIO

    r_txt = p.add_run(text)
    r_txt.font.name = "Calibri"
    r_txt.font.size = Pt(8.5)
    r_txt.font.color.rgb = RGB_TEXTO_OSCURO

    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_figure_with_caption(doc, image_path: Path, caption: str, width_inches: float = 4.6):
    """Inserta imagen centrada con su leyenda enlazada."""
    if not image_path.is_file():
        raise FileNotFoundError(f"No se encontró la figura en: {image_path}")

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(3)
    p_img.paragraph_format.space_after = Pt(1)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(str(image_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(3)
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(8.0)
    run_cap.font.italic = True
    run_cap.font.color.rgb = RGB_MUTED


def setup_document_styles_and_margins(doc):
    """Configura márgenes y encabezados."""
    for section in doc.sections:
        section.top_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.header_distance = Inches(0.35)
        section.footer_distance = Inches(0.35)

        # Encabezado
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("UAO · Maestría en Ciencia de Datos e IA · Aprendizaje Automático (2026-2)  |  Entregable 1")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(7.5)
        hrun.font.color.rgb = RGB_MUTED

        # Pie de página
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun1 = fp.add_run("Proyecto ML: Predicción de Vivienda · Descripción del Dataset  |  Página ")
        frun1.font.name = "Calibri"
        frun1.font.size = Pt(7.5)
        frun1.font.color.rgb = RGB_MUTED

        fld_page = OxmlElement("w:fldSimple")
        fld_page.set(qn("w:instr"), "PAGE")
        fp._p.append(fld_page)

        frun2 = fp.add_run(" de ")
        frun2.font.name = "Calibri"
        frun2.font.size = Pt(7.5)
        frun2.font.color.rgb = RGB_MUTED

        fld_numpages = OxmlElement("w:fldSimple")
        fld_numpages.set(qn("w:instr"), "NUMPAGES")
        fp._p.append(fld_numpages)


def build_full_dataset_document() -> Path:
    """Construye el documento Word formal calibrado en exactamente 9 páginas."""
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    doc = docx.Document()
    setup_document_styles_and_margins(doc)

    # =========================================================================
    # PÁGINA 1: PORTADA COMPACTA + RESUMEN + SECCIÓN 1 (FICHA TÉCNICA)
    # =========================================================================
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(1)
    r_meta = p_meta.add_run("UNIVERSIDAD AUTÓNOMA DE OCCIDENTE (UAO) · FACULTAD DE INGENIERÍA\nMAESTRÍA EN CIENCIA DE DATOS E IA · APRENDIZAJE AUTOMÁTICO (PERIODO 2026-2)")
    r_meta.font.name = "Calibri"
    r_meta.font.size = Pt(8.0)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGB_SECUNDARIO

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(1)
    r_title = p_title.add_run("Consolidación y Descripción Detallada del Conjunto de Datos: Mercado Inmobiliario de Windsor 1987 y Housing Prices Dataset")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(14.0)
    r_title.font.bold = True
    r_title.font.color.rgb = RGB_PRIMARIO

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(2)
    r_sub = p_sub.add_run("Documento Técnico de Soporte · Entregable 1 (Propuesta de Proyecto) · Tarjetas T05, T06, T07 y T08")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(9.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGB_SECUNDARIO

    p_aut = doc.add_paragraph()
    p_aut.paragraph_format.space_before = Pt(0)
    p_aut.paragraph_format.space_after = Pt(3)
    p_aut.paragraph_format.line_spacing = 1.10
    r_aut = p_aut.add_run(
        "Equipo: Mauricio José Mozo Terán (Data Engineer) · Brandon Eduardo Tobar Sánchez (Model Developer) · "
        "Juan Miguel León Gómez (Project Manager & Reporting Lead)\n"
        "Fecha: 7 de noviembre de 2026 · Repositorio: github.com/Juanm0331123/ml-precio-vivienda · Semilla global: 5007"
    )
    r_aut.font.name = "Calibri"
    r_aut.font.size = Pt(8.0)
    r_aut.font.color.rgb = RGB_TEXTO_OSCURO

    add_callout_box(
        doc,
        title="RESUMEN EJECUTIVO / ABSTRACT",
        text=(
            "Este documento consolida la caracterización formal, genealogía histórica, calidad multidimensional, "
            "análisis exploratorio (EDA) y preparación de datos para la predicción del precio residencial mediante "
            "aprendizaje automático supervisado. Se demuestra empíricamente la correspondencia exacta entre el dataset "
            "de Kaggle ('Housing.csv', 545 filas × 13 columnas) y la investigación econométrica original de Anglin & Gençay (1996) "
            "en Windsor, Ontario (Canadá) preservada en el paquete 'AER' de R: el precio de Kaggle equivale exactamente al precio "
            "en dólares canadienses (CAD) multiplicado por 70, habiéndose depurado un registro duplicado original (filas 318 y 535 en AER) "
            "e incorporado una variable sintética ('furnishingstatus'). La auditoría certifica 100 % de completitud (0 nulos), 100 % de validez, "
            "0 duplicados globales, la presencia de ruido irreducible (cota de Bayes σ_ε² en 6 pares idénticos), 15 atípicos superiores "
            "conservados por validez de mercado, desbalance en calentador a gas (4.59 %) y un sesgo de orden descendente en el archivo crudo. "
            "El EDA evidencia que log(price) reduce la asimetría un 88.4 % (de 1.21 a 0.14) y corrige la heterocedasticidad en abanico frente al área. "
            "Finalmente, se sella la partición estratificada 80/20 por quintiles de precio (semilla 5007) sin fuga de datos."
        ),
    )

    add_heading_1(doc, "1. Ficha Técnica del Conjunto de Datos")
    add_body_paragraph(
        doc,
        "La estimación del valor de un inmueble constituye un problema canónico de regresión supervisada dentro de la economía urbana "
        "y el aprendizaje automático. De acuerdo con la teoría clásica de precios hedónicos de Rosen [1] y Anglin & Gençay [2], el precio "
        "de mercado residencial se descompone como una función matemática dependiente de su vector de características estructurales "
        "(S), localización espacial (L) y amenidades de confort (A): P = f(S, L, A) + ε. Para este proyecto, el conjunto de datos de trabajo "
        "corresponde al 'Housing Prices Dataset', publicado en Kaggle por M. Yasser H. [4], derivado de las transacciones residenciales "
        "en Windsor, Ontario (Canadá) durante el verano de 1987. En la Tabla 1 se presenta la ficha técnica formal del conjunto de datos."
    )

    add_heading_3(doc, "Tabla 1: Ficha técnica institucional y operativa del conjunto de datos")
    t1_data = [
        ["Atributo Metodológico", "Especificación Técnica Formal"],
        ["Nombre formal del dataset de trabajo", "Housing Prices Dataset (Kaggle) [4]"],
        ["Dataset histórico base / referencia", "Windsor Housing Dataset (Anglin & Gençay 1996 [2] / Paquete R AER [3])"],
        ["Fuente y autoría de publicación", "Kaggle (M. Yasser H., 2021) / Journal of Applied Econometrics (JAE) Data Archive"],
        ["Dirección URL de acceso primario", "https://www.kaggle.com/datasets/yasserh/housing-prices-dataset"],
        ["Dirección URL del repositorio histórico", "http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/"],
        ["Licencia legal del dataset de trabajo", "Creative Commons Zero 1.0 Universal (CC0: Public Domain) [4]"],
        ["Licencia legal del dataset original", "GNU General Public License (GPL-2 | GPL-3) vía paquete AER de R [3]"],
        ["Dimensiones y granularidad", "545 observaciones (filas) × 13 variables (12 predictoras + 1 objetivo)"],
        ["Formato y almacenamiento", "Archivo de texto plano delimitado por comas (.csv), codificación UTF-8"],
        ["Ubicación física en el repositorio", "ml-precio-vivienda/data/raw/Housing.csv (solo lectura inmutable)"],
        ["Periodo temporal de recolección", "Julio, agosto y septiembre de 1987 (Tercer trimestre de 1987)"],
        ["Ciudad y país de recolección", "Windsor, Provincia de Ontario, Canadá"],
        ["Tipo de problema de aprendizaje", "Regresión supervisada (aprendizaje automático tabular)"],
        ["Variable dependiente / objetivo", "price (precio final de transacción residencial escalado, CAD × 70)"],
        ["Objetivo predictivo y analítico", "Predecir el precio residencial hedónico y contrastar modelos lineales vs. ensambles"],
        ["Integridad criptográfica (SHA-256)", "a8ef508351b4d6e87b23a831cf316a3d53c794ed9d969c7b697a865a60a3d65f"],
    ]

    tbl1 = doc.add_table(rows=len(t1_data), cols=2)
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl1)
    tbl1.columns[0].width = Inches(2.2)
    tbl1.columns[1].width = Inches(4.8)

    for i, row in enumerate(tbl1.rows):
        row.cells[0].paragraphs[0].text = t1_data[i][0]
        row.cells[1].paragraphs[0].text = t1_data[i][1]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=7.5)

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 2: SECCIÓN 2 (ORIGEN, GENEALOGÍA Y PROCESO DE OBTENCIÓN)
    # =========================================================================
    add_heading_1(doc, "2. Origen, Genealogía y Proceso de Obtención")

    add_body_paragraph(
        doc,
        "La caracterización rigurosa de los datos exige comprender en profundidad su contexto de captura y las transformaciones "
        "experimentadas a lo largo de su historia. En la tarjeta T05 se llevó a cabo una auditoría forense de procedencia, "
        "reconstruyendo con exactitud la cadena genealógica desde Windsor 1987 hasta la actualidad."
    )

    add_heading_2(doc, "2.1 Cadena de custodia: Windsor 1987 → JAE/AER → Kaggle")
    add_body_paragraph(
        doc,
        "Los datos provienen de contratos de compraventa inmobiliaria procesados por el Windsor Real Estate Board en Windsor, Ontario "
        "(Canadá), entre julio y septiembre de 1987. Windsor es una ciudad industrial fronteriza clave, situada directamente frente a "
        "Detroit, Michigan. En 1996, Paul M. Anglin y Ramazan Gençay publicaron en el 'Journal of Applied Econometrics' [2] el artículo "
        "'Semiparametric Estimation of a Hedonic Price Function', utilizando esta base para contrastar modelos de precios hedónicos lineales "
        "frente a estimadores no paramétricos de kernel. Posteriormente, el dataset fue preservado en el JAE Data Archive e integrado como "
        "'HousePrices' en el paquete de R 'AER' (Kleiber & Zeileis, 2008) [3] y en Rdatasets [10]. En 2021, M. Yasser H. lo compiló en Kaggle [4]."
    )

    add_heading_2(doc, "2.2 Transformaciones demostradas algorítmicamente (Tarjeta T05)")
    add_body_paragraph(
        doc,
        "Mediante contrastación con multiconjuntos (Counter) en '01_procedencia_y_calidad.ipynb', se demostraron cuatro hechos irrefutables:\n"
        "1. Factor de escala en el precio (× 70): En las 545 observaciones comunes, el precio en Kaggle equivale exactamente al precio original "
        "en CAD multiplicado por 70 (price_Kaggle = 70 × price_Windsor). En el 100 % de las filas, price mod 70 = 0. El rango original "
        "[25,000, 190,000] CAD pasa a [1,750,000, 13,300,000]. Esto descarta que sean precios en dólares modernos o sin escalar.\n"
        "2. Depuración del registro duplicado (546 vs. 545 filas): El dataset original de Windsor (AER) contenía un registro exactamente "
        "idéntico duplicado en dos filas (318 y 535 en índice 1-based): price = 82,000 CAD, lotsize = 5,000 sq ft, bedrooms = 3, "
        "bathrooms = 1, stories = 3, driveway = yes, recreation = no, fullbase = no, gasheat = no, aircon = yes, garage = 0, prefer = no. "
        "Kaggle depuró este duplicado redundante, conservando 545 viviendas únicas.\n"
        "3. Renombramiento de atributos: lotsize → area, driveway → mainroad, recreation → guestroom, fullbase → basement, gasheat → hotwaterheating, "
        "aircon → airconditioning, garage → parking, y prefer → prefarea. Price, bedrooms, bathrooms y stories conservaron sus nombres.\n"
        "4. Variable sintética añadida ('furnishingstatus'): Ausente en el registro original de 1987 y en Anglin & Gençay [2]. Fue introducida "
        "en Kaggle con 3 niveles ('furnished', 'semi-furnished', 'unfurnished'). Se evaluará mediante ablación para vigilar posibles sesgos sintéticos."
    )

    add_heading_3(doc, "Tabla 2: Matriz comparativa entre el dataset histórico original (AER) y el dataset de trabajo (Kaggle)")
    t2_data = [
        ["Criterio Comparativo", "Windsor Housing (AER / Anglin & Gençay 1996)", "Housing Prices Dataset (Kaggle / Yasser H.)"],
        ["Número de observaciones", "546 filas (contiene 1 fila idéntica duplicada)", "545 filas (duplicado depurado)"],
        ["Número de variables", "12 variables (1 target + 11 covariables)", "13 variables (1 target + 12 covariables)"],
        ["Variable de precio", "price en CAD reales (rango: [25,000, 190,000])", "price escalado (rango: [1,750,000, 13,300,000])"],
        ["Relación del precio", "Precio base original (1.0×)", "Precio exactamente multiplicado por 70 (70.0×)"],
        ["Variable amoblado", "No existe en el registro econométrico histórico", "furnishingstatus añadida de manera sintética"],
        ["Nombres de variables", "lotsize, driveway, recreation, fullbase, gasheat", "area, mainroad, guestroom, basement, hotwaterheating"],
        ["Licencia de distribución", "GPL-2 / GPL-3 (Copyleft libre con atribución)", "CC0 1.0 Universal (Dominio público)"],
    ]

    tbl2 = doc.add_table(rows=len(t2_data), cols=3)
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl2)
    tbl2.columns[0].width = Inches(1.8)
    tbl2.columns[1].width = Inches(2.6)
    tbl2.columns[2].width = Inches(2.6)

    for i, row in enumerate(tbl2.rows):
        row.cells[0].paragraphs[0].text = t2_data[i][0]
        row.cells[1].paragraphs[0].text = t2_data[i][1]
        row.cells[2].paragraphs[0].text = t2_data[i][2]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=8.0)

    add_heading_2(doc, "2.3 Marco legal y coexistencia de licencias")
    add_body_paragraph(
        doc,
        "En cuanto al marco legal, coexisten armónicamente la licencia CC0 1.0 Universal de Kaggle (renuncia de dominio público que autoriza "
        "uso, modelado y distribución sin restricciones) y la licencia GPL-2/3 del paquete AER [3] (que autoriza uso académico citando "
        "a Anglin & Gençay [2]). El proyecto satisface ambos marcos con rigor ético e investigativo."
    )

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 3: SECCIÓN 3 (DICCIONARIO DE DATOS COMPLETO)
    # =========================================================================
    add_heading_1(doc, "3. Diccionario de Datos")

    add_body_paragraph(
        doc,
        "El dataset contiene 13 variables que describen las dimensiones físicas, estructurales y amenidades residenciales. "
        "En la Tabla 3 se detalla el diccionario formal de variables, su descripción técnica, tipología, unidades, rango y rol."
    )

    add_heading_3(doc, "Tabla 3: Diccionario de datos formal para las 13 variables del dataset de trabajo")
    t3_data = [
        ["Variable (Kaggle)", "Nombre AER", "Descripción Real y Económica", "Tipo de Dato", "Unidades / Niveles", "Rango Obs.", "Rol en ML"],
        ["price", "price", "Precio final de venta residencial transado (CAD × 70)", "Numérica continua", "Moneda escalada", "[1.75M, 13.3M]", "Objetivo (Target)"],
        ["area", "lotsize", "Área total superficial del lote o terreno del inmueble", "Numérica continua", "Pies cuadrados (sq ft)", "[1,650, 16,200]", "Predictora (Feature)"],
        ["bedrooms", "bedrooms", "Número total de dormitorios o alcobas en la propiedad", "Numérica discreta", "Entero (cuartos)", "[1, 6]", "Predictora (Feature)"],
        ["bathrooms", "bathrooms", "Número de cuartos de baño completos en la vivienda", "Numérica discreta", "Entero (baños)", "[1, 4]", "Predictora (Feature)"],
        ["stories", "stories", "Número de pisos o plantas habitables sobre rasante", "Numérica discreta", "Entero (plantas)", "[1, 4]", "Predictora (Feature)"],
        ["mainroad", "driveway", "Presencia de entrada vehicular privada pavimentada", "Categórica binaria", "{'yes', 'no'}", "{'no', 'yes'}", "Predictora (Feature)"],
        ["guestroom", "recreation", "Presencia de sala de recreación o cuarto de juegos", "Categórica binaria", "{'yes', 'no'}", "{'no', 'yes'}", "Predictora (Feature)"],
        ["basement", "fullbase", "Presencia de sótano completo terminado y habitable", "Categórica binaria", "{'yes', 'no'}", "{'no', 'yes'}", "Predictora (Feature)"],
        ["hotwaterheating", "gasheat", "Sistema de calentamiento de agua accionado por gas", "Categórica binaria", "{'yes', 'no'}", "{'no', 'yes'}", "Predictora (Feature)"],
        ["airconditioning", "aircon", "Sistema de aire acondicionado y climatización central", "Categórica binaria", "{'yes', 'no'}", "{'no', 'yes'}", "Predictora (Feature)"],
        ["parking", "garage", "Plazas o capacidad de vehículos en garaje cubierto", "Numérica discreta", "Entero (plazas)", "[0, 3]", "Predictora (Feature)"],
        ["prefarea", "prefer", "Ubicación en vecindario preferencial de alta cotización", "Categórica binaria", "{'yes', 'no'}", "{'no', 'yes'}", "Predictora (Feature)"],
        ["furnishingstatus", "(Ninguno)", "Grado de dotación o amoblado (columna sintética Kaggle)", "Categórica politómica", "{'unfurnished', 'semi-furnished', 'furnished'}", "3 niveles", "Predictora (Feature)"],
    ]

    tbl3 = doc.add_table(rows=len(t3_data), cols=7)
    tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl3)
    tbl3.columns[0].width = Inches(1.1)
    tbl3.columns[1].width = Inches(0.8)
    tbl3.columns[2].width = Inches(1.9)
    tbl3.columns[3].width = Inches(0.9)
    tbl3.columns[4].width = Inches(0.9)
    tbl3.columns[5].width = Inches(0.8)
    tbl3.columns[6].width = Inches(0.6)

    for i, row in enumerate(tbl3.rows):
        for j in range(7):
            row.cells[j].paragraphs[0].text = t3_data[i][j]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=7.5)

    add_body_paragraph(
        doc,
        "La taxonomía de los atributos revela una estructura balanceada: (a) Variables dimensionales de capacidad física (area, bedrooms, "
        "bathrooms, stories, parking); (b) Amenidades y equipamiento de confort (mainroad, guestroom, basement, hotwaterheating, airconditioning); "
        "(c) Componente hedónico de localización espacial (prefarea); y (d) Dotación de mobiliario (furnishingstatus)."
    )

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 4: SECCIÓN 4 (CALIDAD DE DATOS Y SCORECARD)
    # =========================================================================
    add_heading_1(doc, "4. Auditoría de Calidad de Datos")

    add_body_paragraph(
        doc,
        "La auditoría formal de calidad evaluó seis dimensiones analíticas clave (Tarjeta T06), implementadas en 'src/datos.py' "
        "y consolidadas en el scorecard de la Tabla 4:\n"
        "• Completitud: 100.0 % en todas las variables (0 nulos en las 7,085 celdas). No se requiere imputación.\n"
        "• Unicidad y Ruido Irreducible: 0 duplicados globales exactos. No obstante, se detectaron 6 casos de viviendas con vectores "
        "idénticos de covariables X vendidas a precios Y distintos (16 casos si se omite furnishingstatus). Este hallazgo prueba la presencia "
        "de ruido irreducible (cota inferior de Bayes σ_ε²): existen factores latentes no observados (acabados finos, vista, conservación, "
        "negociación) que impiden a cualquier modelo predecir ambos precios con error cero.\n"
        "• Validez: 100 % de cumplimiento de rangos y dominios inmobiliarios esperados.\n"
        "• Atípicos: 15 observaciones superiores por regla 1.5×IQR en price (> 8,750,000) y 16 por puntaje z robusto MAD (|z_MAD| > 3). "
        "Se decidió formalmente CONSERVARLOS: no son errores sino viviendas premium de alta gama con terrenos extensos y amenidades completas. "
        "Eliminarlos generaría sesgo de truncamiento perjudicial para predecir casas de lujo. El sesgo se corrige mediante log(price).\n"
        "• Desbalance de categorías: Severo desbalance en 'hotwaterheating' (solo 4.59 % 'yes' = 25 casos vs. 95.41 % 'no'). Implica "
        "riesgo de alta varianza en submuestras. Las demás categorías son estables (mainroad 14.1 %, guestroom 17.8 %, basement 35.1 %, "
        "airconditioning 31.6 %, prefarea 23.5 %, furnishingstatus equilibrada).\n"
        "• Artefacto de orden: El CSV crudo viene ordenado descendentemente por precio (de 13.3M a 1.75M). Un corte secuencial provocaría "
        "un desfase de distribución catastrófico (dataset shift). Se neutraliza mediante partición estratificada con barajado previo."
    )

    add_heading_3(doc, "Tabla 4: Scorecard multidimensional de calidad de datos por variable")
    t4_data = [
        ["Variable", "Tipo", "Nulos", "Compl. %", "Únicos", "Dominio Esperado", "Validez", "Atípicos IQR", "% Clase Min.", "Observaciones y Diagnóstico"],
        ["price", "int64", "0", "100.0 %", "219", "Entero > 0 (CAD × 70)", "Cumple", "15 sup.", "N/A", "15 atípicos superiores conservados por validez de mercado"],
        ["area", "int64", "0", "100.0 %", "284", "Entero > 0 (sq ft)", "Cumple", "12 sup.", "N/A", "12 atípicos superiores; asimetría positiva hacia lotes grandes"],
        ["bedrooms", "int64", "0", "100.0 %", "6", "[1, 6] enteros", "Cumple", "12", "N/A", "Calidad óptima; concentración modal en 3 dormitorios (55.0 %)"],
        ["bathrooms", "int64", "0", "100.0 %", "4", "[1, 4] enteros", "Cumple", "1", "N/A", "Calidad óptima; concentración modal en 1 baño (73.6 %)"],
        ["stories", "int64", "0", "100.0 %", "4", "[1, 4] enteros", "Cumple", "41", "N/A", "Calidad óptima; distribución dominada por 1 y 2 plantas (87.7 %)"],
        ["mainroad", "str", "0", "100.0 %", "2", "{'yes', 'no'}", "Cumple", "0", "14.13 %", "Calidad óptima; 77 viviendas sin entrada vehicular principal"],
        ["guestroom", "str", "0", "100.0 %", "2", "{'yes', 'no'}", "Cumple", "0", "17.80 %", "Calidad óptima; 97 viviendas con sala de recreación"],
        ["basement", "str", "0", "100.0 %", "2", "{'yes', 'no'}", "Cumple", "0", "35.05 %", "Calidad óptima; 191 viviendas con sótano terminado"],
        ["hotwaterheating", "str", "0", "100.0 %", "2", "{'yes', 'no'}", "Cumple", "0", "4.59 %", "Desbalance severo (solo 25 casos 'yes'); riesgo de varianza"],
        ["airconditioning", "str", "0", "100.0 %", "2", "{'yes', 'no'}", "Cumple", "0", "31.56 %", "Calidad óptima; 172 viviendas con climatización central"],
        ["parking", "int64", "0", "100.0 %", "4", "[0, 3] enteros", "Cumple", "12", "N/A", "Calidad óptima; 299 viviendas sin garaje cubierto (54.9 %)"],
        ["prefarea", "str", "0", "100.0 %", "2", "{'yes', 'no'}", "Cumple", "0", "23.49 %", "Calidad óptima; 128 viviendas en zona de alta cotización"],
        ["furnishingstatus", "str", "0", "100.0 %", "3", "3 categorías válidas", "Cumple", "0", "25.69 %", "Calidad óptima; semi (41.6 %), unfurn (32.7 %), furn (25.7 %)"],
    ]

    tbl4 = doc.add_table(rows=len(t4_data), cols=10)
    tbl4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl4)
    tbl4.columns[0].width = Inches(0.9)
    tbl4.columns[1].width = Inches(0.4)
    tbl4.columns[2].width = Inches(0.3)
    tbl4.columns[3].width = Inches(0.5)
    tbl4.columns[4].width = Inches(0.4)
    tbl4.columns[5].width = Inches(0.9)
    tbl4.columns[6].width = Inches(0.4)
    tbl4.columns[7].width = Inches(0.5)
    tbl4.columns[8].width = Inches(0.5)
    tbl4.columns[9].width = Inches(2.2)

    for i, row in enumerate(tbl4.rows):
        for j in range(10):
            row.cells[j].paragraphs[0].text = t4_data[i][j]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=7.0)

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 5: SECCIÓN 5 (EDA - FIGURA 1 EFECTO LOGARÍTMICO Y TABLA 5)
    # =========================================================================
    add_heading_1(doc, "5. Análisis Exploratorio de Datos (EDA)")

    add_heading_2(doc, "5.1 Figura 1: Efecto estabilizador de log(price) sobre la asimetría y curtosis")
    add_body_paragraph(
        doc,
        "La Figura 1 compara la distribución del precio en escala natural frente a la transformación logarítmica natural log(price). "
        "En la escala original (arriba), el precio muestra una asimetría positiva pronunciada (Fisher-Pearson = 1.211), curtosis de exceso de "
        "1.960 y 15 atípicos superiores que curvan el Q-Q plot. Al aplicar log(price) (abajo), la asimetría se reduce en un 88.4 % "
        "(descendiendo a 0.141), la curtosis cae a -0.142 (prácticamente mesocúrtica) y los puntos en el Q-Q plot se alinean perfectamente "
        "a lo largo de la normal teórica. Los datos exactos se resumen en la Tabla 5."
    )

    add_figure_with_caption(
        doc,
        DIR_FIGURAS / "fig04_efecto_log_precio.png",
        "Figura 1: Efecto comparativo de la transformación logarítmica natural log(price) frente a la escala original. "
        "Fila superior: escala original (asimetría = 1.21, 15 atípicos superiores, Q-Q divergente). Fila inferior: escala logarítmica "
        "(distribución normalizada, asimetría = 0.14, curtosis = -0.14 y alineación Q-Q casi perfecta).",
        width_inches=4.8,
    )

    add_heading_3(doc, "Tabla 5: Comparación de momentos estadísticos entre la escala natural y logarítmica de price")
    t5_data = [
        ["Métrica Estadística", "Escala Original (price)", "Escala Transformada (log(price))", "Efecto Observado y Diagnóstico Metodológico"],
        ["Media aritmética", "4,766,729.25", "15.3218", "Pasa a interpretarse en escala log (media geométrica: 4,498,655 CAD escalados)"],
        ["Mediana (Percentil 50)", "4,340,000.00", "15.2834", "Convergencia casi idéntica con la media en escala log (diferencia relativa < 0.25 %)"],
        ["Desviación estándar", "1,870,439.62", "0.3667", "Varianza estabilizada; representa dispersión relativa porcentual"],
        ["Rango Intercuartílico (IQR)", "2,075,000.00", "0.4855", "Dispersión intercuartílica condensada proporcionalmente"],
        ["Asimetría (Skewness)", "1.2110", "0.1407", "Reducción drástica del 88.4 %; distribución casi perfectamente simétrica"],
        ["Curtosis de exceso", "1.9601", "-0.1423", "Pasa de leptocúrtica a prácticamente mesocúrtica teórica (-0.14 vs. 0.0)"],
        ["Mínimo / Máximo", "1.75M / 13.30M", "14.3751 / 16.4033", "Atenúa drásticamente el apalancamiento numérico de valores extremos"],
    ]

    tbl5 = doc.add_table(rows=len(t5_data), cols=4)
    tbl5.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl5)
    tbl5.columns[0].width = Inches(1.5)
    tbl5.columns[1].width = Inches(1.3)
    tbl5.columns[2].width = Inches(1.4)
    tbl5.columns[3].width = Inches(2.8)

    for i, row in enumerate(tbl5.rows):
        row.cells[0].paragraphs[0].text = t5_data[i][0]
        row.cells[1].paragraphs[0].text = t5_data[i][1]
        row.cells[2].paragraphs[0].text = t5_data[i][2]
        row.cells[3].paragraphs[0].text = t5_data[i][3]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=7.5)

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 6: SECCIÓN 5 (EDA - FIGURAS 2 Y 3: DISCRETAS Y CATEGÓRICAS)
    # =========================================================================
    add_heading_2(doc, "5.2 Figura 2: Distribución de variables predictoras discretas")
    add_body_paragraph(
        doc,
        "La Figura 2 ilustra las distribuciones de dormitorios, baños, plantas y plazas de garaje. Se evidencia una marcada concentración "
        "habitacional típica del parque residencial de Windsor en 1987: el 55.0 % de las casas tienen 3 alcobas y el 24.9 % tienen 2; "
        "el 73.6 % poseen un único baño completo y el 24.4 % tienen 2; el 87.7 % son de 1 o 2 pisos; y el 54.9 % no disponen de garaje "
        "cubierto privado (frente a 23.1 % con 1 plaza y 19.8 % con 2 plazas)."
    )

    add_figure_with_caption(
        doc,
        DIR_FIGURAS / "fig05_variables_discretas.png",
        "Figura 2: Distribución univariada de variables discretas. Arriba: 'bedrooms' (predominio modal en 3) y 'bathrooms' (73.6 % con 1 baño). "
        "Abajo: 'stories' (87.7 % concentrado en 1 y 2 plantas) y 'parking' (mayoría sin estacionamiento cubierto).",
        width_inches=4.2,
    )

    add_heading_2(doc, "5.3 Figura 3: Frecuencias relativas de variables categóricas")
    add_body_paragraph(
        doc,
        "La Figura 3 detalla las proporciones de las variables cualitativas. Se observa una alta penetración de acceso vial principal "
        "(85.87 % mainroad), un tercio de viviendas con sótano terminado (35.05 %) y climatización central (31.56 %), y un 23.49 % "
        "ubicadas en zona de alta cotización ('prefarea'). 'Furnishingstatus' se distribuye armónicamente (41.6 % semi, 32.7 % unfurn, 25.7 % furn), "
        "mientras que 'hotwaterheating' ratifica su desbalance crítico con solo 4.59 % de casos afirmativos."
    )

    add_figure_with_caption(
        doc,
        DIR_FIGURAS / "fig06_variables_categoricas.png",
        "Figura 3: Frecuencias y proporciones relativas de variables categóricas binarias y politómica. Se ratifica el desbalance "
        "en calentador a gas (4.59 %) y la distribución homogénea de los tres niveles de amoblado.",
        width_inches=4.5,
    )

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 7: SECCIÓN 5 (EDA - FIGURAS 4 Y 5: CORRELACIONES Y LOWESS)
    # =========================================================================
    add_heading_2(doc, "5.4 Figura 4: Correlaciones bivariadas de Pearson y Spearman")
    add_body_paragraph(
        doc,
        "La Figura 4 contrasta los coeficientes de correlación lineal de Pearson y monótona de Spearman contra el precio y log(price). "
        "El tamaño del lote ('area') lidera la correlación lineal (r = 0.536), seguido muy de cerca por el número de baños (r = 0.518), "
        "aire acondicionado (r = 0.453), pisos (r = 0.421) y garajes (r = 0.384). Spearman revela dependencias monótonas crecientes intensas "
        "en baños (r_s = 0.497) y dormitorios (r_s = 0.377). En log(price), la correlación con el área incrementa su linealidad."
    )

    add_figure_with_caption(
        doc,
        DIR_FIGURAS / "fig07_correlaciones_pearson_spearman.png",
        "Figura 4: Jerarquía de correlaciones bivariadas con el precio. Izquierda: Coeficientes de Pearson y Spearman en escala natural. "
        "Derecha: Coeficientes correspondientes evaluados contra la escala logarítmica log(price).",
        width_inches=4.2,
    )

    add_heading_2(doc, "5.5 Figura 5: Dispersión precio vs. área y heterocedasticidad en abanico")
    add_body_paragraph(
        doc,
        "La Figura 5 examina la relación bivariada central entre precio y área de terreno, comparando la escala natural frente a la log-log, "
        "e incorporando tanto la recta de regresión por MCO como la curva no paramétrica LOWESS [7]. En la escala natural (izquierda) se "
        "observa un nítido patrón de heterocedasticidad en abanico: para áreas < 4,000 sq ft la dispersión de precios es muy reducida, "
        "pero por encima de 8,000 sq ft los precios explotan verticalmente desde 4M hasta más de 13M. En escala log-log (derecha), "
        "la dispersión se vuelve completamente homocedástica y la curva LOWESS coincide plenamente con la recta lineal."
    )

    add_figure_with_caption(
        doc,
        DIR_FIGURAS / "fig08_dispersion_precio_area.png",
        "Figura 5: Dispersión bivariada precio vs. área de lote. Panel izquierdo: Escala natural, evidenciando patrón de heterocedasticidad "
        "en abanico. Panel derecho: Escala log-log (log(price) vs. log(area)), logrando homocedasticidad y coincidencia perfecta OLS-LOWESS.",
        width_inches=4.2,
    )

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 8: SECCIÓN 5 (FIGURA 6 Y TABLA 6 VIF) + SECCIÓN 6 (LIMPIEZA Y TABLA 7)
    # =========================================================================
    add_heading_2(doc, "5.6 Figura 6: Primas de precio asociadas a amenidades y amoblado")
    add_body_paragraph(
        doc,
        "La Figura 6 presenta diagramas de caja analíticos de precio condicionado a cada atributo cualitativo. Las primas económicas son "
        "sustanciales: estar en zona preferencial eleva la mediana de 4.34M a 6.12M (+41.1 %); disponer de aire acondicionado la incrementa "
        "de 4.34M a 6.19M (+42.7 %); y el acceso por calle principal la sube de 3.85M a 4.90M (+27.3 %). En amoblado, la mediana asciende "
        "progresivamente de 4.13M (unfurnished) a 4.55M (semi-furnished) y 5.95M (furnished), confirmando su comportamiento ordinal."
    )

    add_figure_with_caption(
        doc,
        DIR_FIGURAS / "fig09_boxplots_precio_categoricas.png",
        "Figura 6: Distribución del precio según amenidades binarias y amoblado, evidenciando primas superiores al 40 % en "
        "zona preferencial y aire acondicionado, así como un incremento monótono según el grado de dotación.",
        width_inches=4.2,
    )

    add_heading_3(doc, "Tabla 6: Factor de Inflación de la Varianza (VIF) de las covariables predictoras")
    t6_data = [
        ["Variable Predictora", "VIF", "R² Auxiliar", "Diagnóstico Metodológico"],
        ["bedrooms", "1.611", "0.379", "Colinealidad baja/segura (asociada moderadamente a stories y bathrooms)"],
        ["stories", "1.458", "0.314", "Colinealidad baja/segura (asociada a bedrooms y basement)"],
        ["area", "1.341", "0.254", "Colinealidad baja/segura (ortogonalidad adecuada respecto a comodidades)"],
        ["bathrooms", "1.306", "0.234", "Colinealidad baja/segura (independencia lineal favorable)"],
        ["parking / aircon", "1.238", "0.192", "Colinealidad muy baja/segura (VIF < 1.25 en ambas)"],
        ["basement / mainroad", "1.229", "0.186", "Colinealidad muy baja/segura (independencia mutua)"],
        ["prefarea / guestroom", "1.185", "0.156", "Colinealidad muy baja/segura (VIF < 1.20)"],
        ["furnishing / hotwater", "1.096", "0.088", "Colinealidad prácticamente nula (VIF ≈ 1.05 - 1.10)"],
    ]

    tbl6 = doc.add_table(rows=len(t6_data), cols=4)
    tbl6.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl6)
    tbl6.columns[0].width = Inches(1.8)
    tbl6.columns[1].width = Inches(0.8)
    tbl6.columns[2].width = Inches(1.0)
    tbl6.columns[3].width = Inches(3.4)

    for i, row in enumerate(tbl6.rows):
        row.cells[0].paragraphs[0].text = t6_data[i][0]
        row.cells[1].paragraphs[0].text = t6_data[i][1]
        row.cells[2].paragraphs[0].text = t6_data[i][2]
        row.cells[3].paragraphs[0].text = t6_data[i][3]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=7.0)

    add_heading_1(doc, "6. Limpieza, Preparación Realizada y Partición Sellada")
    add_body_paragraph(
        doc,
        "El preprocesamiento implementado en 'src/datos.py' (Tarjeta T06) se ejecutó bajo principios rigurosos de reproducibilidad y anti-fuga: "
        "(1) Inserción de ID único determinístico (0 a 544); (2) Codificación binaria determinística {0, 1} y ordinal de amoblado {0, 1, 2}; "
        "(3) Salvaguarda CERO fuga de información (no data leakage): 'housing_limpio.csv' no aplica estandarización ni escalado muestral global, "
        "delegándose a Pipelines de entrenamiento; (4) Partición estratificada 80/20 por quintiles de precio (semilla 5007) con barajado previo "
        "para neutralizar el orden descendente del CSV crudo. El conjunto de prueba queda SELLADO en disco hasta T25."
    )

    add_heading_3(doc, "Tabla 7: Balance muestral y distribución por quintiles en la partición sellada 80/20")
    t7_data = [
        ["Estrato / Quintil", "Rango de Precio en CAD Escalado", "Entrenamiento (80 %)", "Prueba Sellada (20 %)", "Total Observaciones", "Proporción Exacta"],
        ["Q1 (Muy bajo)", "[1,750,000 – 3,430,000]", "88 observaciones", "22 observaciones", "110 observaciones", "80.00 % / 20.00 %"],
        ["Q2 (Bajo)", "[3,430,000 – 4,123,000]", "87 observaciones", "22 observaciones", "109 observaciones", "79.82 % / 20.18 %"],
        ["Q3 (Medio)", "[4,130,000 – 4,900,000]", "87 observaciones", "22 observaciones", "109 observaciones", "79.82 % / 20.18 %"],
        ["Q4 (Alto)", "[4,907,000 – 6,090,000]", "87 observaciones", "22 observaciones", "109 observaciones", "79.82 % / 20.18 %"],
        ["Q5 (Muy alto / lujo)", "[6,090,000 – 13,300,000]", "87 observaciones", "21 observaciones", "108 observaciones", "80.56 % / 19.44 %"],
        ["Total Consolidado", "[1,750,000 – 13,300,000]", "436 observaciones (80.0 %)", "109 observaciones (20.0 %)", "545 observaciones", "100.0 % (Sellado T25)"],
    ]

    tbl7 = doc.add_table(rows=len(t7_data), cols=6)
    tbl7.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl7)
    tbl7.columns[0].width = Inches(1.2)
    tbl7.columns[1].width = Inches(1.8)
    tbl7.columns[2].width = Inches(1.1)
    tbl7.columns[3].width = Inches(1.1)
    tbl7.columns[4].width = Inches(1.0)
    tbl7.columns[5].width = Inches(0.8)

    for i, row in enumerate(tbl7.rows):
        for j in range(6):
            row.cells[j].paragraphs[0].text = t7_data[i][j]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=7.0)

    # doc.add_page_break()

    # =========================================================================
    # PÁGINA 9: SECCIONES 7, 8, 9 (REVISIÓN JUAN MIGUEL) Y REFERENCIAS IEEE
    # =========================================================================
    add_heading_1(doc, "7. Implicaciones Directas para el Modelado")
    add_body_paragraph(
        doc,
        "Los análisis anteriores establecen directrices fundamentales para la fase de modelado supervisado (Entregables 1 y 2):\n"
        "1. Transformación logarítmica de la variable objetivo: Obligatoria en modelos lineales (OLS, Ridge, Lasso, ElasticNet) para asegurar "
        "normalidad y homocedasticidad. En evaluación se aplicará re-transformación exponencial con corrección de Jensen (exp(s²/2)).\n"
        "2. Validación cruzada repetida ante muestra pequeña (N = 545): Se adopta RepeatedStratifiedKFold (5 folds × 5 repeticiones = 25 ajustes) "
        "para reducir la varianza muestral y reportar métricas sólidas (RMSE, MAE, R²) con intervalos de confianza estrechos.\n"
        "3. Inclusión justificada de ensambles: Curvaturas LOWESS y dependencias no lineales (Información Mutua) respaldan el uso de "
        "Random Forest, Gradient Boosting, XGBoost y LightGBM para capturar interacciones de amenidades sin sobreajuste.\n"
        "4. Ablación de 'furnishingstatus': Se medirá el impacto de incluir/excluir la columna sintética de Kaggle para controlar sesgos espurios.\n"
        "5. Control del desbalance en 'hotwaterheating' (4.59 %): Regularización de hojas en árboles (min_samples_leaf ≥ 5) para evitar divisiones frágiles."
    )

    add_heading_1(doc, "8. Reproducibilidad y Auditoría Criptográfica")
    add_body_paragraph(
        doc,
        "El proyecto se rige por trazabilidad total: repositorio oficial en GitHub (github.com/Juanm0331123/ml-precio-vivienda), notebooks "
        "ejecutables localmente y en Google Colab, y semilla global SEMILLA = 5007 fijada en 'src/config.py'. En la Tabla 8 se registran "
        "los checksums SHA-256 de los archivos maestros, calculados sobre los archivos tal como están en el repositorio (fin de línea LF)."
    )

    add_heading_3(doc, "Tabla 8: Trazabilidad criptográfica e integridad de archivos clave (SHA-256)")
    t8_data = [
        ["Archivo del Repositorio", "Tipo y Alcance", "Huella Digital Criptográfica (SHA-256 Oficial)"],
        ["data/raw/Housing.csv", "Datos crudos de trabajo (Kaggle)", "a8ef508351b4d6e87b23a831cf316a3d53c794ed9d969c7b697a865a60a3d65f"],
        ["data/raw/HousePrices_Windsor_1987_AER.csv", "Datos históricos de referencia (AER)", "4c7f6df4c6264cf6f57ea6ec5157958c4033efe1f51b2855efb18e297274bfe3"],
        ["data/processed/housing_limpio.csv", "Dataset procesado reproducible (T06)", "26b609d04237e4b8e69adcef22fc13368a3163d488d58d9c8f656012f09358b4"],
        ["data/processed/particion.csv", "Partición sellada 80/20 train/test (T06)", "16413b3207343d81d5251861f417edcce3bf6de29a2bd18fa3493840fc1ed009"],
        ["reports/tables/calidad_datos.csv", "Scorecard formal de calidad de datos", "05a7fb1631061743996d3b6982fbffb11ddfb03d2af138fcc13eaa4e3a9b99d9"],
        ["reports/tables/diccionario_datos.csv", "Diccionario formal de variables", "e7f6aa193b90a40ad46b38feb3bd37e86d4a3c34f35b9c18bc326d079f08ba0d"],
    ]

    tbl8 = doc.add_table(rows=len(t8_data), cols=3)
    tbl8.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl8)
    tbl8.columns[0].width = Inches(2.2)
    tbl8.columns[1].width = Inches(1.8)
    tbl8.columns[2].width = Inches(3.0)

    for i, row in enumerate(tbl8.rows):
        row.cells[0].paragraphs[0].text = t8_data[i][0]
        row.cells[1].paragraphs[0].text = t8_data[i][1]
        row.cells[2].paragraphs[0].text = t8_data[i][2]
        format_row(row, is_header=(i == 0), is_alternate=(i % 2 == 1 and i > 0), font_size=7.0)

    add_heading_1(doc, "9. Referencias Bibliográficas (Formato IEEE)")
    referencias = [
        "[1] S. Rosen, \"Hedonic prices and implicit markets: product differentiation in pure competition,\" Journal of Political Economy, vol. 82, no. 1, pp. 34–55, 1974.",
        "[2] P. M. Anglin and R. Gençay, \"Semiparametric estimation of a hedonic price function,\" Journal of Applied Econometrics, vol. 11, no. 6, pp. 633–648, Nov. 1996.",
        "[3] C. Kleiber and A. Zeileis, Applied Econometrics with R. New York: Springer-Verlag, 2008. [Online]. Disponible: https://CRAN.R-project.org/package=AER",
        "[4] M. Yasser H., \"Housing Prices Dataset,\" Kaggle, 2021. [Online]. Disponible: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset",
        "[5] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed. New York: Springer, 2009.",
        "[6] A. Kraskov, H. Stögbauer, and P. Grassberger, \"Estimating mutual information,\" Physical Review E, vol. 69, no. 6, p. 066138, Jun. 2004.",
        "[7] W. S. Cleveland, \"Robust locally weighted regression and smoothing scatterplots,\" Journal of the American Statistical Association, vol. 74, no. 368, pp. 829–836, 1979.",
        "[8] D. C. Montgomery, E. A. Peck, and G. G. Vining, Introduction to Linear Regression Analysis, 5th ed. Hoboken, NJ: John Wiley & Sons, 2012.",
        "[9] F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "[10] V. Arel-Bundock, \"Rdatasets: A collection of datasets originally distributed in various R packages,\" 2022. [Online]. Disponible: https://vincentarelbundock.github.io/Rdatasets/",
    ]

    for ref in referencias:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(1)
        p_ref.paragraph_format.line_spacing = 1.10
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p_ref.add_run(ref)
        r.font.name = "Calibri"
        r.font.size = Pt(7.5)
        r.font.color.rgb = RGB_TEXTO_OSCURO

    doc.save(str(RUTA_DOCX))
    print(f"Documento Word guardado exitosamente en: {RUTA_DOCX}")
    return RUTA_DOCX


def convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> Path:
    """Convierte el documento .docx a .pdf usando Microsoft Word vía PowerShell COM."""
    ps_command = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    try {{
        $doc = $word.Documents.Open('{docx_path.resolve()}')
        $doc.ExportAsFixedFormat('{pdf_path.resolve()}', 17) # 17 = wdExportFormatPDF
        $doc.Close($false)
        Write-Output "Conversion a PDF completada exitosamente."
    }} finally {{
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }}
    """
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_command]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(res.stdout)
    if not pdf_path.is_file():
        raise RuntimeError(f"No se generó el archivo PDF en {pdf_path}")
    print(f"Documento PDF generado exitosamente en: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    docx_file = build_full_dataset_document()
    pdf_file = convert_docx_to_pdf(docx_file, RUTA_PDF)
    print("Tarjeta T08: Artefactos generados exitosamente.")
