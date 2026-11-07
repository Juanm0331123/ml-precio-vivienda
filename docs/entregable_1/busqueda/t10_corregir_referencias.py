"""T10 · Correcciones de referencias de T09 con evidencia bibliográfica (Crossref, 5 nov. 2026).

Solo se corrigen datos con evidencia directa en la API de Crossref (https://api.crossref.org/works/<doi>):

  [7]  E07 Park y Bae: el DOI 10.1016/j.eswa.2015.03.005 corresponde a la fe de erratas (vol. 42, no. 19, p. 6806).
       El artículo es 10.1016/j.eswa.2014.11.040 (vol. 42, no. 6, pp. 2928–2934, 2015).
  [8]  E08 Kok et al.: se agrega doi 10.3905/jpm.2017.43.6.202 (título, vol. 43, no. 6, pp. 202–211 coinciden).
  [18] E18 Mora-Garcia et al.: se agrega doi 10.3390/land11112100 (vol. 11, no. 11, art. 2100 coinciden).
  [22] E22 Kim y Hong: se agrega doi 10.3390/land13091436 (vol. 13, no. 9, art. 1436 coinciden).
  [23] E23 Deng y Zhang: Crossref da vol. 74, no. 1, art. 32 (2025), sin páginas; "pp. 1–28" se reemplaza por "art. 32".

Se aplican a: incluidos.py, 02_protocolo_busqueda.md, la hoja Incluidos de registro_busquedas.xlsx y las
referencias de Revision_Literatura.docx. Es idempotente. Los identificadores E01–E26 no cambian.
Desde la raíz del repositorio: python docs/entregable_1/busqueda/t10_corregir_referencias.py
"""
from pathlib import Path

import docx
import openpyxl

AQUI = Path(__file__).resolve().parent
CARPETA = AQUI.parent

DOI_E07_VIEJO, DOI_E07 = "10.1016/j.eswa.2015.03.005", "10.1016/j.eswa.2014.11.040"
# id -> (url que T09 usaba como identificador, doi que se agrega)
AGREGAR_DOI = {
    "E08": ("https://cris.maastrichtuniversity.nl/en/publications/big-data-in-real-estate-from-manual-appraisal-to-automated-valuat/", "10.3905/jpm.2017.43.6.202"),
    "E18": ("https://www.mdpi.com/2073-445X/11/11/2100", "10.3390/land11112100"),
    "E22": ("https://www.mdpi.com/2073-445X/13/9/1436", "10.3390/land13091436"),
}


def corregir_texto(t):
    """Aplica las correcciones a una línea o párrafo de referencia (texto plano o Markdown)."""
    t = t.replace(DOI_E07_VIEJO, DOI_E07)
    t = t.replace("pp. 1–28, 2025", "art. 32, 2025")
    return t


def agregar_doi_en_referencia(t, num):
    """Antepone 'doi: X.' a '[Online]' en las referencias [8], [18] y [22] (si aún no lo tienen)."""
    doi = {8: "10.3905/jpm.2017.43.6.202", 18: "10.3390/land11112100", 22: "10.3390/land13091436"}[num]
    if f"doi: {doi}" in t:
        return t
    return t.replace("[Online]", f"doi: {doi}. [Online]", 1)


def md():
    p = CARPETA / "02_protocolo_busqueda.md"
    nuevas = []
    for linea in p.read_text(encoding="utf-8").split("\n"):
        linea = corregir_texto(linea)
        for n in (8, 18, 22):
            if linea.startswith(f"[{n}] "):
                linea = agregar_doi_en_referencia(linea, n)
        nuevas.append(linea)
    p.write_text("\n".join(nuevas), encoding="utf-8", newline="")


def incluidos_py():
    p = AQUI / "incluidos.py"
    s = p.read_text(encoding="utf-8")
    s = s.replace(f'"doi:{DOI_E07_VIEJO}"', f'"doi:{DOI_E07}"')
    s = s.replace('"vol. 74, no. 1, pp. 1–28"', '"vol. 74, no. 1, art. 32"')
    for url, doi in AGREGAR_DOI.values():
        s = s.replace(f'"{url}"', f'"doi:{doi}"')
    p.write_text(s, encoding="utf-8", newline="")


def xlsx():
    p = CARPETA / "registro_busquedas.xlsx"
    wb = openpyxl.load_workbook(p)
    ws = wb["Incluidos"]
    for fila in ws.iter_rows(min_row=2):
        idv, vol, ident = fila[0].value, fila[5], fila[6]
        if ident.value == f"doi:{DOI_E07_VIEJO}":
            ident.value = f"doi:{DOI_E07}"
        if idv == "E23" and vol.value and "pp. 1–28" in vol.value:
            vol.value = vol.value.replace("pp. 1–28", "art. 32")
        if idv in AGREGAR_DOI and ident.value == AGREGAR_DOI[idv][0]:
            ident.value = f"doi:{AGREGAR_DOI[idv][1]}"
    wb.save(p)


def word():
    p = CARPETA / "Revision_Literatura.docx"
    d = docx.Document(p)
    for par in d.paragraphs:
        if not par.text.startswith("["):
            continue
        num = int(par.text[1:par.text.index("]")])
        for r in par.runs:
            r.text = corregir_texto(r.text)
            if num in (8, 18, 22) and "[Online]" in r.text:
                r.text = agregar_doi_en_referencia(r.text, num)
    d.save(p)


if __name__ == "__main__":
    md()
    incluidos_py()
    xlsx()
    word()
    print("Referencias corregidas: [7] DOI, [8][18][22] DOI agregado, [23] art. 32.")
