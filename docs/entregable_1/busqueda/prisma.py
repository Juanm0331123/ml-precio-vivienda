"""Diagrama de flujo PRISMA 2020 a partir de la hoja PRISMA de registro_busquedas.xlsx."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from openpyxl import load_workbook

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
xlsx = REPO / "docs" / "entregable_1" / "registro_busquedas.xlsx"
salida = REPO / "docs" / "diagramas" / "prisma_2020.png"
v = {r[0].strip(): r[1] for r in load_workbook(xlsx, data_only=True)["PRISMA"].iter_rows(min_row=2, values_only=True)}
g = lambda k: v[k]
mot = {k.split(":")[0]: val for k, val in v.items() if k[:1] == "E" and k[1].isdigit()}

plt.rcParams["font.family"] = "IBM Plex Sans"
fig, ax = plt.subplots(figsize=(13, 11.5), dpi=200)
ax.set_xlim(0, 130); ax.set_ylim(0, 115); ax.axis("off")
TINTA, SEC, BORDE, AZUL, FONDO = "#0b0b0b", "#52514e", "#c3c2b7", "#2a78d6", "#fcfcfb"
fig.patch.set_facecolor("white")

def caja(x, y, w, h, texto, negrita=False, fondo="white", tam=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                linewidth=1.1, edgecolor=BORDE, facecolor=fondo))
    ax.text(x + w/2, y + h/2, texto, ha="center", va="center", fontsize=tam, color=TINTA,
            fontweight="bold" if negrita else "normal", linespacing=1.35, wrap=True)

def flecha(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=SEC, lw=1.2, shrinkA=0, shrinkB=0))

def fase(y, h, texto):
    ax.add_patch(FancyBboxPatch((1, y), 5, h, boxstyle="round,pad=0.2,rounding_size=1", linewidth=0, facecolor=AZUL))
    ax.text(3.5, y + h/2, texto, rotation=90, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

ax.text(65, 112, "Diagrama de flujo PRISMA 2020 · Revisión de literatura (búsqueda del 4 de noviembre de 2026)",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color=TINTA)
ax.text(40, 106.5, "Identificación en bases de datos", ha="center", fontsize=10.5, fontweight="bold", color=AZUL)
ax.text(111, 106.5, "Identificación por otros métodos", ha="center", fontsize=10.5, fontweight="bold", color=AZUL)

# Identificación
fase(83, 21, "Identificación")
caja(9, 86, 46, 16, f"Registros identificados (n = {g('Registros identificados en bases de datos')})\n"
     f"{g('Búsquedas ejecutadas en bases de datos')} búsquedas con filtro de dominio en 11 fuentes:\nIEEE Xplore, ScienceDirect, SpringerLink, MDPI,\nTaylor & Francis, Emerald, arXiv, SciELO, Redalyc,\nWiley y RePEc")
caja(62, 88, 30, 12, f"Registros eliminados antes\ndel tamizaje: duplicados\n(n = {g('Duplicados eliminados antes del tamizaje')})")
caja(98, 89, 28, 10, f"Registros identificados por\nbúsqueda por citas (n = {g('Registros identificados por búsqueda por citas')})")
flecha(55, 94, 62, 94)

# Tamizaje
fase(28, 53, "Tamizaje")
caja(9, 68, 46, 10, f"Registros tamizados por título\ny resumen (n = {g('Registros tamizados (título y resumen)')})")
motivos_txt = "\n".join([
    f"E1 fuera del tema: {mot['E1']}", f"E2 arriendo o alquiler: {mot['E2']}",
    f"E3 no académico o sin título: {mot['E3']}", f"E4 datos no comparables: {mot['E4']}",
    f"E5 preprint sin revisión por pares: {mot['E5']}", f"E6 explicativo, sin predicción ni ML: {mot['E6']}",
    f"E7 no residencial o índices: {mot['E7']}"])
caja(62, 60, 30, 26, f"Registros excluidos (n = {g('Registros excluidos en el tamizaje')})\n\n" + motivos_txt, tam=8.4)
flecha(32, 86, 32, 78); flecha(55, 73, 62, 73)
caja(9, 48, 46, 9, f"Reportes buscados para recuperación\n(n = {g('Reportes buscados para recuperación (bases de datos)')})")
caja(62, 47.5, 30, 10, f"Reportes no recuperados (n = {g('Reportes no recuperados (bases de datos)')})\nsin acceso al resumen o metadatos\nno verificables", tam=8.6)
flecha(32, 68, 32, 57); flecha(55, 52.5, 62, 52.5)
caja(9, 31, 46, 9, f"Reportes evaluados para elegibilidad\npor resumen y metadatos (n = {g('Reportes evaluados para elegibilidad (bases de datos)')})")
caja(62, 31, 30, 9, f"Reportes excluidos\n(n = {g('Reportes excluidos en elegibilidad (bases de datos)')})")
flecha(32, 48, 32, 40); flecha(55, 35.5, 62, 35.5)
# Otros métodos
caja(98, 48, 28, 13, f"Reportes buscados para\nrecuperación (n = {g('Reportes buscados (búsqueda por citas)')})\n\nNo recuperados (n = {g('Reportes no recuperados (búsqueda por citas)')})")
caja(98, 31, 28, 9, f"Evaluados por resumen\ny metadatos (n = {g('Reportes evaluados (búsqueda por citas)')})")
flecha(112, 89, 112, 61); flecha(112, 48, 112, 40)

# Incluidos
fase(5, 20, "Incluidos")
caja(9, 8, 117, 13, f"Estudios incluidos en la revisión (n = {g('Total de estudios incluidos en la revisión')})\n"
     f"{g('Estudios incluidos desde bases de datos')} desde bases de datos + {g('Estudios incluidos desde búsqueda por citas')} por búsqueda por citas",
     negrita=True, fondo="#eef4fc", tam=10.5)
flecha(32, 31, 32, 21); flecha(112, 31, 112, 21)

ax.text(129, 1.5, "Fuente: docs/entregable_1/registro_busquedas.xlsx (hoja PRISMA). Formato: Page et al., PRISMA 2020.",
        ha="right", fontsize=7.8, color=SEC)
fig.savefig(salida, dpi=200, bbox_inches="tight", facecolor="white")
print("Escrito", salida)
