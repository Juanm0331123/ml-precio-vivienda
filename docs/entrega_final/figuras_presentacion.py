"""Figuras de las diapositivas de respaldo de la presentación final (T30).

Desde la raíz del repositorio: python docs/entrega_final/figuras_presentacion.py
Salidas: reports/figures/deck_respaldo_variables.png (variables renombradas, T05) y deck_respaldo_hiperparametros.png (espacios de búsqueda de T19 a T21).
"""
from pathlib import Path

import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
AZUL_MARINO, GRIS = "#0F2A47", "#5A6B7E"

# Mismos valores que el código de T19 (src/modelos_lineales.py), T20 (src/modelos_no_lineales.py) y T21 (src/ensambles.py)
FILAS = [
    ("OLS hedónico", "sin hiperparámetros", "—"),
    ("Ridge", "alpha: 25 valores entre 10⁻³ y 10³", "rejilla, 25"),
    ("Lasso", "alpha: 31 valores entre 10⁻⁶ y 1", "rejilla, 31"),
    ("ElasticNet", "alpha: 20 valores entre 10⁻⁴ y 1 · l1_ratio: 9 valores entre 0,01 y 0,99", "rejilla, 180"),
    ("KNN", "k: 3 a 21 (7 valores) · pesos uniformes o por distancia · Manhattan o euclidiana", "rejilla, 28"),
    ("SVR (RBF)", "C: 0,1 a 500 · gamma: scale, auto, 0,001 a 0,1 · epsilon: 0,01 a 0,2", "rejilla, 120"),
    ("Random Forest", "árboles 100–500 · profundidad 6–14 o libre · mínimos por hoja y división · variables por división", "aleatoria, 30 de 900"),
    ("XGBoost", "árboles 100–500 · profundidad 3–6 · tasa 0,01–0,15 · submuestreo · regularización L1 y L2", "aleatoria, 30 de 25 920"),
]


# Lo que verificó T05 al comparar Kaggle con el archivo original de Anglin y Gençay (1996)
VARIABLES = [
    ("price", "precio de venta en dólares canadienses de 1987, multiplicado por 70"),
    ("area", "tamaño del lote en pies², no el área construida"),
    ("bathrooms · stories", "baños completos · pisos sin contar el sótano"),
    ("mainroad", "tiene entrada vehicular (driveway), no que dé a una vía principal"),
    ("guestroom", "tiene sala de recreación"),
    ("basement", "tiene sótano completo terminado"),
    ("hotwaterheating", "usa gas para calentar el agua"),
    ("airconditioning · parking", "aire acondicionado central · plazas de garaje"),
    ("prefarea", "está en el barrio preferido de la ciudad"),
    ("furnishingstatus", "no existe en el original: la agregó Kaggle sin documentar cómo"),
]


def _tabla(filas, encabezado, anchos, salida):
    """Tabla con el estilo del deck (encabezado azul oscuro y filas alternas)."""
    fig, eje = plt.subplots(figsize=(11.6, 4.55), dpi=200)
    eje.axis("off")
    tabla = eje.table(cellText=[list(f) for f in filas], colLabels=encabezado, cellLoc="left", colLoc="left",
                      colWidths=anchos, loc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10.5)
    tabla.scale(1, 1.9 if len(filas) < 10 else 1.62)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor("#E1E5EC")
        celda.PAD = 0.03
        if fila == 0:
            celda.set_facecolor(AZUL_MARINO)
            celda.get_text().set_color("white")
            celda.get_text().set_fontweight("bold")
        else:
            celda.set_facecolor("#FFFFFF" if fila % 2 else "#F5F6F9")
            celda.get_text().set_color("#142B45")
            if col == 0:
                celda.get_text().set_fontweight("bold")
    fig.tight_layout(pad=0.2)
    fig.savefig(salida, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return salida


def tabla_variables():
    return _tabla(VARIABLES, ["Nombre en Kaggle", "Qué mide en realidad (archivo original)"], [0.25, 0.75],
                  RAIZ / "reports/figures/deck_respaldo_variables.png")


def tabla_hiperparametros():
    fig, eje = plt.subplots(figsize=(11.6, 4.55), dpi=200)
    eje.axis("off")
    encabezado = ["Modelo", "Qué se busca", "Búsqueda y candidatos"]
    tabla = eje.table(cellText=[list(f) for f in FILAS], colLabels=encabezado, cellLoc="left", colLoc="left",
                      colWidths=[0.14, 0.66, 0.20], loc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10.5)
    tabla.scale(1, 1.9)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor("#E1E5EC")
        celda.PAD = 0.03
        if fila == 0:
            celda.set_facecolor(AZUL_MARINO)
            celda.get_text().set_color("white")
            celda.get_text().set_fontweight("bold")
        else:
            celda.set_facecolor("#FFFFFF" if fila % 2 else "#F5F6F9")
            celda.get_text().set_color("#142B45")
            if col == 0:
                celda.get_text().set_fontweight("bold")
    fig.tight_layout(pad=0.2)
    salida = RAIZ / "reports/figures/deck_respaldo_hiperparametros.png"
    fig.savefig(salida, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return salida


if __name__ == "__main__":
    print(tabla_variables())
    print(tabla_hiperparametros())
