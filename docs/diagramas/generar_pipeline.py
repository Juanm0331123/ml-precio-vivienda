"""T11 · Genera el diagrama del pipeline experimental (drawio editable y PNG) desde una sola especificación.

Desde la raíz del repositorio (requiere matplotlib):

    python docs/diagramas/generar_pipeline.py

Salidas:
  - docs/diagramas/pipeline.drawio   editable en diagrams.net (app.diagrams.net → Archivo → Abrir)
  - docs/diagramas/pipeline.png      16:9, 3840 × 2160 px (200 dpi sobre 19,2 × 10,8 pulgadas)

Coordenadas en unidades de un lienzo de 1920 × 1080. El test sellado nunca se conecta a la validación.
"""
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyBboxPatch        # noqa: E402

CARPETA = Path(__file__).resolve().parent
AZUL_MARINO, AZUL, LADRILLO, GRIS = "#0F2A47", "#2B6CC4", "#B5532F", "#5A6B7E"

# kind -> (relleno, borde, discontinuo)
ESTILOS = {
    "datos": ("#E8F0FB", AZUL, False),
    "prep": ("#EEF1F6", GRIS, False),
    "train": ("#E3F4EA", "#2F855A", False),
    "sellado": ("#FBE9E4", LADRILLO, True),
    "final": ("#FFFFFF", LADRILLO, True),
    "p1": ("#E8F0FB", AZUL, False),
    "p2": ("#D6E4F7", AZUL, False),
    "p3": ("#C2D7F2", AZUL, False),
    "etapa": ("#F5F6F9", AZUL_MARINO, False),
    "modelo": ("#FFFFFF", GRIS, False),
    "valid": ("#FFF4D6", "#B7791F", False),
    "contenedor": ("none", AZUL_MARINO, True),
}

# id, tipo, x, y, w, h, título, líneas, tamaño del título, tamaño del texto
N = [
    ("datos", "datos", 40, 110, 330, 220, "DATASET ORIGINAL", ["Housing.csv", "545 filas × 13 columnas", "precio + 12 atributos", "Windsor (Canadá), 1987", "vía Kaggle · T05"], 24, 20),
    ("calidad", "prep", 420, 110, 340, 220, "CALIDAD Y PREPARACIÓN · T06", ["0 faltantes (completitud 100 %)", "binarias → 0/1", "furnishingstatus codificada", "sin escalado ni transformaciones", "(esas se ajustan en el pipeline)"], 19, 19),
    ("particion", "prep", 810, 150, 290, 140, "PARTICIÓN 80/20", ["estratificada por quintiles", "de precio · semilla 5007"], 24, 20),
    ("test", "sellado", 1160, 110, 330, 120, "TEST SELLADO · 109 filas", ["NO SE UTILIZA en validación,", "ajuste ni selección"], 20, 19),
    ("train", "train", 1160, 260, 330, 100, "ENTRENAMIENTO · 436 filas", ["único dato que usa el modelado"], 20, 19),
    ("final", "final", 1560, 110, 320, 145, "EVALUACIÓN FINAL · T25", ["una sola vez, con la", "configuración elegida", "solo por validación"], 19, 19),
    ("contenedor", "contenedor", 30, 470, 1860, 560, "", [], 0, 0),
    ("prepro", "etapa", 60, 540, 330, 460, "PREPROCESAMIENTO", [], 24, 0),
    ("p1", "p1", 80, 590, 290, 120, "P1 · BASE", ["codificación +", "estandarización"], 19, 19),
    ("p2", "p2", 80, 725, 290, 120, "P2 · TRANSFORMACIONES", ["log(precio) · Yeo-Johnson", "escalado robusto"], 18, 18),
    ("p3", "p3", 80, 860, 290, 125, "P3 · INGENIERÍA Y SELECCIÓN", ["derivadas · interacciones", "winsorización · selección"], 16.5, 18),
    ("modelos", "etapa", 430, 540, 250, 460, "6 MODELOS", [], 24, 0),
    ("valid", "etapa", 720, 540, 300, 460, "VALIDACIÓN CRUZADA", [], 20, 0),
    ("v1", "valid", 740, 590, 260, 130, "PILOTO · T12", ["repetida 5 × 3", "hiperparámetros fijos"], 21, 19),
    ("v2", "valid", 740, 735, 260, 150, "FINAL · T19–T22", ["anidada: el bucle interno", "ajusta hiperparámetros"], 21, 19),
    ("v3", "modelo", 740, 900, 260, 85, "mismos pliegues", ["para los 18 pipelines"], 19, 18),
    ("metr", "etapa", 1060, 540, 280, 460, "MÉTRICAS Y\nCOMPARACIÓN", [], 22, 0),
    ("interp", "etapa", 1380, 540, 260, 460, "INTERPRETABILIDAD", [], 19, 0),
    ("res", "etapa", 1680, 540, 180, 460, "RESULTADOS Y\nCONCLUSIONES", [], 18, 0),
]
# Seis cajas de modelos dentro de la columna "6 MODELOS" (alto 58, separación 10)
MODELOS = [("m1", "M1 · OLS hedónico"), ("m2", "M2 · Ridge"), ("m3", "M3 · KNN"), ("m4", "M4 · SVR (RBF)"),
           ("m5", "M5 · Random Forest"), ("m6", "M6 · XGBoost")]
for k, (i, t) in enumerate(MODELOS):
    N.append((i, "modelo", 445, 592 + k * 68, 220, 58, t, [], 17, 0))

# Textos de las etapas de la derecha (lista de líneas dentro de la caja)
TEXTOS_ETAPA = {
    "metr": ["RMSE (principal)", "MAE · MAPE · R²", "", "prueba t corregida", "(Nadeau–Bengio) + Holm", "", "complejidad: tiempo,", "memoria, tamaño"],
    "interp": ["coeficientes (M1, M2)", "permutación", "SHAP (árboles)", "PDP / ICE", "", "asociación,", "no causalidad"],
    "res": ["T12: piloto", "T25: prueba", "sellada"],
}

# Aristas: origen, destino, estilo (continua / sellada), puntos intermedios
A = [
    ("datos", "calidad", "n", []),
    ("calidad", "particion", "n", []),
    ("particion", "test", "s", []),
    ("particion", "train", "n", []),
    ("test", "final", "s", []),
    ("train", "prepro", "n", [(1325, 430), (225, 430)]),
    ("prepro", "modelos", "n", []),
    ("modelos", "valid", "n", []),
    ("valid", "metr", "n", []),
    ("metr", "interp", "n", []),
    ("interp", "res", "n", []),
]


def nodo(i):
    return next(n for n in N if n[0] == i)


def centro(i):
    _, _, x, y, w, h = nodo(i)[:6]
    return x + w / 2, y + h / 2


def borde(i, hacia):
    """Punto del borde del nodo i en dirección al punto (hx, hy)."""
    _, _, x, y, w, h = nodo(i)[:6]
    hx, hy = hacia
    cx, cy = x + w / 2, y + h / 2
    if abs(hx - cx) / (w / 2) >= abs(hy - cy) / (h / 2):
        return (x + w, cy) if hx > cx else (x, cy)
    return (cx, y + h) if hy > cy else (cx, y)


def aristas_geom():
    """Devuelve las aristas como lista de puntos (origen, intermedios, destino)."""
    out = []
    for o, d, est, pts in A:
        po = list(pts)
        if po:
            ini = borde(o, po[0])
            fin = borde(d, po[-1])
            ruta = [ini] + po + [fin]
            if o == "train":                      # entra por arriba de PREPROCESAMIENTO
                ruta[-1] = (225, nodo(d)[3])
        else:
            ini, fin = borde(o, centro(d)), borde(d, centro(o))
            ruta = [ini, fin]
        out.append((ruta, est))
    return out


# ----------------------------------------------------------------------------- PNG
def dibujar_png(ruta):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920); ax.set_ylim(1080, 0); ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.text(40, 45, "Pipeline experimental de regresión: predicción del precio de vivienda (Windsor, 1987)",
            fontsize=21, fontweight="bold", color=AZUL_MARINO, va="center")
    ax.text(40, 82, "T11 · 3 preprocesamientos × 6 modelos = 18 pipelines · todo lo ajustable se ajusta solo con el pliegue de entrenamiento",
            fontsize=12.5, color=GRIS, va="center")
    for i, tipo, x, y, w, h, titulo, lineas, ts, tl in N:
        rel, bor, disc = ESTILOS[tipo]
        parche = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14", linewidth=2.4 if tipo != "contenedor" else 2.6,
                                edgecolor=bor, facecolor=rel, linestyle=(0, (6, 4)) if disc else "-", zorder=1 if tipo == "contenedor" else 2)
        ax.add_patch(parche)
        if tipo == "contenedor":
            ax.text(x + w / 2, y + 30, "PIPELINE DE SCIKIT-LEARN · se ajusta solo con el pliegue de entrenamiento (sin fuga de información)",
                    fontsize=14, fontweight="bold", color=AZUL_MARINO, ha="center", va="center")
            continue
        color_t = LADRILLO if tipo in ("sellado", "final") else AZUL_MARINO
        lineas_all = list(lineas) + TEXTOS_ETAPA.get(i, [])
        es_etapa = tipo == "etapa"
        if es_etapa:
            ax.text(x + w / 2, y + (34 if "\n" not in titulo else 48), titulo, fontsize=ts * 0.72, fontweight="bold", color=AZUL_MARINO, ha="center", va="center", linespacing=1.15)
            if i in TEXTOS_ETAPA:
                for k, l in enumerate(lineas_all):
                    ax.text(x + w / 2, y + 135 + k * 42, l, fontsize=20 * 0.72, color="#1F2933", ha="center", va="center")
        elif lineas:
            ax.text(x + w / 2, y + 30, titulo, fontsize=ts * 0.72, fontweight="bold", color=color_t, ha="center", va="center")
            for k, l in enumerate(lineas):
                ax.text(x + w / 2, y + 30 + (k + 1) * (tl * 1.5) + 6, l, fontsize=tl * 0.72, color="#1F2933", ha="center", va="center")
        else:
            ax.text(x + w / 2, y + h / 2, titulo, fontsize=ts * 0.72, fontweight="bold", color=color_t, ha="center", va="center")
    for ruta_, est in aristas_geom():
        color = LADRILLO if est == "s" else AZUL_MARINO
        xs, ys = zip(*ruta_)
        ax.plot(xs[:-1], ys[:-1], color=color, lw=3, ls=(0, (5, 3)) if est == "s" else "-", zorder=3, solid_capstyle="butt")
        ax.annotate("", xy=ruta_[-1], xytext=ruta_[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=3, mutation_scale=26,
                                                                       ls=(0, (5, 3)) if est == "s" else "-", shrinkA=0, shrinkB=0), zorder=4)
    # Leyenda
    leyenda = [("datos", "Datos originales"), ("prep", "Preparación T06"), ("train", "Entrenamiento"), ("sellado", "Test sellado"), ("p2", "Preprocesamiento"), ("valid", "Validación")]
    x0 = 330
    for tipo, txt in leyenda:
        rel, bor, disc = ESTILOS[tipo]
        ax.add_patch(FancyBboxPatch((x0, 1048), 34, 22, boxstyle="round,pad=0,rounding_size=5", edgecolor=bor, facecolor=rel, lw=2,
                                    linestyle=(0, (4, 3)) if disc else "-"))
        ax.text(x0 + 46, 1059, txt, fontsize=12, va="center", color="#1F2933")
        x0 += 46 + 12 * 11 + 28
    fig.savefig(ruta, dpi=200, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- drawio
def html(titulo, lineas, ts, tl, color):
    partes = []
    if titulo:
        partes.append(f'<b style="font-size:{ts}px;color:{color}">{escape(titulo).replace(chr(10), "<br>")}</b>')
    partes += [f'<span style="font-size:{tl}px">{escape(l)}</span>' if l else "&nbsp;" for l in lineas]
    return "<br>".join(partes)


def dibujar_drawio(ruta):
    celdas = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    celdas.append('<mxCell id="titulo" value="%s" style="text;html=1;align=left;verticalAlign=middle;fontSize=28;fontStyle=1;fontColor=%s;" vertex="1" parent="1">'
                  '<mxGeometry x="40" y="20" width="1840" height="50" as="geometry"/></mxCell>'
                  % (escape("Pipeline experimental de regresión: predicción del precio de vivienda (Windsor, 1987)"), AZUL_MARINO))
    celdas.append('<mxCell id="subtitulo" value="%s" style="text;html=1;align=left;verticalAlign=middle;fontSize=17;fontColor=%s;" vertex="1" parent="1">'
                  '<mxGeometry x="40" y="64" width="1840" height="36" as="geometry"/></mxCell>'
                  % (escape("T11 · 3 preprocesamientos × 6 modelos = 18 pipelines · todo lo ajustable se ajusta solo con el pliegue de entrenamiento"), GRIS))
    for i, tipo, x, y, w, h, titulo, lineas, ts, tl in N:
        rel, bor, disc = ESTILOS[tipo]
        base = f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={rel};strokeColor={bor};strokeWidth=2;" + ("dashed=1;" if disc else "")
        if tipo == "contenedor":
            valor = html("PIPELINE DE SCIKIT-LEARN · se ajusta solo con el pliegue de entrenamiento (sin fuga de información)", [], 19, 0, AZUL_MARINO)
            estilo = base + "verticalAlign=top;align=center;spacingTop=10;fillColor=none;"
        else:
            color_t = LADRILLO if tipo in ("sellado", "final") else AZUL_MARINO
            extra = TEXTOS_ETAPA.get(i, [])
            valor = html(titulo, list(lineas) + extra, ts or 20, tl or 18, color_t)
            estilo = base + ("verticalAlign=top;spacingTop=12;" if tipo == "etapa" else "verticalAlign=middle;") + "align=center;fontColor=#1F2933;"
        celdas.append(f'<mxCell id="{i}" value="{escape(valor, {chr(34): "&quot;"})}" style="{estilo}" vertex="1" parent="1">'
                      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for k, (ruta_, est) in enumerate(aristas_geom()):
        o, d = A[k][0], A[k][1]
        color = LADRILLO if est == "s" else AZUL_MARINO
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in ruta_[1:-1])
        arr = f'<Array as="points">{pts}</Array>' if pts else ""
        estilo = (f"endArrow=block;endFill=1;html=1;strokeWidth=3;strokeColor={color};" + ("dashed=1;" if est == "s" else "") +
                  "edgeStyle=orthogonalEdgeStyle;rounded=0;")
        celdas.append(f'<mxCell id="e{k}" style="{estilo}" edge="1" parent="1" source="{o}" target="{d}">'
                      f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
    xml = ('<mxfile host="app.diagrams.net"><diagram id="pipeline" name="Pipeline T11">'
           '<mxGraphModel dx="1920" dy="1080" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           'pageWidth="1920" pageHeight="1080" math="0" shadow="0"><root>' + "".join(celdas) + "</root></mxGraphModel></diagram></mxfile>")
    ruta.write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    dibujar_png(CARPETA / "pipeline.png")
    dibujar_drawio(CARPETA / "pipeline.drawio")
    print("Diagrama generado:", CARPETA / "pipeline.png", CARPETA / "pipeline.drawio")
