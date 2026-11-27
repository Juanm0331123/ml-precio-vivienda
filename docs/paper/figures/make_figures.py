"""Figuras del paper (en inglés). Solo usan la partición de entrenamiento.

Desde la raíz del repositorio: python docs/paper/figures/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
AQUI = Path(__file__).resolve().parent
AZUL = "#1f4e79"
FACTOR_KAGGLE = 70  # price de Kaggle = CAD de 1987 x 70 (verificado en T05)

# Nombres del diccionario original de Windsor, que son los que usa el paper
NOMBRES = {
    "area": "Lot size", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms", "stories": "Stories",
    "mainroad": "Driveway", "guestroom": "Recreation room", "basement": "Finished basement",
    "hotwaterheating": "Gas water heating", "airconditioning": "Central air cond.",
    "parking": "Garage places", "prefarea": "Preferred area", "furnishingstatus": "Furnishing status",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300,
})


def entrenamiento():
    """Une el dataset limpio con la partición y deja solo entrenamiento (el test sigue sellado)."""
    datos = pd.read_csv(RAIZ / "data/processed/housing_limpio.csv")
    particion = pd.read_csv(RAIZ / "data/processed/particion.csv")
    df = datos.merge(particion, on="id")
    return df[df["conjunto"] == "entrenamiento"].drop(columns=["id", "conjunto"])


def fig_precio(df):
    """Distribución del precio en CAD de 1987 y de su logaritmo."""
    cad = df["price"] / FACTOR_KAGGLE / 1000
    fig, ejes = plt.subplots(1, 2, figsize=(3.45, 1.75))
    for eje, serie, etiqueta in [(ejes[0], cad, "Price (thousand 1987 CAD)"),
                                 (ejes[1], np.log(df["price"] / FACTOR_KAGGLE), "log(price)")]:
        eje.hist(serie, bins=22, color=AZUL, edgecolor="white", linewidth=0.4)
        eje.set_xlabel(etiqueta)
        eje.set_title(f"skewness = {serie.skew():.2f}", fontsize=7, loc="right", pad=2)
    ejes[0].set_ylabel("Houses")
    fig.tight_layout(pad=0.4)
    fig.savefig(AQUI / "fig_price_distribution.pdf")
    plt.close(fig)


def fig_correlacion(df):
    """Correlación de Spearman de cada atributo con el precio."""
    rho = df.corr(method="spearman")["price"].drop("price").sort_values()
    fig, eje = plt.subplots(figsize=(3.45, 2.3))
    eje.barh([NOMBRES[c] for c in rho.index], rho.values, color=AZUL, height=0.62)
    for y, v in enumerate(rho.values):
        eje.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7)
    eje.set_xlim(0, rho.max() + 0.09)
    eje.set_xlabel("Spearman correlation with price")
    eje.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(AQUI / "fig_correlation_price.pdf")
    plt.close(fig)
    return rho


def fig_ablacion():
    """Ablación de furnishingstatus (T18): diferencia de RMSE con su IC 95 % corregido, en CAD de 1987.

    Lee la tabla que deja el notebook 04; no vuelve a entrenar nada.
    """
    tabla = pd.read_csv(RAIZ / "reports/tables/t18_ablacion_furnishingstatus.csv")
    nombres = {"OLS (hedónico)": "OLS (hedonic)", "Ridge": "Ridge", "KNN": "KNN", "SVR (RBF)": "SVR",
               "Random Forest": "Random forest", "XGBoost": "XGBoost"}
    tabla = tabla.iloc[::-1]  # OLS arriba
    delta = tabla["delta_rmse_media"] / FACTOR_KAGGLE
    inf = tabla["ic95_inf"] / FACTOR_KAGGLE
    sup = tabla["ic95_sup"] / FACTOR_KAGGLE
    y = np.arange(len(tabla))
    fig, eje = plt.subplots(figsize=(3.45, 2.0))
    eje.axvline(0, color="0.55", linewidth=0.8, linestyle="--")
    eje.errorbar(delta, y, xerr=[delta - inf, sup - delta], fmt="o", color=AZUL, markersize=4,
                 elinewidth=1, capsize=2)
    eje.set_yticks(y, [nombres[n] for n in tabla["modelo_nombre"]])
    eje.set_xlabel(r"$\Delta$RMSE without $-$ with (CAD, 1987)")
    eje.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(AQUI / "fig_ablation_furnishing.pdf")
    plt.close(fig)


def fig_pipeline():
    """Esquema del experimento para el paper (ancho de dos columnas). Versión en inglés del diagrama de T11.

    Arriba va el flujo de los datos; abajo, dentro del recuadro punteado, lo que se ajusta solo con el
    pliegue de entrenamiento. El test sellado va aparte y solo se usa en la evaluación final.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ladrillo, gris = "#B5532F", "#5A6B7E"
    fig = plt.figure(figsize=(7.16, 2.55))
    eje = fig.add_axes([0, 0, 1, 1])
    eje.set_xlim(0, 716)
    eje.set_ylim(255, 0)
    eje.axis("off")

    def caja(x, y, w, h, titulo, lineas, relleno="#EEF1F6", borde=gris, punteada=False, color_t=AZUL):
        eje.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", linewidth=0.9,
                                     edgecolor=borde, facecolor=relleno, linestyle=(0, (3, 2)) if punteada else "-"))
        alto = 10.5
        y0 = y + h / 2 - alto * len(lineas) / 2
        eje.text(x + w / 2, y0 - 1, titulo, ha="center", va="center", fontsize=7, fontweight="bold", color=color_t)
        for k, linea in enumerate(lineas):
            eje.text(x + w / 2, y0 + alto * (k + 1) - 1, linea, ha="center", va="center", fontsize=6.3, color="#1F2933")

    def flecha(p, q, punteada=False, color=AZUL, conexion="arc3"):
        eje.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color=color,
                                      linestyle=(0, (3, 2)) if punteada else "-", connectionstyle=conexion,
                                      shrinkA=0, shrinkB=0))

    # Fila superior: datos -> calidad -> partición -> entrenamiento / test sellado -> evaluación final
    caja(8, 12, 118, 62, "Windsor data, 1987", ["545 sales, 12 attributes", "provenance verified", "against the original"], "#E8F0FB", AZUL)
    caja(146, 12, 118, 62, "Quality and cleaning", ["no missing values", "binary attributes to 0/1", "no fitted transforms here"])
    caja(284, 12, 118, 62, "Stratified 80/20 split", ["strata: price quintiles", "seed 5007"])
    caja(422, 6, 120, 34, "Sealed test set", ["109 sales, used once"], "#FBE9E4", ladrillo, True, ladrillo)
    caja(422, 46, 120, 34, "Training set", ["436 sales"], "#E3F4EA", "#2F855A", color_t="#2F855A")
    caja(590, 2, 118, 42, "Final evaluation", ["chosen model and the", "hedonic baseline"], "#FFFFFF", ladrillo, True, ladrillo)
    flecha((126, 43), (146, 43))
    flecha((264, 43), (284, 43))
    flecha((402, 36), (422, 23), punteada=True, color=ladrillo)
    flecha((402, 50), (422, 63))
    flecha((542, 23), (590, 23), punteada=True, color=ladrillo)

    # Recuadro inferior: todo lo que se ajusta dentro del pliegue de entrenamiento
    eje.add_patch(FancyBboxPatch((8, 100), 700, 147, boxstyle="round,pad=0,rounding_size=5", linewidth=0.9,
                                 edgecolor=AZUL, facecolor="none", linestyle=(0, (3, 2))))
    eje.text(358, 110, "scikit-learn pipelines: every fitted step uses only the training part of each fold",
             ha="center", va="center", fontsize=6.8, fontweight="bold", color=AZUL)
    flecha((482, 80), (482, 100))

    cajas = [
        (18, "Preprocessing", ["P1: encoding and", "standardization", "P2: + log price,", "Yeo-Johnson, robust scaling", "P3: + derived attributes,", "winsorization"]),
        (157, "Models", ["OLS (hedonic baseline)", "ridge, lasso, elastic net", "KNN", "SVR (RBF kernel)", "random forest", "XGBoost"]),
        (296, "Repeated 5×3 CV", ["same 15 folds for", "the 18 pipelines", "hyperparameters tuned", "in an inner loop", "(nested)", ""]),
        (435, "Comparison", ["RMSE, MAE, MAPE, R²", "on the price scale", "corrected t-test", "(Nadeau-Bengio)", "time and model size", ""]),
        (574, "Interpretation", ["hedonic coefficients", "SHAP values", "of the tree ensembles", "residuals by", "price segment", ""]),
    ]
    for x, titulo, lineas in cajas:
        caja(x, 122, 124, 116, titulo, [l for l in lineas if l], "#F5F6F9", AZUL)
    for x in (142, 281, 420, 559):
        flecha((x, 180), (x + 15, 180))

    fig.savefig(AQUI / "fig_pipeline.pdf")
    plt.close(fig)


if __name__ == "__main__":
    df = entrenamiento()
    fig_precio(df)
    print(fig_correlacion(df).round(2).to_string())
    fig_ablacion()
    fig_pipeline()
