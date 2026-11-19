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


if __name__ == "__main__":
    df = entrenamiento()
    fig_precio(df)
    print(fig_correlacion(df).round(2).to_string())
