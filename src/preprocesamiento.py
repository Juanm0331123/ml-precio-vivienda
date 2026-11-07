"""Estrategias de preprocesamiento P1, P2 y P3 del piloto (Tarjeta T12).

Implementan la definición de ``docs/entregable_1/03_metodologia.md`` (T11, sección 3.4). Los tres
preprocesamientos son **acumulativos**: P2 = P1 + transformaciones y P3 = P2 + ingeniería y selección.

Todo lo que se aprende de los datos (escaladores, Yeo-Johnson, percentiles de winsorización y la
selección de variables) vive dentro de un ``Pipeline`` de scikit-learn, de modo que se ajusta solo con
el pliegue de entrenamiento y se aplica sin cambios al de validación (sin fuga de información).

Entradas esperadas: las 12 predictoras de ``data/processed/housing_limpio.csv`` (T06), sin ``id`` ni
``price``. Las binarias ya vienen como 0/1 y ``furnishingstatus`` como entero 0/1/2 (ver
``src.datos.MAPA_AMOBLADO``). Aquí ``furnishingstatus`` se trata como **nominal**: T06 la dejó ordinal,
pero su orden no está documentado, así que se expande con *one-hot* sin modificar el archivo de T06.

Ejemplo::

    from src.preprocesamiento import construir_preprocesamiento
    prepro = construir_preprocesamiento("P2", escalar=True)
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler

from src.config import SEMILLA

# Grupos de columnas de housing_limpio.csv (nombres reales de T06).
COLUMNA_AREA = "area"
COLUMNAS_NUMERICAS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
COLUMNAS_CONTEOS = ["bedrooms", "bathrooms", "stories", "parking"]  # enteros pequeños (sin Yeo-Johnson)
COLUMNAS_BINARIAS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
COLUMNA_AMOBLADO = "furnishingstatus"
CODIGOS_AMOBLADO = [0, 1, 2]  # unfurnished, semi-furnished, furnished (src.datos.MAPA_AMOBLADO)
COLUMNAS_PREDICTORAS = COLUMNAS_NUMERICAS + COLUMNAS_BINARIAS + [COLUMNA_AMOBLADO]

# Variables derivadas de P3 (fórmulas fijas por fila: no usan estadísticos de la muestra).
COLUMNAS_DERIVADAS = ["area_por_dormitorio", "n_comodidades", "area_x_prefarea", "area_x_airconditioning"]
COMODIDADES = ["guestroom", "basement", "hotwaterheating", "airconditioning"]

# Selección de variables de P3 con k fijo en el piloto (T11: k ∈ {8, 10, 12, todas}; el ajuste de k es de T19–T22).
K_SELECCION_PILOTO = 10
PERCENTILES_WINSORIZACION = (1, 99)


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles ``inferior`` y ``superior`` aprendidos en ``fit``.

    Limita la influencia de los valores extremos sin eliminar filas. Los percentiles se calculan solo
    con los datos que recibe ``fit`` (el pliegue de entrenamiento) y se reutilizan en ``transform``,
    por lo que sirve también con datos nuevos.

    Parameters
    ----------
    inferior, superior : float
        Percentiles (0-100) de recorte.
    """

    def __init__(self, inferior: float = PERCENTILES_WINSORIZACION[0], superior: float = PERCENTILES_WINSORIZACION[1]):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.limite_inferior_ = np.percentile(X, self.inferior, axis=0)
        self.limite_superior_ = np.percentile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.limite_inferior_, self.limite_superior_)

    def get_feature_names_out(self, input_features=None):
        nombres = input_features if input_features is not None else getattr(self, "feature_names_in_", None)
        return np.asarray(nombres, dtype=object)


class DerivadorVariables(BaseEstimator, TransformerMixin):
    """Agrega las variables derivadas de P3 como columnas nuevas (sin quitar las originales).

    - ``area_por_dormitorio = area / bedrooms`` (``bedrooms >= 1`` según T06, no hay división por cero).
    - ``n_comodidades``: suma de guestroom, basement, hotwaterheating y airconditioning.
    - ``area_x_prefarea`` y ``area_x_airconditioning``: interacciones hedónicas de diseño (T11, 3.4).

    Son fórmulas por fila: no aprenden nada, así que no pueden filtrar información entre pliegues.
    """

    def fit(self, X, y=None):
        self.columnas_entrada_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        X["area_por_dormitorio"] = X["area"] / X["bedrooms"]
        X["n_comodidades"] = X[COMODIDADES].sum(axis=1)
        X["area_x_prefarea"] = X["area"] * X["prefarea"]
        X["area_x_airconditioning"] = X["area"] * X["airconditioning"]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.columnas_entrada_) + COLUMNAS_DERIVADAS, dtype=object)


def _codificador_amoblado() -> OneHotEncoder:
    """One-hot de furnishingstatus (códigos 0/1/2 de T06); ``drop='first'`` evita la colinealidad perfecta."""
    return OneHotEncoder(categories=[CODIGOS_AMOBLADO], drop="first", sparse_output=False)


def _transformador_columnas(p: str, escalar: bool) -> ColumnTransformer:
    """Arma el ColumnTransformer de P1, P2 o P3.

    Parameters
    ----------
    p : {"P1", "P2", "P3"}
    escalar : bool
        True para los modelos sensibles a la escala (Ridge, KNN, SVR). Los árboles y OLS no necesitan
        escalado: se dejan en su escala original (T11, 3.4).
    """
    if p == "P1":
        numericas = [("numericas", StandardScaler() if escalar else "passthrough", COLUMNAS_NUMERICAS)]
    else:
        escalador = RobustScaler() if escalar else "passthrough"
        if p == "P2":
            # Yeo-Johnson solo en area (cola derecha); los conteos se escalan sin transformar.
            numericas = [
                ("area", PowerTransformer(method="yeo-johnson"), [COLUMNA_AREA]),
                ("conteos", escalador, COLUMNAS_CONTEOS),
            ]
        else:
            # P3: area se winsoriza antes de Yeo-Johnson; las derivadas se winsorizan y escalan.
            # Las derivadas (sobre todo las interacciones con binarias) son casi todas 0: su rango intercuartílico
            # es 0 y RobustScaler no las escala (deja la escala original, de miles). Por eso se usa StandardScaler
            # (discrepancia local con T11, que no fijaba el escalador de las derivadas; ver notebook 03, sección 4).
            derivadas = [("winsor", Winsorizador())] + ([("escala", StandardScaler())] if escalar else [])
            numericas = [
                ("area", Pipeline([("winsor", Winsorizador()), ("yeo", PowerTransformer(method="yeo-johnson"))]), [COLUMNA_AREA]),
                ("conteos", escalador, COLUMNAS_CONTEOS),
                ("derivadas", Pipeline(derivadas), COLUMNAS_DERIVADAS),
            ]
    return ColumnTransformer(
        numericas + [("amoblado", _codificador_amoblado(), [COLUMNA_AMOBLADO])],
        remainder="passthrough",          # las 6 binarias (ya 0/1 desde T06) pasan sin cambio
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _informacion_mutua(X, y):
    """Información mutua con semilla fija (el estimador usa vecinos y aleatoriedad interna)."""
    return mutual_info_regression(X, y, random_state=SEMILLA)


def construir_preprocesamiento(p: str, escalar: bool) -> Pipeline:
    """Devuelve el preprocesamiento P1, P2 o P3 como ``Pipeline`` (sin el modelo).

    P3 incluye, en orden: derivadas -> transformaciones -> selección por información mutua (``sel``).
    Cada paso se ajusta únicamente con las filas que recibe ``fit``.

    Parameters
    ----------
    p : {"P1", "P2", "P3"}
    escalar : bool
        Si el modelo que sigue es sensible a la escala de las predictoras.
    """
    if p not in ("P1", "P2", "P3"):
        raise ValueError(f"Preprocesamiento desconocido: {p}")
    pasos: List = []
    if p == "P3":
        pasos.append(("derivar", DerivadorVariables()))
    pasos.append(("columnas", _transformador_columnas(p, escalar)))
    if p == "P3":
        pasos.append(("sel", SelectKBest(score_func=_informacion_mutua, k=K_SELECCION_PILOTO)))
    return Pipeline(pasos)


def nombres_seleccionados(prepro: Pipeline) -> List[str]:
    """Nombres de las columnas que conserva la selección de P3 (para dejar evidencia de la regla)."""
    if "sel" not in prepro.named_steps:
        return []
    nombres = Pipeline(prepro.steps[:-1]).get_feature_names_out()
    return list(np.asarray(nombres)[prepro.named_steps["sel"].get_support()])


DESCRIPCION = {
    "P1": "Base: one-hot de furnishingstatus, StandardScaler en Ridge/KNN/SVR, objetivo en escala original",
    "P2": "P1 + log(price), Yeo-Johnson en area, RobustScaler en conteos (Ridge/KNN/SVR)",
    "P3": "P2 + variables derivadas, 2 interacciones, winsorización 1-99 % (+ StandardScaler en derivadas) y selección por información mutua (k=10)",
}
