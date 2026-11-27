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

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler

from src.config import SEMILLA
from src.datos import MAPA_AMOBLADO, MAPA_BINARIO

# Grupos de columnas de housing_limpio.csv (nombres reales de T06).
COLUMNA_AREA = "area"
COLUMNAS_NUMERICAS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
COLUMNAS_CONTEOS = ["bedrooms", "bathrooms", "stories", "parking"]  # enteros pequeños (sin Yeo-Johnson)
COLUMNAS_BINARIAS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
COLUMNA_AMOBLADO = "furnishingstatus"
CODIGOS_AMOBLADO = [0, 1, 2]  # unfurnished, semi-furnished, furnished (src.datos.MAPA_AMOBLADO)
COLUMNAS_PREDICTORAS = COLUMNAS_NUMERICAS + COLUMNAS_BINARIAS + [COLUMNA_AMOBLADO]

# Variables derivadas de P3 alineadas con T18 (fórmulas fijas por fila: no usan estadísticos de la muestra).
COLUMNAS_DERIVADAS = ["habitaciones_totales", "banos_por_habitacion", "n_comodidades", "log_area"]
COMODIDADES = ["guestroom", "basement", "hotwaterheating", "airconditioning"]

# En T12 (piloto) se usó k=10 con MI; tras T18, P3 conserva todas las variables por defecto (k=None).
K_SELECCION_PILOTO = 10
PERCENTILES_WINSORIZACION = (1, 99)


class CodificadorBase(BaseEstimator, TransformerMixin):
    """Codifica variables binarias (``yes``/``no`` -> 1/0) y ``furnishingstatus`` (a 0/1/2) de forma idempotente.

    Si las columnas ya vienen codificadas como enteros desde ``housing_limpio.csv`` (T06), las conserva
    intactas. Si llegan en formato crudo (``yes``/``no`` y cadenas de amoblado), aplica los mapeos fijos
    ``MAPA_BINARIO`` y ``MAPA_AMOBLADO``. Al ser mapeos deterministas por fila, no aprende estadísticos
    ni introduce fuga de información.
    """

    def fit(self, X, y=None):
        self.columnas_entrada_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for col in COLUMNAS_BINARIAS:
            if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
                mapeado = X[col].map(MAPA_BINARIO)
                if mapeado.isna().any():
                    raise ValueError(f"Valores inesperados en columna binaria '{col}'")
                X[col] = mapeado.astype(int)
        if COLUMNA_AMOBLADO in X.columns and not pd.api.types.is_numeric_dtype(X[COLUMNA_AMOBLADO]):
            mapeado = X[COLUMNA_AMOBLADO].map(MAPA_AMOBLADO)
            if mapeado.isna().any():
                raise ValueError(f"Valores inesperados en '{COLUMNA_AMOBLADO}'")
            X[COLUMNA_AMOBLADO] = mapeado.astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        nombres = input_features if input_features is not None else getattr(self, "columnas_entrada_", None)
        return np.asarray(nombres, dtype=object)


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
    """Agrega las 4 variables derivadas de P3 alineadas con T18 como columnas nuevas (sin quitar las originales).

    - ``habitaciones_totales = bedrooms + bathrooms``: capacidad funcional total de la vivienda.
    - ``banos_por_habitacion = bathrooms / bedrooms``: densidad sanitaria (``bedrooms >= 1`` según T06).
    - ``n_comodidades``: suma de guestroom, basement, hotwaterheating y airconditioning (0 a 4).
    - ``log_area = np.log(area)``: logaritmo natural del tamaño del lote.

    Son fórmulas por fila: no aprenden nada, así que no pueden filtrar información entre pliegues.
    """

    def fit(self, X, y=None):
        self.columnas_entrada_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        if (X["bedrooms"] <= 0).any():
            raise ValueError("bedrooms debe ser >= 1 para calcular banos_por_habitacion")
        if (X["area"] <= 0).any():
            raise ValueError("area debe ser > 0 para calcular log_area")
        X["habitaciones_totales"] = X["bedrooms"] + X["bathrooms"]
        X["banos_por_habitacion"] = X["bathrooms"] / X["bedrooms"]
        X["n_comodidades"] = X[COMODIDADES].sum(axis=1)
        X["log_area"] = np.log(X["area"].astype(float))
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.columnas_entrada_)
        return np.asarray(base + COLUMNAS_DERIVADAS, dtype=object)


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


def construir_preprocesamiento(p: str, escalar: bool = True, k: Optional[int] = None) -> Pipeline:
    """Devuelve el preprocesamiento P1, P2 o P3 como ``Pipeline`` (sin el modelo).

    P3 incluye, en orden: derivadas T18 -> transformaciones (winsorización y escalado) y, solo si se
    indica ``k``, selección por información mutua (``sel``). Tras T18, por defecto ``k=None`` (sin
    recorte fijo por información mutua). Cada paso se ajusta únicamente con las filas que recibe ``fit``.

    Parameters
    ----------
    p : {"P1", "P2", "P3"}
    escalar : bool
        Si el modelo que sigue es sensible a la escala de las predictoras.
    k : int or None, default=None
        Si es entero, aplica ``SelectKBest(mutual_info_regression, k=k)`` al final de P3.
    """
    if p not in ("P1", "P2", "P3"):
        raise ValueError(f"Preprocesamiento desconocido: {p}")
    pasos: List = [("codificar", CodificadorBase())]
    if p == "P3":
        pasos.append(("derivar", DerivadorVariables()))
    pasos.append(("columnas", _transformador_columnas(p, escalar)))
    if p == "P3" and k is not None:
        pasos.append(("sel", SelectKBest(score_func=_informacion_mutua, k=k)))
    return Pipeline(pasos)


def envolver_objetivo(estimador, p: str):
    """Envuelve un estimador/pipeline con ``TransformedTargetRegressor`` (log/exp) para P2 y P3.

    En P1 devuelve el estimador sin transformar el objetivo (escala original). En P2 y P3 aplica
    ``np.log`` al ajustar y ``np.exp`` al predecir, garantizando comparabilidad en la escala original.
    """
    if p not in ("P1", "P2", "P3"):
        raise ValueError(f"Preprocesamiento desconocido: {p}")
    if p == "P1":
        return estimador
    return TransformedTargetRegressor(
        regressor=estimador,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def nombres_seleccionados(prepro: Pipeline) -> List[str]:
    """Nombres de las columnas que conserva P3 (con o sin paso ``sel``)."""
    if "sel" not in prepro.named_steps:
        if "derivar" in prepro.named_steps:
            return list(prepro.get_feature_names_out())
        return []
    nombres = Pipeline(prepro.steps[:-1]).get_feature_names_out()
    return list(np.asarray(nombres)[prepro.named_steps["sel"].get_support()])


DESCRIPCION = {
    "P1": "Base: one-hot de furnishingstatus, StandardScaler en Ridge/KNN/SVR, objetivo en escala original",
    "P2": "P1 + log(price), Yeo-Johnson en area, RobustScaler en conteos (Ridge/KNN/SVR)",
    "P3": "P2 + derivadas T18 (habitaciones_totales, banos_por_habitacion, n_comodidades, log_area), winsorización 1-99 % (+ StandardScaler en derivadas), sin recorte fijo MI",
}
