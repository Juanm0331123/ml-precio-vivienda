"""Modelos M1-M6 del piloto y armado de los 18 pipelines (Tarjeta T12).

Seis modelos de cuatro familias, definidos en ``docs/entregable_1/03_metodologia.md`` (T11, 3.5):
M1 OLS (línea base hedónica), M2 Ridge, M3 KNN, M4 SVR con kernel RBF, M5 Random Forest y M6 XGBoost.

**Hiperparámetros fijos.** El piloto no ajusta hiperparámetros (eso es de T19-T22 con validación
anidada). Se eligieron **antes de ejecutar el experimento**, tomando valores centrales de las rejillas
que propone T11, y no se modificaron después de ver resultados. Así el error de validación cruzada del
piloto no queda optimista por selección de hiperparámetros.

**Paralelismo.** Todos los modelos usan ``n_jobs=1`` y ``random_state=5007``: el piloto es pequeño (436
filas) y un solo hilo da tiempos comparables y resultados reproducibles, sin paralelismo anidado en
Windows ni Colab.
"""

from typing import Dict

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.config import SEMILLA
from src.preprocesamiento import construir_preprocesamiento

# id -> (nombre, familia, ¿sensible a la escala de las predictoras?)
MODELOS: Dict[str, tuple] = {
    "M1": ("OLS (hedónico)", "lineal", False),
    "M2": ("Ridge", "lineal", True),
    "M3": ("KNN", "distancia", True),
    "M4": ("SVR (RBF)", "kernel", True),
    "M5": ("Random Forest", "árboles", False),
    "M6": ("XGBoost", "árboles", False),
}

# Hiperparámetros fijos del piloto (valores centrales de las rejillas de T11, sección 3.5).
HIPERPARAMETROS: Dict[str, dict] = {
    "M1": {},
    "M2": {"alpha": 1.0},
    "M3": {"n_neighbors": 11, "weights": "uniform", "p": 2, "n_jobs": 1},
    "M4": {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"},
    "M5": {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 3, "max_features": 0.6, "n_jobs": 1, "random_state": SEMILLA},
    "M6": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.85, "colsample_bytree": 0.85,
           "reg_lambda": 3.0, "n_jobs": 1, "random_state": SEMILLA, "tree_method": "hist"},
}


def crear_modelo(modelo_id: str):
    """Crea el estimador de M1-M6 con los hiperparámetros fijos del piloto."""
    clases = {"M1": LinearRegression, "M2": Ridge, "M3": KNeighborsRegressor, "M4": SVR,
              "M5": RandomForestRegressor, "M6": XGBRegressor}
    return clases[modelo_id](**HIPERPARAMETROS[modelo_id])


def _log_directo():
    """Transformación log del objetivo con su inversa exp (sin parámetros que aprender)."""
    return FunctionTransformer(np.log, inverse_func=np.exp, check_inverse=False)


def construir_pipeline(prepro_id: str, modelo_id: str):
    """Arma el pipeline completo P×M, listo para ``fit``/``predict`` sobre las 12 predictoras.

    - P1 deja el objetivo en escala original. Solo SVR estandariza el objetivo dentro del pliegue,
      porque ``price`` es del orden de 10⁶ y los parámetros de SVR no están pensados para esa magnitud
      (necesidad numérica, no una estrategia de mejora; T11, 3.4).
    - P2 y P3 modelan ``log(price)`` con ``TransformedTargetRegressor``: las predicciones se devuelven
      ya en la escala original, así que las métricas de P1, P2 y P3 son comparables. Para SVR el
      log-precio además se estandariza (``log`` -> ``StandardScaler``), también dentro del pliegue.

    Returns
    -------
    sklearn.base.BaseEstimator
        Estimador con ``fit(X, y)`` y ``predict(X)``; ``predict`` devuelve precios en escala original.
    """
    escalar = MODELOS[modelo_id][2]
    interno = Pipeline([("prepro", construir_preprocesamiento(prepro_id, escalar)), ("modelo", crear_modelo(modelo_id))])
    if prepro_id == "P1":
        if modelo_id == "M4":
            return TransformedTargetRegressor(regressor=interno, transformer=StandardScaler())
        return interno
    if modelo_id == "M4":
        return TransformedTargetRegressor(regressor=interno, transformer=Pipeline([("log", _log_directo()), ("std", StandardScaler())]))
    return TransformedTargetRegressor(regressor=interno, func=np.log, inverse_func=np.exp, check_inverse=False)


def pipeline_interno(estimador) -> Pipeline:
    """Devuelve el ``Pipeline`` ajustado (prepro + modelo), con o sin ``TransformedTargetRegressor``."""
    return estimador.regressor_ if hasattr(estimador, "regressor_") else estimador
