"""Búsqueda de hiperparámetros con validación cruzada interna (utilidades compartidas de T19, T20 y T21).

**Alcance.** Estas funciones sirven para el *desarrollo y ajuste* de los modelos sobre las filas de
entrenamiento. El mejor puntaje de una búsqueda es **optimista**: se eligió el máximo entre muchas
configuraciones con los mismos pliegues con que se mide. La estimación de error sin ese sesgo es la
validación anidada de T22. Nada aquí toca el conjunto de prueba sellado: ``X`` e ``y`` deben venir de
``src.evaluacion.cargar_entrenamiento``.

Convenciones:

- Los pliegues son una lista fija de ``(indices_ajuste, indices_validacion)``: los mismos para todos los
  modelos y preprocesamientos (``src.evaluacion.crear_pliegues``: validación repetida 5 × 3, semilla 5007).
- Las métricas se calculan en la escala original de ``price``: el estimador devuelve precios, y los
  transformadores del objetivo se invierten dentro de ``TransformedTargetRegressor``.
- Toda transformación aprendida vive dentro del ``Pipeline`` que se pasa a la búsqueda, de modo que se
  ajusta solo con el pliegue de ajuste de cada candidato (sin fuga de información).
"""

import json
import platform
import time
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import SEMILLA

# Puntajes de scikit-learn sobre la escala original de price. MAPE sale como fracción; se pasa a % al resumir.
PUNTAJES = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "r2": "r2",
}


def entorno(extra: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Versiones, equipo y semilla (los tiempos dependen del hardware; los resultados, de las versiones)."""
    import os

    datos = {
        "python": platform.python_version(), "sistema": f"{platform.system()} {platform.release()}",
        "procesador": platform.processor() or "no disponible", "nucleos_logicos": os.cpu_count(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "semilla": SEMILLA,
    }
    for nombre in ("xgboost", "lightgbm", "statsmodels", "shap"):
        try:
            datos[nombre] = __import__(nombre).__version__
        except Exception:  # biblioteca opcional no instalada
            pass
    datos.update(extra or {})
    return datos


def ejecutar_busqueda(estimador, espacio: dict, X: pd.DataFrame, y: pd.Series, pliegues, tipo: str = "grid",
                      n_iter: Optional[int] = None, n_jobs: int = 1, semilla: int = SEMILLA):
    """Ejecuta una búsqueda en rejilla (``grid``) o aleatoria (``random``) con los pliegues dados.

    El criterio de selección (``refit``) es el RMSE de validación. Con ``n_jobs=1`` se capturan los avisos de
    convergencia; con más procesos, los avisos se pierden en los trabajadores y se cuentan como ``None``.

    Returns
    -------
    dict
        ``busqueda`` (objeto de scikit-learn ya reajustado con el mejor candidato sobre todo ``X``),
        ``duracion_s``, ``n_candidatos`` y ``avisos_convergencia``.
    """
    comunes = dict(scoring=PUNTAJES, refit="rmse", cv=pliegues, n_jobs=n_jobs, error_score="raise", return_train_score=False)
    if tipo == "grid":
        busqueda = GridSearchCV(estimador, espacio, **comunes)
    elif tipo == "random":
        if n_iter is None:
            raise ValueError("La búsqueda aleatoria requiere n_iter (presupuesto explícito)")
        busqueda = RandomizedSearchCV(estimador, espacio, n_iter=n_iter, random_state=semilla, **comunes)
    else:
        raise ValueError(f"Tipo de búsqueda desconocido: {tipo}")
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        busqueda.fit(X, y)
    duracion = time.perf_counter() - t0
    n_conv = sum(issubclass(a.category, ConvergenceWarning) for a in avisos) if n_jobs == 1 else None
    return {"busqueda": busqueda, "duracion_s": duracion, "n_candidatos": len(busqueda.cv_results_["params"]), "avisos_convergencia": n_conv}


def tabla_candidatos(busqueda, preprocesamiento: str, modelo: str, prefijo: str = "regressor__modelo__") -> pd.DataFrame:
    """Todos los candidatos probados con su desempeño medio de validación cruzada (RMSE, MAE, MAPE %, R²).

    Una fila por candidato: se conservan los hiperparámetros probados, no solo el mejor.
    """
    r = busqueda.cv_results_
    c = _correccion_ddof(r)
    filas = []
    for i, params in enumerate(r["params"]):
        filas.append({
            "preprocesamiento": preprocesamiento, "modelo": modelo,
            "hiperparametros": json.dumps({k.replace(prefijo, ""): _a_nativo(v) for k, v in params.items()}, ensure_ascii=False, sort_keys=True),
            "rmse_media": -r["mean_test_rmse"][i], "rmse_desv": c * r["std_test_rmse"][i],
            "mae_media": -r["mean_test_mae"][i], "mape_media": -100 * r["mean_test_mape"][i],
            "r2_media": r["mean_test_r2"][i], "tiempo_ajuste_s_media": r["mean_fit_time"][i],
            "rango_rmse": int(r["rank_test_rmse"][i]),
        })
    return pd.DataFrame(filas)


def fila_mejor(busqueda, preprocesamiento: str, modelo: str, duracion_s: float, n_candidatos: int,
               prefijo: str = "regressor__modelo__") -> dict:
    """Resumen del mejor candidato: hiperparámetros, métricas medias y desviaciones entre pliegues."""
    r, i = busqueda.cv_results_, busqueda.best_index_
    c = _correccion_ddof(r)
    return {
        "preprocesamiento": preprocesamiento, "modelo": modelo,
        "hiperparametros_seleccionados": json.dumps({k.replace(prefijo, ""): _a_nativo(v) for k, v in busqueda.best_params_.items()}, ensure_ascii=False, sort_keys=True),
        "n_pliegues": sum(1 for k in r if k.startswith("split") and k.endswith("_test_rmse")),
        "rmse_media": -r["mean_test_rmse"][i], "rmse_desv": c * r["std_test_rmse"][i],
        "mae_media": -r["mean_test_mae"][i], "mae_desv": c * r["std_test_mae"][i],
        "mape_media": -100 * r["mean_test_mape"][i], "mape_desv": 100 * c * r["std_test_mape"][i],
        "r2_media": r["mean_test_r2"][i], "r2_desv": c * r["std_test_r2"][i],
        "n_candidatos": n_candidatos, "duracion_busqueda_s": duracion_s,
    }


def _correccion_ddof(cv_results) -> float:
    """Factor ``sqrt(n/(n-1))`` que lleva la desviación de scikit-learn (ddof=0) a la muestral (ddof=1), como en el piloto T12."""
    n = sum(1 for k in cv_results if k.startswith("split") and k.endswith("_test_rmse"))
    return float(np.sqrt(n / (n - 1)))


def _a_nativo(v):
    """Convierte tipos de numpy a tipos nativos de Python para serializar a JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
