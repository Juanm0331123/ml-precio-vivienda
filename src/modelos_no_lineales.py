"""Modelos no lineales: K-Nearest Neighbors (KNN) y Support Vector Regression (SVR).

Tarjeta T20 · Modelos KNN y SVR (kernel RBF) [Brandon].
Responsable: Brandon Eduardo Tobar Sánchez (Model Developer).
Revisa el PR: Mauricio José Mozo Terán (Data Engineer).

Este módulo implementa:
1. Definición y configuración de estimadores KNN y SVR con kernel RBF.
2. Rejillas de búsqueda y adaptación de nombres de parámetros para pipelines directos (P1)
   y con objetivo transformado (P2, P3 y SVR en P1 vía TransformedTargetRegressor).
3. Búsqueda exhaustiva con validación cruzada interna (GridSearchCV) con KFold(n_splits=5, shuffle=True, random_state=5007).
4. Experimento pareado sobre los pliegues de validación cruzada 5×3 de T12 (RepeatedStratifiedKFold).
5. Evaluación del impacto crítico del escalado de variables (con vs sin escalado en P1).
6. Generación de tablas y figuras de desempeño e interpretabilidad técnica.

Referencias:
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.
- Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. Statistics and Computing, 14(3), 199-222.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import SEMILLA
from src.evaluacion import FACTOR_KAGGLE, calcular_metricas
from src.modelos import _log_directo, construir_pipeline
from src.preprocesamiento import construir_preprocesamiento

# -----------------------------------------------------------------------------
# Rejillas de hiperparámetros para búsqueda interna
# -----------------------------------------------------------------------------

REJILLA_KNN: Dict[str, list] = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2],  # 1: Manhattan (L1), 2: Euclidiana (L2)
}

REJILLA_SVR: Dict[str, list] = {
    "C": [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0],
    "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
}


def obtener_prefijo_modelo(modelo_id: str, prepro_id: str) -> str:
    """Determina el prefijo necesario para acceder a los hiperparámetros del estimador.
    
    En P1 con KNN, el pipeline es directo: Pipeline([('prepro', ...), ('modelo', KNN)]).
    El prefijo es 'modelo__'.
    
    En P2 y P3 (y en P1 para SVR), el estimador está envuelto en TransformedTargetRegressor:
    TransformedTargetRegressor(regressor=Pipeline([('prepro', ...), ('modelo', ...)])).
    El prefijo es 'regressor__modelo__'.
    """
    if prepro_id == "P1" and modelo_id != "M4":
        return "modelo__"
    return "regressor__modelo__"


def obtener_grilla_no_lineal(modelo_id: str, prepro_id: str) -> Dict[str, list]:
    """Genera la rejilla de parámetros de GridSearchCV con los prefijos correctos."""
    prefijo = obtener_prefijo_modelo(modelo_id, prepro_id)
    base_grid = REJILLA_KNN if modelo_id == "M3" else REJILLA_SVR
    return {f"{prefijo}{param}": valores for param, valores in base_grid.items()}


def crear_pipeline_knn_svr(
    modelo_id: str,
    prepro_id: str,
    escalar_predictoras: bool = True,
    hiperparametros: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construye un pipeline para KNN o SVR, permitiendo controlar el escalado y parámetros.
    
    Parameters
    ----------
    modelo_id : str
        'M3' para KNN o 'M4' para SVR.
    prepro_id : str
        'P1', 'P2' o 'P3'.
    escalar_predictoras : bool, default=True
        Si False, desactiva el escalado en el preprocesamiento para demostrar
        el impacto de la falta de normalización en algoritmos basados en distancias/kernels.
    hiperparametros : dict, optional
        Diccionario de hiperparámetros para inicializar el estimador base.
    """
    params = hiperparametros.copy() if hiperparametros else {}
    if modelo_id == "M3":
        if "n_jobs" not in params:
            params["n_jobs"] = 1
        estimador_base = KNeighborsRegressor(**params)
    elif modelo_id == "M4":
        if "kernel" not in params:
            params["kernel"] = "rbf"
        estimador_base = SVR(**params)
    else:
        raise ValueError(f"modelo_id debe ser 'M3' o 'M4', recibido: {modelo_id}")

    prepro = construir_preprocesamiento(prepro_id, escalar=escalar_predictoras)
    interno = Pipeline([("prepro", prepro), ("modelo", estimador_base)])

    if prepro_id == "P1":
        if modelo_id == "M4":
            # SVR en escala Kaggle (~10^6) colapsa numéricamente sin estandarizar el objetivo
            return TransformedTargetRegressor(regressor=interno, transformer=StandardScaler())
        return interno

    if modelo_id == "M4":
        # SVR en P2/P3: log(price) + StandardScaler
        transformer = Pipeline([("log", _log_directo()), ("std", StandardScaler())])
        return TransformedTargetRegressor(regressor=interno, transformer=transformer)

    # KNN en P2/P3: modelado en log(price) con inversión exp
    return TransformedTargetRegressor(
        regressor=interno,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def extraer_estimador_interno(pipeline_ajustado: Any) -> Tuple[Any, Any]:
    """Extrae la etapa de preprocesamiento y el modelo base ya ajustados."""
    if hasattr(pipeline_ajustado, "regressor_"):
        interno = pipeline_ajustado.regressor_
    elif hasattr(pipeline_ajustado, "regressor"):
        interno = pipeline_ajustado.regressor
    else:
        interno = pipeline_ajustado

    prepro = interno.named_steps["prepro"]
    modelo = interno.named_steps["modelo"]
    return prepro, modelo


def optimizar_hiperparametros(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modelo_id: str,
    prepro_id: str,
    cv_splits: int = 5,
    semilla: int = SEMILLA,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Ejecuta GridSearchCV sobre el conjunto de ajuste dado y devuelve:
    (mejor_pipeline, mejores_hiperparametros, df_resultados_cv).
    """
    pipeline = crear_pipeline_knn_svr(modelo_id, prepro_id, escalar_predictoras=True)
    grilla = obtener_grilla_no_lineal(modelo_id, prepro_id)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=semilla)
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=grilla,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )

    busqueda.fit(X_train, y_train)

    prefijo = obtener_prefijo_modelo(modelo_id, prepro_id)
    mejores_params = {
        k.replace(prefijo, ""): v for k, v in busqueda.best_params_.items()
    }

    df_cv = pd.DataFrame(busqueda.cv_results_)
    # Limpiar nombres de columnas de parámetros para facilitar análisis
    renombrar = {c: c.replace(f"param_{prefijo}", "param_") for c in df_cv.columns if c.startswith("param_")}
    df_cv = df_cv.rename(columns=renombrar)
    df_cv["modelo"] = modelo_id
    df_cv["preprocesamiento"] = prepro_id

    return busqueda.best_estimator_, mejores_params, df_cv


def evaluar_experimento_cv5x3(
    X: pd.DataFrame,
    y: pd.Series,
    pliegues: List[Tuple[np.ndarray, np.ndarray]],
    mejores_hiperparametros: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa los modelos KNN y SVR con sus mejores hiperparámetros encontrados
    en la validación cruzada repetida 5×3 de T12, garantizando comparación pareada.
    """
    filas_pliegues = []
    
    for (prepro_id, modelo_id), params in mejores_hiperparametros.items():
        for k, (ajuste, validacion) in enumerate(pliegues):
            pipeline = crear_pipeline_knn_svr(
                modelo_id=modelo_id,
                prepro_id=prepro_id,
                escalar_predictoras=True,
                hiperparametros=params,
            )
            
            X_ajuste, y_ajuste = X.iloc[ajuste], y.iloc[ajuste]
            X_val, y_val = X.iloc[validacion], y.iloc[validacion]
            
            pipeline.fit(X_ajuste, y_ajuste)
            y_pred_val = pipeline.predict(X_val)
            y_pred_ajuste = pipeline.predict(X_ajuste)
            
            metricas_val = calcular_metricas(y_val, y_pred_val)
            metricas_ajuste = calcular_metricas(y_ajuste, y_pred_ajuste)
            
            fila = {
                "preprocesamiento": prepro_id,
                "modelo": modelo_id,
                "modelo_nombre": "KNN" if modelo_id == "M3" else "SVR (RBF)",
                "pliegue": k,
                "repeticion": k // 5,
                "rmse": metricas_val["rmse"],
                "mae": metricas_val["mae"],
                "mape": metricas_val["mape"],
                "r2": metricas_val["r2"],
                "rmse_cad": metricas_val["rmse"] / FACTOR_KAGGLE,
                "mae_cad": metricas_val["mae"] / FACTOR_KAGGLE,
                "rmse_ajuste": metricas_ajuste["rmse"],
                "rmse_ajuste_cad": metricas_ajuste["rmse"] / FACTOR_KAGGLE,
            }
            filas_pliegues.append(fila)

    df_pliegues = pd.DataFrame(filas_pliegues)
    
    # Resumen agregado
    g = df_pliegues.groupby(["preprocesamiento", "modelo", "modelo_nombre"])
    df_resumen = pd.DataFrame({
        "n_pliegues": g.size(),
        "rmse_media": g["rmse"].mean(),
        "rmse_std": g["rmse"].std(ddof=1),
        "rmse_cad_media": g["rmse_cad"].mean(),
        "rmse_cad_std": g["rmse_cad"].std(ddof=1),
        "mae_cad_media": g["mae_cad"].mean(),
        "mae_cad_std": g["mae_cad"].std(ddof=1),
        "mape_media": g["mape"].mean(),
        "mape_std": g["mape"].std(ddof=1),
        "r2_media": g["r2"].mean(),
        "r2_std": g["r2"].std(ddof=1),
        "rmse_ajuste_cad_media": g["rmse_ajuste_cad"].mean(),
    }).reset_index().sort_values("rmse_cad_media").reset_index(drop=True)
    
    return df_resumen, df_pliegues


def experimento_impacto_escalado(
    X: pd.DataFrame,
    y: pd.Series,
    pliegues: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Evalúa el desempeño de KNN y SVR con y sin escalado bajo P1.
    
    Esto responde directamente al criterio de aceptación de T20:
    'Justificar por qué estos modelos requieren escalado y comparar su desempeño'.
    """
    filas = []
    configuraciones = [
        ("M3", "KNN", True),
        ("M3", "KNN", False),
        ("M4", "SVR (RBF)", True),
        ("M4", "SVR (RBF)", False),
    ]

    for modelo_id, nombre, con_escalado in configuraciones:
        for k, (ajuste, validacion) in enumerate(pliegues):
            pipeline = crear_pipeline_knn_svr(
                modelo_id=modelo_id,
                prepro_id="P1",
                escalar_predictoras=con_escalado,
            )
            X_ajuste, y_ajuste = X.iloc[ajuste], y.iloc[ajuste]
            X_val, y_val = X.iloc[validacion], y.iloc[validacion]
            
            pipeline.fit(X_ajuste, y_ajuste)
            pred = pipeline.predict(X_val)
            metricas = calcular_metricas(y_val, pred)
            
            filas.append({
                "modelo": modelo_id,
                "nombre": nombre,
                "escalado_predictoras": "Con escalado" if con_escalado else "Sin escalado",
                "pliegue": k,
                "rmse_cad": metricas["rmse"] / FACTOR_KAGGLE,
                "mae_cad": metricas["mae"] / FACTOR_KAGGLE,
                "mape": metricas["mape"],
                "r2": metricas["r2"],
            })
            
    df = pd.DataFrame(filas)
    resumen = df.groupby(["nombre", "escalado_predictoras"]).agg(
        rmse_cad_media=("rmse_cad", "mean"),
        rmse_cad_std=("rmse_cad", lambda s: s.std(ddof=1)),
        mae_cad_media=("mae_cad", "mean"),
        mae_cad_std=("mae_cad", lambda s: s.std(ddof=1)),
        mape_media=("mape", "mean"),
        mape_std=("mape", lambda s: s.std(ddof=1)),
        r2_media=("r2", "mean"),
        r2_std=("r2", lambda s: s.std(ddof=1)),
    ).reset_index()
    
    return resumen
