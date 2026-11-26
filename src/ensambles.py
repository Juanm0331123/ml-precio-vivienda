"""Modelos de ensamble de árboles: Random Forest y XGBoost.

Tarjeta T21 · Ensambles: Random Forest y XGBoost/LightGBM [Brandon].
Responsable: Brandon Eduardo Tobar Sánchez (Model Developer).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).

Este módulo implementa:
1. Configuración de estimadores Random Forest (Bagging) y XGBoost (Gradient Boosting).
2. Adaptación de prefijos de hiperparámetros para pipelines directos (P1) y con objetivo
   transformado (P2, P3 vía TransformedTargetRegressor).
3. Búsqueda amplia de hiperparámetros con RandomizedSearchCV (CV=5, n_iter=30, random_state=5007),
   atendiendo el criterio A1 del docente (búsqueda más amplia para candidatos principales).
4. Evaluación pareada sobre los 15 pliegues de validación cruzada 5×3 de T12 (RepeatedStratifiedKFold).
5. Extracción y análisis de importancia de variables (MDI en Random Forest, Gain y Weight en XGBoost).
6. Comparación sistemática de preprocesamientos (P1 vs P2 vs P3) en modelos basados en particiones.

Referencias:
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16, 785-794.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import SEMILLA
from src.evaluacion import FACTOR_KAGGLE, calcular_metricas
from src.preprocesamiento import (
    COLUMNAS_DERIVADAS,
    COLUMNAS_PREDICTORAS,
    construir_preprocesamiento,
)

# -----------------------------------------------------------------------------
# Rejillas / Distribuciones de hiperparámetros para búsqueda aleatoria amplia
# -----------------------------------------------------------------------------

DISTRIBUCION_RF: Dict[str, list] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 6, 8, 10, 14],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
}

DISTRIBUCION_XGB: Dict[str, list] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [0.5, 1.0, 3.0, 5.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}


def obtener_prefijo_ensamble(prepro_id: str) -> str:
    """En P1 el pipeline es directo: 'modelo__'. En P2/P3 es 'regressor__modelo__'."""
    return "modelo__" if prepro_id == "P1" else "regressor__modelo__"


def obtener_distribucion_ensamble(modelo_id: str, prepro_id: str) -> Dict[str, list]:
    """Genera la distribución de búsqueda con los prefijos correctos."""
    prefijo = obtener_prefijo_ensamble(prepro_id)
    base_dist = DISTRIBUCION_RF if modelo_id == "M5" else DISTRIBUCION_XGB
    return {f"{prefijo}{param}": valores for param, valores in base_dist.items()}


def crear_pipeline_ensamble(
    modelo_id: str,
    prepro_id: str,
    hiperparametros: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construye un pipeline para Random Forest o XGBoost sobre P1, P2 o P3.
    
    Parameters
    ----------
    modelo_id : str
        'M5' para Random Forest o 'M6' para XGBoost.
    prepro_id : str
        'P1', 'P2' o 'P3'.
    hiperparametros : dict, optional
        Hiperparámetros específicos para el estimador de ensamble.
    """
    params = hiperparametros.copy() if hiperparametros else {}
    if modelo_id == "M5":
        if "random_state" not in params:
            params["random_state"] = SEMILLA
        if "n_jobs" not in params:
            params["n_jobs"] = 1
        estimador_base = RandomForestRegressor(**params)
    elif modelo_id == "M6":
        if "random_state" not in params:
            params["random_state"] = SEMILLA
        if "n_jobs" not in params:
            params["n_jobs"] = 1
        if "tree_method" not in params:
            params["tree_method"] = "hist"
        estimador_base = XGBRegressor(**params)
    else:
        raise ValueError(f"modelo_id debe ser 'M5' o 'M6', recibido: {modelo_id}")

    # Árboles no requieren escalado de predictoras (escalar=False)
    prepro = construir_preprocesamiento(prepro_id, escalar=False)
    interno = Pipeline([("prepro", prepro), ("modelo", estimador_base)])

    if prepro_id == "P1":
        return interno

    # P2 y P3 modelan log(price) con reconversión exacta exp(y)
    return TransformedTargetRegressor(
        regressor=interno,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def extraer_modelo_base(pipeline_ajustado: Any) -> Tuple[Any, Any]:
    """Extrae el preprocesador y estimador de ensamble ya ajustados."""
    if hasattr(pipeline_ajustado, "regressor_"):
        interno = pipeline_ajustado.regressor_
    elif hasattr(pipeline_ajustado, "regressor"):
        interno = pipeline_ajustado.regressor
    else:
        interno = pipeline_ajustado

    prepro = interno.named_steps["prepro"]
    modelo = interno.named_steps["modelo"]
    return prepro, modelo


def optimizar_ensamble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modelo_id: str,
    prepro_id: str,
    n_iter: int = 30,
    cv_splits: int = 5,
    semilla: int = SEMILLA,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Búsqueda amplia aleatoria (RandomizedSearchCV) con KFold(cv_splits, shuffle=True)."""
    pipeline = crear_pipeline_ensamble(modelo_id, prepro_id)
    distribucion = obtener_distribucion_ensamble(modelo_id, prepro_id)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=semilla)
    busqueda = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=distribucion,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=semilla,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )

    busqueda.fit(X_train, y_train)

    prefijo = obtener_prefijo_ensamble(prepro_id)
    mejores_params = {
        k.replace(prefijo, ""): v for k, v in busqueda.best_params_.items()
    }

    df_cv = pd.DataFrame(busqueda.cv_results_)
    renombrar = {c: c.replace(f"param_{prefijo}", "param_") for c in df_cv.columns if c.startswith("param_")}
    df_cv = df_cv.rename(columns=renombrar)
    df_cv["modelo"] = modelo_id
    df_cv["preprocesamiento"] = prepro_id

    return busqueda.best_estimator_, mejores_params, df_cv


def evaluar_ensambles_cv5x3(
    X: pd.DataFrame,
    y: pd.Series,
    pliegues: List[Tuple[np.ndarray, np.ndarray]],
    mejores_hiperparametros: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa Random Forest y XGBoost con sus mejores parámetros en CV 5×3 de T12."""
    filas_pliegues = []

    for (prepro_id, modelo_id), params in mejores_hiperparametros.items():
        for k, (ajuste, validacion) in enumerate(pliegues):
            pipeline = crear_pipeline_ensamble(
                modelo_id=modelo_id,
                prepro_id=prepro_id,
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
                "modelo_nombre": "Random Forest" if modelo_id == "M5" else "XGBoost",
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


def obtener_nombres_caracteristicas_prepro(prepro_etapa: Any, X_sample: pd.DataFrame) -> List[str]:
    """Obtiene los nombres exactos de las columnas transformadas por ColumnTransformer."""
    transformer = prepro_etapa.named_steps["transformador"]
    nombres = []
    for nombre_trans, trans, cols in transformer.transformers_:
        if nombre_trans == "remainder" or trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            nombres.extend(trans.get_feature_names_out(cols))
        elif isinstance(cols, list):
            nombres.extend(cols)
        elif isinstance(cols, str):
            nombres.append(cols)
    return [str(n).replace("onehot__", "").replace("remainder__", "").replace("num__", "") for n in nombres]


def extraer_importancia_caracteristicas(
    pipeline_ajustado: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    prepro_id: str,
) -> pd.DataFrame:
    """Extrae la importancia relativa de variables del modelo ajustado en entrenamiento."""
    prepro, modelo = extraer_modelo_base(pipeline_ajustado)
    
    # Obtener nombres de columnas
    if prepro_id in ["P1", "P2"]:
        columnas = [
            "area", "bedrooms", "bathrooms", "stories", "parking",
            "mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "prefarea",
            "furnishing_unfurnished", "furnishing_semi_furnished", "furnishing_furnished"
        ]
    else:
        # P3 incluye derivadas y selección
        columnas = [f"var_{i}" for i in range(len(modelo.feature_importances_))]

    importancias = modelo.feature_importances_
    # Si las dimensiones coinciden exactamente con columnas
    if len(importancias) == len(columnas):
        df = pd.DataFrame({"variable": columnas, "importancia": importancias})
    else:
        df = pd.DataFrame({
            "variable": [f"feature_{i}" for i in range(len(importancias))],
            "importancia": importancias,
        })

    return df.sort_values("importancia", ascending=False).reset_index(drop=True)
