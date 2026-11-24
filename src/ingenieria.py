"""Ingeniería de variables, comparación de selección en CV y ablación de furnishingstatus (Tarjeta T18).

Implementa los tres bloques experimentales de T18 sobre las 436 filas de entrenamiento
(sin tocar el conjunto de prueba sellado de T06):

1. **Variables derivadas justificadas** (``DerivadorVariablesT18``):
   - ``habitaciones_totales = bedrooms + bathrooms``
   - ``banos_por_habitacion = bathrooms / bedrooms``
   - ``n_comodidades = guestroom + basement + hotwaterheating + airconditioning``
   - ``log_area = np.log(area)``

2. **Comparación de selección de variables dentro de la CV (5×3 = 15 pliegues)**:
   - Información Mutua (``SelectKBest(mutual_info_regression)``)
   - Lasso (``LassoCV``)
   - RFE (``RFE`` con ``Ridge``)
   Se reporta estabilidad por pliegue, importancia/rango y desempeño fuera de muestra (RMSE).

3. **Ablación de ``furnishingstatus`` (Hipótesis H3)**:
   - Entrenamiento pareado con y sin ``furnishingstatus`` en los mismos 15 pliegues.
   - Diferencia de RMSE (``sin - con``), desviación estándar por pliegue e intervalo/prueba t
     corregida de Nadeau & Bengio (2003) para CV repetida.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.linear_model import LassoCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SEMILLA
from src.evaluacion import N_ESTRATOS, N_REPETICIONES, N_SPLITS, calcular_metricas
from src.modelos import MODELOS, crear_modelo
from src.preprocesamiento import (
    CODIGOS_AMOBLADO,
    COLUMNA_AMOBLADO,
    COLUMNAS_BINARIAS,
    COLUMNAS_NUMERICAS,
    COMODIDADES,
    CodificadorBase,
)

COLUMNAS_DERIVADAS_T18: List[str] = [
    "habitaciones_totales",
    "banos_por_habitacion",
    "n_comodidades",
    "log_area",
]

K_SELECCION_T18 = 10


class DerivadorVariablesT18(BaseEstimator, TransformerMixin):
    """Agrega las 4 variables derivadas justificadas de T18 (fórmulas fijas por fila).

    - ``habitaciones_totales = bedrooms + bathrooms``: capacidad funcional total de la vivienda.
    - ``banos_por_habitacion = bathrooms / bedrooms``: densidad sanitaria / confort relativo
      (``bedrooms >= 1`` verificado en T06).
    - ``n_comodidades``: conteo de comodidades (``guestroom``, ``basement``, ``hotwaterheating``,
      ``airconditioning``), entre 0 y 4.
    - ``log_area = np.log(area)``: logaritmo natural del tamaño del lote, que corrige la asimetría
      derecha de ``area`` y refleja rendimientos marginales decrecientes en precios hedónicos.
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
        return np.asarray(base + COLUMNAS_DERIVADAS_T18, dtype=object)


class _EliminadorColumna(BaseEstimator, TransformerMixin):
    """Elimina columnas indicadas si están presentes (usado en la ablación de furnishingstatus)."""

    def __init__(self, columnas: Tuple[str, ...]):
        self.columnas = columnas

    def fit(self, X, y=None):
        self.columnas_entrada_ = list(X.columns)
        self.columnas_salida_ = [c for c in self.columnas_entrada_ if c not in self.columnas]
        return self

    def transform(self, X):
        return X[self.columnas_salida_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_salida_, dtype=object)


def _info_mutua(X, y):
    return mutual_info_regression(X, y, random_state=SEMILLA)


def _crear_selector(metodo: str, k: int = K_SELECCION_T18):
    """Crea el selector de variables para ejecutar dentro del pliegue de entrenamiento."""
    if metodo == "mi":
        return SelectKBest(score_func=_info_mutua, k=k)
    if metodo == "lasso":
        return SelectFromModel(
            LassoCV(cv=5, random_state=SEMILLA, max_iter=10000),
            threshold=1e-6,
        )
    if metodo == "rfe":
        return RFE(
            estimator=Ridge(alpha=1.0, random_state=SEMILLA),
            n_features_to_select=k,
            step=1,
        )
    raise ValueError(f"Método de selección desconocido: {metodo}")


def construir_preprocesamiento_t18(
    escalar: bool = True,
    incluir_amoblado: bool = True,
    selector: Optional[str] = None,
    k: int = K_SELECCION_T18,
) -> Pipeline:
    """Construye el pipeline de preprocesamiento de T18 (sin fuga de información).

    Parameters
    ----------
    escalar : bool
        Si True, aplica ``StandardScaler`` a las columnas numéricas y derivadas.
    incluir_amoblado : bool
        Si False, excluye ``furnishingstatus`` antes de codificar (ablación de H3).
    selector : {None, "mi", "lasso", "rfe"}
        Selector de variables a aplicar después de transformar las columnas.
    k : int
        Número de variables a conservar para ``"mi"`` y ``"rfe"``.
    """
    cols_continuas = COLUMNAS_NUMERICAS + COLUMNAS_DERIVADAS_T18
    escalador = StandardScaler() if escalar else "passthrough"

    transformadores = [("continuas", escalador, cols_continuas)]
    if incluir_amoblado:
        transformadores.append(
            (
                "amoblado",
                OneHotEncoder(categories=[CODIGOS_AMOBLADO], drop="first", sparse_output=False),
                [COLUMNA_AMOBLADO],
            )
        )

    ct = ColumnTransformer(
        transformadores,
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    pasos: List[Tuple[str, BaseEstimator]] = [("codificar", CodificadorBase())]
    if not incluir_amoblado:
        pasos.append(("sin_amoblado", _EliminadorColumna((COLUMNA_AMOBLADO,))))
    pasos.append(("derivar", DerivadorVariablesT18()))
    pasos.append(("columnas", ct))
    if selector is not None:
        pasos.append(("sel", _crear_selector(selector, k=k)))
    return Pipeline(pasos)


def construir_pipeline_t18(
    modelo_id: str,
    incluir_amoblado: bool = True,
    selector: Optional[str] = None,
    k: int = K_SELECCION_T18,
) -> TransformedTargetRegressor:
    """Arma el pipeline completo de T18 con ``log(price)`` en ``TransformedTargetRegressor``."""
    escalar = MODELOS[modelo_id][2] or (selector in ("lasso", "rfe"))
    prepro = construir_preprocesamiento_t18(
        escalar=escalar,
        incluir_amoblado=incluir_amoblado,
        selector=selector,
        k=k,
    )
    interno = Pipeline([("prepro", prepro), ("modelo", crear_modelo(modelo_id))])
    if modelo_id == "M4":
        from sklearn.preprocessing import FunctionTransformer

        return TransformedTargetRegressor(
            regressor=interno,
            transformer=Pipeline(
                [
                    ("log", FunctionTransformer(np.log, inverse_func=np.exp, check_inverse=False)),
                    ("std", StandardScaler()),
                ]
            ),
        )
    return TransformedTargetRegressor(
        regressor=interno,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def justificar_variables_derivadas(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Calcula correlaciones (Pearson y Spearman) con ``price`` y ``log(price)`` para las 4 derivadas."""
    cod = CodificadorBase().fit_transform(X)
    der = DerivadorVariablesT18().fit_transform(cod)
    y_log = np.log(y.astype(float))

    filas = []
    for col in COLUMNAS_DERIVADAS_T18:
        s = der[col].astype(float)
        r_p, p_p = stats.pearsonr(s, y)
        r_log, p_log = stats.pearsonr(s, y_log)
        rho, p_s = stats.spearmanr(s, y)
        filas.append(
            {
                "variable": col,
                "min": float(s.min()),
                "media": float(s.mean()),
                "mediana": float(s.median()),
                "max": float(s.max()),
                "asimetria": float(s.skew()),
                "pearson_price": float(r_p),
                "pearson_log_price": float(r_log),
                "spearman_price": float(rho),
                "p_valor_log": float(p_log),
            }
        )
    return pd.DataFrame(filas)


def comparar_seleccion_cv(
    X: pd.DataFrame,
    y: pd.Series,
    pliegues: List[Tuple[np.ndarray, np.ndarray]],
    modelos: Iterable[str] = ("M1", "M2", "M4", "M5", "M6"),
    k: int = K_SELECCION_T18,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compara Información Mutua, Lasso y RFE dentro de los pliegues de CV.

    Returns
    -------
    resumen_vars : pd.DataFrame
        Frecuencia de selección (%) y métricas promedio por variable en MI, Lasso y RFE.
    por_pliegue_modelos : pd.DataFrame
        RMSE, MAE, MAPE y R² por (selector, modelo, pliegue) fuera de muestra.
    resumen_modelos : pd.DataFrame
        Media y desviación estándar de las métricas por (selector, modelo).
    """
    registros_vars = []
    registros_modelos = []

    for pliegue_id, (ajuste, validacion) in enumerate(pliegues):
        X_aj, y_aj = X.iloc[ajuste], y.iloc[ajuste]
        y_aj_log = np.log(y_aj.astype(float))

        pre_base = construir_preprocesamiento_t18(escalar=True, incluir_amoblado=True, selector=None)
        X_tr = pre_base.fit_transform(X_aj, y_aj_log)
        nombres = list(X_tr.columns)

        # 1. Información Mutua
        mi_scores = _info_mutua(X_tr, y_aj_log)
        mi_top = set(np.argsort(mi_scores)[::-1][:k])

        # 2. Lasso (CV interno)
        lasso = LassoCV(cv=5, random_state=SEMILLA, max_iter=10000).fit(X_tr, y_aj_log)
        lasso_coefs = np.abs(lasso.coef_)
        lasso_sel = lasso_coefs > 1e-6

        # 3. RFE con Ridge
        rfe = RFE(estimator=Ridge(alpha=1.0, random_state=SEMILLA), n_features_to_select=k, step=1).fit(X_tr, y_aj_log)

        for idx, nombre in enumerate(nombres):
            registros_vars.append(
                {
                    "pliegue": pliegue_id,
                    "variable": nombre,
                    "mi_score": float(mi_scores[idx]),
                    "mi_sel": int(idx in mi_top),
                    "lasso_abs_coef": float(lasso_coefs[idx]),
                    "lasso_sel": int(lasso_sel[idx]),
                    "rfe_rank": int(rfe.ranking_[idx]),
                    "rfe_sel": int(rfe.support_[idx]),
                }
            )

        # Evaluación fuera de muestra con cada selector
        for sel_nombre in (None, "mi", "lasso", "rfe"):
            etiqueta_sel = "Sin selección (todas)" if sel_nombre is None else sel_nombre.upper()
            for m in modelos:
                est = clone(construir_pipeline_t18(m, incluir_amoblado=True, selector=sel_nombre, k=k))
                est.fit(X_aj, y_aj)
                pred = est.predict(X.iloc[validacion])
                metricas = calcular_metricas(y.iloc[validacion], pred)
                metricas.update({"selector": etiqueta_sel, "modelo": m, "modelo_nombre": MODELOS[m][0], "pliegue": pliegue_id})
                registros_modelos.append(metricas)

    df_vars = pd.DataFrame(registros_vars)
    resumen_vars = (
        df_vars.groupby("variable", as_index=False)
        .agg(
            freq_mi_pct=("mi_sel", lambda s: 100.0 * s.mean()),
            mi_score_media=("mi_score", "mean"),
            freq_lasso_pct=("lasso_sel", lambda s: 100.0 * s.mean()),
            lasso_abs_coef_media=("lasso_abs_coef", "mean"),
            freq_rfe_pct=("rfe_sel", lambda s: 100.0 * s.mean()),
            rfe_rank_media=("rfe_rank", "mean"),
        )
    )
    resumen_vars["freq_promedio_pct"] = resumen_vars[["freq_mi_pct", "freq_lasso_pct", "freq_rfe_pct"]].mean(axis=1)
    resumen_vars["consenso_3_metodos"] = (
        (resumen_vars["freq_mi_pct"] >= 50.0).astype(int)
        + (resumen_vars["freq_lasso_pct"] >= 50.0).astype(int)
        + (resumen_vars["freq_rfe_pct"] >= 50.0).astype(int)
    )
    resumen_vars = resumen_vars.sort_values(
        ["consenso_3_metodos", "freq_promedio_pct", "mi_score_media"], ascending=[False, False, False]
    ).reset_index(drop=True)

    por_pliegue_modelos = pd.DataFrame(registros_modelos)
    resumen_modelos = (
        por_pliegue_modelos.groupby(["selector", "modelo", "modelo_nombre"], as_index=False)
        .agg(
            rmse_media=("rmse", "mean"),
            rmse_desv=("rmse", lambda s: s.std(ddof=1)),
            mae_media=("mae", "mean"),
            mape_media=("mape", "mean"),
            r2_media=("r2", "mean"),
        )
        .sort_values(["modelo", "rmse_media"])
        .reset_index(drop=True)
    )
    return resumen_vars, por_pliegue_modelos, resumen_modelos


def prueba_t_corregida_cv(
    diferencias: np.ndarray,
    n_ajuste: int,
    n_validacion: int,
) -> Dict[str, float]:
    """Prueba t corregida de Nadeau & Bengio (2003) para diferencias pareadas en CV repetida.

    Parameters
    ----------
    diferencias : array-like
        Diferencias pareadas por pliegue (ej. ``rmse_sin - rmse_con``).
    n_ajuste : int
        Número de observaciones en el subconjunto de entrenamiento de cada pliegue.
    n_validacion : int
        Número de observaciones en el subconjunto de validación de cada pliegue.
    """
    d = np.asarray(diferencias, dtype=float)
    n_pliegues = len(d)
    media = float(np.mean(d))
    desv = float(np.std(d, ddof=1))
    var = desv**2
    factor = (1.0 / n_pliegues) + (n_validacion / n_ajuste)
    se_corr = float(np.sqrt(factor * var))
    if se_corr == 0.0:
        t_stat = 0.0
        p_valor = 1.0
    else:
        t_stat = float(media / se_corr)
        p_valor = float(2.0 * stats.t.sf(np.abs(t_stat), df=n_pliegues - 1))
    t_crit = float(stats.t.ppf(0.975, df=n_pliegues - 1))
    return {
        "media_diff": media,
        "desv_diff": desv,
        "min_diff": float(np.min(d)),
        "max_diff": float(np.max(d)),
        "se_corregida": se_corr,
        "ic95_inf": media - t_crit * se_corr,
        "ic95_sup": media + t_crit * se_corr,
        "t_stat": t_stat,
        "p_valor": p_valor,
        "significativo_005": bool(p_valor < 0.05),
    }


def ablar_furnishingstatus(
    X: pd.DataFrame,
    y: pd.Series,
    pliegues: List[Tuple[np.ndarray, np.ndarray]],
    modelos: Iterable[str] = tuple(MODELOS.keys()),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta la ablación de ``furnishingstatus`` (con vs sin) en los mismos pliegues de CV.

    Diferencia reportada: ``delta_rmse = rmse_sin - rmse_con``.
    Un valor positivo indica que incluir ``furnishingstatus`` reduce el RMSE; un valor negativo
    indica que incluirla empeora el RMSE.
    """
    filas_pliegue = []
    resumen = []

    for m in modelos:
        diffs = []
        rmses_con, rmses_sin = [], []
        n_aj_lista, n_val_lista = [], []

        for k, (ajuste, validacion) in enumerate(pliegues):
            X_aj, y_aj = X.iloc[ajuste], y.iloc[ajuste]
            X_val, y_val = X.iloc[validacion], y.iloc[validacion]

            est_con = clone(construir_pipeline_t18(m, incluir_amoblado=True, selector=None))
            est_sin = clone(construir_pipeline_t18(m, incluir_amoblado=False, selector=None))

            est_con.fit(X_aj, y_aj)
            est_sin.fit(X_aj, y_aj)

            met_con = calcular_metricas(y_val, est_con.predict(X_val))
            met_sin = calcular_metricas(y_val, est_sin.predict(X_val))
            delta = met_sin["rmse"] - met_con["rmse"]

            rmses_con.append(met_con["rmse"])
            rmses_sin.append(met_sin["rmse"])
            diffs.append(delta)
            n_aj_lista.append(len(ajuste))
            n_val_lista.append(len(validacion))

            filas_pliegue.append(
                {
                    "modelo": m,
                    "modelo_nombre": MODELOS[m][0],
                    "pliegue": k,
                    "rmse_con": met_con["rmse"],
                    "rmse_sin": met_sin["rmse"],
                    "delta_rmse": delta,
                    "r2_con": met_con["r2"],
                    "r2_sin": met_sin["r2"],
                }
            )

        prueba = prueba_t_corregida_cv(
            diffs,
            n_ajuste=int(np.mean(n_aj_lista)),
            n_validacion=int(np.mean(n_val_lista)),
        )
        rmse_con_media = float(np.mean(rmses_con))
        rmse_sin_media = float(np.mean(rmses_sin))
        resumen.append(
            {
                "modelo": m,
                "modelo_nombre": MODELOS[m][0],
                "rmse_con_media": rmse_con_media,
                "rmse_con_desv": float(np.std(rmses_con, ddof=1)),
                "rmse_sin_media": rmse_sin_media,
                "rmse_sin_desv": float(np.std(rmses_sin, ddof=1)),
                "delta_rmse_media": prueba["media_diff"],
                "delta_rmse_desv": prueba["desv_diff"],
                "delta_rmse_pct": 100.0 * prueba["media_diff"] / rmse_sin_media,
                "se_corregida": prueba["se_corregida"],
                "ic95_inf": prueba["ic95_inf"],
                "ic95_sup": prueba["ic95_sup"],
                "t_stat": prueba["t_stat"],
                "p_valor": prueba["p_valor"],
                "significativo_005": prueba["significativo_005"],
            }
        )

    return pd.DataFrame(filas_pliegue), pd.DataFrame(resumen)
