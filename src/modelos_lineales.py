"""Modelos lineales de T19: OLS hedónico, Ridge, Lasso y ElasticNet.

Los cuatro se arman sobre los preprocesamientos de ``src.preprocesamiento`` (T12, T17, T18), con la misma
estructura que el piloto: ``TransformedTargetRegressor(regressor=Pipeline([prepro, modelo]))``. Así toda
transformación aprendida (escaladores, Yeo-Johnson) se ajusta solo con el pliegue de ajuste y las
predicciones salen siempre en la **escala original de price**, comparable entre P1 y P2.

**Escala del objetivo en los modelos penalizados.** Ridge, Lasso y ElasticNet estandarizan el objetivo
dentro del pliegue (en P2, primero ``log`` y luego estandarización). Así ``alpha`` tiene el mismo significado
en P1 (precio del orden de 10⁶) y en P2 (log-precio): sin eso, ninguna rejilla razonable de ``alpha`` serviría
en las dos escalas. Para OLS y Ridge no cambia las predicciones: la solución escala con el objetivo (hay una
prueba en ``tests/test_modelos_lineales.py``). Con Lasso y ElasticNet cambia solo la escala efectiva de ``alpha``.

**Nota sobre P3.** Este módulo evalúa P1 y P2. P3 no se usa hasta que se integre en ``main`` la corrección de
Mauricio (variables derivadas de T18 sin la selección por información mutua con k = 10); no se implementa
aquí una versión alternativa.

Nombres: ``OLS``, ``Ridge``, ``Lasso`` y ``ElasticNet`` (en la tarjeta T19, M1 a M4). No se reutilizan los
identificadores M1–M6 del piloto de ``src/modelos.py`` para no confundirlos.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import SEMILLA
from src.preprocesamiento import construir_preprocesamiento

PREPROCESAMIENTOS_T19: Tuple[str, ...] = ("P1", "P2")  # P3 pendiente de la corrección de Mauricio
PREFIJO = "regressor__modelo__"                          # prefijo de los hiperparámetros en la búsqueda
MAX_ITER_CD = 200_000                                    # descenso por coordenadas: tope generoso para vigilar la convergencia

# nombre -> (descripción, ¿modelo penalizado?)
MODELOS_LINEALES: Dict[str, Tuple[str, bool]] = {
    "OLS": ("OLS hedónico (línea base, sin penalización)", False),
    "Ridge": ("Ridge (penalización L2)", True),
    "Lasso": ("Lasso (penalización L1)", True),
    "ElasticNet": ("ElasticNet (L1 + L2)", True),
}

# Rejillas de búsqueda (alpha en la escala del objetivo estandarizado). Listas explícitas y reproducibles.
ESPACIOS: Dict[str, dict] = {
    "OLS": {},
    "Ridge": {PREFIJO + "alpha": list(np.logspace(-3, 3, 25))},
    "Lasso": {PREFIJO + "alpha": list(np.logspace(-6, 0, 31))},
    "ElasticNet": {PREFIJO + "alpha": list(np.logspace(-4, 0, 20)), PREFIJO + "l1_ratio": [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]},
}


def crear_modelo_lineal(nombre: str):
    """Estimador lineal con valores iniciales de ``alpha`` (la búsqueda los reemplaza)."""
    if nombre == "OLS":
        return LinearRegression()
    if nombre == "Ridge":
        return Ridge(alpha=1.0)
    if nombre == "Lasso":
        return Lasso(alpha=0.01, max_iter=MAX_ITER_CD)
    if nombre == "ElasticNet":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=MAX_ITER_CD)
    raise ValueError(f"Modelo lineal desconocido: {nombre}")


def _log_directo() -> FunctionTransformer:
    """log del objetivo con su inversa exp (sin parámetros que aprender)."""
    return FunctionTransformer(np.log, inverse_func=np.exp, check_inverse=False)


def construir_pipeline_lineal(prepro_id: str, nombre: str) -> TransformedTargetRegressor:
    """Pipeline completo preprocesamiento + modelo lineal; ``predict`` devuelve precios en escala original.

    - OLS no escala las predictoras (sus predicciones no dependen de la escala y los coeficientes quedan en
      unidades naturales); los modelos penalizados sí, porque la penalización depende de la escala.
    - OLS usa el objetivo en escala original (P1) o ``log`` (P2). Los penalizados estandarizan además el
      objetivo (ver el docstring del módulo).
    """
    if prepro_id not in PREPROCESAMIENTOS_T19:
        raise ValueError(f"T19 solo evalúa {PREPROCESAMIENTOS_T19}; P3 está pendiente de la corrección de Mauricio")
    penalizado = MODELOS_LINEALES[nombre][1]
    interno = Pipeline([("prepro", construir_preprocesamiento(prepro_id, escalar=penalizado)), ("modelo", crear_modelo_lineal(nombre))])
    if prepro_id == "P1":
        return TransformedTargetRegressor(regressor=interno, transformer=StandardScaler() if penalizado else None)
    if penalizado:
        return TransformedTargetRegressor(regressor=interno, transformer=Pipeline([("log", _log_directo()), ("std", StandardScaler())]))
    return TransformedTargetRegressor(regressor=interno, func=np.log, inverse_func=np.exp, check_inverse=False)


def sigma_objetivo(estimador_ajustado) -> float:
    """Desviación estándar del objetivo (en su escala de modelado) usada para estandarizar; 1.0 si no se estandarizó."""
    t = getattr(estimador_ajustado, "transformer_", None)
    if t is None:
        return 1.0
    escalador = t.named_steps["std"] if hasattr(t, "named_steps") else t
    return float(escalador.scale_[0]) if hasattr(escalador, "scale_") else 1.0  # solo log (OLS en P2): sin estandarizar


def tabla_coeficientes(estimador_ajustado, X_entrenamiento: pd.DataFrame, prepro_id: str, nombre: str) -> pd.DataFrame:
    """Coeficientes de un modelo ajustado, con una medida comparable entre variables.

    Para cada variable del diseño (después del preprocesamiento) devuelve:

    - ``coef_modelo``: coeficiente en el objetivo de modelado (log-precio en P2; precio en P1), sin la
      estandarización interna del objetivo (se multiplica por ``sigma_objetivo``).
    - ``sd_variable``: desviación estándar de la variable en entrenamiento, ya transformada.
    - ``efecto_1DE``: ``coef_modelo × sd_variable``: cambio del objetivo por +1 desviación estándar de la
      variable. **Es la medida que permite comparar variables con escalas distintas**; ``coef_modelo`` solo
      se compara entre variables con la misma escala.
    - ``efecto_1DE_pct`` (solo P2): ``100·(exp(efecto_1DE) − 1)``, cambio porcentual aproximado del precio.
    - ``efecto_1DE_cad1987`` (solo P1): ``efecto_1DE / 70``, en dólares canadienses de 1987.

    Son asociaciones del modelo, no efectos causales. Se calculan con filas de entrenamiento únicamente.
    """
    interno = estimador_ajustado.regressor_
    prepro, modelo = interno.named_steps["prepro"], interno.named_steps["modelo"]
    nombres = list(prepro.get_feature_names_out())
    Z = prepro.transform(X_entrenamiento)
    sd = np.asarray(Z, dtype=float).std(axis=0, ddof=1)
    coef = np.asarray(modelo.coef_, dtype=float).ravel() * sigma_objetivo(estimador_ajustado)
    efecto = coef * sd
    tabla = pd.DataFrame({
        "preprocesamiento": prepro_id, "modelo": nombre, "variable": nombres, "coef_modelo": coef,
        "sd_variable": sd, "efecto_1DE": efecto, "coef_nulo": np.isclose(coef, 0.0),
    })
    if prepro_id == "P2":
        tabla["efecto_1DE_pct"] = 100 * (np.exp(efecto) - 1)
    else:
        tabla["efecto_1DE_cad1987"] = efecto / 70
    return tabla


def diagnostico_convergencia(estimador_ajustado, nombre: str) -> Dict[str, object]:
    """Iteraciones usadas por el descenso por coordenadas frente al tope (Lasso y ElasticNet); None si no aplica."""
    modelo = estimador_ajustado.regressor_.named_steps["modelo"]
    if nombre in ("Lasso", "ElasticNet"):
        n_iter = int(np.max(np.atleast_1d(modelo.n_iter_)))
        return {"n_iter": n_iter, "max_iter": int(modelo.max_iter), "convergio": n_iter < int(modelo.max_iter)}
    return {"n_iter": None, "max_iter": None, "convergio": None}


def ols_hedonico_statsmodels(X: pd.DataFrame, y: pd.Series, logaritmico: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Regresión hedónica clásica con inferencia (statsmodels), solo con filas de entrenamiento.

    Conecta la línea base de T19 con la formulación del curso de Inferencia Estadística (T04, 1.8): modelo
    lineal o log-lineal del precio sobre las 12 atributos, con ``furnishingstatus`` como dos indicadoras.
    Devuelve la tabla de coeficientes (error estándar, valor p, IC 95 %) y R² y R² ajustado **dentro de muestra**.
    Con ``logaritmico=True`` se agrega ``cambio_pct = 100·(exp(β) − 1)`` por unidad de la variable, una lectura
    aproximada y asociativa (no causal). Los valores p no se usan para seleccionar modelos.
    """
    import statsmodels.api as sm

    diseno = X.drop(columns=["furnishingstatus"]).copy()
    diseno["amoblado_semi"] = (X["furnishingstatus"] == 1).astype(int)
    diseno["amoblado_completo"] = (X["furnishingstatus"] == 2).astype(int)
    objetivo = np.log(y) if logaritmico else y
    ajuste = sm.OLS(objetivo.to_numpy(float), sm.add_constant(diseno.to_numpy(float))).fit()
    nombres = ["intercepto"] + list(diseno.columns)
    ic = ajuste.conf_int(0.05)
    tabla = pd.DataFrame({"variable": nombres, "coeficiente": ajuste.params, "error_estandar": ajuste.bse, "valor_p": ajuste.pvalues,
                          "ic95_inferior": ic[:, 0], "ic95_superior": ic[:, 1]})
    if logaritmico:
        tabla["cambio_pct"] = 100 * (np.exp(tabla["coeficiente"]) - 1)
        tabla.loc[tabla["variable"] == "intercepto", "cambio_pct"] = np.nan  # el intercepto no tiene lectura porcentual
    resumen = {"n": int(ajuste.nobs), "r2": float(ajuste.rsquared), "r2_ajustado": float(ajuste.rsquared_adj)}
    return tabla, resumen
