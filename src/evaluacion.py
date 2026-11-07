"""Carga aislada del entrenamiento, pliegues, métricas y ejecución del piloto (Tarjeta T12).

**Regla crítica: el conjunto de prueba está sellado (T06).** ``cargar_entrenamiento`` lee primero solo
los ``id`` de ``particion.csv`` y luego carga de ``housing_limpio.csv`` únicamente las filas de
``entrenamiento`` (con ``skiprows``): los valores de las filas ``prueba`` nunca entran a memoria. Todo
lo demás (pliegues, ajuste, métricas, curva de aprendizaje) trabaja sobre esas filas.

Las métricas se calculan **por pliegue y en la escala original de ``price``** (unidades de Kaggle =
dólares canadienses de 1987 × 70, T05), sin redondear antes de agregar.
"""

import hashlib
import time
import traceback
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, learning_curve

from src.config import RUTA_DATOS_LIMPIOS, RUTA_PARTICION, SEMILLA
from src.modelos import MODELOS, construir_pipeline
from src.preprocesamiento import COLUMNAS_PREDICTORAS, nombres_seleccionados

N_SPLITS, N_REPETICIONES, N_ESTRATOS = 5, 3, 5
FACTOR_KAGGLE = 70  # price (Kaggle) = precio original en CAD de 1987 × 70 (T05)


# ----------------------------------------------------------------------------- datos aislados
def huella_sha256(ruta) -> str:
    """Huella SHA-256 de un archivo (para dejar evidencia de que no se modificó)."""
    return hashlib.sha256(open(ruta, "rb").read()).hexdigest()


def cargar_entrenamiento(ruta_limpios=RUTA_DATOS_LIMPIOS, ruta_particion=RUTA_PARTICION) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Carga SOLO las filas de entrenamiento definidas por T06 y devuelve ``(X, y, evidencia)``.

    El archivo de limpios se lee en dos pasos: (1) solo la columna ``id`` para saber en qué línea está
    cada fila; (2) el archivo completo saltando las líneas cuyo ``id`` es de ``prueba``. Así los valores
    de prueba no se cargan. ``evidencia`` registra recuentos, huellas y verificaciones del aislamiento.
    """
    particion = pd.read_csv(ruta_particion)
    ids_prueba = set(particion.loc[particion["conjunto"] == "prueba", "id"])
    ids_entrenamiento = set(particion.loc[particion["conjunto"] == "entrenamiento", "id"])

    ids_en_orden = pd.read_csv(ruta_limpios, usecols=["id"])["id"]
    lineas_prueba = [pos + 1 for pos, i in enumerate(ids_en_orden) if i in ids_prueba]  # +1 por el encabezado
    df = pd.read_csv(ruta_limpios, skiprows=lineas_prueba).set_index("id")

    assert set(df.index) == ids_entrenamiento, "El conjunto cargado no coincide con el entrenamiento de T06"
    assert not (set(df.index) & ids_prueba), "Se cargaron filas del conjunto sellado"
    assert df.isna().sum().sum() == 0, "Se esperaban 0 valores faltantes (T06)"

    evidencia = {
        "n_entrenamiento_cargado": int(len(df)),
        "n_prueba_en_particion": int(len(ids_prueba)),          # solo el recuento, nunca sus valores
        "n_prueba_cargado": int(len(set(df.index) & ids_prueba)),
        "interseccion_ids_entrenamiento_prueba": int(len(ids_entrenamiento & ids_prueba)),
        "sha256_particion": huella_sha256(ruta_particion),
        "sha256_housing_limpio": huella_sha256(ruta_limpios),
        "semilla": SEMILLA,
    }
    X, y = df[COLUMNAS_PREDICTORAS].copy(), df["price"].copy()
    return X, y, evidencia


# ----------------------------------------------------------------------------- pliegues
def crear_pliegues(y: pd.Series, n_splits: int = N_SPLITS, n_repeticiones: int = N_REPETICIONES, semilla: int = SEMILLA) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Lista fija de pliegues ``(indices_ajuste, indices_validacion)`` de la validación cruzada repetida 5×3.

    Se usa ``RepeatedStratifiedKFold`` con estratos = quintiles del precio de entrenamiento (como en T06
    y T11, 3.6), de modo que cada pliegue cubre el rango de precios. Los estratos solo salen de ``y`` de
    entrenamiento. La misma lista se reutiliza para los 18 pipelines: las comparaciones son pareadas.
    """
    estratos = pd.qcut(y, q=N_ESTRATOS, labels=False)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeticiones, random_state=semilla)
    return [(a, v) for a, v in cv.split(np.zeros(len(y)), estratos)]


# ----------------------------------------------------------------------------- métricas
def calcular_metricas(y_real, y_pred) -> Dict[str, float]:
    """RMSE, MAE, MAPE (%) y R² en la escala de ``y_real``. R² puede ser negativo fuera de muestra."""
    y_real, y_pred = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_real - y_pred
    ss_res, ss_tot = np.sum(error**2), np.sum((y_real - y_real.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(100 * np.mean(np.abs(error / y_real))),
        "r2": float(1 - ss_res / ss_tot),
    }


# ----------------------------------------------------------------------------- experimento
def ejecutar_experimento(X: pd.DataFrame, y: pd.Series, pliegues, preprocesamientos: Iterable[str], modelos: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalúa cada combinación P×M en los mismos pliegues y devuelve ``(por_pliegue, fallos, seleccion_p3)``.

    ``por_pliegue`` tiene una fila por (preprocesamiento, modelo, pliegue) con las cuatro métricas y los
    tiempos de ajuste e inferencia. Si un ajuste falla, se registra en ``fallos`` (con el pliegue y el
    mensaje) y **no** se sustituye por ceros. ``seleccion_p3`` lista las variables que conservó la
    selección de P3 en cada pliegue.
    """
    filas, fallos, seleccion = [], [], []
    for p in preprocesamientos:
        for m in modelos:
            for k, (ajuste, validacion) in enumerate(pliegues):
                try:
                    estimador = clone(construir_pipeline(p, m))
                    t0 = time.perf_counter()
                    estimador.fit(X.iloc[ajuste], y.iloc[ajuste])
                    t_ajuste = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pred = estimador.predict(X.iloc[validacion])
                    t_inferencia = time.perf_counter() - t0
                    if not np.all(np.isfinite(pred)):
                        raise ValueError("La predicción contiene NaN o infinitos")
                    pred_ajuste = estimador.predict(X.iloc[ajuste])  # error sobre las mismas filas de ajuste (señal de sobreajuste)
                    fila = {"preprocesamiento": p, "modelo": m, "pliegue": k, "repeticion": k // N_SPLITS,
                            "n_ajuste": len(ajuste), "n_validacion": len(validacion),
                            "tiempo_ajuste_s": t_ajuste, "tiempo_inferencia_s": t_inferencia}
                    fila.update(calcular_metricas(y.iloc[validacion], pred))
                    fila["rmse_ajuste"] = calcular_metricas(y.iloc[ajuste], pred_ajuste)["rmse"]
                    filas.append(fila)
                    if p == "P3":
                        interno = estimador.regressor_ if hasattr(estimador, "regressor_") else estimador
                        seleccion.append({"modelo": m, "pliegue": k, "variables": ";".join(nombres_seleccionados(interno.named_steps["prepro"]))})
                except Exception as e:  # se registra y se continúa: no se oculta ni se reemplaza por ceros
                    fallos.append({"preprocesamiento": p, "modelo": m, "pliegue": k, "error": f"{type(e).__name__}: {e}",
                                   "traza": traceback.format_exc(limit=3)})
    return pd.DataFrame(filas), pd.DataFrame(fallos), pd.DataFrame(seleccion)


def resumir(por_pliegue: pd.DataFrame) -> pd.DataFrame:
    """Resume por (P, M): media y desviación estándar (ddof=1) de cada métrica, folds válidos y tiempos medios.

    Se agrega sin redondear. Se ordena por RMSE medio ascendente (métrica principal).
    """
    g = por_pliegue.groupby(["preprocesamiento", "modelo"])
    r = pd.DataFrame({
        "n_pliegues_validos": g.size(),
        "rmse_media": g["rmse"].mean(), "rmse_desv": g["rmse"].std(ddof=1),
        "mae_media": g["mae"].mean(), "mae_desv": g["mae"].std(ddof=1),
        "mape_media": g["mape"].mean(), "mape_desv": g["mape"].std(ddof=1),
        "r2_media": g["r2"].mean(), "r2_desv": g["r2"].std(ddof=1),
        "rmse_ajuste_media": g["rmse_ajuste"].mean(),
        "tiempo_ajuste_s_media": g["tiempo_ajuste_s"].mean(),
        "tiempo_inferencia_s_media": g["tiempo_inferencia_s"].mean(),
    }).reset_index()
    r.insert(2, "modelo_nombre", r["modelo"].map(lambda m: MODELOS[m][0]))
    r.insert(3, "familia", r["modelo"].map(lambda m: MODELOS[m][1]))
    return r.sort_values("rmse_media").reset_index(drop=True)


# ----------------------------------------------------------------------------- curva de aprendizaje
def curva_aprendizaje(X: pd.DataFrame, y: pd.Series, pliegues, prepro_id: str, modelo_id: str, fracciones=(0.2, 0.35, 0.5, 0.65, 0.8, 1.0)) -> pd.DataFrame:
    """RMSE de ajuste y de validación en función del tamaño de la muestra de ajuste (solo entrenamiento).

    Usa ``learning_curve`` con los mismos pliegues del piloto. ``shuffle=True`` es necesario porque
    ``housing_limpio.csv`` viene ordenado por precio descendente (T06): sin barajar, las primeras filas
    de cada submuestra serían las más caras. Devuelve una fila por (tamaño, pliegue).
    """
    tamanos, ent, val = learning_curve(
        construir_pipeline(prepro_id, modelo_id), X, y, cv=pliegues, train_sizes=list(fracciones),
        scoring="neg_root_mean_squared_error", shuffle=True, random_state=SEMILLA, n_jobs=1, return_times=False,
    )
    filas = []
    for i, n in enumerate(tamanos):
        for k in range(ent.shape[1]):
            filas.append({"preprocesamiento": prepro_id, "modelo": modelo_id, "n_ajuste": int(n), "pliegue": k,
                          "rmse_ajuste": float(-ent[i, k]), "rmse_validacion": float(-val[i, k])})
    return pd.DataFrame(filas)
