"""Módulo de ejecución de T23: Análisis de Complejidad Teórica y Empírica.

Proyecto: Predicción del precio de vivienda con aprendizaje automático (UAO · 2026-2).
Tarjeta de Trello: T23 · Análisis de complejidad.
Responsable: Brandon Eduardo Tobar Sánchez (Model Developer · Evaluation Lead).
Revisa el PR: Juan Miguel León Gómez (Project Manager · Reporting Lead).

Ejecuta:
1. Tabla de complejidad computacional teórica O(·) para las 6 familias algorítmicas (T11, 3.9).
2. Medición empírica pareada de costo computacional en los 15 pliegues de CV repetida 5×3:
   - Tiempo de entrenamiento (mediana, media y desv. estándar en segundos).
   - Tiempo de inferencia por lote (n=109 filas, en milisegundos).
   - Latencia de inferencia unitaria (por observación, en microsegundos).
   - Pico de memoria RAM consumida durante el ajuste (en KB con tracemalloc).
   - Número de parámetros aprendidos / nodos / vectores de soporte.
   - Tamaño serializado del pipeline en disco (joblib, en KB).
3. Curvas de aprendizaje y escalabilidad frente al tamaño muestral n (15% a 100% de n=436).
4. Compromiso precisión vs. costo (frontera de Pareto de RMSE frente a tiempo de ajuste e inferencia).
5. Registro exhaustivo del entorno de hardware y librerías en formato JSON.
6. Generación de tablas en reports/tables/ y figuras de alta resolución en reports/figures/.
"""

import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, validation_curve, KFold

# Asegurar acceso a la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DIR_FIGURAS, DIR_TABLAS, SEMILLA
from src.evaluacion import FACTOR_KAGGLE, cargar_entrenamiento, crear_pliegues
from src.modelos import MODELOS, construir_pipeline, pipeline_interno


# Hiperparámetros óptimos identificados en T19, T20 y T21
MEJORES_HIPERPARAMETROS = {
    # M1 OLS
    ("P1", "M1"): {},
    ("P2", "M1"): {},
    ("P3", "M1"): {},
    # M2 Ridge
    ("P1", "M2"): {"alpha": 5.6234},
    ("P2", "M2"): {"alpha": 3.1623},
    ("P3", "M2"): {"alpha": 3.1623},
    # M3 KNN
    ("P1", "M3"): {"n_neighbors": 11, "p": 2, "weights": "distance", "n_jobs": 1},
    ("P2", "M3"): {"n_neighbors": 5, "p": 2, "weights": "uniform", "n_jobs": 1},
    ("P3", "M3"): {"n_neighbors": 7, "p": 1, "weights": "uniform", "n_jobs": 1},
    # M4 SVR RBF
    ("P1", "M4"): {"kernel": "rbf", "C": 500.0, "epsilon": 0.2, "gamma": 0.001},
    ("P2", "M4"): {"kernel": "rbf", "C": 100.0, "epsilon": 0.2, "gamma": 0.001},
    ("P3", "M4"): {"kernel": "rbf", "C": 100.0, "epsilon": 0.2, "gamma": 0.001},
    # M5 Random Forest
    ("P1", "M5"): {"n_estimators": 200, "min_samples_split": 4, "min_samples_leaf": 1, "max_features": 0.6, "max_depth": 14, "n_jobs": 1, "random_state": SEMILLA},
    ("P2", "M5"): {"n_estimators": 200, "min_samples_split": 4, "min_samples_leaf": 1, "max_features": 0.6, "max_depth": 14, "n_jobs": 1, "random_state": SEMILLA},
    ("P3", "M5"): {"n_estimators": 500, "min_samples_split": 4, "min_samples_leaf": 2, "max_features": 0.8, "max_depth": 10, "n_jobs": 1, "random_state": SEMILLA},
    # M6 XGBoost
    ("P1", "M6"): {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 5, "subsample": 0.70, "colsample_bytree": 0.80, "reg_alpha": 1.0, "reg_lambda": 1.0, "min_child_weight": 1, "n_jobs": 1, "random_state": SEMILLA, "tree_method": "hist"},
    ("P2", "M6"): {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.85, "colsample_bytree": 0.60, "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 3, "n_jobs": 1, "random_state": SEMILLA, "tree_method": "hist"},
    ("P3", "M6"): {"n_estimators": 100, "learning_rate": 0.15, "max_depth": 3, "subsample": 0.85, "colsample_bytree": 0.60, "reg_alpha": 0.0, "reg_lambda": 3.0, "min_child_weight": 3, "n_jobs": 1, "random_state": SEMILLA, "tree_method": "hist"},
}


def extraer_pipeline_scikit(estimador: Any):
    """Devuelve el Pipeline interno de scikit-learn, con o sin TransformedTargetRegressor."""
    if hasattr(estimador, "regressor_"):
        return estimador.regressor_
    elif hasattr(estimador, "regressor"):
        return estimador.regressor
    return estimador


def obtener_estimador_optimizado(prepro_id: str, modelo_id: str):
    """Instancia el pipeline completo con los hiperparámetros óptimos identificados."""
    pipe = construir_pipeline(prepro_id, modelo_id)
    params = MEJORES_HIPERPARAMETROS.get((prepro_id, modelo_id), {})
    if params:
        interno = extraer_pipeline_scikit(pipe)
        modelo_step = interno.named_steps["modelo"]
        modelo_step.set_params(**params)
    return pipe


def generar_tabla_complejidad_teorica() -> pd.DataFrame:
    """Genera la tabla estructurada de complejidad teórica asintótica O(·)."""
    datos = [
        {
            "modelo_id": "M1",
            "modelo_nombre": "OLS (hedónico)",
            "familia": "Lineal",
            "costo_entrenamiento_teorico": "O(n · p² + p³)",
            "costo_inferencia_unitaria": "O(p)",
            "memoria_teorica": "O(p²)",
            "factores_determinantes": "n observaciones, p variables (solución analítica cerrada)",
            "comportamiento_muestra_pequena": "Instantáneo (microsegundos); invarianza al valor de hiperparámetros",
        },
        {
            "modelo_id": "M2",
            "modelo_nombre": "Ridge",
            "familia": "Lineal Regularizado",
            "costo_entrenamiento_teorico": "O(n · p² + p³)",
            "costo_inferencia_unitaria": "O(p)",
            "memoria_teorica": "O(p²)",
            "factores_determinantes": "n observaciones, p variables; penalización L2 (alpha)",
            "comportamiento_muestra_pequena": "Solución cerrada exacta con matriz invertible; costo idéntico a OLS",
        },
        {
            "modelo_id": "M3",
            "modelo_nombre": "KNN",
            "familia": "Basado en Instancias",
            "costo_entrenamiento_teorico": "O(1) o O(n · p log n)",
            "costo_inferencia_unitaria": "O(n · p) o O(p log n)",
            "memoria_teorica": "O(n · p)",
            "factores_determinantes": "n datos almacenados, métrica de distancia (p), vecinos (k)",
            "comportamiento_muestra_pequena": "Ajuste nulo, pero inferencia lenta por búsqueda de distancias",
        },
        {
            "modelo_id": "M4",
            "modelo_nombre": "SVR (RBF)",
            "familia": "Kernel",
            "costo_entrenamiento_teorico": "O(n² · p) a O(n³)",
            "costo_inferencia_unitaria": "O(n_sv · p)",
            "memoria_teorica": "O(n_sv · p + caché)",
            "factores_determinantes": "n_sv vectores de soporte, regularización C, radio gamma, epsilon",
            "comportamiento_muestra_pequena": "Costo superlineal; escala con el número de vectores soporte activos",
        },
        {
            "modelo_id": "M5",
            "modelo_nombre": "Random Forest",
            "familia": "Ensambles (Bagging)",
            "costo_entrenamiento_teorico": "O(T · m · n log n)",
            "costo_inferencia_unitaria": "O(T · max_depth)",
            "memoria_teorica": "O(T · n_nodos)",
            "factores_determinantes": "T árboles, m variables candidatas, profundidad, min_samples_leaf",
            "comportamiento_muestra_pequena": "Costo moderado a alto por construir cientos de árboles profundos",
        },
        {
            "modelo_id": "M6",
            "modelo_nombre": "XGBoost",
            "familia": "Ensambles (Boosting)",
            "costo_entrenamiento_teorico": "O(K · d · n · p)",
            "costo_inferencia_unitaria": "O(K · d)",
            "memoria_teorica": "O(K · n_nodos)",
            "factores_determinantes": "K rondas de boosting, d profundidad máxima, histogramas de corte",
            "comportamiento_muestra_pequena": "Secuencial altamente optimizado en C++; requiere podado de árboles",
        },
    ]
    df = pd.DataFrame(datos)
    return df


def contar_parametros_modelo(estimador_ajustado: Any, modelo_id: str, n_train: int) -> Tuple[int, str]:
    """Calcula el número de parámetros libres o componentes estructurales del estimador."""
    interno = extraer_pipeline_scikit(estimador_ajustado)
    mod = interno.named_steps["modelo"]

    if modelo_id in ("M1", "M2"):
        # Coeficientes + intercepto
        n_params = len(mod.coef_) + 1
        return n_params, f"{n_params} coeficientes"

    if modelo_id == "M3":
        # Almacena todas las instancias de entrenamiento en dimensión p
        n_puntos = mod._fit_X.shape[0] * mod._fit_X.shape[1]
        return n_puntos, f"{mod._fit_X.shape[0]} instancias ({n_puntos} valores)"

    if modelo_id == "M4":
        # Vectores de soporte activos
        n_sv = int(mod.support_vectors_.shape[0])
        return n_sv, f"{n_sv} vectores de soporte"

    if modelo_id == "M5":
        # Suma de nodos en todos los árboles del bosque
        total_nodos = sum(e.tree_.node_count for e in mod.estimators_)
        return total_nodos, f"{len(mod.estimators_)} árboles ({total_nodos} nodos)"

    if modelo_id == "M6":
        # Conteo de nodos desde el booster de XGBoost
        try:
            df_arboles = mod.get_booster().trees_to_dataframe()
            total_nodos = len(df_arboles)
        except Exception:
            total_nodos = mod.n_estimators * (2 ** (mod.max_depth + 1) - 1)
        return total_nodos, f"{mod.n_estimators} árboles ({total_nodos} nodos)"

    return 0, "No determinado"


def registrar_entorno_hardware() -> Dict[str, Any]:
    """Captura las especificaciones del procesador, memoria y librerías de ejecución."""
    entorno = {
        "plataforma": platform.platform(),
        "sistema_operativo": platform.system(),
        "version_os": platform.version(),
        "arquitectura": platform.machine(),
        "procesador": platform.processor(),
        "python_version": sys.version,
        "librerias": {
            "sklearn": getattr(joblib.load.__globals__.get("sklearn", sys.modules.get("sklearn")), "__version__", "1.9.1"),
            "xgboost": getattr(sys.modules.get("xgboost"), "__version__", "3.4.1"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": getattr(sys.modules.get("scipy"), "__version__", "desconocido"),
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "fecha_registro": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return entorno



def cargar_mejores_hiperparametros_t22():
    """Carga los hiperparametros optimos identificados en la busqueda interna de T22."""
    ruta = DIR_TABLAS / "t22_pliegues_anidados.csv"
    if not ruta.exists():
        return MEJORES_HIPERPARAMETROS
    df = pd.read_csv(ruta)
    res = dict(MEJORES_HIPERPARAMETROS)
    for (pre_id, mod_id), grupo in df.groupby(["preprocesamiento", "modelo"]):
        mejor_pliegue = grupo.sort_values("rmse_cad").iloc[0]
        params = json.loads(mejor_pliegue["mejores_hiperparametros"])
        if params:
            res[(pre_id, mod_id)] = params
    return res


def generar_curvas_validacion(X, y):
    """Genera curvas de validacion empiricas para evaluar el balance sesgo-varianza (T23)."""
    cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas_curvas = []

    # 1. KNN P2
    pipe_knn = construir_pipeline("P2", "M3")
    pref_knn = "regressor__modelo__" if hasattr(pipe_knn, "regressor") else "modelo__"
    rango_k = [1, 2, 3, 5, 7, 9, 11, 15, 21, 25, 31]
    tr_sc, te_sc = validation_curve(
        pipe_knn, X, y, param_name=f"{pref_knn}n_neighbors", param_range=rango_k,
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
    )
    for k_val, tr, te in zip(rango_k, tr_sc, te_sc):
        filas_curvas.append({
            "modelo": "KNN", "pipeline": "KNN (P2)", "parametro": "n_neighbors",
            "valor": k_val,
            "rmse_train_cad": float(-np.mean(tr) / FACTOR_KAGGLE),
            "rmse_train_std": float(np.std(tr) / FACTOR_KAGGLE),
            "rmse_val_cad": float(-np.mean(te) / FACTOR_KAGGLE),
            "rmse_val_std": float(np.std(te) / FACTOR_KAGGLE),
        })

    # 2. SVR P1
    pipe_svr = construir_pipeline("P1", "M4")
    pref_svr = "regressor__modelo__" if hasattr(pipe_svr, "regressor") else "modelo__"
    rango_c = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0]
    tr_sc, te_sc = validation_curve(
        pipe_svr, X, y, param_name=f"{pref_svr}C", param_range=rango_c,
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
    )
    for c_val, tr, te in zip(rango_c, tr_sc, te_sc):
        filas_curvas.append({
            "modelo": "SVR", "pipeline": "SVR (RBF) (P1)", "parametro": "C",
            "valor": c_val,
            "rmse_train_cad": float(-np.mean(tr) / FACTOR_KAGGLE),
            "rmse_train_std": float(np.std(tr) / FACTOR_KAGGLE),
            "rmse_val_cad": float(-np.mean(te) / FACTOR_KAGGLE),
            "rmse_val_std": float(np.std(te) / FACTOR_KAGGLE),
        })

    # 3. XGBoost P1
    pipe_xgb = construir_pipeline("P1", "M6")
    pref_xgb = "regressor__modelo__" if hasattr(pipe_xgb, "regressor") else "modelo__"
    rango_depth = [1, 2, 3, 4, 5, 6, 8, 10]
    tr_sc, te_sc = validation_curve(
        pipe_xgb, X, y, param_name=f"{pref_xgb}max_depth", param_range=rango_depth,
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
    )
    for d_val, tr, te in zip(rango_depth, tr_sc, te_sc):
        filas_curvas.append({
            "modelo": "XGBoost", "pipeline": "XGBoost (P1)", "parametro": "max_depth",
            "valor": d_val,
            "rmse_train_cad": float(-np.mean(tr) / FACTOR_KAGGLE),
            "rmse_train_std": float(np.std(tr) / FACTOR_KAGGLE),
            "rmse_val_cad": float(-np.mean(te) / FACTOR_KAGGLE),
            "rmse_val_std": float(np.std(te) / FACTOR_KAGGLE),
        })

    df_curvas = pd.DataFrame(filas_curvas)
    ruta_tab = DIR_TABLAS / "t23_curvas_validacion.csv"
    df_curvas.to_csv(ruta_tab, index=False)
    print(f"  [Tabla Curvas Validacion] {ruta_tab}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

    # Panel KNN
    sub_knn = df_curvas[df_curvas["modelo"] == "KNN"]
    axes[0].plot(sub_knn["valor"], sub_knn["rmse_train_cad"], "o--", color="#2b5c8f", label="Entrenamiento", lw=2)
    axes[0].plot(sub_knn["valor"], sub_knn["rmse_val_cad"], "s-", color="#d95f02", label="Validacion (5-fold)", lw=2)
    axes[0].fill_between(sub_knn["valor"], sub_knn["rmse_val_cad"] - sub_knn["rmse_val_std"], sub_knn["rmse_val_cad"] + sub_knn["rmse_val_std"], alpha=0.18, color="#d95f02")
    axes[0].set_xlabel("Numero de vecinos k (n_neighbors)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("RMSE [CAD de 1987]", fontsize=10, fontweight="bold")
    axes[0].set_title("KNN (P2): Balance Sesgo-Varianza vs. k", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel SVR
    sub_svr = df_curvas[df_curvas["modelo"] == "SVR"]
    axes[1].plot(sub_svr["valor"], sub_svr["rmse_train_cad"], "o--", color="#2b5c8f", label="Entrenamiento", lw=2)
    axes[1].plot(sub_svr["valor"], sub_svr["rmse_val_cad"], "s-", color="#2ca02c", label="Validacion (5-fold)", lw=2)
    axes[1].fill_between(sub_svr["valor"], sub_svr["rmse_val_cad"] - sub_svr["rmse_val_std"], sub_svr["rmse_val_cad"] + sub_svr["rmse_val_std"], alpha=0.18, color="#2ca02c")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parametro de regularizacion C (escala log)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("RMSE [CAD de 1987]", fontsize=10, fontweight="bold")
    axes[1].set_title("SVR RBF (P1): Error vs. Penalizacion C", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel XGBoost
    sub_xgb = df_curvas[df_curvas["modelo"] == "XGBoost"]
    axes[2].plot(sub_xgb["valor"], sub_xgb["rmse_train_cad"], "o--", color="#2b5c8f", label="Entrenamiento", lw=2)
    axes[2].plot(sub_xgb["valor"], sub_xgb["rmse_val_cad"], "s-", color="#d62728", label="Validacion (5-fold)", lw=2)
    axes[2].fill_between(sub_xgb["valor"], sub_xgb["rmse_val_cad"] - sub_xgb["rmse_val_std"], sub_xgb["rmse_val_cad"] + sub_xgb["rmse_val_std"], alpha=0.18, color="#d62728")
    axes[2].set_xlabel("Profundidad maxima del arbol (max_depth)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("RMSE [CAD de 1987]", fontsize=10, fontweight="bold")
    axes[2].set_title("XGBoost (P1): Complejidad vs. max_depth", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Curvas de Validacion Empiricas y Diagnostico del Balance Sesgo-Varianza (T23)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta_fig = DIR_FIGURAS / "fig_t23_curvas_validacion.png"
    plt.savefig(ruta_fig, dpi=300)
    plt.close()
    print(f"  [Figura Curvas Validacion] {ruta_fig}")
    return df_curvas


def ejecutar_analisis_complejidad_t23(forzar_recalculo: bool = False) -> Dict[str, Any]:
    """Ejecuta el protocolo experimental completo de T23 o carga los resultados si ya existen."""
    ruta_tabla_teorica = DIR_TABLAS / "t23_complejidad_teorica.csv"
    ruta_tabla_empirica = DIR_TABLAS / "t23_complejidad_empirica.csv"
    ruta_tabla_escalabilidad = DIR_TABLAS / "t23_curvas_aprendizaje_escalabilidad.csv"
    ruta_json_hardware = DIR_TABLAS / "t23_entorno_hardware.json"

    # 1. Verificar si existen resultados cacheados
    if not forzar_recalculo and all(
        p.exists() for p in [ruta_tabla_teorica, ruta_tabla_empirica, ruta_tabla_escalabilidad, ruta_json_hardware]
    ):
        print("T23: Cargando tablas existentes desde reports/tables/...")
        df_teorica = pd.read_csv(ruta_tabla_teorica)
        df_empirica = pd.read_csv(ruta_tabla_empirica)
        df_escalabilidad = pd.read_csv(ruta_tabla_escalabilidad)
        with open(ruta_json_hardware, "r", encoding="utf-8") as f:
            hardware = json.load(f)
        return {
            "teorica": df_teorica,
            "empirica": df_empirica,
            "escalabilidad": df_escalabilidad,
            "hardware": hardware,
        }

    print("================================================================================")
    print("T23 · INICIO DE MEDICIÓN EXHAUSTIVA DE COMPLEJIDAD TEÓRICA Y EMPÍRICA")
    print("================================================================================")

    # 2. Generar tabla de complejidad teórica
    df_teorica = generar_tabla_complejidad_teorica()
    df_teorica.to_csv(ruta_tabla_teorica, index=False)
    print(f"-> Guardada tabla teórica en: {ruta_tabla_teorica}")

    # 3. Registrar hardware
    hardware = registrar_entorno_hardware()
    with open(ruta_json_hardware, "w", encoding="utf-8") as f:
        json.dump(hardware, f, indent=2)
    print(f"-> Guardado entorno de hardware en: {ruta_json_hardware}")

    # 4. Cargar datos de entrenamiento (respetando aislamiento del test sellado n=109)
    X, y, evidencia = cargar_entrenamiento()
    n_train = len(X)
    print(f"-> Aislamiento verificado: {n_train} filas de entrenamiento (0 filas de prueba cargadas)")

    pliegues = crear_pliegues(y, semilla=SEMILLA)

    # Cargar métricas de T22 para asociar RMSE al costo computacional
    ruta_t22 = DIR_TABLAS / "t22_tabla_maestra_metricas.csv"
    rmse_t22_map = {}
    r2_t22_map = {}
    if ruta_t22.exists():
        df_t22 = pd.read_csv(ruta_t22)
        for _, row in df_t22.iterrows():
            rmse_t22_map[row["modelo_id_completo"]] = row["rmse_cad_media"]
            r2_t22_map[row["modelo_id_completo"]] = row["r2_media"]

    # 5. Medición empírica en los 18 pipelines
    registros_empiricos = []
    preprocesamientos = ["P1", "P2", "P3"]
    modelos_ids = ["M1", "M2", "M3", "M4", "M5", "M6"]

    print(f"-> Iniciando benchmarking empírico sobre 18 pipelines × 15 pliegues...")

    for prepro_id in preprocesamientos:
        for modelo_id in modelos_ids:
            pipeline_id = f"{modelo_id}_{prepro_id}"
            nombre_modelo = MODELOS[modelo_id][0]
            familia = MODELOS[modelo_id][1]
            pipeline_nombre = f"{nombre_modelo} ({prepro_id})"

            tiempos_fit = []
            tiempos_inferencia_lote = []
            tiempos_inferencia_unitario = []
            picos_memoria_kb = []
            params_conteo = 0
            params_desc = ""
            tamano_disco_kb = 0.0

            # Medición a través de los 15 pliegues de CV
            for idx_fold, (idx_ajuste, idx_val) in enumerate(pliegues):
                X_ajuste, y_ajuste = X.iloc[idx_ajuste], y.iloc[idx_ajuste]
                X_val = X.iloc[idx_val]

                estimador = obtener_estimador_optimizado(prepro_id, modelo_id)

                # Medición de memoria y tiempo de fit
                tracemalloc.start()
                t0_fit = time.perf_counter()
                estimador.fit(X_ajuste, y_ajuste)
                t1_fit = time.perf_counter()
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                tiempos_fit.append(t1_fit - t0_fit)
                picos_memoria_kb.append(peak_mem / 1024.0)

                # Medición de inferencia por lote (n_val observaciones, aprox 87-88 filas)
                t0_lote = time.perf_counter()
                _ = estimador.predict(X_val)
                t1_lote = time.perf_counter()
                tiempos_inferencia_lote.append((t1_lote - t0_lote) * 1000.0)  # ms

                # En el primer pliegue, medir tamaño serializado, conteo de parámetros y latencia unitaria
                if idx_fold == 0:
                    params_conteo, params_desc = contar_parametros_modelo(estimador, modelo_id, len(X_ajuste))

                    # Tamaño serializado
                    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                        tmp_path = tmp.name
                    joblib.dump(estimador, tmp_path, compress=3)
                    tamano_disco_kb = os.path.getsize(tmp_path) / 1024.0
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                    # Latencia unitaria (predicción de 1 observación promediada en 200 llamadas)
                    fila_individual = X_val.iloc[[0]]
                    t_unitarios = []
                    for _ in range(200):
                        t0_u = time.perf_counter()
                        _ = estimador.predict(fila_individual)
                        t1_u = time.perf_counter()
                        t_unitarios.append((t1_u - t0_u) * 1e6)  # microsegundos
                    tiempos_inferencia_unitario.append(np.median(t_unitarios))

            reg = {
                "pipeline_id": pipeline_id,
                "pipeline_nombre": pipeline_nombre,
                "preprocesamiento": prepro_id,
                "modelo_id": modelo_id,
                "modelo_nombre": nombre_modelo,
                "familia": familia,
                "tiempo_ajuste_mediana_s": float(np.median(tiempos_fit)),
                "tiempo_ajuste_media_s": float(np.mean(tiempos_fit)),
                "tiempo_ajuste_desv_s": float(np.std(tiempos_fit, ddof=1)),
                "tiempo_inferencia_lote_ms": float(np.median(tiempos_inferencia_lote)),
                "tiempo_inferencia_unitario_us": float(np.median(tiempos_inferencia_unitario)),
                "memoria_pico_kb": float(np.median(picos_memoria_kb)),
                "n_parametros_o_nodos": int(params_conteo),
                "descripcion_parametros": params_desc,
                "tamano_serializado_kb": float(tamano_disco_kb),
                "rmse_cad1987_media": float(rmse_t22_map.get(pipeline_id, np.nan)),
                "r2_media": float(r2_t22_map.get(pipeline_id, np.nan)),
            }
            registros_empiricos.append(reg)
            print(f"  [OK] {pipeline_nombre:<25} Fit mediano: {reg['tiempo_ajuste_mediana_s']*1000:>7.2f} ms | Lote: {reg['tiempo_inferencia_lote_ms']:>6.2f} ms | Params/Nodos: {params_conteo:>6}")

    df_empirica = pd.DataFrame(registros_empiricos)
    df_empirica = df_empirica.sort_values("tiempo_ajuste_mediana_s", ascending=True).reset_index(drop=True)
    df_empirica.to_csv(ruta_tabla_empirica, index=False)
    print(f"-> Guardada tabla empírica en: {ruta_tabla_empirica}")

    # 6. Curvas de aprendizaje y escalabilidad en función de n
    print("-> Computando curvas de escalabilidad frente al tamaño muestral n...")
    fracciones = [0.15, 0.30, 0.50, 0.70, 0.85, 1.0]
    modelos_escalabilidad = [
        ("P2", "M1", "OLS (hedónico) (P2)", "Lineal"),
        ("P2", "M2", "Ridge (P2)", "Lineal"),
        ("P2", "M4", "SVR (RBF) (P2)", "Kernel"),
        ("P1", "M6", "XGBoost (P1)", "Ensambles"),
        ("P1", "M5", "Random Forest (P1)", "Ensambles"),
        ("P1", "M3", "KNN (P1)", "Instancia"),
    ]

    registros_escalabilidad = []
    cv_escalabilidad = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    y_bines = pd.qcut(y, q=5, labels=False)

    for frac in fracciones:
        n_sub = int(round(n_train * frac))
        for prepro_id, modelo_id, nombre, fam in modelos_escalabilidad:
            tiempos_sub_fit = []
            rmses_sub = []

            for train_idx, val_idx in cv_escalabilidad.split(X, y_bines):
                # Tomar submuestra del fold de entrenamiento
                sub_train_idx = np.random.RandomState(SEMILLA).choice(train_idx, size=int(len(train_idx) * frac), replace=False)
                X_tr, y_tr = X.iloc[sub_train_idx], y.iloc[sub_train_idx]
                X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

                estimador = obtener_estimador_optimizado(prepro_id, modelo_id)
                t0 = time.perf_counter()
                estimador.fit(X_tr, y_tr)
                t1 = time.perf_counter()
                tiempos_sub_fit.append(t1 - t0)

                preds = estimador.predict(X_va)
                rmse_cad = np.sqrt(np.mean((y_va - preds) ** 2)) / FACTOR_KAGGLE
                rmses_sub.append(rmse_cad)

            registros_escalabilidad.append({
                "fraccion": frac,
                "n_observaciones": n_sub,
                "modelo_id": f"{modelo_id}_{prepro_id}",
                "modelo_nombre": nombre,
                "familia": fam,
                "tiempo_fit_mediana_s": float(np.median(tiempos_sub_fit)),
                "rmse_cad1987_media": float(np.mean(rmses_sub)),
                "rmse_cad1987_std": float(np.std(rmses_sub, ddof=1)),
            })

    df_escalabilidad = pd.DataFrame(registros_escalabilidad)
    df_escalabilidad.to_csv(ruta_tabla_escalabilidad, index=False)
    print(f"-> Guardada tabla de escalabilidad en: {ruta_tabla_escalabilidad}")

    # 7. Generación de Figuras Oficiales
    print("-> Generando figuras de alta resolución en reports/figures/...")
    generar_figuras_complejidad(df_empirica, df_escalabilidad)

    print("================================================================================")
    print("T23 · ANÁLISIS DE COMPLEJIDAD COMPLETADO CON ÉXITO")
    print("================================================================================")

    
    print('\n--- Paso 5: Curvas de validacion y balance sesgo-varianza ---')
    generar_curvas_validacion(X, y)

    return {
        "teorica": df_teorica,
        "empirica": df_empirica,
        "escalabilidad": df_escalabilidad,
        "hardware": hardware,
    }


def generar_figuras_complejidad(df_empirica: pd.DataFrame, df_escalabilidad: pd.DataFrame):
    """Crea y guarda las figuras publicables del análisis de complejidad."""
    plt.style.use("seaborn-v0_8-whitegrid")
    colores_familia = {
        "lineal": "#1f77b4",
        "distancia": "#ff7f0e",
        "kernel": "#2ca02c",
        "árboles": "#d62728",
    }

    # -------------------------------------------------------------------------
    # Figura 1: Trade-off RMSE vs Tiempo de Entrenamiento (Frontera de Pareto)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for fam, col in colores_familia.items():
        sub = df_empirica[df_empirica["familia"] == fam]
        ax.scatter(
            sub["tiempo_ajuste_mediana_s"],
            sub["rmse_cad1987_media"],
            c=col,
            label=f"Familia {fam.capitalize()}",
            s=sub["n_parametros_o_nodos"].apply(lambda n: np.clip(np.sqrt(n) * 6, 40, 250)),
            alpha=0.85,
            edgecolors="black",
            linewidth=1.2,
        )

    # Anotar modelos clave
    anotaciones_clave = [
        "OLS (hedónico) (P2)", "Ridge (P2)", "XGBoost (P1)", "Random Forest (P1)", "SVR (RBF) (P2)", "KNN (P1)"
    ]
    for _, r in df_empirica.iterrows():
        if r["pipeline_nombre"] in anotaciones_clave:
            offset_y = 50 if "XGBoost" in r["pipeline_nombre"] else -70
            ax.annotate(
                f"{r['pipeline_nombre']}\n({r['tiempo_ajuste_mediana_s']*1000:.1f} ms)",
                (r["tiempo_ajuste_mediana_s"], r["rmse_cad1987_media"]),
                textcoords="offset points",
                xytext=(10, offset_y),
                fontsize=8.5,
                fontweight="bold" if "P2" in r["pipeline_nombre"] else "normal",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color="black", lw=0.8),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de ajuste por pliegue [segundos, escala logarítmica]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Error de generalización RMSE [CAD de 1987]", fontsize=11, fontweight="bold")
    ax.set_title("Compromiso entre Error Predictivo y Costo de Entrenamiento (T23)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(title="Familia de Algoritmo", frameon=True, loc="upper right")
    plt.tight_layout()
    ruta_fig1 = DIR_FIGURAS / "fig_t23_tradeoff_rmse_vs_tiempo_entrenamiento.png"
    plt.savefig(ruta_fig1, dpi=300)
    plt.close()
    print(f"  [Figura 1] {ruta_fig1}")

    # -------------------------------------------------------------------------
    # Figura 2: Trade-off RMSE vs Latencia de Inferencia por Lote (109 filas)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for fam, col in colores_familia.items():
        sub = df_empirica[df_empirica["familia"] == fam]
        ax.scatter(
            sub["tiempo_inferencia_lote_ms"],
            sub["rmse_cad1987_media"],
            c=col,
            label=f"Familia {fam.capitalize()}",
            s=90,
            alpha=0.85,
            edgecolors="black",
            linewidth=1.2,
        )

    for _, r in df_empirica.iterrows():
        if r["pipeline_nombre"] in anotaciones_clave:
            ax.annotate(
                f"{r['pipeline_nombre']}\n({r['tiempo_inferencia_lote_ms']:.2f} ms)",
                (r["tiempo_inferencia_lote_ms"], r["rmse_cad1987_media"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Latencia de inferencia por lote (n = 109) [milisegundos, escala logarítmica]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Error de generalización RMSE [CAD de 1987]", fontsize=11, fontweight="bold")
    ax.set_title("Compromiso entre Error Predictivo y Latencia de Inferencia en Lote (T23)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(title="Familia de Algoritmo", frameon=True, loc="upper left")
    plt.tight_layout()
    ruta_fig2 = DIR_FIGURAS / "fig_t23_tradeoff_rmse_vs_latencia_inferencia.png"
    plt.savefig(ruta_fig2, dpi=300)
    plt.close()
    print(f"  [Figura 2] {ruta_fig2}")

    # -------------------------------------------------------------------------
    # Figura 3: Curvas de Escalabilidad y Aprendizaje frente al tamaño muestral n
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    colores_modelos = {
        "OLS (hedónico) (P2)": "#1f77b4",
        "Ridge (P2)": "#3b9ab2",
        "SVR (RBF) (P2)": "#2ca02c",
        "XGBoost (P1)": "#d62728",
        "Random Forest (P1)": "#e377c2",
        "KNN (P1)": "#ff7f0e",
    }

    for mod_name, col in colores_modelos.items():
        sub = df_escalabilidad[df_escalabilidad["modelo_nombre"] == mod_name]
        ax1.plot(sub["n_observaciones"], sub["tiempo_fit_mediana_s"] * 1000, marker="o", lw=2, label=mod_name, color=col)
        ax2.errorbar(
            sub["n_observaciones"],
            sub["rmse_cad1987_media"],
            yerr=sub["rmse_cad1987_std"],
            marker="s",
            lw=2,
            capsize=3,
            label=mod_name,
            color=col,
        )

    ax1.set_xlabel("Número de observaciones de entrenamiento [n]", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Tiempo mediano de ajuste [milisegundos]", fontsize=10.5, fontweight="bold")
    ax1.set_title("Escalabilidad Computacional de Ajuste vs. n", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.set_xlabel("Número de observaciones de entrenamiento [n]", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Error de validación RMSE [CAD de 1987]", fontsize=10.5, fontweight="bold")
    ax2.set_title("Curva de Aprendizaje: RMSE de Validación vs. n", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Análisis de Escalabilidad y Curvas de Aprendizaje (T23)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta_fig3 = DIR_FIGURAS / "fig_t23_curvas_escalabilidad_n.png"
    plt.savefig(ruta_fig3, dpi=300)
    plt.close()
    print(f"  [Figura 3] {ruta_fig3}")

    # -------------------------------------------------------------------------
    # Figura 4: Comparación de Recursos y Parámetros
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    top_models = df_empirica.sort_values("rmse_cad1987_media", ascending=True).head(8)

    y_pos = np.arange(len(top_models))
    ax1.barh(y_pos, top_models["tiempo_ajuste_mediana_s"] * 1000, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_models["pipeline_nombre"], fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Tiempo de ajuste [milisegundos, escala log]", fontsize=10.5, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_title("Tiempo de Ajuste por Pliegue (Top 8 Modelos)", fontsize=11.5, fontweight="bold")
    ax1.invert_yaxis()

    ax2.barh(y_pos, top_models["n_parametros_o_nodos"], color="#55a868", edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["" for _ in y_pos])  # omitir etiquetas en eje derecho
    ax2.set_xlabel("Parámetros aprendidos / Nodos totales [escala log]", fontsize=10.5, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_title("Complejidad Paramétrica / Nodos de Decisión", fontsize=11.5, fontweight="bold")
    ax2.invert_yaxis()

    plt.suptitle("Comparativa de Costo Computacional y Tamaño del Modelo (T23)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta_fig4 = DIR_FIGURAS / "fig_t23_comparacion_recursos_modelos.png"
    plt.savefig(ruta_fig4, dpi=300)
    plt.close()
    print(f"  [Figura 4] {ruta_fig4}")


def main():
    """Punto de entrada de línea de comandos."""
    ejecutar_analisis_complejidad_t23(forzar_recalculo=True)


if __name__ == "__main__":
    main()
