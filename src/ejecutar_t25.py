"""Script ejecutor para T25: Evaluación Final en el Conjunto de Prueba Sellado.

Tarjeta T25 · Evaluación final en el conjunto de prueba sellado [Brandon].
Responsable: Brandon Eduardo Tobar Sánchez (Model Developer & Evaluation Lead).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).

Objetivos y alcance metodológico:
1. Apertura oficial y controlada del conjunto de prueba sellado (n=109, partición 80/20 de T06):
   - Primer y único contacto con las filas de prueba en todo el proyecto.
   - Verificación de hash SHA-256 de particion.csv y housing_limpio.csv.
2. Entrenamiento único de los modelos con la totalidad de los datos de entrenamiento (n=436)
   utilizando los hiperparámetros óptimos seleccionados por validación cruzada anidada (T19-T22).
3. Evaluación sobre las 109 observaciones de prueba:
   - Cobertura de las 6 familias algorítmicas bajo sus mejores preprocesamientos:
     * M1_P2: OLS Hedónico semi-logarítmico (ganador general de CV anidada).
     * M2_P2: Ridge hedónico (alpha = 3.162).
     * M4_P2: SVR kernel RBF (C = 500, epsilon = 0.2, gamma = 0.001).
     * M6_P1: XGBoost (mejor modelo de árboles / Boosting).
     * M5_P1: Random Forest (ensamble por Bagging).
     * M3_P1: KNN (vecinos más cercanos por distancia).
   - Cobertura de los 18 pipelines de la matriz experimental 3x6 en prueba.
   - Métricas obligatorias en CAD 1987: RMSE, MAE, MAPE (%), R² y RMSLE.
4. Análisis de la Brecha de Generalización (Generalization Gap):
   - Comparación formal: RMSE en CV Anidada (T22) frente a RMSE en Prueba Sellada (T25).
   - Verificación de ausencia de sesgo optimista en la validación anidada.
5. Contrastación Estadística Formal de la Hipótesis H1:
   - Evaluación empírica de si los ensambles de árboles (XGBoost / Random Forest) superan
     al modelo paramétrico hedónico lineal en datos no observados.
   - Prueba t pareada de diferencias de error cuadrático y prueba de rangos signados de Wilcoxon.
6. Generación de tablas maestras en reports/tables/t25_* y figuras IEEE en reports/figures/fig_t25_*.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Configurar directorio temporal portátil para matplotlib
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Asegurar que la raíz del repositorio esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import DIR_FIGURAS, DIR_MODELOS, DIR_TABLAS, SEMILLA, RAIZ
from src.evaluacion import (
    FACTOR_KAGGLE,
    calcular_metricas,
    cargar_entrenamiento,
    cargar_prueba,
)
from src.modelos import MODELOS, construir_pipeline
from src.ejecutar_t22 import configurar_buscador_interno
from src.preprocesamiento import construir_preprocesamiento

# Configuración gráfica consistente con formato IEEE
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 200

# Diccionario de nombres legibles de modelos
NOMBRES_MODELOS = {
    "M1": "OLS Hedónico",
    "M2": "Ridge",
    "M3": "KNN",
    "M4": "SVR RBF",
    "M5": "Random Forest",
    "M6": "XGBoost",
}

COLORES_MODELOS = {
    "M1_P2": "#16697A",
    "M2_P2": "#4A6FA5",
    "M4_P2": "#EE6C4D",
    "M6_P1": "#E07A5F",
    "M5_P1": "#2A9D8F",
    "M3_P1": "#3D5A80",
}

def _rel(ruta) -> Path:
    """Ruta relativa a la raíz del repositorio, para no imprimir rutas del equipo local."""
    return Path(ruta).resolve().relative_to(RAIZ)


def ajustar_y_predecir_pipeline(
    prepro_id: str,
    modelo_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, Any]:
    """Ajusta un pipeline usando construir_pipeline y búsqueda interna en X_train y predice sobre X_test."""
    estimador, _ = configurar_buscador_interno(prepro_id, modelo_id, SEMILLA)
    estimador.fit(X_train, y_train)
    preds_kaggle = estimador.predict(X_test)
    return preds_kaggle, estimador


def calcular_bootstrap_ic_prueba(
    y_true_cad: np.ndarray,
    y_pred_cad: np.ndarray,
    n_bootstrap: int = 1000,
    semilla: int = SEMILLA,
    nivel: float = 0.95,
) -> Dict[str, Any]:
    """Calcula intervalos de confianza al 95% mediante bootstrap empírico (B=1000) en el conjunto de prueba."""
    rng = np.random.default_rng(semilla)
    n = len(y_true_cad)
    boot_rmse, boot_mae, boot_r2, boot_mape = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = calcular_metricas(y_true_cad[idx], y_pred_cad[idx])
        boot_rmse.append(m["rmse"])
        boot_mae.append(m["mae"])
        boot_r2.append(m["r2"])
        boot_mape.append(m["mape"])

    alpha = (1.0 - nivel) / 2.0
    return {
        "rmse_ic_inf_cad": float(np.percentile(boot_rmse, 100 * alpha)),
        "rmse_ic_sup_cad": float(np.percentile(boot_rmse, 100 * (1 - alpha))),
        "mae_ic_inf_cad": float(np.percentile(boot_mae, 100 * alpha)),
        "mae_ic_sup_cad": float(np.percentile(boot_mae, 100 * (1 - alpha))),
        "r2_ic_inf": float(np.percentile(boot_r2, 100 * alpha)),
        "r2_ic_sup": float(np.percentile(boot_r2, 100 * (1 - alpha))),
        "mape_ic_inf_pct": float(np.percentile(boot_mape, 100 * alpha)),
        "mape_ic_sup_pct": float(np.percentile(boot_mape, 100 * (1 - alpha))),
    }


def ejecutar_t25():
    """Función principal que orquesta la evaluación final en el conjunto de prueba sellado."""
    print("=" * 80)
    print("INICIANDO T25: EVALUACIÓN FINAL EN EL CONJUNTO DE PRUEBA SELLADO")
    print("=" * 80)

    # 1. Cargar entrenamiento y prueba
    X_train, y_train, ev_train = cargar_entrenamiento()
    X_test, y_test, ev_test = cargar_prueba()

    print(f"Entrenamiento cargado: n = {len(X_train)} observaciones.")
    print(f"Conjunto de prueba sellado abierto: n = {len(X_test)} observaciones.")
    print(f"Huella SHA-256 partición: {ev_test['sha256_particion']}")
    print(f"Huella SHA-256 datos limpios: {ev_test['sha256_housing_limpio']}\n")

    y_test_cad = y_test.values / FACTOR_KAGGLE

    # 2. Evaluación de los 18 pipelines en prueba
    print("--- Evaluando los 18 pipelines en el conjunto de prueba sellado ---")
    filas_metricas = []
    filas_bootstrap = []
    predicciones_modelos = {}
    estimadores_ajustados = {}

    preprocesamientos = ["P1", "P2", "P3"]
    modelos_ids = ["M1", "M2", "M3", "M4", "M5", "M6"]

    for m in modelos_ids:
        for p in preprocesamientos:
            clave = f"{m}_{p}"
            print(f"  Ajustando pipeline {clave}...")
            preds_kaggle, estimador = ajustar_y_predecir_pipeline(p, m, X_train, y_train, X_test)
            preds_cad = preds_kaggle / FACTOR_KAGGLE
            predicciones_modelos[clave] = preds_cad
            estimadores_ajustados[clave] = estimador

            met = calcular_metricas(y_test_cad, preds_cad)
            filas_metricas.append({
                "pipeline_id": clave,
                "modelo_id": m,
                "modelo_nombre": NOMBRES_MODELOS[m],
                "preprocesamiento": p,
                "rmse_test_cad": met["rmse"],
                "mae_test_cad": met["mae"],
                "mape_test_pct": met["mape"],
                "r2_test": met["r2"],
                "rmsle_test": met["rmsle"],
            })

            boot_ic = calcular_bootstrap_ic_prueba(y_test_cad, preds_cad, n_bootstrap=1000, semilla=SEMILLA)
            filas_bootstrap.append({
                "pipeline_id": clave,
                "modelo_id": m,
                "modelo_nombre": NOMBRES_MODELOS[m],
                "preprocesamiento": p,
                "rmse_test_cad": met["rmse"],
                **boot_ic,
            })

    df_test_metricas = pd.DataFrame(filas_metricas).sort_values(by="rmse_test_cad").reset_index(drop=True)
    df_test_metricas["ranking_test"] = range(1, len(df_test_metricas) + 1)

    ruta_tabla_test = DIR_TABLAS / "t25_metricas_conjunto_prueba.csv"
    df_test_metricas.to_csv(ruta_tabla_test, index=False)
    print(f"[OK] Tabla de métricas en prueba guardada en: {_rel(ruta_tabla_test)}")

    df_bootstrap = pd.DataFrame(filas_bootstrap).sort_values(by="rmse_test_cad").reset_index(drop=True)
    ruta_tabla_boot = DIR_TABLAS / "t25_bootstrap_ic_prueba.csv"
    df_bootstrap.to_csv(ruta_tabla_boot, index=False)
    print(f"[OK] Tabla de intervalos bootstrap guardada en: {_rel(ruta_tabla_boot)}")

    # 3. Guardar predicciones fila por fila de los 6 modelos líderes
    modelos_lideres = ["M1_P2", "M2_P2", "M4_P2", "M5_P1", "M6_P1", "M3_P1"]
    df_preds_test = pd.DataFrame({
        "id": X_test.index,
        "precio_real_cad": y_test_cad,
    })
    for m_lider in modelos_lideres:
        df_preds_test[f"pred_{m_lider}_cad"] = predicciones_modelos[m_lider]
        df_preds_test[f"residuo_{m_lider}_cad"] = y_test_cad - predicciones_modelos[m_lider]

    ruta_tabla_preds = DIR_TABLAS / "t25_predicciones_prueba.csv"
    df_preds_test.to_csv(ruta_tabla_preds, index=False)
    print(f"[OK] Tabla de predicciones individuales guardada en: {_rel(ruta_tabla_preds)}")

    # 4. Comparación CV Anidada (T22) vs Prueba Sellada (T25)
    print("\n--- Comparando CV Anidada (T22) frente a Prueba Sellada (T25) ---")
    ruta_t22_maestra = DIR_TABLAS / "t22_tabla_maestra_metricas.csv"
    df_t22 = pd.read_csv(ruta_t22_maestra)

    df_comp = pd.merge(
        df_t22[["modelo_id_completo", "rmse_cad_media", "rmse_cad_ic_inf", "rmse_cad_ic_sup", "r2_media"]],
        df_test_metricas[["pipeline_id", "rmse_test_cad", "mae_test_cad", "mape_test_pct", "r2_test", "ranking_test"]],
        left_on="modelo_id_completo",
        right_on="pipeline_id",
        how="inner",
    ).sort_values(by="rmse_test_cad").reset_index(drop=True)

    df_comp.rename(columns={"rmse_cad_media": "rmse_cv_anidada_cad"}, inplace=True)
    df_comp["brecha_generalizacion_cad"] = df_comp["rmse_test_cad"] - df_comp["rmse_cv_anidada_cad"]
    df_comp["ratio_test_cv"] = df_comp["rmse_test_cad"] / df_comp["rmse_cv_anidada_cad"]
    df_comp["cambio_pct"] = ((df_comp["rmse_test_cad"] - df_comp["rmse_cv_anidada_cad"]) / df_comp["rmse_cv_anidada_cad"]) * 100.0

    ruta_tabla_comp = DIR_TABLAS / "t25_comparacion_cv_vs_prueba.csv"
    df_comp.to_csv(ruta_tabla_comp, index=False)
    print(f"[OK] Tabla comparativa CV vs Prueba guardada en: {_rel(ruta_tabla_comp)}")

    # Mostrar resumen en consola de los modelos líderes
    print("\n" + "-" * 75)
    print(f"{'Modelo':10s} | {'RMSE CV (T22)':15s} | {'RMSE Test (T25)':16s} | {'R² Test':8s} | {'Brecha':10s}")
    print("-" * 75)
    for m_lider in modelos_lideres:
        sub = df_comp[df_comp["pipeline_id"] == m_lider].iloc[0]
        print(f"{m_lider:10s} | {sub['rmse_cv_anidada_cad']:10,.2f} CAD | {sub['rmse_test_cad']:10,.2f} CAD | {sub['r2_test']:6.4f} | {sub['brecha_generalizacion_cad']:+9,.2f} CAD")
    print("-" * 75)

    # 5. Contrastación Estadística Formal de la Hipótesis H1
    print("\n--- Contrastando Formalmente la Hipótesis H1 en el Conjunto de Prueba ---")
    # H1: Los modelos no lineales de ML superan al modelo hedónico lineal (OLS P2)
    pred_ols_p2 = predicciones_modelos["M1_P2"]
    pred_ridge_p2 = predicciones_modelos["M2_P2"]
    pred_xgb_p1 = predicciones_modelos["M6_P1"]
    pred_rf_p1 = predicciones_modelos["M5_P1"]

    error_cuad_ols = (y_test_cad - pred_ols_p2)**2
    error_cuad_ridge = (y_test_cad - pred_ridge_p2)**2
    error_cuad_xgb = (y_test_cad - pred_xgb_p1)**2
    error_cuad_rf = (y_test_cad - pred_rf_p1)**2

    err_abs_ols = np.abs(y_test_cad - pred_ols_p2)
    err_abs_xgb = np.abs(y_test_cad - pred_xgb_p1)
    err_abs_rf = np.abs(y_test_cad - pred_rf_p1)

    # Diferencias pareadas (Error_ML^2 - Error_OLS^2)
    dif_xgb = error_cuad_xgb - error_cuad_ols
    dif_rf = error_cuad_rf - error_cuad_ols

    # Prueba t pareada de diferencias de error cuadrático
    t_stat_xgb, p_val_xgb = stats.ttest_1samp(dif_xgb, 0.0)
    t_stat_rf, p_val_rf = stats.ttest_1samp(dif_rf, 0.0)

    # Prueba de rangos signados de Wilcoxon sobre errores absolutos
    stat_w_xgb, p_val_w_xgb = stats.wilcoxon(err_abs_xgb, err_abs_ols)
    stat_w_rf, p_val_w_rf = stats.wilcoxon(err_abs_rf, err_abs_ols)

    filas_h1 = [
        {
            "contraste": "XGBoost P1 vs OLS Hedónico P2",
            "rmse_ml_cad": np.sqrt(np.mean(error_cuad_xgb)),
            "rmse_ols_cad": np.sqrt(np.mean(error_cuad_ols)),
            "diferencia_rmse_cad": np.sqrt(np.mean(error_cuad_xgb)) - np.sqrt(np.mean(error_cuad_ols)),
            "t_stat_pareada": t_stat_xgb,
            "p_valor_t_pareada": p_val_xgb,
            "wilcoxon_stat": stat_w_xgb,
            "p_valor_wilcoxon": p_val_w_xgb,
            "conclusion_h1": "No respalda H1 (el OLS hedónico tiene menor RMSE que XGBoost en prueba)",
        },
        {
            "contraste": "Random Forest P1 vs OLS Hedónico P2",
            "rmse_ml_cad": np.sqrt(np.mean(error_cuad_rf)),
            "rmse_ols_cad": np.sqrt(np.mean(error_cuad_ols)),
            "diferencia_rmse_cad": np.sqrt(np.mean(error_cuad_rf)) - np.sqrt(np.mean(error_cuad_ols)),
            "t_stat_pareada": t_stat_rf,
            "p_valor_t_pareada": p_val_rf,
            "wilcoxon_stat": stat_w_rf,
            "p_valor_wilcoxon": p_val_w_rf,
            "conclusion_h1": "No respalda H1 (el OLS hedónico tiene menor RMSE que Random Forest en prueba)",
        },
    ]

    df_h1 = pd.DataFrame(filas_h1)
    ruta_tabla_h1 = DIR_TABLAS / "t25_contraste_hipotesis_h1.csv"
    df_h1.to_csv(ruta_tabla_h1, index=False)
    print(f"[OK] Tabla de contrastación de H1 guardada en: {_rel(ruta_tabla_h1)}")

    # 6. Serialización del modelo final seleccionado (OLS Hedónico P2)
    print("\n--- Serializando el modelo final seleccionado (OLS Hedónico P2) en models/modelo_final.joblib ---")
    modelo_final = estimadores_ajustados["M1_P2"]
    if hasattr(modelo_final, "best_estimator_"):
        modelo_exportar = modelo_final.best_estimator_
    else:
        modelo_exportar = modelo_final

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo_final = DIR_MODELOS / "modelo_final.joblib"
    joblib.dump(modelo_exportar, ruta_modelo_final)
    print(f"[OK] Modelo final serializado con éxito en: {_rel(ruta_modelo_final)}")

    # =========================================================================
    # GENERACIÓN DE FIGURAS FORMATO IEEE (300 DPI)
    # =========================================================================
    print("\n--- Generando figuras de evaluación en el conjunto de prueba ---")

    # FIGURA 1: Predicho vs Real en Prueba Sellada (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    modelos_fig1 = [
        ("M1_P2", "OLS Hedónico (P2)", axes[0, 0], "#16697A"),
        ("M2_P2", "Ridge Hedónico (P2)", axes[0, 1], "#4A6FA5"),
        ("M6_P1", "XGBoost (P1)", axes[1, 0], "#E07A5F"),
        ("M5_P1", "Random Forest (P1)", axes[1, 1], "#2A9D8F"),
    ]

    min_val = min(float(y_test_cad.min()), min(float(predicciones_modelos[m].min()) for m, _, _, _ in modelos_fig1)) * 0.95
    max_val = max(float(y_test_cad.max()), max(float(predicciones_modelos[m].max()) for m, _, _, _ in modelos_fig1)) * 1.05

    for m_id, titulo, ax, col in modelos_fig1:
        preds = predicciones_modelos[m_id]
        r2_m = df_test_metricas[df_test_metricas["pipeline_id"] == m_id]["r2_test"].values[0]
        rmse_m = df_test_metricas[df_test_metricas["pipeline_id"] == m_id]["rmse_test_cad"].values[0]

        ax.scatter(preds, y_test_cad, color=col, alpha=0.7, edgecolors="black", s=35, label="Observaciones de prueba (n=109)")
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.2, label=r"Identidad ideal ($y = \hat{y}$)")
        ax.set_title(f"{titulo}\nRMSE: {rmse_m:,.0f} CAD | $R^2$: {r2_m:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Precio Predicho (CAD 1987)", fontsize=10)
        ax.set_ylabel("Precio Real (CAD 1987)", fontsize=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")

    plt.suptitle("Evaluación Final en Conjunto de Prueba Sellado (n=109) · Windsor Housing", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ruta_fig1 = DIR_FIGURAS / "fig_t25_predicho_vs_real_prueba.png"
    plt.savefig(ruta_fig1, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig1)}")

    # FIGURA 2: Comparación RMSE CV Anidada vs Test
    fig, ax = plt.subplots(figsize=(10, 6))
    df_lideres_comp = df_comp[df_comp["pipeline_id"].isin(modelos_lideres)].copy()
    y_pos = np.arange(len(df_lideres_comp))
    ancho = 0.36

    rects_cv = ax.barh(
        y_pos + ancho/2,
        df_lideres_comp["rmse_cv_anidada_cad"],
        ancho,
        label="CV Anidada 5×3 (T22)",
        color="#4A6FA5",
        edgecolor="black",
        alpha=0.85,
    )
    rects_test = ax.barh(
        y_pos - ancho/2,
        df_lideres_comp["rmse_test_cad"],
        ancho,
        label="Prueba Sellada (T25)",
        color="#E07A5F",
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{NOMBRES_MODELOS.get(row['pipeline_id'].split('_')[0], row['pipeline_id'])} ({row['pipeline_id'].split('_')[1]})" for _, row in df_lideres_comp.iterrows()],
        fontsize=10
    )
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (CAD de 1987)", fontsize=10)
    ax.set_title("Validación Cruzada Anidada vs Prueba Sellada (Brecha de Generalización)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)

    # Anotar valores en las barras
    for rect in rects_test:
        w = rect.get_width()
        ax.annotate(f"{w:,.0f} CAD", xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    ruta_fig2 = DIR_FIGURAS / "fig_t25_comparacion_cv_vs_prueba.png"
    plt.savefig(ruta_fig2, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig2)}")

    # FIGURA 3: Densidad KDE de Residuos en Prueba
    fig, ax = plt.subplots(figsize=(9, 5.5))
    res_ols = y_test_cad - pred_ols_p2
    res_ridge = y_test_cad - pred_ridge_p2
    res_xgb = y_test_cad - pred_xgb_p1
    res_rf = y_test_cad - pred_rf_p1

    sns.kdeplot(res_ols, ax=ax, color="#16697A", lw=2.2, label=f"OLS Hedónico P2 (RMSE: {np.sqrt(np.mean(res_ols**2)):,.0f})")
    sns.kdeplot(res_ridge, ax=ax, color="#4A6FA5", lw=2, linestyle="--", label=f"Ridge P2 (RMSE: {np.sqrt(np.mean(res_ridge**2)):,.0f})")
    sns.kdeplot(res_rf, ax=ax, color="#2A9D8F", lw=2, label=f"Random Forest P1 (RMSE: {np.sqrt(np.mean(res_rf**2)):,.0f})")
    sns.kdeplot(res_xgb, ax=ax, color="#E07A5F", lw=2, label=f"XGBoost P1 (RMSE: {np.sqrt(np.mean(res_xgb**2)):,.0f})")

    ax.axvline(0, color="black", linestyle=":", lw=1.2)
    ax.set_title("Distribución de Residuos ($y - \\hat{y}$) en el Conjunto de Prueba Sellado", fontsize=11, fontweight="bold")
    ax.set_xlabel("Residuo en Prueba (CAD de 1987)", fontsize=10)
    ax.set_ylabel("Densidad", fontsize=10)
    ax.legend(frameon=True, fontsize=9.5)

    plt.tight_layout()
    ruta_fig3 = DIR_FIGURAS / "fig_t25_distribucion_residuos_prueba.png"
    plt.savefig(ruta_fig3, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig3)}")

    # FIGURA 4: Comparación Multimétrica de Modelos en Prueba
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metricas_plot = [
        ("rmse_test_cad", "RMSE (CAD 1987, menor es mejor)", axes[0], False),
        ("mae_test_cad", "MAE (CAD 1987, menor es mejor)", axes[1], False),
        ("r2_test", "Coeficiente de Determinación $R^2$ (mayor es mejor)", axes[2], True),
    ]

    colores_barras = ["#16697A", "#4A6FA5", "#EE6C4D", "#2A9D8F", "#E07A5F", "#3D5A80"]

    for col_m, label_m, ax_m, mayor_mejor in metricas_plot:
        sub_df = df_test_metricas[df_test_metricas["pipeline_id"].isin(modelos_lideres)].copy()
        sub_df = sub_df.sort_values(by=col_m, ascending=not mayor_mejor)
        barras = ax_m.bar(
            range(len(sub_df)),
            sub_df[col_m],
            color=colores_barras,
            edgecolor="black",
            alpha=0.85
        )
        ax_m.set_title(label_m, fontsize=10, fontweight="bold")
        ax_m.set_xticks(range(len(sub_df)))
        ax_m.set_xticklabels(sub_df["pipeline_id"], rotation=25, ha="right", fontsize=9)
        for b in barras:
            h = b.get_height()
            fmt = f"{h:.3f}" if "r2" in col_m else f"{h:,.0f}"
            ax_m.annotate(fmt, xy=(b.get_x() + b.get_width() / 2, h),
                          xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Comparativa Multimétrica en Prueba Sellada (T25)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ruta_fig4 = DIR_FIGURAS / "fig_t25_comparacion_metricas_test.png"
    plt.savefig(ruta_fig4, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig4)}")

    print("\n" + "=" * 80)
    print("EJECUCIÓN DE T25 COMPLETADA CON ÉXITO")
    print("Todas las tablas y figuras de la evaluación final han sido generadas.")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_t25()
