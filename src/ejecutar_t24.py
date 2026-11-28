"""Script ejecutor para T24: Análisis de Errores e Interpretabilidad (SHAP).

Tarjeta T24 · Análisis de errores e interpretabilidad (SHAP) [Brandon].
Responsable: Brandon Eduardo Tobar Sánchez (Model Developer & Evaluation Lead).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).

Objetivos y alcance metodológico:
1. Análisis riguroso de residuos de los modelos líderes identificados en T22:
   - OLS hedónico semi-logarítmico (M1_P2) y Ridge (M2_P2).
   - XGBoost (M6_P1) y Random Forest (M5_P1).
   - Diagnósticos: Predicho vs Real, Residuos vs Predicho, normalidad (Shapiro-Wilk, D'Agostino),
     asimetría, curtosis y heterocedasticidad (correlación Spearman error-predicción).
   - Detección y caracterización física de viviendas atípicas (top sobrepredichas y subpredichas).
2. Interpretabilidad global post-hoc mediante TreeExplainer (Lundberg & Lee, 2017):
   - Modelos de ensambles basados en árboles: XGBoost P1 y Random Forest P1.
   - Ranking por Mean Absolute SHAP (CAD 1987) y porcentaje relativo.
   - Gráfico Beeswarm (summary plot) revelando dirección y magnitud del impacto.
   - Gráficos de dependencia SHAP para detectar no linealidades (saturación de área) y saltos discretos.
3. Interpretabilidad local detallada:
   - Desglose atributivo (waterfall / barras locales) para 4 arquetipos de vivienda:
     (a) Vivienda mediana representativa.
     (b) Vivienda de alto valor / lujo.
     (c) Vivienda con máxima subpredicción.
     (d) Vivienda con máxima sobrepredicción.
4. Contraste epistemológico y empírico:
   - Coeficientes econométricos hedónicos intrínsecos (semi-elasticidad 100(exp(beta)-1)% y beta estandarizado)
     frente al orden de impacto algorítmico SHAP.
   - Cumplimiento de las recomendaciones docentes (A2 y A3): la explicabilidad mide el
     impacto de las predictoras en la salida del algoritmo, sin atribuir causalidad ni describir
     el mecanismo de equilibrio en el mercado inmobiliario.
5. Generación de tablas maestras en reports/tables/t24_* y figuras IEEE en reports/figures/fig_t24_*.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import DIR_FIGURAS, DIR_TABLAS, SEMILLA, RAIZ
from src.evaluacion import FACTOR_KAGGLE, cargar_entrenamiento, crear_pliegues
from src.preprocesamiento import construir_preprocesamiento

# Configuración gráfica consistente con formato IEEE
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 200

# Diccionario de nombres legibles para gráficos y tablas
NOMBRES_VARIABLES = {
    "area": "Área del lote (sq ft)",
    "bedrooms": "Dormitorios",
    "bathrooms": "Baños",
    "stories": "Pisos (plantas)",
    "parking": "Plazas de estacionamiento",
    "mainroad": "Acceso a vía principal",
    "guestroom": "Habitación de huéspedes",
    "basement": "Sótano",
    "hotwaterheating": "Calefacción por agua",
    "airconditioning": "Aire acondicionado",
    "prefarea": "Zona de alta plusvalía",
    "furnishingstatus_1": "Amoblado parcial (semi-furn.)",
    "furnishingstatus_2": "Amoblado completo (furn.)",
}

# Hiperparámetros óptimos consolidados de T19-T22
def _rel(ruta) -> Path:
    """Ruta relativa a la raíz del repositorio, para no imprimir rutas del equipo local."""
    return Path(ruta).resolve().relative_to(RAIZ)


def _cargar_hp_t22(modelo_id: str, prepro_id: str, defaults: dict) -> dict:
    ruta = DIR_TABLAS / "t22_pliegues_anidados.csv"
    if ruta.exists():
        try:
            df = pd.read_csv(ruta)
            sub = df[(df["modelo"] == modelo_id) & (df["preprocesamiento"] == prepro_id)]
            if len(sub) > 0:
                mejor = sub.sort_values("rmse_cad").iloc[0]
                params = json.loads(mejor["mejores_hiperparametros"])
                res = dict(defaults)
                res.update(params)
                res["random_state"] = SEMILLA
                res["n_jobs"] = -1
                return res
        except Exception:
            pass
    return defaults



HP_OLS_P2 = {}
HP_RIDGE_P2 = {"alpha": 3.162}
HP_XGB_P1 = _cargar_hp_t22("M6", "P1", {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.0,
    "reg_lambda": 3.0,
    "min_child_weight": 1,
    "random_state": SEMILLA,
    "n_jobs": -1,
})
HP_RF_P1 = _cargar_hp_t22("M5", "P1", {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": SEMILLA,
    "n_jobs": -1,
})


def calcular_predicciones_oof(
    X: pd.DataFrame,
    y: pd.Series,
    prepro_id: str,
    modelo_nombre: str,
    pliegues: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Calcula predicciones out-of-fold (OOF) para diagnóstico no sesgado de residuos."""
    oof_preds = np.zeros(len(X))
    conteo_visitas = np.zeros(len(X))

    for idx_train, idx_val in pliegues:
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_val = X.iloc[idx_val]

        if modelo_nombre == "OLS_P2":
            pre = construir_preprocesamiento("P2", escalar=False)
            X_tr = pre.fit_transform(X_train)
            X_vl = pre.transform(X_val)
            reg = LinearRegression()
            reg.fit(X_tr, np.log(y_train))
            preds_fold = np.exp(reg.predict(X_vl))

        elif modelo_nombre == "Ridge_P2":
            pre = construir_preprocesamiento("P2", escalar=True)
            X_tr = pre.fit_transform(X_train)
            X_vl = pre.transform(X_val)
            reg = Ridge(alpha=HP_RIDGE_P2["alpha"], random_state=SEMILLA)
            reg.fit(X_tr, np.log(y_train))
            preds_fold = np.exp(reg.predict(X_vl))

        elif modelo_nombre == "XGBoost_P1":
            pre = construir_preprocesamiento("P1", escalar=False)
            X_tr = pre.fit_transform(X_train)
            X_vl = pre.transform(X_val)
            reg = XGBRegressor(**HP_XGB_P1)
            reg.fit(X_tr, y_train)
            preds_fold = reg.predict(X_vl)

        elif modelo_nombre == "RandomForest_P1":
            pre = construir_preprocesamiento("P1", escalar=False)
            X_tr = pre.fit_transform(X_train)
            X_vl = pre.transform(X_val)
            reg = RandomForestRegressor(**HP_RF_P1)
            reg.fit(X_tr, y_train)
            preds_fold = reg.predict(X_vl)
        else:
            raise ValueError(f"Modelo no reconocido: {modelo_nombre}")

        oof_preds[idx_val] += preds_fold
        conteo_visitas[idx_val] += 1

    return oof_preds / conteo_visitas


def analizar_residuos(
    y_real_cad: np.ndarray,
    y_pred_cad: np.ndarray,
    modelo_nombre: str,
) -> Dict[str, Any]:
    """Calcula métricas estadísticas descriptivas y de contraste sobre los residuos."""
    residuos = y_real_cad - y_pred_cad
    errores_pct = (residuos / y_real_cad) * 100.0

    media_res = np.mean(residuos)
    std_res = np.std(residuos, ddof=1)
    mediana_res = np.median(residuos)
    iqr_res = stats.iqr(residuos)
    asimetria = float(stats.skew(residuos))
    curtosis = float(stats.kurtosis(residuos))  # exceso de curtosis (0 = normal)

    # Prueba de normalidad Shapiro-Wilk
    stat_shapiro, p_shapiro = stats.shapiro(residuos)

    # Prueba de normalidad D'Agostino-Pearson
    stat_dagostino, p_dagostino = stats.normaltest(residuos)

    # Heterocedasticidad: Correlación Spearman entre valor predicho y residuo absoluto
    rho_het, p_het = stats.spearmanr(y_pred_cad, np.abs(residuos))

    # Métricas de error clásicas
    rmse = np.sqrt(np.mean(residuos**2))
    mae = np.mean(np.abs(residuos))
    mape = np.mean(np.abs(errores_pct))
    ss_tot = np.sum((y_real_cad - np.mean(y_real_cad))**2)
    ss_res = np.sum(residuos**2)
    r2 = 1.0 - (ss_res / ss_tot)

    return {
        "modelo": modelo_nombre,
        "rmse_cad": rmse,
        "mae_cad": mae,
        "mape_pct": mape,
        "r2": r2,
        "media_residuos_cad": media_res,
        "std_residuos_cad": std_res,
        "mediana_residuos_cad": mediana_res,
        "iqr_residuos_cad": iqr_res,
        "asimetria": asimetria,
        "curtosis_exceso": curtosis,
        "shapiro_stat": stat_shapiro,
        "shapiro_p_valor": p_shapiro,
        "dagostino_stat": stat_dagostino,
        "dagostino_p_valor": p_dagostino,
        "spearman_rho_heterocedasticidad": rho_het,
        "spearman_p_valor_heterocedasticidad": p_het,
    }



def analizar_residuos_por_cuartil(y_real: np.ndarray, y_pred: np.ndarray, modelo_nombre: str = "XGBoost (P1)") -> pd.DataFrame:
    """Calcula metricas de error y sesgo estratificadas por cuartiles de precio real (T24)."""
    error = y_real - y_pred
    cuartiles = pd.qcut(y_real, q=4, labels=["Q1 (Bajo)", "Q2 (Medio-Bajo)", "Q3 (Medio-Alto)", "Q4 (Alto)"])
    
    df_res = pd.DataFrame({
        "y_real": y_real,
        "y_pred": y_pred,
        "error": error,
        "abs_error": np.abs(error),
        "pct_error": 100 * np.abs(error / y_real),
        "cuartil": cuartiles,
    })
    
    filas = []
    for q_label, grp in df_res.groupby("cuartil"):
        filas.append({
            "modelo": modelo_nombre,
            "cuartil": str(q_label),
            "n_observaciones": len(grp),
            "precio_min_cad": float(grp["y_real"].min()),
            "precio_max_cad": float(grp["y_real"].max()),
            "precio_medio_cad": float(grp["y_real"].mean()),
            "prediccion_media_cad": float(grp["y_pred"].mean()),
            "sesgo_medio_cad": float(grp["error"].mean()),
            "mae_cad": float(grp["abs_error"].mean()),
            "rmse_cad": float(np.sqrt(np.mean(grp["error"] ** 2))),
            "mape_pct": float(grp["pct_error"].mean()),
        })
    df_cuartiles = pd.DataFrame(filas)
    ruta_tab = DIR_TABLAS / "t24_residuos_por_cuartil.csv"
    df_cuartiles.to_csv(ruta_tab, index=False)
    print(f"  [Tabla Residuos por Cuartil] {_rel(ruta_tab)}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    
    datos_box = [df_res[df_res["cuartil"] == q]["error"].values for q in ["Q1 (Bajo)", "Q2 (Medio-Bajo)", "Q3 (Medio-Alto)", "Q4 (Alto)"]]
    bp = ax1.boxplot(datos_box, patch_artist=True, tick_labels=["Q1 (Bajo)", "Q2 (Med-Bajo)", "Q3 (Med-Alto)", "Q4 (Alto)"])
    colores_box = ["#4a6fa5", "#16697a", "#e07a5f", "#d62728"]
    for patch, col in zip(bp["boxes"], colores_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax1.axhline(0, color="black", linestyle="--", lw=1.2)
    ax1.set_xlabel("Cuartiles de Precio Real de Vivienda", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Residuo (Real - Predicho) [CAD de 1987]", fontsize=10.5, fontweight="bold")
    ax1.set_title(f"Distribución de Residuos por Cuartil ({modelo_nombre})", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    x_pos = np.arange(len(df_cuartiles))
    ancho = 0.35
    ax2.bar(x_pos - ancho/2, df_cuartiles["rmse_cad"], width=ancho, label="RMSE", color="#2b5c8f", alpha=0.85, edgecolor="black")
    ax2.bar(x_pos + ancho/2, df_cuartiles["mae_cad"], width=ancho, label="MAE", color="#d95f02", alpha=0.85, edgecolor="black")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_cuartiles["cuartil"], fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Cuartiles de Precio", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Error en CAD de 1987", fontsize=10.5, fontweight="bold")
    ax2.set_title("RMSE y MAE Desagregados por Estrato de Precio", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Diagnóstico de Residuos por Cuartil de Precio (T24)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta_fig = DIR_FIGURAS / "fig_t24_residuos_por_cuartil.png"
    plt.savefig(ruta_fig, dpi=300)
    plt.close()
    print(f"  [Figura Residuos por Cuartil] {_rel(ruta_fig)}")
    return df_cuartiles


def generar_pdp_ice(modelo_ajustado, X_trans: pd.DataFrame, nombres_columnas: list) -> None:
    """Genera Partial Dependence Plots (PDP) y curvas ICE para los atributos principales (T24)."""
    # scikit-learn no acepta columnas enteras en PDP; el modelo da lo mismo con flotantes
    X_trans = X_trans.astype(float)
    indices_top = []
    features_top = []
    for f in ["area", "bathrooms", "stories", "airconditioning"]:
        if f in nombres_columnas:
            indices_top.append(nombres_columnas.index(f))
            features_top.append(f)
            
    if not indices_top:
        indices_top = [0, 1, 2, 3]
        features_top = [nombres_columnas[i] for i in indices_top]
        
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    axes_flat = axes.flatten()
    
    PartialDependenceDisplay.from_estimator(
        modelo_ajustado,
        X_trans,
        features=indices_top,
        feature_names=nombres_columnas,
        kind="both",
        subsample=50,
        n_jobs=1,
        grid_resolution=30,
        random_state=SEMILLA,
        ax=axes_flat,
        line_kw={"color": "#2b5c8f", "linewidth": 2.2, "label": "PDP (Promedio)"},
        ice_lines_kw={"color": "#7fa9c6", "alpha": 0.25, "linewidth": 0.7},
    )
    
    for ax, f_name in zip(axes_flat, features_top):
        legible = NOMBRES_VARIABLES.get(f_name, f_name)
        ax.set_title(f"PDP + ICE: {legible}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Impacto marginal en precio", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("Gráficos de Dependencia Parcial (PDP) e Individual Conditional Expectation (ICE) - XGBoost P1 (T24)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    ruta_fig = DIR_FIGURAS / "fig_t24_pdp_ice.png"
    plt.savefig(ruta_fig, dpi=300)
    plt.close()
    print(f"  [Figura PDP / ICE] {_rel(ruta_fig)}")


def ejecutar_t24():
    """Función principal que orquesta el análisis de residuos, SHAP y comparación econométrica."""
    print("=" * 80)
    print("INICIANDO T24: ANÁLISIS DE ERRORES E INTERPRETABILIDAD (SHAP)")
    print("=" * 80)

    # 1. Cargar conjunto de entrenamiento aislado (n=436)
    X, y, evidencia = cargar_entrenamiento()
    y_cad = y.values / FACTOR_KAGGLE
    print(f"Datos de entrenamiento cargados: {X.shape[0]} observaciones, {X.shape[1]} columnas.")
    print(f"Conjunto de prueba protegido: {evidencia['n_prueba_en_particion']} filas selladas intactas.\n")

    # 2. Configurar pliegues pareados para residuos Out-Of-Fold (OOF)
    pliegues = crear_pliegues(y, semilla=SEMILLA)
    print(f"Pliegues pareados configurados: {len(pliegues)} divisiones (5x3 CV).")

    # 3. Predicciones OOF para diagnóstico de residuos no sesgado
    print("\n--- Calculando predicciones Out-Of-Fold para diagnóstico de residuos ---")
    modelos_diagnostico = ["OLS_P2", "Ridge_P2", "XGBoost_P1", "RandomForest_P1"]
    oof_predicciones = {}
    estadisticas_residuos = []

    for m in modelos_diagnostico:
        preds = calcular_predicciones_oof(X, y, "P1" if "P1" in m else "P2", m, pliegues)
        oof_predicciones[m] = preds / FACTOR_KAGGLE
        stats_m = analizar_residuos(y_cad, oof_predicciones[m], m)
        estadisticas_residuos.append(stats_m)
        print(f"  {m:16s} -> RMSE: {stats_m['rmse_cad']:,.2f} CAD | Sesgo medio: {stats_m['media_residuos_cad']:+,.2f} CAD | Asimetría: {stats_m['asimetria']:.3f} | p-Het: {stats_m['spearman_p_valor_heterocedasticidad']:.4e}")

    df_residuos_resumen = pd.DataFrame(estadisticas_residuos)
    ruta_tabla_residuos = DIR_TABLAS / "t24_analisis_residuos.csv"
    df_residuos_resumen.to_csv(ruta_tabla_residuos, index=False)
    print(f"\n[OK] Tabla de residuos guardada en: {_rel(ruta_tabla_residuos)}")

    # 4. Detección y caracterización de casos extremos (outliers de predicción)
    print("\n--- Identificando casos extremos de predicción (Top Over/Under predicted) ---")
    residuos_xgb = y_cad - oof_predicciones["XGBoost_P1"]
    residuos_ols = y_cad - oof_predicciones["OLS_P2"]

    df_casos = X.copy()
    df_casos["precio_real_cad"] = y_cad
    df_casos["pred_xgb_cad"] = oof_predicciones["XGBoost_P1"]
    df_casos["residuo_xgb_cad"] = residuos_xgb
    df_casos["pred_ols_cad"] = oof_predicciones["OLS_P2"]
    df_casos["residuo_ols_cad"] = residuos_ols
    df_casos["error_abs_xgb_cad"] = np.abs(residuos_xgb)

    # Top 5 subpredichas (precio real mucho mayor que la predicción: residuo positivo grande)
    top_subpredichas = df_casos.sort_values(by="residuo_xgb_cad", ascending=False).head(5).copy()
    top_subpredichas["tipo_error"] = "Subpredicha extrema (Real >> Predicho)"

    # Top 5 sobrepredichas (predicción mucho mayor que el precio real: residuo negativo grande)
    top_sobrepredichas = df_casos.sort_values(by="residuo_xgb_cad", ascending=True).head(5).copy()
    top_sobrepredichas["tipo_error"] = "Sobrepredicha extrema (Real << Predicho)"

    df_extremos = pd.concat([top_subpredichas, top_sobrepredichas])
    ruta_tabla_extremos = DIR_TABLAS / "t24_casos_extremos_residuos.csv"
    df_extremos.to_csv(ruta_tabla_extremos)
    print(f"[OK] Tabla de casos extremos guardada en: {_rel(ruta_tabla_extremos)}")

    # 5. Ajuste de modelos finales sobre entrenamiento completo para SHAP
    print("\n--- Entrenando modelos en conjunto completo (n=436) para TreeExplainer ---")
    pre_p1 = construir_preprocesamiento("P1", escalar=False)
    X_p1 = pre_p1.fit_transform(X)
    nombres_cols_p1 = list(X_p1.columns)

    # XGBoost P1
    xgb_final = XGBRegressor(**HP_XGB_P1)
    xgb_final.fit(X_p1, y)
    print("  XGBoost P1 entrenado con éxito.")

    # Random Forest P1
    rf_final = RandomForestRegressor(**HP_RF_P1)
    rf_final.fit(X_p1, y)
    print("  Random Forest P1 entrenado con éxito.")

    # OLS Hedónico P2 (para comparación intrínseca)
    pre_p2 = construir_preprocesamiento("P2", escalar=False)
    X_p2 = pre_p2.fit_transform(X)
    nombres_cols_p2 = list(X_p2.columns)
    ols_p2_final = LinearRegression()
    ols_p2_final.fit(X_p2, np.log(y))
    print("  OLS Hedónico P2 entrenado con éxito.")

    # 6. Cálculo de explicaciones SHAP con TreeExplainer
    print("\n--- Calculando valores SHAP con TreeExplainer ---")
    explainer_xgb = shap.TreeExplainer(xgb_final)
    shap_vals_xgb_raw = explainer_xgb(X_p1)
    # Convertir valores SHAP a CAD 1987 dividiendo por FACTOR_KAGGLE
    shap_vals_xgb_cad = shap_vals_xgb_raw.values / FACTOR_KAGGLE
    base_val_raw_xgb = explainer_xgb.expected_value
    base_value_xgb_cad = float(np.ravel(base_val_raw_xgb)[0]) / FACTOR_KAGGLE

    # SHAP Explanation object en CAD 1987 para gráficos de shap
    shap_exp_xgb = shap.Explanation(
        values=shap_vals_xgb_cad,
        base_values=np.full(len(X_p1), base_value_xgb_cad),
        data=X_p1.values,
        feature_names=[NOMBRES_VARIABLES.get(c, c) for c in nombres_cols_p1],
    )

    # Random Forest SHAP
    explainer_rf = shap.TreeExplainer(rf_final)
    shap_vals_rf_raw = explainer_rf(X_p1)
    shap_vals_rf_cad = shap_vals_rf_raw.values / FACTOR_KAGGLE
    base_val_raw_rf = explainer_rf.expected_value
    base_value_rf_cad = float(np.ravel(base_val_raw_rf)[0]) / FACTOR_KAGGLE

    # 7. Importancias globales SHAP vs Permutación
    print("\n--- Calculando rankings de importancia global ---")
    mean_abs_shap_xgb = np.mean(np.abs(shap_vals_xgb_cad), axis=0)
    std_shap_xgb = np.std(shap_vals_xgb_cad, axis=0)
    pct_shap_xgb = (mean_abs_shap_xgb / np.sum(mean_abs_shap_xgb)) * 100.0

    mean_abs_shap_rf = np.mean(np.abs(shap_vals_rf_cad), axis=0)
    pct_shap_rf = (mean_abs_shap_rf / np.sum(mean_abs_shap_rf)) * 100.0

    # Permutación sobre XGBoost (en CAD 1987)
    perm_res = permutation_importance(
        xgb_final, X_p1, y, n_repeats=10, random_state=SEMILLA, scoring="neg_root_mean_squared_error"
    )
    perm_importancia_cad = np.abs(perm_res.importances_mean) / FACTOR_KAGGLE

    df_importancias = pd.DataFrame({
        "variable": nombres_cols_p1,
        "nombre_legible": [NOMBRES_VARIABLES.get(c, c) for c in nombres_cols_p1],
        "shap_mean_abs_xgb_cad": mean_abs_shap_xgb,
        "shap_std_xgb_cad": std_shap_xgb,
        "shap_pct_xgb": pct_shap_xgb,
        "shap_mean_abs_rf_cad": mean_abs_shap_rf,
        "shap_pct_rf": pct_shap_rf,
        "permutacion_importancia_cad": perm_importancia_cad,
    }).sort_values(by="shap_mean_abs_xgb_cad", ascending=False).reset_index(drop=True)
    df_importancias["ranking_shap_xgb"] = range(1, len(df_importancias) + 1)

    ruta_tabla_shap = DIR_TABLAS / "t24_shap_importancias_globales.csv"
    df_importancias.to_csv(ruta_tabla_shap, index=False)
    print(f"[OK] Tabla de importancia global SHAP guardada en: {_rel(ruta_tabla_shap)}")

    # 8. Contraste: Coeficientes Hedónicos (OLS P2) vs SHAP (XGBoost P1)
    print("\n--- Comparando Coeficientes Hedónicos OLS P2 vs SHAP XGBoost ---")
    coefs_ols = ols_p2_final.coef_
    std_x_p2 = X_p2.std(axis=0).values
    std_log_y = np.std(np.log(y))
    beta_estandarizados = coefs_ols * (std_x_p2 / std_log_y)
    semi_elasticidad_pct = (np.exp(coefs_ols) - 1.0) * 100.0

    df_ols_comparacion = pd.DataFrame({
        "variable": nombres_cols_p2,
        "nombre_legible": [NOMBRES_VARIABLES.get(c, c) for c in nombres_cols_p2],
        "coef_ols_p2": coefs_ols,
        "beta_estandarizado_ols": beta_estandarizados,
        "semi_elasticidad_pct": semi_elasticidad_pct,
        "abs_beta_estandarizado": np.abs(beta_estandarizados),
    }).sort_values(by="abs_beta_estandarizado", ascending=False).reset_index(drop=True)
    df_ols_comparacion["ranking_ols"] = range(1, len(df_ols_comparacion) + 1)

    # Fusionar con SHAP para contraste sistemático
    df_contraste = pd.merge(
        df_ols_comparacion,
        df_importancias[["variable", "shap_mean_abs_xgb_cad", "shap_pct_xgb", "ranking_shap_xgb"]],
        on="variable",
        how="inner",
    ).sort_values(by="ranking_shap_xgb").reset_index(drop=True)

    rho_spearman_rank, p_spearman_rank = stats.spearmanr(df_contraste["ranking_ols"], df_contraste["ranking_shap_xgb"])
    print(f"  Correlación de rangos Spearman entre OLS y SHAP: rho = {rho_spearman_rank:.3f} (p = {p_spearman_rank:.4f})")

    ruta_tabla_contraste = DIR_TABLAS / "t24_comparacion_coeficientes_vs_shap.csv"
    df_contraste.to_csv(ruta_tabla_contraste, index=False)
    print(f"[OK] Tabla comparativa OLS vs SHAP guardada en: {_rel(ruta_tabla_contraste)}")

    # 9. Selección y análisis de 4 casos locales representativos
    print("\n--- Seleccionando 4 arquetipos de vivienda para explicación local ---")
    # Caso 1: Vivienda Mediana (error cercano a 0, precio cercano a la mediana)
    mediana_y = np.median(y_cad)
    candidatos_mediana = df_casos[
        (df_casos["precio_real_cad"] >= mediana_y * 0.90) &
        (df_casos["precio_real_cad"] <= mediana_y * 1.10)
    ]
    idx_mediana = candidatos_mediana["error_abs_xgb_cad"].idxmin()

    # Caso 2: Vivienda de Alto Valor / Lujo
    idx_lujo = df_casos["precio_real_cad"].idxmax()

    # Caso 3: Máxima Subpredicción (Real >> Predicho)
    idx_subpredicha = df_casos["residuo_xgb_cad"].idxmax()

    # Caso 4: Máxima Sobrepredicción (Real << Predicho)
    idx_sobrepredicha = df_casos["residuo_xgb_cad"].idxmin()

    casos_seleccionados = [
        ("Caso A: Mediana representativa", idx_mediana),
        ("Caso B: Alto valor / Lujo", idx_lujo),
        ("Caso C: Máxima subpredicción", idx_subpredicha),
        ("Caso D: Máxima sobrepredicción", idx_sobrepredicha),
    ]

    filas_casos_locales = []
    for etiqueta, idx_caso in casos_seleccionados:
        pos_fila = X.index.get_loc(idx_caso)
        y_r = y_cad[pos_fila]
        y_p = oof_predicciones["XGBoost_P1"][pos_fila]
        y_p_modelo = float(xgb_final.predict(X_p1.iloc[[pos_fila]])[0]) / FACTOR_KAGGLE
        res = y_r - y_p

        fila_info = {
            "arquetipo": etiqueta,
            "id_vivienda": idx_caso,
            "precio_real_cad": y_r,
            "prediccion_xgb_oof_cad": y_p,
            "prediccion_xgb_modelo_cad": y_p_modelo,
            "prediccion_xgb_cad": y_p,
            "residuo_cad": res,
            "valor_base_cad": base_value_xgb_cad,
        }
        for j, col in enumerate(nombres_cols_p1):
            fila_info[f"val_{col}"] = X_p1.iloc[pos_fila, j]
            fila_info[f"shap_{col}_cad"] = shap_vals_xgb_cad[pos_fila, j]
        filas_casos_locales.append(fila_info)

    df_casos_locales = pd.DataFrame(filas_casos_locales)
    ruta_tabla_locales = DIR_TABLAS / "t24_casos_locales_shap.csv"
    df_casos_locales.to_csv(ruta_tabla_locales, index=False)
    print(f"[OK] Tabla de casos locales guardada en: {_rel(ruta_tabla_locales)}")

    # =========================================================================
    # GENERACIÓN DE FIGURAS FORMATO IEEE
    # =========================================================================
    print("\n--- Generando figuras de diagnóstico y SHAP ---")

    # FIGURA 1: Diagnóstico de Residuos 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Predicho vs Real
    ax = axes[0, 0]
    min_val = min(y_cad.min(), oof_predicciones["XGBoost_P1"].min()) * 0.95
    max_val = max(y_cad.max(), oof_predicciones["XGBoost_P1"].max()) * 1.05
    ax.scatter(oof_predicciones["OLS_P2"], y_cad, alpha=0.45, color="#16697A", s=25, label="OLS Hedónico P2")
    ax.scatter(oof_predicciones["XGBoost_P1"], y_cad, alpha=0.55, color="#E07A5F", s=25, label="XGBoost P1")
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label=r"Identidad ideal ($y = \hat{y}$)")
    ax.set_title(r"(a) Predicción Out-Of-Fold vs Precio Real", fontsize=11, fontweight="bold")
    ax.set_xlabel("Precio Predicho (CAD 1987)", fontsize=10)
    ax.set_ylabel("Precio Real (CAD 1987)", fontsize=10)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(frameon=True, fontsize=9)

    # (b) Residuos vs Predicho
    ax = axes[0, 1]
    res_ols = y_cad - oof_predicciones["OLS_P2"]
    res_xgb = y_cad - oof_predicciones["XGBoost_P1"]
    ax.scatter(oof_predicciones["OLS_P2"], res_ols, alpha=0.45, color="#16697A", s=25, label="OLS Hedónico P2")
    ax.scatter(oof_predicciones["XGBoost_P1"], res_xgb, alpha=0.55, color="#E07A5F", s=25, label="XGBoost P1")
    ax.axhline(0, color="black", linestyle="--", lw=1.2)
    # Suavizado de tendencia
    sns.regplot(x=oof_predicciones["XGBoost_P1"], y=res_xgb, ax=ax, scatter=False, color="#D9381E", line_kws={"lw": 1.5, "label": "Tendencia XGBoost"})
    ax.set_title(r"(b) Residuos ($y - \hat{y}$) vs Precio Predicho", fontsize=11, fontweight="bold")
    ax.set_xlabel("Precio Predicho (CAD 1987)", fontsize=10)
    ax.set_ylabel("Residuo (CAD 1987)", fontsize=10)
    ax.legend(frameon=True, fontsize=9)

    # (c) Histograma y densidad KDE de residuos
    ax = axes[1, 0]
    sns.kdeplot(res_ols, ax=ax, color="#16697A", lw=2, label="OLS Hedónico P2")
    sns.kdeplot(res_xgb, ax=ax, color="#E07A5F", lw=2, label="XGBoost P1")
    # Curva normal de referencia
    x_grid = np.linspace(min(res_xgb.min(), res_ols.min()), max(res_xgb.max(), res_ols.max()), 200)
    pdf_norm = stats.norm.pdf(x_grid, 0, np.std(res_xgb))
    ax.plot(x_grid, pdf_norm, "k--", lw=1.2, label=rf"Normal teórica ($\sigma$={np.std(res_xgb):,.0f})")
    ax.axvline(0, color="gray", linestyle=":", lw=1)
    ax.set_title("(c) Distribución de Densidad de Residuos", fontsize=11, fontweight="bold")
    ax.set_xlabel("Residuo (CAD 1987)", fontsize=10)
    ax.set_ylabel("Densidad", fontsize=10)
    ax.legend(frameon=True, fontsize=9)

    # (d) Q-Q Plot comparativo
    ax = axes[1, 1]
    (osm_ols, osr_ols), (slope_ols, intercept_ols, r_ols) = stats.probplot(res_ols, dist="norm")
    (osm_xgb, osr_xgb), (slope_xgb, intercept_xgb, r_xgb) = stats.probplot(res_xgb, dist="norm")
    ax.plot(osm_ols, osr_ols, "o", color="#16697A", alpha=0.45, markersize=4, label=f"OLS P2 ($R^2$={r_ols**2:.3f})")
    ax.plot(osm_xgb, osr_xgb, "^", color="#E07A5F", alpha=0.55, markersize=4, label=f"XGBoost P1 ($R^2$={r_xgb**2:.3f})")
    ax.plot(osm_xgb, slope_xgb * osm_xgb + intercept_xgb, "k--", lw=1.2, label="Referencia normal")
    ax.set_title("(d) Gráfico Cuantil-Cuantil (Q-Q Plot Normal)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cuantiles Teóricos", fontsize=10)
    ax.set_ylabel("Cuantiles Muestrales", fontsize=10)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    ruta_fig_residuos = DIR_FIGURAS / "fig_t24_diagnostico_residuos.png"
    plt.savefig(ruta_fig_residuos, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig_residuos)}")

    # FIGURA 2: Beeswarm Summary Plot (SHAP)
    fig = plt.figure(figsize=(10, 6.5))
    shap.plots.beeswarm(shap_exp_xgb, max_display=13, show=False)
    plt.title("Impacto Global de Atributos sobre el Precio (XGBoost P1) · Beeswarm SHAP", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Valor SHAP (Impacto en la predicción en CAD de 1987)", fontsize=10)
    plt.tight_layout()
    ruta_fig_beeswarm = DIR_FIGURAS / "fig_t24_shap_summary_beeswarm.png"
    plt.savefig(ruta_fig_beeswarm, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig_beeswarm)}")

    # FIGURA 3: Importancia Global de Barras (XGBoost vs Random Forest)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(df_importancias))
    ancho = 0.38
    rects1 = ax.barh(y_pos + ancho/2, df_importancias["shap_mean_abs_xgb_cad"], ancho, label="XGBoost P1", color="#E07A5F", edgecolor="black", alpha=0.85)
    rects2 = ax.barh(y_pos - ancho/2, df_importancias["shap_mean_abs_rf_cad"], ancho, label="Random Forest P1", color="#2A9D8F", edgecolor="black", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_importancias["nombre_legible"], fontsize=9.5)
    ax.invert_yaxis()  # mayor arriba
    ax.set_xlabel(r"Mean Absolute SHAP Value: $\mathbb{E}[|\phi_j|]$ (CAD 1987)", fontsize=10)
    ax.set_title("Importancia Global de Atributos: Mean |SHAP| (XGBoost vs Random Forest)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)

    # Anotar valores en las barras de XGBoost
    for rect in rects1:
        w = rect.get_width()
        ax.annotate(f"{w:,.0f}", xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=8)

    plt.tight_layout()
    ruta_fig_barras = DIR_FIGURAS / "fig_t24_shap_importancia_barras.png"
    plt.savefig(ruta_fig_barras, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig_barras)}")

    # FIGURA 4: Curvas de Dependencia SHAP (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    vars_dependencia = [
        ("area", "Área del lote (sq ft)"),
        ("bathrooms", "Número de baños"),
        ("stories", "Número de plantas (pisos)"),
        ("parking", "Plazas de estacionamiento"),
        ("airconditioning", "Aire acondicionado (0 = No, 1 = Sí)"),
        ("prefarea", "Zona preferencial (0 = No, 1 = Sí)"),
    ]

    for ax, (var, titulo_var) in zip(axes.flatten(), vars_dependencia):
        j_col = nombres_cols_p1.index(var)
        x_vals = X_p1.iloc[:, j_col].values
        shap_vals = shap_vals_xgb_cad[:, j_col]

        # Scatter coloreado por baños o área
        scatter = ax.scatter(x_vals, shap_vals, alpha=0.6, c=X_p1["bathrooms"].values, cmap="viridis", s=25, edgecolor="none")
        # Ajuste LOWESS o media condicional
        if var in ["bathrooms", "stories", "parking", "airconditioning", "prefarea"]:
            medias_cond = pd.DataFrame({"x": x_vals, "shap": shap_vals}).groupby("x").mean().reset_index()
            ax.plot(medias_cond["x"], medias_cond["shap"], "r-o", lw=2, markersize=6, label="Aporte medio")
        else:
            sns.regplot(x=x_vals, y=shap_vals, ax=ax, scatter=False, lowess=True, color="#D9381E", line_kws={"lw": 2.5, "label": "Tendencia LOWESS"})

        ax.axhline(0, color="gray", linestyle="--", lw=1)
        ax.set_title(f"Dependencia SHAP: {titulo_var}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(titulo_var, fontsize=9.5)
        ax.set_ylabel("Aporte SHAP (CAD 1987)", fontsize=9.5)
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Número de baños (variable de interacción visual)", fontsize=9)

    plt.suptitle("Curvas de Dependencia Parcial SHAP (XGBoost P1) · Windsor Housing", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 0.94, 0.97])
    ruta_fig_dependencia = DIR_FIGURAS / "fig_t24_shap_dependencia_atributos.png"
    plt.savefig(ruta_fig_dependencia, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig_dependencia)}")

    # FIGURA 5: Explicación Local de 4 Arquetipos de Vivienda (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (etiqueta, idx_caso) in zip(axes.flatten(), casos_seleccionados):
        pos_fila = X.index.get_loc(idx_caso)
        y_r = y_cad[pos_fila]
        y_p = oof_predicciones["XGBoost_P1"][pos_fila]
        res = y_r - y_p

        shaps_caso = shap_vals_xgb_cad[pos_fila, :]
        # Ordenar por magnitud de aporte
        idx_orden = np.argsort(np.abs(shaps_caso))[::-1][:8]  # top 8 aportes
        cols_top = [nombres_cols_p1[i] for i in idx_orden]
        nombres_top = [f"{NOMBRES_VARIABLES.get(c, c)} = {X_p1.iloc[pos_fila, nombres_cols_p1.index(c)]}" for c in cols_top]
        vals_top = shaps_caso[idx_orden]

        colores = ["#2A9D8F" if v >= 0 else "#E76F51" for v in vals_top]
        y_bar = np.arange(len(vals_top))
        ax.barh(y_bar, vals_top, color=colores, edgecolor="black", alpha=0.85)
        ax.set_yticks(y_bar)
        ax.set_yticklabels(nombres_top, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linestyle="--", lw=1)

        ax.set_title(
            f"{etiqueta} (ID: {idx_caso})\n"
            f"Real: {y_r:,.0f} CAD | Predicho: {y_p:,.0f} CAD | Error: {res:+,.0f} CAD",
            fontsize=10,
            fontweight="bold"
        )
        ax.set_xlabel("Aporte marginal SHAP (CAD 1987)", fontsize=9)

        # Anotar valores
        for j, (v, rect_y) in enumerate(zip(vals_top, y_bar)):
            offset = 4 if v >= 0 else -4
            ha = "left" if v >= 0 else "right"
            ax.annotate(f"{v:+,.0f}", xy=(v, rect_y), xytext=(offset, 0),
                        textcoords="offset points", ha=ha, va="center", fontsize=8)

    plt.suptitle(
        f"Explicaciones Locales de Predicción (XGBoost P1) · Base: {base_value_xgb_cad:,.0f} CAD",
        fontsize=12,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ruta_fig_locales = DIR_FIGURAS / "fig_t24_shap_explicacion_local_casos.png"
    plt.savefig(ruta_fig_locales, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig_locales)}")

    # FIGURA 6: Concordancia Coeficientes Hedónicos (OLS) vs SHAP (XGBoost)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # (a) Barras pareadas normalizadas
    df_sort_ols = df_contraste.sort_values(by="abs_beta_estandarizado", ascending=True)
    y_pos = np.arange(len(df_sort_ols))
    ancho = 0.38
    # Normalizar al máximo para comparar forma de orden
    norm_ols = df_sort_ols["abs_beta_estandarizado"] / df_sort_ols["abs_beta_estandarizado"].max()
    norm_shap = df_sort_ols["shap_mean_abs_xgb_cad"] / df_sort_ols["shap_mean_abs_xgb_cad"].max()

    ax1.barh(y_pos + ancho/2, norm_ols, ancho, label="OLS Hedónico ($|\\beta^*|$ normalizado)", color="#16697A", edgecolor="black", alpha=0.85)
    ax1.barh(y_pos - ancho/2, norm_shap, ancho, label="XGBoost (Mean |SHAP| normalizado)", color="#E07A5F", edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sort_ols["nombre_legible"], fontsize=9)
    ax1.set_xlabel("Importancia relativa normalizada (máx = 1.0)", fontsize=9.5)
    ax1.set_title("(a) Concordancia de Importancia Relativa: OLS vs SHAP", fontsize=10.5, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)

    # (b) Dispersión de ranking OLS vs ranking SHAP
    ax2.scatter(df_contraste["ranking_ols"], df_contraste["ranking_shap_xgb"], color="#4A6FA5", s=80, edgecolors="black", zorder=3)
    ax2.plot([1, 13], [1, 13], "k--", lw=1.2, label="Concordancia perfecta (1:1)")
    for _, fila in df_contraste.iterrows():
        ax2.annotate(fila["variable"], (fila["ranking_ols"] + 0.2, fila["ranking_shap_xgb"] - 0.1), fontsize=8.5)

    ax2.set_xlabel("Ranking según OLS Hedónico (|$\\beta^*$|)", fontsize=9.5)
    ax2.set_ylabel("Ranking según XGBoost SHAP (Mean |SHAP|)", fontsize=9.5)
    ax2.set_title(f"(b) Correlación de Rangos: $\\rho_{{\\mathrm{{Spearman}}}}$ = {rho_spearman_rank:.3f} ($p$ = {p_spearman_rank:.4f})", fontsize=10.5, fontweight="bold")
    ax2.set_xlim(0.5, 13.5)
    ax2.set_ylim(0.5, 13.5)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    ruta_fig_contraste = DIR_FIGURAS / "fig_t24_coeficientes_hedonicos_vs_shap.png"
    plt.savefig(ruta_fig_contraste, dpi=300)
    plt.close()
    print(f"[OK] Figura guardada: {_rel(ruta_fig_contraste)}")

    print("\n" + "=" * 80)
    
    # --- Residuos por cuartil y PDP / ICE (T24) ---
    print("\n=== Paso 9: Diagnóstico de Residuos por Cuartil y Curvas PDP / ICE ===")
    analizar_residuos_por_cuartil(y_cad, oof_predicciones["XGBoost_P1"], "XGBoost (P1)")
    generar_pdp_ice(xgb_final, X_p1, nombres_cols_p1)

    print("EJECUCIÓN DE T24 COMPLETADA EXITOSAMENTE")
    print("Todas las tablas y figuras han sido generadas rigurosamente.")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_t24()
