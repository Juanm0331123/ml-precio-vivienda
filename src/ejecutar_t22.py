"""Script ejecutor para T22: Validación Cruzada Anidada y Comparación Estadística Formal.

Tarjeta T22 · Validación cruzada anidada y comparación estadística de modelos [Brandon].
Responsable: Brandon Eduardo Tobar Sánchez (Evaluation Lead).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).

Ejecuta:
1. Validación Cruzada Anidada Genuina (Nested CV):
   - Bucle externo: 15 pliegues pareados de CV repetida 5×3 de T12 (RepeatedStratifiedKFold).
   - Bucle interno: Selección de hiperparámetros por GridSearchCV / RandomizedSearchCV (5 pliegues)
     estrictamente dentro de la partición de ajuste de cada pliegue externo (sin fuga de información).
   - Cobertura completa de la matriz experimental: 3 preprocesamientos (P1, P2, P3) × 6 familias de modelos
     (M1: OLS hedónico, M2: Ridge, M3: KNN, M4: SVR RBF, M5: Random Forest, M6: XGBoost) = 18 pipelines.
2. Cálculo de métricas obligatorias en cada pliegue externo:
   - RMSE (CAD 1987), MAE (CAD 1987), MAPE (%), R² y RMSLE.
   - Medias, desviaciones estándar e Intervalos de Confianza t-Student al 95%.
3. Inferencia estadística formal:
   - Prueba t corregida para validación cruzada repetida (Nadeau & Bengio, 2003).
   - Corrección por comparaciones múltiples (Holm-Bonferroni).
   - Prueba omnibus de Friedman y ranking medio de algoritmos (Demšar, 2006).
4. Generación de tablas maestras definitivas en reports/tables/ y figuras en reports/figures/.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Configurar directorio temporal portátil para matplotlib
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Asegurar que la raíz del repositorio esté en el path para ejecución directa
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

from src.config import DIR_FIGURAS, DIR_TABLAS, SEMILLA
from src.evaluacion import (
    FACTOR_KAGGLE,
    calcular_intervalo_confianza,
    calcular_metricas,
    cargar_entrenamiento,
    crear_pliegues,
    prueba_friedman_ranking,
    prueba_t_corregida_nadeau_bengio,
)
from src.modelos import MODELOS, construir_pipeline

# Estilo gráfico consistente con el paper IEEE
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 200

COLORES_PREPRO = {
    "P1": "#4A6FA5",  # Azul base
    "P2": "#16697A",  # Verde petróleo / azul oscuro
    "P3": "#E07A5F",  # Coral / terracota
}

COLORES_FAMILIA = {
    "lineal": "#4A6FA5",
    "distancia": "#3D5A80",
    "kernel": "#EE6C4D",
    "árboles": "#2A9D8F",
}


def obtener_prefijo_pipeline(pipeline_base: Any) -> str:
    """Devuelve 'regressor__modelo__' si usa TransformedTargetRegressor o 'modelo__' si es directo."""
    return "regressor__modelo__" if hasattr(pipeline_base, "regressor") else "modelo__"


def configurar_buscador_interno(prepro_id: str, modelo_id: str, semilla_fold: int) -> Tuple[Any, str]:
    """Crea el estimador con búsqueda interna para un pliegue externo específico.
    
    Para M1 (OLS), no hay hiperparámetros que ajustar; se devuelve el pipeline directo.
    Para M2, M3, M4 se usa GridSearchCV con rejillas rigurosas.
    Para M5, M6 se usa RandomizedSearchCV con 15 iteraciones controladas para optimizar tiempo/cobertura.
    """
    pipe = construir_pipeline(prepro_id, modelo_id)
    prefijo = obtener_prefijo_pipeline(pipe)
    cv_interno = KFold(n_splits=5, shuffle=True, random_state=semilla_fold)

    if modelo_id == "M1":
        return pipe, prefijo

    if modelo_id == "M2":
        grid = {
            f"{prefijo}alpha": [0.01, 0.1, 1.0, 3.162, 10.0, 31.62, 100.0, 316.2, 1000.0]
        }
        buscador = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv_interno,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            refit=True,
        )
        return buscador, prefijo

    if modelo_id == "M3":
        grid = {
            f"{prefijo}n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            f"{prefijo}weights": ["uniform", "distance"],
            f"{prefijo}p": [1, 2],
        }
        buscador = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv_interno,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            refit=True,
        )
        return buscador, prefijo

    if modelo_id == "M4":
        grid = {
            f"{prefijo}C": [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0],
            f"{prefijo}gamma": ["scale", 0.0001, 0.001, 0.01],
            f"{prefijo}epsilon": [0.05, 0.1, 0.2, 0.5],
        }
        buscador = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv_interno,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            refit=True,
        )
        return buscador, prefijo

    if modelo_id == "M5":
        dist = {
            f"{prefijo}n_estimators": [100, 200, 300, 500],
            f"{prefijo}max_depth": [None, 6, 8, 10, 14],
            f"{prefijo}min_samples_split": [2, 4, 8],
            f"{prefijo}min_samples_leaf": [1, 2, 4],
            f"{prefijo}max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        }
        buscador = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=dist,
            n_iter=15,
            cv=cv_interno,
            scoring="neg_root_mean_squared_error",
            random_state=semilla_fold,
            n_jobs=1,
            refit=True,
        )
        return buscador, prefijo

    if modelo_id == "M6":
        dist = {
            f"{prefijo}n_estimators": [100, 200, 300],
            f"{prefijo}max_depth": [3, 4, 5, 6],
            f"{prefijo}learning_rate": [0.01, 0.03, 0.05, 0.1],
            f"{prefijo}subsample": [0.7, 0.85, 1.0],
            f"{prefijo}colsample_bytree": [0.6, 0.8, 1.0],
            f"{prefijo}reg_lambda": [0.5, 1.0, 3.0, 5.0],
            f"{prefijo}reg_alpha": [0.0, 0.1, 1.0],
        }
        buscador = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=dist,
            n_iter=15,
            cv=cv_interno,
            scoring="neg_root_mean_squared_error",
            random_state=semilla_fold,
            n_jobs=1,
            refit=True,
        )
        return buscador, prefijo

    raise ValueError(f"Modelo no reconocido: {modelo_id}")


def _evaluar_un_pliegue_externo(
    k: int,
    ajuste_idx: np.ndarray,
    val_idx: np.ndarray,
    prepro_id: str,
    modelo_id: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """Ejecuta la optimización interna en X_ajuste y evalúa en X_val para el pliegue externo k."""
    X_tr, y_tr = X.iloc[ajuste_idx], y.iloc[ajuste_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    semilla_fold = SEMILLA + k * 101
    estimador, prefijo = configurar_buscador_interno(prepro_id, modelo_id, semilla_fold)

    t0 = time.perf_counter()
    estimador.fit(X_tr, y_tr)
    t_ajuste = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = estimador.predict(X_val)
    t_inferencia = time.perf_counter() - t0

    metricas = calcular_metricas(y_val, y_pred)

    if hasattr(estimador, "best_params_"):
        params_limpios = {k.replace(prefijo, ""): v for k, v in estimador.best_params_.items()}
    else:
        params_limpios = {}

    modelo_nombre = MODELOS[modelo_id][0]
    familia = MODELOS[modelo_id][1]
    pipeline_nombre = f"{modelo_nombre} ({prepro_id})"
    modelo_id_completo = f"{modelo_id}_{prepro_id}"

    return {
        "pipeline_nombre": pipeline_nombre,
        "modelo_id_completo": modelo_id_completo,
        "preprocesamiento": prepro_id,
        "modelo": modelo_id,
        "modelo_nombre": modelo_nombre,
        "familia": familia,
        "pliegue": k,
        "repeticion": k // 5,
        "rmse_cad": metricas["rmse"] / FACTOR_KAGGLE,
        "mae_cad": metricas["mae"] / FACTOR_KAGGLE,
        "mape": metricas["mape"],
        "r2": metricas["r2"],
        "rmsle": metricas["rmsle"],
        "tiempo_ajuste_s": t_ajuste,
        "tiempo_inferencia_s": t_inferencia,
        "mejores_hiperparametros": json.dumps(params_limpios),
    }


def ejecutar_experimento_t22(forzar_recalculo: bool = False) -> Dict[str, Any]:
    """Ejecuta la validación cruzada anidada para los 18 pipelines y genera tablas y figuras."""
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    ruta_pliegues = DIR_TABLAS / "t22_pliegues_anidados.csv"

    if not forzar_recalculo and ruta_pliegues.exists():
        print(f"Cargando resultados previos de CV anidada desde {ruta_pliegues}...")
        df_pliegues = pd.read_csv(ruta_pliegues)
    else:
        print("=== Paso 1: Carga aislada del conjunto de entrenamiento (n=436) ===")
        X, y, evidencia = cargar_entrenamiento()
        print(f"Cargadas {len(X)} observaciones de entrenamiento. Huella partición: {evidencia['sha256_particion'][:12]}...")

        pliegues = crear_pliegues(y)
        print(f"Generados {len(pliegues)} pliegues pareados (CV externa repetida 5×3).")

        preprocesamientos = ["P1", "P2", "P3"]
        modelos_ids = ["M1", "M2", "M3", "M4", "M5", "M6"]

        total_pipelines = len(preprocesamientos) * len(modelos_ids)
        print(f"\n=== Paso 2: Validación Cruzada Anidada de los {total_pipelines} pipelines (15 pliegues externos c/u) ===")
        print("  (Ajuste de hiperparámetros estrictamente interno por pliegue vía 5-fold CV)")

        todas_filas_pliegues: List[Dict[str, Any]] = []

        for prepro_id in preprocesamientos:
            for modelo_id in modelos_ids:
                nombre = f"{MODELOS[modelo_id][0]} ({prepro_id})"
                print(f"  -> Ejecutando CV anidada en 15 pliegues externos para: {nombre}...")
                t_inicio_pipe = time.time()

                resultados_pliegues = Parallel(n_jobs=-1)(
                    delayed(_evaluar_un_pliegue_externo)(
                        k=k,
                        ajuste_idx=ajuste,
                        val_idx=val,
                        prepro_id=prepro_id,
                        modelo_id=modelo_id,
                        X=X,
                        y=y,
                    )
                    for k, (ajuste, val) in enumerate(pliegues)
                )

                duracion_pipe = time.time() - t_inicio_pipe
                rmse_medio_cad = np.mean([r["rmse_cad"] for r in resultados_pliegues])
                print(f"     Completado en {duracion_pipe:.1f}s | RMSE medio: ${rmse_medio_cad:,.2f} CAD")
                todas_filas_pliegues.extend(resultados_pliegues)

        df_pliegues = pd.DataFrame(todas_filas_pliegues)
        df_pliegues.to_csv(ruta_pliegues, index=False)
        print(f"\nGuardados resultados por pliegue externo en {ruta_pliegues}")

    print("\n=== Paso 3: Agregación de métricas y cálculo de Intervalos de Confianza (95%) ===")
    filas_resumen = []
    metricas_nombres = ["rmse_cad", "mae_cad", "mape", "r2", "rmsle"]

    for (pipe_nombre, mod_id_comp, pre_id, m_id, mod_nom, fam), grupo in df_pliegues.groupby(
        ["pipeline_nombre", "modelo_id_completo", "preprocesamiento", "modelo", "modelo_nombre", "familia"]
    ):
        fila: Dict[str, Any] = {
            "pipeline_nombre": pipe_nombre,
            "modelo_id_completo": mod_id_comp,
            "preprocesamiento": pre_id,
            "modelo": m_id,
            "modelo_nombre": mod_nom,
            "familia": fam,
            "n_pliegues": len(grupo),
        }
        for met in metricas_nombres:
            ic = calcular_intervalo_confianza(grupo[met].values, nivel=0.95)
            fila[f"{met}_media"] = ic["media"]
            fila[f"{met}_std"] = ic["desv"]
            fila[f"{met}_ic_inf"] = ic["ic_inf"]
            fila[f"{met}_ic_sup"] = ic["ic_sup"]
        filas_resumen.append(fila)

    df_tabla_maestra = pd.DataFrame(filas_resumen).sort_values("rmse_cad_media").reset_index(drop=True)
    ruta_maestra = DIR_TABLAS / "t22_tabla_maestra_metricas.csv"
    df_tabla_maestra.to_csv(ruta_maestra, index=False)
    print(f"Guardada tabla maestra de métricas en {ruta_maestra}")

    print("\nTop 5 Pipelines según RMSE medio en CV anidada (CAD 1987):")
    print(df_tabla_maestra[["pipeline_nombre", "rmse_cad_media", "rmse_cad_std", "mae_cad_media", "mape_media", "r2_media", "rmsle_media"]].head(5).to_string(index=False))

    print("\n=== Paso 4: Inferencia Estadística Formal ===")
    # 4.1 Prueba Omnibus de Friedman y Ranking
    stat_friedman, p_friedman, df_ranking = prueba_friedman_ranking(df_pliegues, "rmse_cad")
    ruta_ranking = DIR_TABLAS / "t22_ranking_modelos.csv"
    df_ranking.to_csv(ruta_ranking, index=False)
    print(f"Prueba de Friedman: Chi2 = {stat_friedman:.4f}, p-valor = {p_friedman:.4e}")
    print(f"Guardado ranking de algoritmos en {ruta_ranking}")

    # 4.2 Pruebas t pareadas corregidas de Nadeau & Bengio contra el mejor modelo
    mejor_modelo_id = df_tabla_maestra.iloc[0]["modelo_id_completo"]
    mejor_modelo_nombre = df_tabla_maestra.iloc[0]["pipeline_nombre"]
    print(f"\nComparando todos los modelos contra el líder: {mejor_modelo_nombre} ({mejor_modelo_id})")

    valores_lider = df_pliegues[df_pliegues["modelo_id_completo"] == mejor_modelo_id].sort_values("pliegue")["rmse_cad"].values

    filas_t = []
    for _, fila_m in df_tabla_maestra.iterrows():
        otro_id = fila_m["modelo_id_completo"]
        if otro_id == mejor_modelo_id:
            continue
        valores_otro = df_pliegues[df_pliegues["modelo_id_completo"] == otro_id].sort_values("pliegue")["rmse_cad"].values
        diferencias = valores_otro - valores_lider  # Positivo si el otro modelo tiene mayor RMSE (es peor)

        res_t = prueba_t_corregida_nadeau_bengio(diferencias, n_ajuste=348, n_val=88)
        filas_t.append({
            "modelo_referencia": mejor_modelo_nombre,
            "modelo_comparado": fila_m["pipeline_nombre"],
            "modelo_id_completo": otro_id,
            "diferencia_rmse_cad_media": res_t["diferencia_media"],
            "error_estandar_corregido": res_t["error_estandar_corregido"],
            "t_estadistico": res_t["t_estadistico"],
            "p_valor_crudo": res_t["p_valor"],
        })

    df_pruebas_t = pd.DataFrame(filas_t).sort_values("p_valor_crudo").reset_index(drop=True)
    # Corrección Holm-Bonferroni
    m_tests = len(df_pruebas_t)
    df_pruebas_t["p_valor_holm"] = [min(1.0, p * (m_tests - i)) for i, p in enumerate(df_pruebas_t["p_valor_crudo"])]
    df_pruebas_t["significativo_al_5pct"] = df_pruebas_t["p_valor_holm"] < 0.05

    ruta_t = DIR_TABLAS / "t22_pruebas_estadisticas_t_corregida.csv"
    df_pruebas_t.to_csv(ruta_t, index=False)
    print(f"Guardadas pruebas t corregidas en {ruta_t}")

    print("\n=== Paso 5: Generación de Figuras para el Paper IEEE (T28) ===")

    # 1. Figura: Comparación de RMSE con Intervalos de Confianza 95% para los 18 pipelines
    fig, ax = plt.subplots(figsize=(10, 6.5))
    df_plot = df_tabla_maestra.sort_values("rmse_cad_media", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_plot))

    for p_id, color in COLORES_PREPRO.items():
        sub_mask = df_plot["preprocesamiento"] == p_id
        if not sub_mask.any():
            continue
        sub = df_plot[sub_mask]
        sub_indices = sub.index.values
        err_sub = [
            sub["rmse_cad_media"] - sub["rmse_cad_ic_inf"],
            sub["rmse_cad_ic_sup"] - sub["rmse_cad_media"],
        ]
        ax.errorbar(
            sub["rmse_cad_media"],
            sub_indices,
            xerr=err_sub,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=4,
            capthick=1.5,
            markersize=6,
            label=f"Preprocesamiento {p_id}",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["pipeline_nombre"], fontsize=9.5)
    ax.set_xlabel("RMSE medio en Validación Cruzada Anidada 5×3 (CAD 1987)", fontsize=10, fontweight="bold")
    ax.set_title("Comparación de Desempeño (Nested CV 5×3) con Intervalos de Confianza al 95%", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    ruta_fig1 = DIR_FIGURAS / "fig_t22_comparacion_rmse_ic.png"
    plt.savefig(ruta_fig1)
    plt.close()
    print(f"Guardada figura 1 en {ruta_fig1}")

    # 2. Figura: Ranking Medio de Algoritmos (Prueba de Friedman)
    fig, ax = plt.subplots(figsize=(8, 5))
    df_rank_plot = df_ranking.merge(
        df_tabla_maestra[["modelo_id_completo", "pipeline_nombre", "preprocesamiento"]],
        on="modelo_id_completo",
    ).sort_values("ranking_medio", ascending=False)

    colores_rank = [COLORES_PREPRO[p] for p in df_rank_plot["preprocesamiento"]]
    ax.barh(df_rank_plot["pipeline_nombre"], df_rank_plot["ranking_medio"], color=colores_rank, edgecolor="#333333", height=0.6)
    ax.set_xlabel("Ranking Medio en CV 5×3 (1 = Mejor desempeño)", fontsize=10, fontweight="bold")
    ax.set_title(f"Ranking Promedio de Algoritmos (Friedman $\\chi^2={stat_friedman:.1f}$, $p={p_friedman:.1e}$)", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    ruta_fig2 = DIR_FIGURAS / "fig_t22_ranking_modelos.png"
    plt.savefig(ruta_fig2)
    plt.close()
    print(f"Guardada figura 2 en {ruta_fig2}")

    # 3. Figura: Matriz de p-valores Nadeau-Bengio pareados (18x18)
    n_pips = len(df_tabla_maestra)
    p_matrix = np.ones((n_pips, n_pips))
    pips = df_tabla_maestra["modelo_id_completo"].tolist()
    labels = df_tabla_maestra["pipeline_nombre"].tolist()

    for i in range(n_pips):
        v_i = df_pliegues[df_pliegues["modelo_id_completo"] == pips[i]].sort_values("pliegue")["rmse_cad"].values
        for j in range(i + 1, n_pips):
            v_j = df_pliegues[df_pliegues["modelo_id_completo"] == pips[j]].sort_values("pliegue")["rmse_cad"].values
            res = prueba_t_corregida_nadeau_bengio(v_i - v_j, n_ajuste=348, n_val=88)
            p_matrix[i, j] = res["p_valor"]
            p_matrix[j, i] = res["p_valor"]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        p_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues_r",
        vmin=0.0,
        vmax=0.1,
        annot=False,
        cbar_kws={"label": "p-valor (Prueba t Nadeau-Bengio)"},
        ax=ax,
    )
    ax.set_title("Matriz de p-valores Pareados Nadeau-Bengio (Comparación de Modelos)", fontsize=11, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    ruta_fig3 = DIR_FIGURAS / "fig_t22_matriz_pvalores_nadeau_bengio.png"
    plt.savefig(ruta_fig3)
    plt.close()
    print(f"Guardada figura 3 en {ruta_fig3}")

    print("\n¡Validación Cruzada Anidada T22 completada con éxito total!")
    return {
        "pliegues": df_pliegues,
        "tabla_maestra": df_tabla_maestra,
        "pruebas_t": df_pruebas_t,
        "ranking": df_ranking,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="T22 CV Anidada")
    parser.add_argument("--forzar", action="store_true", default=True)
    args, _ = parser.parse_known_args()
    ejecutar_experimento_t22(forzar_recalculo=args.forzar)


if __name__ == "__main__":
    main()
