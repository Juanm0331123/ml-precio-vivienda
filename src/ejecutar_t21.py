"""Script ejecutor para los experimentos de T21 (Random Forest y XGBoost).

Ejecuta:
1. Búsqueda amplia de hiperparámetros (RandomizedSearchCV, CV=5, n_iter=30) para RF y XGBoost en P1, P2 y P3.
2. Evaluación pareada sobre la CV repetida 5×3 de T12.
3. Extracción de importancia de variables para ambos ensambles.
4. Exportación de tablas en reports/tables/.
5. Generación de figuras en reports/figures/.
"""

import os
os.environ["MPLCONFIGDIR"] = r"D:\venvs\tmp"
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.evaluacion import cargar_entrenamiento, crear_pliegues, FACTOR_KAGGLE
from src.ensambles import (
    crear_pipeline_ensamble,
    optimizar_ensamble,
    evaluar_ensambles_cv5x3,
    extraer_importancia_caracteristicas,
)

# Estilo gráfico consistente
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 200


def main():
    print("=== Paso 1: Carga aislada del conjunto de entrenamiento (n=436) ===")
    X, y, evidencia = cargar_entrenamiento()
    print(f"Cargadas {len(X)} filas. Huella: {evidencia['sha256_particion'][:12]}...")

    pliegues = crear_pliegues(y)
    print(f"Generados {len(pliegues)} pliegues pareados (CV 5×3).")

    print("\n=== Paso 2: Búsqueda amplia de hiperparámetros (RandomizedSearchCV, 30 iteraciones) ===")
    mejores_params = {}
    dfs_busqueda = []
    mejores_estimadores = {}

    modelos = [("M5", "Random Forest"), ("M6", "XGBoost")]
    preprocesamientos = ["P1", "P2", "P3"]

    for m_id, m_nombre in modelos:
        for p_id in preprocesamientos:
            print(f"  Optimizando {m_nombre} sobre {p_id}...")
            best_est, params, df_cv = optimizar_ensamble(X, y, m_id, p_id, n_iter=30)
            mejores_params[(p_id, m_id)] = params
            mejores_estimadores[(p_id, m_id)] = best_est
            dfs_busqueda.append(df_cv)
            print(f"    -> Mejores parámetros: {params}")

    df_busqueda_completa = pd.concat(dfs_busqueda, ignore_index=True)

    ruta_busqueda = r"D:\ml-precio-vivienda\reports\tables\t21_busqueda_hiperparametros.csv"
    df_busqueda_completa.to_csv(ruta_busqueda, index=False)
    print(f"Guardada búsqueda de hiperparámetros en {ruta_busqueda}")

    # Tabla con los mejores parámetros
    filas_mejores = []
    for (p_id, m_id), params in mejores_params.items():
        fila = {
            "preprocesamiento": p_id,
            "modelo": m_id,
            "modelo_nombre": "Random Forest" if m_id == "M5" else "XGBoost",
            "mejores_hiperparametros": json.dumps(params),
        }
        fila.update(params)
        filas_mejores.append(fila)
    df_mejores = pd.DataFrame(filas_mejores)
    ruta_mejores = r"D:\ml-precio-vivienda\reports\tables\t21_mejores_hiperparametros.csv"
    df_mejores.to_csv(ruta_mejores, index=False)
    print(f"Guardados mejores hiperparámetros en {ruta_mejores}")

    print("\n=== Paso 3: Evaluación pareada en CV repetida 5×3 (T12) ===")
    df_resumen, df_pliegues = evaluar_ensambles_cv5x3(X, y, pliegues, mejores_params)

    ruta_resumen = r"D:\ml-precio-vivienda\reports\tables\t21_resultados.csv"
    ruta_pliegues = r"D:\ml-precio-vivienda\reports\tables\t21_pliegues.csv"
    df_resumen.to_csv(ruta_resumen, index=False)
    df_pliegues.to_csv(ruta_pliegues, index=False)
    print(f"Guardados resultados en {ruta_resumen}")

    print("\nResumen comparativo ensambles (ordenado por RMSE en CAD 1987):")
    print(df_resumen[["preprocesamiento", "modelo_nombre", "rmse_cad_media", "rmse_cad_std", "mae_cad_media", "mape_media", "r2_media"]].to_string(index=False))

    print("\n=== Paso 4: Extracción de importancia de variables ===")
    # Extraer de RF y XGBoost bajo P2 (mejor preprocesamiento global)
    pipe_rf_p2 = mejores_estimadores[("P2", "M5")]
    pipe_xgb_p2 = mejores_estimadores[("P2", "M6")]

    df_imp_rf = extraer_importancia_caracteristicas(pipe_rf_p2, X, y, "P2")
    df_imp_rf["modelo"] = "Random Forest"
    df_imp_rf["preprocesamiento"] = "P2"

    df_imp_xgb = extraer_importancia_caracteristicas(pipe_xgb_p2, X, y, "P2")
    df_imp_xgb["modelo"] = "XGBoost"
    df_imp_xgb["preprocesamiento"] = "P2"

    df_importancias = pd.concat([df_imp_rf, df_imp_xgb], ignore_index=True)
    ruta_imp = r"D:\ml-precio-vivienda\reports\tables\t21_importancia_caracteristicas.csv"
    df_importancias.to_csv(ruta_imp, index=False)
    print(f"Guardada importancia de variables en {ruta_imp}")

    print("\n=== Paso 5: Generación de figuras ===")
    os.makedirs(r"D:\ml-precio-vivienda\reports\figures", exist_ok=True)

    # 1. Figura: Importancia de variables Random Forest (P2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_rf_plot = df_imp_rf.sort_values("importancia", ascending=True)
    ax.barh(df_rf_plot["variable"], df_rf_plot["importancia"] * 100, color="#2B5C8F", edgecolor="#1a3b5c")
    ax.set_title("Random Forest (P2): Importancia Relativa de Variables (MDI)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Importancia relativa (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    ruta_fig1 = r"D:\ml-precio-vivienda\reports\figures\fig_t21_importancia_rf.png"
    plt.savefig(ruta_fig1)
    plt.close()
    print(f"Guardada figura 1 en {ruta_fig1}")

    # 2. Figura: Importancia de variables XGBoost (P2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_xgb_plot = df_imp_xgb.sort_values("importancia", ascending=True)
    ax.barh(df_xgb_plot["variable"], df_xgb_plot["importancia"] * 100, color="#D95F02", edgecolor="#8c3d01")
    ax.set_title("XGBoost (P2): Importancia Relativa de Variables (Gain)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Importancia relativa (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    ruta_fig2 = r"D:\ml-precio-vivienda\reports\figures\fig_t21_importancia_xgb.png"
    plt.savefig(ruta_fig2)
    plt.close()
    print(f"Guardada figura 2 en {ruta_fig2}")

    # 3. Figura: Comparación de preprocesamientos para Ensambles
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    width = 0.25

    p1_vals = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P1")]["rmse_cad_media"].values[0] for m in ["M5", "M6"]]
    p2_vals = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P2")]["rmse_cad_media"].values[0] for m in ["M5", "M6"]]
    p3_vals = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P3")]["rmse_cad_media"].values[0] for m in ["M5", "M6"]]

    p1_err = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P1")]["rmse_cad_std"].values[0] for m in ["M5", "M6"]]
    p2_err = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P2")]["rmse_cad_std"].values[0] for m in ["M5", "M6"]]
    p3_err = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P3")]["rmse_cad_std"].values[0] for m in ["M5", "M6"]]

    ax.bar(x - width, p1_vals, width, yerr=p1_err, label="P1 (base)", color="#4A6FA5", capsize=4)
    ax.bar(x, p2_vals, width, yerr=p2_err, label="P2 (log-precio + Yeo-Johnson)", color="#16697A", capsize=4)
    ax.bar(x + width, p3_vals, width, yerr=p3_err, label="P3 (ingeniería + selección)", color="#E07A5F", capsize=4)

    ax.set_title("Comparación de Preprocesamientos en Ensambles de Árboles", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("RMSE medio CV 5×3 (CAD 1987)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["Random Forest (M5)", "XGBoost (M6)"], fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    ruta_fig3 = r"D:\ml-precio-vivienda\reports\figures\fig_t21_comparacion_ensambles.png"
    plt.savefig(ruta_fig3)
    plt.close()
    print(f"Guardada figura 3 en {ruta_fig3}")

    # 4. Figura: Distribución de RMSE pliegue a pliegue (RF vs XGBoost)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_plot_pliegues = df_pliegues[df_pliegues["preprocesamiento"] == "P2"]
    sns.boxplot(
        data=df_plot_pliegues,
        x="modelo_nombre",
        y="rmse_cad",
        palette=["#2B5C8F", "#D95F02"],
        ax=ax,
        width=0.4,
    )
    sns.stripplot(
        data=df_plot_pliegues,
        x="modelo_nombre",
        y="rmse_cad",
        color="black",
        alpha=0.6,
        jitter=0.15,
        ax=ax,
    )
    ax.set_title("Distribución de RMSE en CV 5×3 bajo Preprocesamiento P2", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Modelo", fontsize=10)
    ax.set_ylabel("RMSE por pliegue (CAD 1987)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    ruta_fig4 = r"D:\ml-precio-vivienda\reports\figures\fig_t21_rf_vs_xgb_prepro.png"
    plt.savefig(ruta_fig4)
    plt.close()
    print(f"Guardada figura 4 en {ruta_fig4}")

    print("\n¡Experimento T21 completado con éxito!")


if __name__ == "__main__":
    main()
