"""Script ejecutor para los experimentos de T20 (KNN y SVR).

Ejecuta:
1. Búsqueda de hiperparámetros de KNN (k, weights, p) para P1, P2 y P3.
2. Búsqueda de hiperparámetros de SVR (C, gamma, epsilon) para P1, P2 y P3.
3. Evaluación pareada sobre CV repetida 5×3 de T12.
4. Experimento de ablación / justificación de la necesidad de escalado de predictoras en P1.
5. Exportación de tablas en reports/tables/.
6. Generación de figuras en reports/figures/.
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
from src.modelos_no_lineales import (
    optimizar_hiperparametros,
    evaluar_experimento_cv5x3,
    experimento_impacto_escalado,
)

# Estilo gráfico consistente con el paper y entregables
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 200

COLORES = {
    "P1": "#4A6FA5",  # Azul base
    "P2": "#16697A",  # Verde petróleo / azul oscuro
    "P3": "#E07A5F",  # Coral / terracota
    "KNN": "#3D5A80",
    "SVR": "#EE6C4D",
    "Con escalado": "#2A9D8F",
    "Sin escalado": "#E76F51",
}


def main():
    print("=== Paso 1: Carga aislada del conjunto de entrenamiento (n=436) ===")
    X, y, evidencia = cargar_entrenamiento()
    print(f"Cargadas {len(X)} observaciones de entrenamiento. Huella partición: {evidencia['sha256_particion'][:12]}...")
    
    pliegues = crear_pliegues(y)
    print(f"Generados {len(pliegues)} pliegues pareados (CV 5×3).")

    print("\n=== Paso 2: Búsqueda de hiperparámetros (GridSearchCV, 5-fold) ===")
    mejores_params = {}
    dfs_busqueda = []

    modelos = [("M3", "KNN"), ("M4", "SVR")]
    preprocesamientos = ["P1", "P2", "P3"]

    for m_id, m_nombre in modelos:
        for p_id in preprocesamientos:
            print(f"  Optimizando {m_nombre} sobre {p_id}...")
            _, params, df_cv = optimizar_hiperparametros(X, y, m_id, p_id)
            mejores_params[(p_id, m_id)] = params
            dfs_busqueda.append(df_cv)
            print(f"    -> Mejores parámetros: {params}")

    df_busqueda_completa = pd.concat(dfs_busqueda, ignore_index=True)
    
    # Exportar cuadrícula de búsqueda
    ruta_busqueda = r"D:\ml-precio-vivienda\reports\tables\t20_busqueda_hiperparametros.csv"
    df_busqueda_completa.to_csv(ruta_busqueda, index=False)
    print(f"Guardada cuadrícula de búsqueda en {ruta_busqueda}")

    # Tabla con los mejores hiperparámetros por modelo y preprocesamiento
    filas_mejores = []
    for (p_id, m_id), params in mejores_params.items():
        fila = {
            "preprocesamiento": p_id,
            "modelo": m_id,
            "modelo_nombre": "KNN" if m_id == "M3" else "SVR (RBF)",
            "mejores_hiperparametros": json.dumps(params),
        }
        fila.update(params)
        filas_mejores.append(fila)
    df_mejores = pd.DataFrame(filas_mejores)
    ruta_mejores = r"D:\ml-precio-vivienda\reports\tables\t20_mejores_hiperparametros.csv"
    df_mejores.to_csv(ruta_mejores, index=False)
    print(f"Guardados mejores hiperparámetros en {ruta_mejores}")

    print("\n=== Paso 3: Evaluación pareada en CV repetida 5×3 (T12) ===")
    df_resumen, df_pliegues = evaluar_experimento_cv5x3(X, y, pliegues, mejores_params)
    
    ruta_resumen = r"D:\ml-precio-vivienda\reports\tables\t20_resultados.csv"
    ruta_pliegues = r"D:\ml-precio-vivienda\reports\tables\t20_pliegues.csv"
    df_resumen.to_csv(ruta_resumen, index=False)
    df_pliegues.to_csv(ruta_pliegues, index=False)
    print(f"Guardados resultados agregados en {ruta_resumen}")
    print(f"Guardados resultados por pliegue en {ruta_pliegues}")

    print("\nResumen comparativo (ordenado por RMSE en CAD 1987):")
    print(df_resumen[["preprocesamiento", "modelo_nombre", "rmse_cad_media", "rmse_cad_std", "mae_cad_media", "mape_media", "r2_media"]].to_string(index=False))

    print("\n=== Paso 4: Experimento de ablación - Justificación del escalado (P1) ===")
    df_escalado = experimento_impacto_escalado(X, y, pliegues)
    ruta_escalado = r"D:\ml-precio-vivienda\reports\tables\t20_impacto_escalado.csv"
    df_escalado.to_csv(ruta_escalado, index=False)
    print(f"Guardado impacto de escalado en {ruta_escalado}")
    print(df_escalado.to_string(index=False))

    print("\n=== Paso 5: Generación de figuras ===")
    os.makedirs(r"D:\ml-precio-vivienda\reports\figures", exist_ok=True)

    # 1. Figura: Impacto de k y ponderación de pesos en KNN (bajo P2)
    df_knn_p2 = df_busqueda_completa[(df_busqueda_completa["modelo"] == "M3") & (df_busqueda_completa["preprocesamiento"] == "P2")].copy()
    df_knn_p2["rmse_cad"] = -df_knn_p2["mean_test_score"] / FACTOR_KAGGLE
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for weights_val, marker, color in [("uniform", "o", "#2B5C8F"), ("distance", "s", "#D95F02")]:
        sub = df_knn_p2[(df_knn_p2["param_weights"] == weights_val) & (df_knn_p2["param_p"] == 2)].sort_values("param_n_neighbors")
        ax.plot(
            sub["param_n_neighbors"],
            sub["rmse_cad"],
            marker=marker,
            linewidth=2,
            label=f"Pesos: {weights_val} (distancia L2)",
            color=color,
        )
    ax.set_title("KNN bajo Preprocesamiento P2: RMSE vs Número de Vecinos ($k$)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Número de vecinos ($k$)", fontsize=10)
    ax.set_ylabel("RMSE de validación cruzada (CAD 1987)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    ruta_fig1 = r"D:\ml-precio-vivienda\reports\figures\fig_t20_knn_k_weights.png"
    plt.savefig(ruta_fig1)
    plt.close()
    print(f"Guardada figura 1 en {ruta_fig1}")

    # 2. Figura: Mapa de calor de hiperparámetros SVR (C vs gamma) bajo P2
    df_svr_p2 = df_busqueda_completa[(df_busqueda_completa["modelo"] == "M4") & (df_busqueda_completa["preprocesamiento"] == "P2")].copy()
    # Tomamos el mejor epsilon
    mejor_eps = mejores_params[("P2", "M4")]["epsilon"]
    sub_svr = df_svr_p2[df_svr_p2["param_epsilon"] == mejor_eps].copy()
    sub_svr["rmse_cad"] = -sub_svr["mean_test_score"] / FACTOR_KAGGLE
    pivot_svr = sub_svr.pivot(index="param_C", columns="param_gamma", values="rmse_cad")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot_svr,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu_r",
        cbar_kws={"label": "RMSE CV (CAD 1987)"},
        ax=ax,
    )
    ax.set_title(f"SVR (RBF) bajo P2: RMSE vs $C$ y $\\gamma$ (con $\\epsilon={mejor_eps}$)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Parámetro $\\gamma$", fontsize=10)
    ax.set_ylabel("Parámetro de regularización $C$", fontsize=10)
    plt.tight_layout()
    ruta_fig2 = r"D:\ml-precio-vivienda\reports\figures\fig_t20_svr_heatmap.png"
    plt.savefig(ruta_fig2)
    plt.close()
    print(f"Guardada figura 2 en {ruta_fig2}")

    # 3. Figura: Comparación de preprocesamientos P1, P2 y P3 para KNN y SVR
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    width = 0.25

    p1_vals = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P1")]["rmse_cad_media"].values[0] for m in ["M3", "M4"]]
    p2_vals = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P2")]["rmse_cad_media"].values[0] for m in ["M3", "M4"]]
    p3_vals = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P3")]["rmse_cad_media"].values[0] for m in ["M3", "M4"]]

    p1_err = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P1")]["rmse_cad_std"].values[0] for m in ["M3", "M4"]]
    p2_err = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P2")]["rmse_cad_std"].values[0] for m in ["M3", "M4"]]
    p3_err = [df_resumen[(df_resumen["modelo"] == m) & (df_resumen["preprocesamiento"] == "P3")]["rmse_cad_std"].values[0] for m in ["M3", "M4"]]

    rects1 = ax.bar(x - width, p1_vals, width, yerr=p1_err, label="P1 (base: StandardScaler)", color="#4A6FA5", capsize=4)
    rects2 = ax.bar(x, p2_vals, width, yerr=p2_err, label="P2 (transformación: log-precio + RobustScaler)", color="#16697A", capsize=4)
    rects3 = ax.bar(x + width, p3_vals, width, yerr=p3_err, label="P3 (ingeniería + selección)", color="#E07A5F", capsize=4)

    ax.set_title("Comparación de Preprocesamientos (P1, P2, P3) en Modelos No Lineales", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("RMSE medio CV 5×3 (CAD 1987)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["K-Nearest Neighbors (M3)", "Support Vector Regression RBF (M4)"], fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    ruta_fig3 = r"D:\ml-precio-vivienda\reports\figures\fig_t20_comparacion_preprocesamientos.png"
    plt.savefig(ruta_fig3)
    plt.close()
    print(f"Guardada figura 3 en {ruta_fig3}")

    # 4. Figura: Impacto dramático del escalado de variables (P1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    df_esc_plot = df_escalado.pivot(index="nombre", columns="escalado_predictoras", values="rmse_cad_media")
    df_esc_err = df_escalado.pivot(index="nombre", columns="escalado_predictoras", values="rmse_cad_std")

    x = np.arange(len(df_esc_plot.index))
    width = 0.35

    ax.bar(x - width/2, df_esc_plot["Con escalado"], width, yerr=df_esc_err["Con escalado"], label="Con escalado (StandardScaler)", color="#2A9D8F", capsize=4)
    ax.bar(x + width/2, df_esc_plot["Sin escalado"], width, yerr=df_esc_err["Sin escalado"], label="Sin escalado (Variables crudas)", color="#E76F51", capsize=4)

    ax.set_title("Impacto del Escalado de Predictoras en KNN y SVR (Preprocesamiento P1)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("RMSE medio CV 5×3 (CAD 1987)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df_esc_plot.index, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    ruta_fig4 = r"D:\ml-precio-vivienda\reports\figures\fig_t20_impacto_escalado.png"
    plt.savefig(ruta_fig4)
    plt.close()
    print(f"Guardada figura 4 en {ruta_fig4}")

    print("\n¡Experimento T20 completado con éxito!")


if __name__ == "__main__":
    main()
