"""Figuras y tablas de resultados del paper (en inglés), a partir de las tablas de T22 a T25.

Desde la raíz del repositorio:

    python docs/paper/figures/make_result_figures.py

Lee de reports/tables/ (o de la carpeta indicada en la variable de entorno TABLAS_PAPER):
t22_pliegues_anidados.csv, t23_complejidad_empirica.csv, t24_comparacion_coeficientes_vs_shap.csv
y t25_predicciones_prueba.csv. Escribe las figuras en docs/paper/figures/ y las tablas LaTeX en
docs/paper/tables/, que main.tex incluye con \\input.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[3]
AQUI = Path(__file__).resolve().parent
TABLAS = Path(os.environ.get("TABLAS_PAPER", RAIZ / "reports" / "tables"))
SALIDA_TEX = AQUI.parent / "tables"
SEMILLA = 5007
N_AJUSTE, N_VAL = 348, 88  # tamaños de ajuste y validación en cada pliegue externo (5 pliegues sobre 436 filas)

MODELOS = {"M1": "OLS (hedonic)", "M2": "Ridge", "M3": "KNN", "M4": "SVR (RBF)", "M5": "Random forest", "M6": "XGBoost"}
COLOR_P = {"P1": "#1f4e79", "P2": "#2e8b57", "P3": "#c0502f"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300,
})


def fmt_p(p):
    return "$<$0.01" if p < 0.01 else f"{p:.2f}"


def miles(v):
    return f"{v:,.0f}"


def t_corregida(d):
    """Prueba t corregida de Nadeau y Bengio (2003) para CV repetida; devuelve (media, p)."""
    d = np.asarray(d, dtype=float)
    j = len(d)
    ee = np.sqrt((1 / j + N_VAL / N_AJUSTE) * np.var(d, ddof=1))
    t = d.mean() / ee
    return d.mean(), 2 * stats.t.sf(abs(t), df=j - 1)


def resumen_cv(pl):
    """Media e IC 95 % (t de Student sobre los 15 pliegues) de cada pipeline."""
    filas = []
    for (pid, p, m), g in pl.groupby(["modelo_id_completo", "preprocesamiento", "modelo"]):
        fila = {"pipeline": pid, "prepro": p, "modelo": m}
        for met in ["rmse_cad", "mae_cad", "mape", "r2"]:
            v = g[met].values
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            fila[met], fila[met + "_lo"], fila[met + "_hi"] = v.mean(), v.mean() - h, v.mean() + h
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("rmse_cad").reset_index(drop=True)


def fig_cv(res):
    fig, ax = plt.subplots(figsize=(3.45, 3.3))
    res = res.iloc[::-1].reset_index(drop=True)
    for i, r in res.iterrows():
        ax.errorbar(r.rmse_cad / 1000, i, xerr=[[(r.rmse_cad - r.rmse_cad_lo) / 1000], [(r.rmse_cad_hi - r.rmse_cad) / 1000]],
                    fmt="o", ms=3, color=COLOR_P[r.prepro], elinewidth=0.9, capsize=1.5)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels([f"{MODELOS[m]} {p}" for m, p in zip(res.modelo, res.prepro)], fontsize=6.5)
    ax.set_xlabel("Nested CV RMSE (thousand CAD, 1987)")
    for p, c in COLOR_P.items():
        ax.plot([], [], "o", color=c, ms=3, label=p)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_cv_rmse.pdf")
    plt.close(fig)


def fig_pareto(costo, res):
    d = costo.merge(res[["pipeline", "rmse_cad"]], left_on="pipeline_id", right_on="pipeline")
    familias = {"M1": "#1f4e79", "M2": "#1f4e79", "M3": "#d9a441", "M4": "#7a6c9d", "M5": "#2e8b57", "M6": "#2e8b57"}
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.scatter(d.tiempo_ajuste_mediana_s * 1000, d.rmse_cad / 1000, s=14, c=[familias[m] for m in d.modelo_id], zorder=3)
    s = d.sort_values("tiempo_ajuste_mediana_s")
    frente = s[s.rmse_cad == s.rmse_cad.cummin()]
    ax.plot(frente.tiempo_ajuste_mediana_s * 1000, frente.rmse_cad / 1000, "--", color="0.4", lw=0.8, zorder=2)
    etiquetas = {"M1_P1": (4, 4), "M1_P2": (4, -9), "M6_P1": (4, 3), "M5_P1": (-30, 6), "M3_P1": (4, 3)}
    for _, r in d[d.pipeline_id.isin(etiquetas)].iterrows():
        nombre = f"{MODELOS[r.modelo_id].split(' (')[0]} {r.preprocesamiento}"
        ax.annotate(nombre, (r.tiempo_ajuste_mediana_s * 1000, r.rmse_cad / 1000), xytext=etiquetas[r.pipeline_id],
                    textcoords="offset points", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("Median training time per fold (ms, log scale)")
    ax.set_ylabel("Nested CV RMSE (thousand CAD)")
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_pareto.pdf")
    plt.close(fig)


def metricas(y, p):
    r = y - p
    return np.sqrt(np.mean(r ** 2)), np.mean(np.abs(r)), 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)


def prueba_sellada(pred):
    y = pred.precio_real_cad.values
    rng = np.random.default_rng(SEMILLA)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(5000)]
    filas = []
    for pid, nombre in [("M1_P2", "OLS (hedonic) P2"), ("M6_P1", "XGBoost P1"), ("M5_P1", "Random forest P1")]:
        p = pred[f"pred_{pid}_cad"].values
        base = metricas(y, p)
        b = np.array([metricas(y[i], p[i]) for i in idx])
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        filas.append((nombre, base, lo, hi))
    # Figura: real contra predicho del modelo elegido por CV
    p = pred.pred_M1_P2_cad.values
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.scatter(p / 1000, y / 1000, s=7, color="#1f4e79", alpha=0.75, lw=0)
    lim = [20, 190]
    ax.plot(lim, lim, "--", color="0.4", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted price (thousand CAD, 1987)")
    ax.set_ylabel("Actual price (thousand CAD, 1987)")
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_test.pdf")
    plt.close(fig)
    return filas


def escribir(nombre, texto):
    SALIDA_TEX.mkdir(exist_ok=True)
    (SALIDA_TEX / nombre).write_text(texto, encoding="utf-8")


def main():
    pl = pd.read_csv(TABLAS / "t22_pliegues_anidados.csv")
    res = resumen_cv(pl)
    fig_cv(res)

    # Tabla de RMSE por modelo y preprocesamiento (CV anidada)
    piv = res.pivot(index="modelo", columns="prepro", values="rmse_cad")
    mape = res.pivot(index="modelo", columns="prepro", values="mape")
    r2 = res.pivot(index="modelo", columns="prepro", values="r2")
    lineas = []
    for m in MODELOS:
        mejor = piv.loc[m].idxmin()
        celdas = [(r"\textbf{%s}" if p == mejor else "%s") % miles(piv.loc[m, p]) for p in ["P1", "P2", "P3"]]
        lineas.append(f"{MODELOS[m]} & " + " & ".join(celdas) + f" & {mape.loc[m, mejor]:.1f} & {r2.loc[m, mejor]:.2f} \\\\")
    escribir("tab_cv.tex", "\n".join(lineas) + "\n")

    # Pruebas t corregidas contra el mejor pipeline (OLS P2), con Holm
    ref = pl[pl.modelo_id_completo == res.pipeline[0]].sort_values(["repeticion", "pliegue"]).rmse_cad.values
    comp = []
    for pid in res.pipeline[1:]:
        otro = pl[pl.modelo_id_completo == pid].sort_values(["repeticion", "pliegue"]).rmse_cad.values
        dm, p = t_corregida(otro - ref)
        comp.append([pid, dm, p])
    comp = pd.DataFrame(comp, columns=["pipeline", "dif", "p"]).sort_values("p").reset_index(drop=True)
    k = len(comp)
    comp["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(comp.p)]))
    comp = comp.set_index("pipeline")
    mejores = res.groupby("modelo").first().pipeline  # mejor preprocesamiento de cada modelo
    lineas = []
    for m in ["M2", "M4", "M6", "M5", "M3"]:
        pid = mejores[m]
        c = comp.loc[pid]
        lineas.append(f"{MODELOS[m]} {pid[-2:]} & +{miles(c.dif)} & {fmt_p(c.p)} & {fmt_p(c.p_holm)} \\\\")
    escribir("tab_ttest.tex", "\n".join(lineas) + "\n")
    comp.round(4).to_csv(AQUI.parent / "tables" / "ttest_vs_best.csv")
    print(res[["pipeline", "rmse_cad", "mape", "r2"]].round(3).to_string())
    print(comp.round(3).to_string())

    # Costo empírico
    costo = pd.read_csv(TABLAS / "t23_complejidad_empirica.csv")
    fig_pareto(costo, res)
    lineas = []
    for m in MODELOS:
        r = costo[costo.pipeline_id == mejores[m]].iloc[0]
        lineas.append(f"{MODELOS[m]} {r.preprocesamiento} & {r.tiempo_ajuste_mediana_s * 1000:,.0f} & "
                      f"{r.tiempo_inferencia_lote_ms:.1f} & {miles(r.n_parametros_o_nodos)} & {r.tamano_serializado_kb:,.0f} \\\\")
    escribir("tab_cost.tex", "\n".join(lineas) + "\n")

    # Prueba sellada con IC por bootstrap
    filas = prueba_sellada(pd.read_csv(TABLAS / "t25_predicciones_prueba.csv"))
    lineas = [f"{n} & {miles(b[0])} [{miles(lo[0])}, {miles(hi[0])}] & {miles(b[1])} [{miles(lo[1])}, {miles(hi[1])}] & "
              f"{b[2]:.2f} [{lo[2]:.2f}, {hi[2]:.2f}] \\\\" for n, b, lo, hi in filas]
    escribir("tab_test.tex", "\n".join(lineas) + "\n")
    for n, b, lo, hi in filas:
        print(n, np.round(b, 3), np.round(lo, 3), np.round(hi, 3))

    # Coeficientes del hedónico contra SHAP de XGBoost (cinco atributos con más peso)
    shap_tab = pd.read_csv(TABLAS / "t24_comparacion_coeficientes_vs_shap.csv").sort_values("ranking_shap_xgb").head(6)
    nombres = {"area": "Lot size", "bathrooms": "Bathrooms", "airconditioning": "Central air cond.", "parking": "Garage places",
               "stories": "Stories", "furnishingstatus_1": "Semi-furnished", "prefarea": "Preferred area"}
    lineas = [f"{nombres.get(r.variable, r.variable)} & {int(r.ranking_ols)} & {r.abs_beta_estandarizado:.2f} & "
              f"{int(r.ranking_shap_xgb)} & {miles(r.shap_mean_abs_xgb_cad)} \\\\" for r in shap_tab.itertuples()]
    escribir("tab_shap.tex", "\n".join(lineas) + "\n")


if __name__ == "__main__":
    main()
