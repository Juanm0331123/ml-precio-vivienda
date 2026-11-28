"""Figuras de resultados para la presentación final (T30): diapositivas 9, 10 y 12.

Desde la raíz del repositorio, cuando T22 a T25 estén en main:
    python docs/entrega_final/figuras_resultados.py
Lee reports/tables/t22_tabla_maestra_metricas.csv, t23_complejidad_empirica.csv y t25_predicciones_prueba.csv.
Las cifras de la tabla de hipótesis se copian de t22_pruebas_estadisticas_t_corregida.csv,
t22_tabla_maestra_metricas.csv y t25_contraste_hipotesis_h1.csv: si se vuelve a correr T22 o T25, hay que actualizarlas.
Salidas en reports/figures/: deck_hipotesis.png, deck_precision_vs_costo.png y deck_prueba_sellada.png.
"""
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
TABLAS = Path(os.environ.get("DIR_TABLAS_DECK", RAIZ / "reports" / "tables"))
FIGURAS = RAIZ / "reports" / "figures"
SEMILLA = 5007
AZ="#0f2a47"; AC="#2b6cc4"; GR="#5a6b7e"; LAD="#c0502f"; VER="#2e8b57"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":15,"axes.edgecolor":"#c9d1dc","axes.labelcolor":AZ,"xtick.color":GR,"ytick.color":GR})

# --- 10: Pareto precisión vs costo
t=pd.read_csv(TABLAS / "t23_complejidad_empirica.csv")
fam_col={"lineal":AC,"kernel":"#7a6c9d","árboles":VER,"distancia":"#d9a441"}
fig,ax=plt.subplots(figsize=(11,6.2),dpi=200)
for f,g in t.groupby("familia"):
    ax.scatter(g.tiempo_ajuste_mediana_s*1000,g.rmse_cad1987_media,s=130,color=fam_col[f],edgecolor="white",lw=1.2,label={"lineal":"Lineales","kernel":"SVR","árboles":"Árboles","distancia":"KNN"}[f],zorder=3)
s=t.sort_values("tiempo_ajuste_mediana_s"); front=[]; best=np.inf
for _,r in s.iterrows():
    if r.rmse_cad1987_media<best: front.append(r); best=r.rmse_cad1987_media
fr=pd.DataFrame(front)
ax.plot(fr.tiempo_ajuste_mediana_s*1000,fr.rmse_cad1987_media,"--",color=GR,lw=1.5,zorder=2,label="Frente de Pareto")
lab={"M1_P2":"OLS P2","M1_P1":"OLS P1","M6_P1":"XGBoost P1","M5_P1":"Random Forest P1","M3_P1":"KNN P1","M4_P2":"SVR P2"}
off={"M1_P2":(-92,-14),"M1_P1":(-6,-46),"M6_P1":(-30,22),"M5_P1":(-40,14),"M3_P1":(8,8),"M4_P2":(14,-30)}
for _,r in t[t.pipeline_id.isin(lab)].iterrows():
    ax.annotate(f"{lab[r.pipeline_id]}\n{r.tiempo_ajuste_mediana_s*1000:,.0f} ms".replace(",","."),(r.tiempo_ajuste_mediana_s*1000,r.rmse_cad1987_media),xytext=off[r.pipeline_id],textcoords="offset points",fontsize=12.5,color=AZ)
ax.set_xscale("log"); ax.set_xlabel("Tiempo de entrenamiento por pliegue (ms, escala log)"); ax.set_ylabel("RMSE en CV anidada (CAD de 1987)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:,.0f}".replace(",",".")))
ax.grid(alpha=.35); ax.legend(frameon=False,loc="upper right",fontsize=12.5)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIGURAS / "deck_precision_vs_costo.png"); plt.close()

# --- 12: prueba sellada OLS P2 + IC bootstrap
d=pd.read_csv(TABLAS / "t25_predicciones_prueba.csv"); y=d.precio_real_cad.values; p=d.pred_M1_P2_cad.values
fig,ax=plt.subplots(figsize=(11,6.2),dpi=200)
ax.scatter(p/1000,y/1000,s=55,color=AC,alpha=.8,edgecolor="white",lw=.6,zorder=3)
lim=[20,190]; ax.plot(lim,lim,"--",color=GR,lw=1.3,label="Predicción perfecta"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Precio predicho (miles de CAD de 1987)"); ax.set_ylabel("Precio real (miles de CAD de 1987)")
ax.set_title("OLS hedónico con P2 en las 109 ventas de prueba",color=AZ,fontsize=16,loc="left",fontweight="bold")
def metricas(y, p):
    r = y - p
    return np.sqrt(np.mean(r ** 2)), np.mean(np.abs(r)), 1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)
rng = np.random.default_rng(SEMILLA)
base = metricas(y, p)
B = np.array([metricas(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(5000))])
lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
cv = pd.read_csv(TABLAS / "t22_tabla_maestra_metricas.csv").set_index("modelo_id_completo").loc["M1_P2", "rmse_cad_media"]
miles = lambda v: f"{v:,.0f}".replace(",", ".")
dec = lambda v: f"{v:.2f}".replace(".", ",")
txt = (f"RMSE  {miles(base[0])}  [{miles(lo[0])} – {miles(hi[0])}]\n"
       f"MAE   {miles(base[1])}  [{miles(lo[1])} – {miles(hi[1])}]\n"
       f"R²    {dec(base[2])}    [{dec(lo[2])} – {dec(hi[2])}]\n\n"
       f"IC 95 % por bootstrap (5.000 remuestreos)\nCV anidada esperaba RMSE {miles(cv)}")
ax.text(0.98,0.04,txt,transform=ax.transAxes,ha="right",va="bottom",fontsize=13.5,family="DejaVu Sans Mono",color=AZ,bbox=dict(boxstyle="round,pad=.6",fc="#f5f6f9",ec="#c9d1dc"))
ax.grid(alpha=.35); ax.legend(frameon=False,loc="upper left",fontsize=12.5)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIGURAS / "deck_prueba_sellada.png"); plt.close()

# --- 9: tabla H1 y H2
filas=[("H1","Ensambles vs hedónico\nen CV anidada","XGBoost P1: +190 CAD de RMSE\np = 0,52 (t corregida y Holm)","Empate"),
       ("H1","Ensambles vs hedónico\nen prueba sellada","XGBoost P1: +3.301 CAD, p = 0,02\nRandom Forest P1: +2.376 CAD, p = 0,03","Gana el\nhedónico"),
       ("H2","MAPE al pasar\nde P1 a P2","OLS 18,2 → 16,7 %   Ridge 18,1 → 16,7 %\nSVR 17,3 → 16,6 %   KNN 17,8 → 17,6 %","Baja en\nlos cuatro")]
ver=[("No se respalda H1",LAD),("",LAD),("Se cumple H2",VER)]
fig,ax=plt.subplots(figsize=(11,6.2),dpi=200); ax.axis("off"); ax.set_xlim(0,11); ax.set_ylim(0,6.2)
cols=[0.25,1.05,3.95,8.9]; heads=["","Contraste","Resultado","Lectura"]
for x,h in zip(cols,heads): ax.text(x,5.75,h,fontsize=14,color=AC,fontweight="bold",va="center")
ys=[4.65,3.05,1.3]
for (h,c,r,l),yv in zip(filas,ys):
    ax.add_patch(plt.Rectangle((0.1,yv-0.68),10.8,1.36,fc="#f5f6f9",ec="#e1e5ec"))
    ax.text(cols[0]+0.1,yv,h,fontsize=22,fontweight="bold",color=AZ,va="center")
    ax.text(cols[1],yv,c,fontsize=13,color=AZ,va="center",linespacing=1.4)
    ax.text(cols[2],yv,r,fontsize=13,color=AZ,va="center",linespacing=1.5)
    ax.text(cols[3],yv,l,fontsize=14,color=VER if h=="H2" else (LAD if "hedónico" in l else GR),va="center",fontweight="bold",linespacing=1.3)
ax.plot([0.1,10.9],[0.38,0.38],color="#c9d1dc",lw=1)
ax.text(0.2,0.08,"Veredicto:  H1 no se respalda  ·  H2 se cumple",fontsize=16,color=AZ,fontweight="bold",va="center")
fig.savefig(FIGURAS / "deck_hipotesis.png"); plt.close()
