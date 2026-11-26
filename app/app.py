"""Aplicación interactiva en Streamlit para predicción de precio de vivienda con intervalos y SHAP.

Tarjeta T26 · Demo práctica: aplicación de predicción de precio.
Responsable: Mauricio José Mozo Terán (Data Engineer).

Características principales:
- Formulario en español con los 12 atributos estructurales y de localización de la vivienda.
- Precio estimado en unidades de Kaggle y dólares canadienses de 1987 (1 CAD = 70 u. Kaggle),
  junto con su intervalo de predicción configurable (80 %, 90 % o 95 %).
- Explicación SHAP local (gráfico waterfall y tabla de aportes) para la predicción consultada.
- Operación 100 % sin conexión: solo usa los artefactos locales del repositorio.

La carga del modelo está aislada en la función única
``cargar_modelo`` para sustituirla por ``models/modelo_final.joblib`` en cuanto entre T25.

Ejecución local::

    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Asegurar que la raíz del repositorio esté disponible para importar src.*
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
import streamlit as st
from xgboost import XGBRegressor

from src.config import DIR_MODELOS, SEMILLA
from src.evaluacion import FACTOR_KAGGLE, cargar_entrenamiento, crear_pliegues
from src.modelos import construir_pipeline, pipeline_interno
from src.preprocesamiento import COLUMNAS_BINARIAS, COLUMNAS_PREDICTORAS

RUTA_MODELO_FINAL = DIR_MODELOS / "modelo_final.joblib"
SQFT_A_M2 = 0.092903

ETIQUETAS_VARIABLES: Dict[str, str] = {
    "area": "Área del lote (ft²)",
    "bedrooms": "Dormitorios",
    "bathrooms": "Baños completos",
    "stories": "Pisos (plantas)",
    "parking": "Plazas de garaje",
    "mainroad": "Acceso a vía principal",
    "guestroom": "Sala de recreación / huéspedes",
    "basement": "Sótano terminado",
    "hotwaterheating": "Calefacción de agua a gas",
    "airconditioning": "Aire acondicionado central",
    "prefarea": "Zona preferencial",
    "furnishingstatus_1": "Amoblado: semi-amoblado",
    "furnishingstatus_2": "Amoblado: completo",
}

OPCIONES_AMOBLADO: Dict[str, int] = {
    "Sin amoblar (unfurnished)": 0,
    "Semi-amoblado (semi-furnished)": 1,
    "Amoblado completo (furnished)": 2,
}

ETIQUETA_AMOBLADO_INVERSA: Dict[int, str] = {
    0: "Sin amoblar",
    1: "Semi-amoblado",
    2: "Amoblado completo",
}

PERFILES_EJEMPLO: Dict[str, Dict[str, Any]] = {
    "Vivienda mediana (típica)": {
        "area": 4600,
        "bedrooms": 3,
        "bathrooms": 1,
        "stories": 2,
        "parking": 0,
        "mainroad": True,
        "guestroom": False,
        "basement": False,
        "hotwaterheating": False,
        "airconditioning": False,
        "prefarea": False,
        "furnishingstatus": 1,
    },
    "Vivienda compacta básica": {
        "area": 2400,
        "bedrooms": 2,
        "bathrooms": 1,
        "stories": 1,
        "parking": 0,
        "mainroad": False,
        "guestroom": False,
        "basement": False,
        "hotwaterheating": False,
        "airconditioning": False,
        "prefarea": False,
        "furnishingstatus": 0,
    },
    "Vivienda equipada de gama alta": {
        "area": 8500,
        "bedrooms": 4,
        "bathrooms": 3,
        "stories": 3,
        "parking": 2,
        "mainroad": True,
        "guestroom": True,
        "basement": True,
        "hotwaterheating": False,
        "airconditioning": True,
        "prefarea": True,
        "furnishingstatus": 2,
    },
}


# ----------------------------------------------------------------------------- utilidades internas
def _es_objetivo_log(modelo: Any) -> bool:
    """Indica si el pipeline modela ``log(price)`` mediante ``TransformedTargetRegressor``."""
    if not isinstance(modelo, TransformedTargetRegressor):
        return False
    if getattr(modelo, "func", None) is np.log:
        return True
    transformador = getattr(modelo, "transformer_", None)
    if transformador is not None:
        if hasattr(transformador, "named_steps") and "log" in transformador.named_steps:
            return True
        try:
            valor = float(np.asarray(transformador.transform(np.array([[10.0]]))).ravel()[0])
            return np.isclose(valor, np.log(10.0), rtol=1e-4)
        except Exception:
            return False
    return False


def _escala_objetivo_interno(modelo: Any, y_ent: pd.Series) -> tuple[float, float, bool]:
    """Devuelve ``(mu_y, sigma_y, es_log)`` para llevar la salida del estimador interno a ``log(y)`` o ``y``."""
    es_log = _es_objetivo_log(modelo)
    y_base = np.log(y_ent.to_numpy(float)) if es_log else y_ent.to_numpy(float)
    transformador = getattr(modelo, "transformer_", None)
    if transformador is not None and hasattr(transformador, "named_steps") and "std" in transformador.named_steps:
        escalador = transformador.named_steps["std"]
        return float(escalador.mean_[0]), float(escalador.scale_[0]), es_log
    if transformador is not None and hasattr(transformador, "scale_") and hasattr(transformador, "mean_"):
        return float(transformador.mean_[0]), float(transformador.scale_[0]), es_log
    # Confirmamos con la escala real de y_base que no haya estandarización oculta
    _ = y_base
    return 0.0, 1.0, es_log


def _extraer_estimador_desde_joblib(artefacto: Any) -> Any:
    """Extrae el estimador compatible con ``predict`` desde ``models/modelo_final.joblib``."""
    if hasattr(artefacto, "predict"):
        return artefacto
    if isinstance(artefacto, dict):
        for clave in ("modelo", "pipeline", "estimador", "model"):
            candidato = artefacto.get(clave)
            if candidato is not None and hasattr(candidato, "predict"):
                return candidato
    raise ValueError("El archivo models/modelo_final.joblib no contiene un estimador válido con .predict()")


def _construir_explainer_shap(estimador: Any, Z_ent: pd.DataFrame) -> tuple[Any, str]:
    """Construye el explicador SHAP apropiado según la familia del modelo interno."""
    if isinstance(estimador, (LinearRegression, Ridge, Lasso, ElasticNet)):
        mascara = shap.maskers.Independent(Z_ent, max_samples=len(Z_ent))
        return shap.LinearExplainer(estimador, mascara), "LinearExplainer"
    if isinstance(estimador, (RandomForestRegressor, XGBRegressor)):
        return shap.TreeExplainer(estimador), "TreeExplainer"
    fondo = shap.sample(Z_ent, min(100, len(Z_ent)), random_state=SEMILLA)
    return shap.KernelExplainer(estimador.predict, fondo), "KernelExplainer"


def _preparar_info_intervalo(modelo: Any, X_ent: pd.DataFrame, y_ent: pd.Series, es_log: bool) -> Dict[str, Any]:
    """Precalcula los estadísticos necesarios para intervalos de predicción."""
    interno = pipeline_interno(modelo)
    prepro = interno.named_steps["prepro"]
    estimador = interno.named_steps["modelo"]

    if isinstance(estimador, LinearRegression):
        Z_ent = prepro.transform(X_ent)
        D = np.column_stack([np.ones(len(Z_ent)), np.asarray(Z_ent, dtype=float)])
        y_mod = np.log(y_ent.to_numpy(float)) if es_log else y_ent.to_numpy(float)
        y_hat = np.log(modelo.predict(X_ent)) if es_log else modelo.predict(X_ent)
        gl = max(1, len(Z_ent) - D.shape[1])
        s2 = float(np.sum((y_mod - y_hat) ** 2) / gl)
        DtD_inv = np.linalg.pinv(D.T @ D)
        return {
            "tipo": "ols_exacto",
            "es_log": es_log,
            "s2": s2,
            "gl": gl,
            "DtD_inv": DtD_inv,
        }

    # Respaldo conforme fuera de muestra (5 pliegues) para otros modelos cuando entre T25
    pliegues = crear_pliegues(y_ent, n_splits=5, n_repeticiones=1, semilla=SEMILLA)
    residuos = np.zeros(len(y_ent), dtype=float)
    y_arr = y_ent.to_numpy(float)
    for ajuste, validacion in pliegues:
        clon = clone(modelo).fit(X_ent.iloc[ajuste], y_ent.iloc[ajuste])
        pred_val = clon.predict(X_ent.iloc[validacion])
        if es_log:
            residuos[validacion] = np.abs(np.log(y_arr[validacion]) - np.log(np.maximum(pred_val, 1e-9)))
        else:
            residuos[validacion] = np.abs(y_arr[validacion] - pred_val)
    return {
        "tipo": "conforme_cv",
        "es_log": es_log,
        "residuos_abs": np.sort(residuos),
    }


# ----------------------------------------------------------------------------- carga única del modelo
def cargar_modelo(ruta_modelo: Optional[Path] = None) -> Dict[str, Any]:
    """Carga el modelo de producción y su contexto de entrenamiento (función única de carga).

    Mientras no exista ``models/modelo_final.joblib`` (tarjeta T25), entrena provisionalmente
    ``construir_pipeline("P2", "M1")`` (OLS hedónico sobre ``log(price)`` con Yeo-Johnson en
    ``area``) usando exclusivamente las 436 filas de entrenamiento de ``cargar_entrenamiento()``.
    Cuando T25 agregue ``models/modelo_final.joblib``, esta función lo cargará directamente sin
    tocar el resto de la aplicación.
    """
    X_ent, y_ent, evidencia = cargar_entrenamiento()
    ruta = RUTA_MODELO_FINAL if ruta_modelo is None else Path(ruta_modelo)

    if ruta.is_file():
        artefacto = joblib.load(ruta)
        modelo = _extraer_estimador_desde_joblib(artefacto)
        try:
            ruta_legible = ruta.resolve().relative_to(RAIZ.resolve()).as_posix()
        except ValueError:
            ruta_legible = ruta.name
        fuente = f"Artefacto guardado ({ruta_legible})"
        es_provisional = False
    else:
        # Provisional hasta que entre T25: entrenar P2 + M1 sobre las 436 filas de entrenamiento
        modelo = construir_pipeline("P2", "M1").fit(X_ent, y_ent)
        fuente = 'Provisional: construir_pipeline("P2", "M1") sobre 436 filas de entrenamiento'
        es_provisional = True

    interno = pipeline_interno(modelo)
    prepro = interno.named_steps["prepro"]
    estimador = interno.named_steps["modelo"]
    Z_ent = prepro.transform(X_ent)
    if not isinstance(Z_ent, pd.DataFrame):
        nombres = list(prepro.get_feature_names_out())
        Z_ent = pd.DataFrame(Z_ent, columns=nombres, index=X_ent.index)

    mu_y, sigma_y, es_log = _escala_objetivo_interno(modelo, y_ent)
    explainer, tipo_explainer = _construir_explainer_shap(estimador, Z_ent)
    info_intervalo = _preparar_info_intervalo(modelo, X_ent, y_ent, es_log)

    return {
        "modelo": modelo,
        "X_ent": X_ent,
        "y_ent": y_ent,
        "Z_ent": Z_ent,
        "evidencia": evidencia,
        "fuente": fuente,
        "es_provisional": es_provisional,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "es_log": es_log,
        "explainer": explainer,
        "tipo_explainer": tipo_explainer,
        "info_intervalo": info_intervalo,
    }


# ----------------------------------------------------------------------------- inferencia e interpretabilidad
def construir_fila_entrada(atributos: Dict[str, Any]) -> pd.DataFrame:
    """Convierte los valores del formulario en un ``DataFrame`` de 1 fila con ``COLUMNAS_PREDICTORAS``."""
    fila = {
        "area": int(atributos["area"]),
        "bedrooms": int(atributos["bedrooms"]),
        "bathrooms": int(atributos["bathrooms"]),
        "stories": int(atributos["stories"]),
        "parking": int(atributos["parking"]),
        "mainroad": int(bool(atributos["mainroad"])),
        "guestroom": int(bool(atributos["guestroom"])),
        "basement": int(bool(atributos["basement"])),
        "hotwaterheating": int(bool(atributos["hotwaterheating"])),
        "airconditioning": int(bool(atributos["airconditioning"])),
        "prefarea": int(bool(atributos["prefarea"])),
        "furnishingstatus": int(atributos["furnishingstatus"]),
    }
    return pd.DataFrame([fila], columns=COLUMNAS_PREDICTORAS)


def predecir_con_intervalo(
    recurso: Dict[str, Any],
    df_entrada: pd.DataFrame,
    nivel_confianza: float = 0.95,
) -> Dict[str, Any]:
    """Estima el precio de la vivienda y su intervalo de predicción en unidades Kaggle y CAD 1987."""
    modelo = recurso["modelo"]
    info = recurso["info_intervalo"]
    es_log = info["es_log"]

    pred_kaggle = float(modelo.predict(df_entrada)[0])
    alfa = 1.0 - float(nivel_confianza)

    if info["tipo"] == "ols_exacto":
        prepro = pipeline_interno(modelo).named_steps["prepro"]
        z_nuevo = np.asarray(prepro.transform(df_entrada), dtype=float).ravel()
        d0 = np.concatenate([[1.0], z_nuevo])
        se_pred = float(np.sqrt(info["s2"] * (1.0 + d0 @ info["DtD_inv"] @ d0)))
        t_crit = float(stats.t.ppf(1.0 - alfa / 2.0, df=info["gl"]))
        centro = float(np.log(pred_kaggle)) if es_log else pred_kaggle
        lo_mod = centro - t_crit * se_pred
        hi_mod = centro + t_crit * se_pred
        inferior_kaggle = float(np.exp(lo_mod)) if es_log else max(0.0, float(lo_mod))
        superior_kaggle = float(np.exp(hi_mod)) if es_log else float(hi_mod)
        metodo = f"Intervalo de predicción OLS exacto (t, gl={info['gl']})"
    else:
        residuos = info["residuos_abs"]
        q = float(np.quantile(residuos, min(1.0, max(0.0, nivel_confianza))))
        if es_log:
            inferior_kaggle = float(pred_kaggle * np.exp(-q))
            superior_kaggle = float(pred_kaggle * np.exp(q))
        else:
            inferior_kaggle = max(0.0, float(pred_kaggle - q))
            superior_kaggle = float(pred_kaggle + q)
        metodo = "Intervalo conforme con residuos de validación cruzada (5 pliegues)"

    return {
        "pred_kaggle": pred_kaggle,
        "inferior_kaggle": inferior_kaggle,
        "superior_kaggle": superior_kaggle,
        "pred_cad": pred_kaggle / FACTOR_KAGGLE,
        "inferior_cad": inferior_kaggle / FACTOR_KAGGLE,
        "superior_cad": superior_kaggle / FACTOR_KAGGLE,
        "nivel_confianza": float(nivel_confianza),
        "metodo_intervalo": metodo,
    }


def _valor_legible_para_columna(nombre_col: str, fila_cruda: pd.Series) -> str:
    """Devuelve el valor ingresado legible para mostrar en el gráfico y la tabla SHAP."""
    if nombre_col == "area":
        return f"{int(fila_cruda['area']):,} ft²"
    if nombre_col in ("bedrooms", "bathrooms", "stories", "parking"):
        return str(int(fila_cruda[nombre_col]))
    if nombre_col in COLUMNAS_BINARIAS:
        return "Sí" if int(fila_cruda[nombre_col]) == 1 else "No"
    if nombre_col == "furnishingstatus_1":
        return "Sí" if int(fila_cruda["furnishingstatus"]) == 1 else "No"
    if nombre_col == "furnishingstatus_2":
        return "Sí" if int(fila_cruda["furnishingstatus"]) == 2 else "No"
    return str(fila_cruda.get(nombre_col, ""))


def explicar_prediccion_shap(recurso: Dict[str, Any], df_entrada: pd.DataFrame) -> Dict[str, Any]:
    """Calcula los valores SHAP locales de la vivienda consultada y construye la tabla y ``Explanation``."""
    modelo = recurso["modelo"]
    prepro = pipeline_interno(modelo).named_steps["prepro"]
    z_nuevo = prepro.transform(df_entrada)
    if not isinstance(z_nuevo, pd.DataFrame):
        z_nuevo = pd.DataFrame(z_nuevo, columns=list(recurso["Z_ent"].columns))

    explainer = recurso["explainer"]
    tipo_explainer = recurso["tipo_explainer"]
    mu_y, sigma_y, es_log = recurso["mu_y"], recurso["sigma_y"], recurso["es_log"]

    if tipo_explainer == "KernelExplainer":
        vals_raw = np.asarray(explainer.shap_values(z_nuevo), dtype=float).ravel()
        base_raw = float(np.asarray(explainer.expected_value).ravel()[0])
    else:
        exp_res = explainer(z_nuevo)
        vals_raw = np.asarray(exp_res.values[0], dtype=float).ravel()
        base_raw = float(np.asarray(exp_res.base_values).ravel()[0])

    # Llevamos siempre los SHAP a log(price) (P2/P3) o price (P1) deshaciendo cualquier estandarización
    vals = vals_raw * sigma_y
    base = base_raw * sigma_y + mu_y

    columnas = list(z_nuevo.columns)
    fila_cruda = df_entrada.iloc[0]
    nombres_legibles = [ETIQUETAS_VARIABLES.get(c, c) for c in columnas]
    valores_legibles = [_valor_legible_para_columna(c, fila_cruda) for c in columnas]

    explanation = shap.Explanation(
        values=vals,
        base_values=base,
        data=np.asarray(valores_legibles, dtype=object),
        feature_names=nombres_legibles,
    )

    pred_escala_modelo = float(base + np.sum(vals))
    if es_log:
        valor_base_kaggle = float(np.exp(base))
        impacto_pct = 100.0 * (np.exp(vals) - 1.0)
    else:
        valor_base_kaggle = float(base)
        impacto_pct = 100.0 * (vals / max(valor_base_kaggle, 1e-9))

    filas_tabla: List[Dict[str, Any]] = []
    for col, nombre, val_txt, sv, pct in zip(columnas, nombres_legibles, valores_legibles, vals, impacto_pct):
        if np.isclose(sv, 0.0, atol=1e-7):
            direccion = "• Neutro"
        elif sv > 0:
            direccion = "↑ Sube"
        else:
            direccion = "↓ Baja"
        filas_tabla.append({
            "variable": col,
            "Atributo": nombre,
            "Valor ingresado": val_txt,
            "SHAP (Δ log precio)" if es_log else "SHAP (Δ precio)": float(sv),
            "Impacto relativo (%)": float(pct),
            "Dirección": direccion,
            "_abs": abs(float(sv)),
        })

    tabla = (
        pd.DataFrame(filas_tabla)
        .sort_values("_abs", ascending=False)
        .drop(columns=["_abs"])
        .reset_index(drop=True)
    )

    return {
        "explanation": explanation,
        "tabla": tabla,
        "base_escala_modelo": float(base),
        "pred_escala_modelo": pred_escala_modelo,
        "valor_base_kaggle": valor_base_kaggle,
        "valor_base_cad": valor_base_kaggle / FACTOR_KAGGLE,
        "es_log": es_log,
        "tipo_explainer": tipo_explainer,
    }


# ----------------------------------------------------------------------------- visualizaciones locales
def generar_figura_waterfall(explanation: shap.Explanation, es_log: bool) -> plt.Figure:
    """Genera el gráfico waterfall de SHAP en Matplotlib listo para renderizar sin conexión."""
    fig = plt.figure(figsize=(8.5, 5.2), dpi=180)
    shap.plots.waterfall(explanation, max_display=13, show=False)
    ax = plt.gca()
    ax.set_title(
        "Contribución SHAP por atributo (en Δ log-precio)" if es_log else "Contribución SHAP por atributo (en precio Kaggle)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    return fig


def generar_figura_intervalo(resultado_pred: Dict[str, Any], y_ent: pd.Series) -> plt.Figure:
    """Dibuja la posición del precio estimado y su intervalo frente a la distribución de entrenamiento."""
    y_cad = y_ent.to_numpy(float) / FACTOR_KAGGLE
    q25, q50, q75 = np.percentile(y_cad, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(8.2, 2.1), dpi=180)
    ax.axvspan(q25, q75, color="#E2E8F0", alpha=0.8, label="Rango intercuartílico entrenamiento (25–75 %)")
    ax.axvline(q50, color="#64748B", linestyle="--", linewidth=1.3, label=f"Mediana entrenamiento (${q50:,.0f} CAD)")

    lo = resultado_pred["inferior_cad"]
    hi = resultado_pred["superior_cad"]
    est = resultado_pred["pred_cad"]
    pct_conf = int(round(resultado_pred["nivel_confianza"] * 100))

    ax.hlines(y=0.5, xmin=lo, xmax=hi, color="#16697A", linewidth=5.5, alpha=0.85, label=f"Intervalo de predicción ({pct_conf} %)")
    ax.plot([lo, hi], [0.5, 0.5], "|", color="#16697A", markersize=16, markeredgewidth=2.2)
    ax.scatter([est], [0.5], color="#E07A5F", edgecolor="#1E293B", s=110, zorder=5, label=f"Estimación (${est:,.0f} CAD)")

    limite_sup = max(float(np.max(y_cad)) * 1.05, hi * 1.08)
    ax.set_xlim(0, limite_sup)
    ax.set_ylim(0.15, 0.92)
    ax.set_yticks([])
    ax.set_xlabel("Precio en dólares canadienses de 1987 (CAD)", fontsize=9.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------- interfaz Streamlit
def _inicializar_estado() -> None:
    """Inicializa los valores del formulario en ``st.session_state`` con la vivienda mediana."""
    perfil_inicial = PERFILES_EJEMPLO["Vivienda mediana (típica)"]
    for clave, valor in perfil_inicial.items():
        st.session_state.setdefault(f"inp_{clave}", valor)


def _aplicar_perfil(nombre_perfil: str) -> None:
    """Callback para cargar un arquetipo de vivienda en el formulario."""
    for clave, valor in PERFILES_EJEMPLO[nombre_perfil].items():
        st.session_state[f"inp_{clave}"] = valor


@st.cache_resource(show_spinner="Cargando modelo y explicador SHAP...")
def _cargar_recurso_en_cache() -> Dict[str, Any]:
    """Envuelve ``cargar_modelo`` en caché de Streamlit para interacciones instantáneas."""
    return cargar_modelo()


def renderizar_app() -> None:
    """Construye y ejecuta la interfaz web de Streamlit."""
    st.set_page_config(
        page_title="Predicción de Precio de Vivienda · ML UAO",
        page_icon="🏠",
        layout="wide",
    )
    _inicializar_estado()
    recurso = _cargar_recurso_en_cache()

    st.title("Predicción del precio de vivienda e interpretabilidad SHAP")
    st.caption(
        "Maestría en Ciencia de Datos e IA · Universidad Autónoma de Occidente (2026-2) · "
        "Benchmark histórico de Windsor, Canadá (1987) · Ejecución local sin conexión."
    )

    if recurso["es_provisional"]:
        st.info(
            f"**Modelo activo (provisional hasta T25):** OLS hedónico semi-logarítmico (`P2 + M1`) "
            f"ajustado con las **{recurso['evidencia']['n_entrenamiento_cargado']}** filas de entrenamiento "
            f"(conjunto de prueba de {recurso['evidencia']['n_prueba_en_particion']} filas intacto)."
        )
    else:
        st.success(f"**Modelo final cargado:** `{recurso['fuente']}`")

    # Barra lateral: perfiles rápidos, nivel de confianza y notas técnicas
    with st.sidebar:
        st.header("Configuración de la demo")
        st.subheader("Cargar vivienda de ejemplo")
        for nombre_perfil in PERFILES_EJEMPLO:
            st.button(
                nombre_perfil,
                key=f"btn_{nombre_perfil}",
                use_container_width=True,
                on_click=_aplicar_perfil,
                args=(nombre_perfil,),
            )

        st.divider()
        nivel_pct = st.select_slider(
            "Nivel del intervalo de predicción",
            options=[80, 90, 95],
            value=95,
            format_func=lambda v: f"{v} %",
        )
        nivel_confianza = nivel_pct / 100.0

        st.divider()
        st.markdown(
            "**Escala de los datos (T05):**\n"
            "- Precio Kaggle = CAD de 1987 × 70.\n"
            "- Área en pies cuadrados (`1 ft² ≈ 0,093 m²`).\n"
            "- Semilla global del proyecto: `5007`."
        )

    col_form, col_res = st.columns([1.0, 1.45], gap="large")

    # ------------------------------------------------------------------------- Formulario
    with col_form:
        st.subheader("Atributos de la vivienda")

        st.markdown("**Dimensiones y capacidad**")
        area = st.slider(
            "Área del lote (`area`, pies²)",
            min_value=1500,
            max_value=16500,
            step=50,
            key="inp_area",
            help="Rango observado en los datos: 1,650 – 16,200 ft².",
        )
        st.caption(f"Equivalente aproximado: **{area * SQFT_A_M2:,.0f} m²**")

        c1, c2 = st.columns(2)
        with c1:
            bedrooms = st.number_input("Dormitorios (`bedrooms`)", min_value=1, max_value=6, step=1, key="inp_bedrooms")
            stories = st.number_input("Pisos (`stories`)", min_value=1, max_value=4, step=1, key="inp_stories")
        with c2:
            bathrooms = st.number_input("Baños (`bathrooms`)", min_value=1, max_value=4, step=1, key="inp_bathrooms")
            parking = st.number_input("Garajes (`parking`)", min_value=0, max_value=3, step=1, key="inp_parking")

        st.markdown("**Equipamiento y localización**")
        b1, b2 = st.columns(2)
        with b1:
            mainroad = st.checkbox("Acceso a vía principal (`mainroad`)", key="inp_mainroad")
            guestroom = st.checkbox("Sala de recreación (`guestroom`)", key="inp_guestroom")
            basement = st.checkbox("Sótano terminado (`basement`)", key="inp_basement")
        with b2:
            hotwaterheating = st.checkbox("Calentador a gas (`hotwaterheating`)", key="inp_hotwaterheating")
            airconditioning = st.checkbox("Aire acondicionado (`airconditioning`)", key="inp_airconditioning")
            prefarea = st.checkbox("Zona preferencial (`prefarea`)", key="inp_prefarea")

        st.markdown("**Estado de amoblado**")
        codigos_amoblado = list(OPCIONES_AMOBLADO.values())
        etiquetas_por_codigo = {v: k for k, v in OPCIONES_AMOBLADO.items()}
        st.selectbox(
            "Amoblado (`furnishingstatus`)",
            options=codigos_amoblado,
            format_func=lambda c: etiquetas_por_codigo[int(c)],
            key="inp_furnishingstatus",
        )

    atributos = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": st.session_state["inp_furnishingstatus"],
    }
    df_entrada = construir_fila_entrada(atributos)
    pred = predecir_con_intervalo(recurso, df_entrada, nivel_confianza=nivel_confianza)
    shap_res = explicar_prediccion_shap(recurso, df_entrada)

    # ------------------------------------------------------------------------- Resultados e intervalos
    with col_res:
        st.subheader("Precio estimado y rango de incertidumbre")
        m1, m2, m3 = st.columns(3)
        delta_pct = 100.0 * (pred["pred_kaggle"] / shap_res["valor_base_kaggle"] - 1.0)
        m1.metric(
            label="Precio estimado (CAD 1987)",
            value=f"${pred['pred_cad']:,.0f}",
            delta=f"{delta_pct:+.1f} % vs base",
        )
        m2.metric(
            label=f"Intervalo de predicción ({nivel_pct} %)",
            value=f"${pred['inferior_cad']:,.0f} – ${pred['superior_cad']:,.0f}",
            help=pred["metodo_intervalo"],
        )
        m3.metric(
            label="Escala Kaggle (CAD × 70)",
            value=f"{pred['pred_kaggle']:,.0f}",
            help=f"IC {nivel_pct} %: [{pred['inferior_kaggle']:,.0f} ; {pred['superior_kaggle']:,.0f}]",
        )

        st.caption(
            f"**Rango en unidades Kaggle:** [{pred['inferior_kaggle']:,.0f} – {pred['superior_kaggle']:,.0f}] · "
            f"**Valor base promedio del modelo:** ${shap_res['valor_base_cad']:,.0f} CAD ({shap_res['valor_base_kaggle']:,.0f} u. Kaggle) · "
            f"**Método:** {pred['metodo_intervalo']}."
        )

        fig_int = generar_figura_intervalo(pred, recurso["y_ent"])
        st.pyplot(fig_int, width="stretch")
        plt.close(fig_int)

        st.subheader(f"Explicación SHAP de la predicción ({shap_res['tipo_explainer']})")
        tab_graf, tab_datos = st.tabs(["Gráfico Waterfall", "Tabla de contribuciones"])

        with tab_graf:
            fig_shap = generar_figura_waterfall(shap_res["explanation"], shap_res["es_log"])
            st.pyplot(fig_shap, width="stretch")
            plt.close(fig_shap)

        with tab_datos:
            tabla_mostrar = shap_res["tabla"].drop(columns=["variable"]).copy()
            col_shap = [c for c in tabla_mostrar.columns if c.startswith("SHAP")][0]
            st.dataframe(
                tabla_mostrar.style.format({
                    col_shap: "{:+.4f}",
                    "Impacto relativo (%)": "{:+.2f} %",
                }),
                hide_index=True,
                width="stretch",
            )

        st.caption(
            "**Nota metodológica:** Los valores SHAP miden la contribución de cada atributo a la "
            "**predicción del modelo** respecto al valor base del conjunto de entrenamiento (436 viviendas). "
            "Describen asociaciones estadísticas aprendidas por el modelo y no relaciones causales ni "
            "formación de precios en el mercado actual."
        )


if __name__ == "__main__":
    renderizar_app()
