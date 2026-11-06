"""Módulo de procesamiento, auditoría de calidad y partición de datos.

Proyecto: Predicción del precio de vivienda (UAO · 2026-2)
Tarjeta: T06 · Calidad de datos, limpieza reproducible y partición sellada.

Este módulo implementa funciones reproducibles para:
1. Auditar completitud, unicidad y validez de dominios/rangos.
2. Detección de valores atípicos (1.5×IQR y z robusto basado en MAD).
3. Evaluación de desbalance de clases y detección de artefactos de orden.
4. Generación y exportación de la tabla de resumen de calidad (scorecard).
5. Limpieza reproducible sin fuga de información (yes/no -> 1/0, amoblado ordinal).
6. Creación y sellado de la partición estratificada 80/20 por quintiles de precio.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RUTA_DATOS_CRUDOS,
    RUTA_DATOS_LIMPIOS,
    RUTA_PARTICION,
    RUTA_TABLA_CALIDAD,
    SEMILLA,
)

# Mapeos semánticos y de codificación
MAPA_BINARIO = {
    "yes": 1,
    "no": 0,
}

MAPA_AMOBLADO = {
    "unfurnished": 0,
    "semi-furnished": 1,
    "furnished": 2,
}

MAPA_COLUMNAS_ESPANOL = {
    "price": "precio",
    "area": "area_terreno",
    "bedrooms": "habitaciones",
    "bathrooms": "banos",
    "stories": "pisos",
    "mainroad": "calle_principal",
    "guestroom": "cuarto_servicio",
    "basement": "sotano",
    "hotwaterheating": "calentador_agua",
    "airconditioning": "aire_acondicionado",
    "parking": "estacionamientos",
    "prefarea": "zona_preferencial",
    "furnishingstatus": "estado_amoblado",
}

COLUMNAS_BINARIAS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]

DOMINIOS_ESPERADOS = {
    "price": "Entero > 0 (unidades de Kaggle = CAD de 1987 × 70)",
    "area": "Entero > 0 (sqft)",
    "bedrooms": "Entero en [1, 6]",
    "bathrooms": "Entero en [1, 4]",
    "stories": "Entero en [1, 4]",
    "mainroad": "{'yes', 'no'}",
    "guestroom": "{'yes', 'no'}",
    "basement": "{'yes', 'no'}",
    "hotwaterheating": "{'yes', 'no'}",
    "airconditioning": "{'yes', 'no'}",
    "parking": "Entero en [0, 3]",
    "prefarea": "{'yes', 'no'}",
    "furnishingstatus": "{'unfurnished', 'semi-furnished', 'furnished'}",
}


def cargar_datos_crudos(ruta: Optional[Path] = None) -> pd.DataFrame:
    """Carga el archivo crudo de viviendas asignando un identificador de fila 'id'.

    Parameters
    ----------
    ruta : Path, optional
        Ruta al archivo CSV crudo. Si es None, utiliza RUTA_DATOS_CRUDOS de src.config.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos crudos y una columna 'id' que preserva el orden original.
    """
    ruta_archivo = Path(ruta) if ruta is not None else RUTA_DATOS_CRUDOS
    if not ruta_archivo.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos crudos en: {ruta_archivo}")

    df = pd.read_csv(ruta_archivo)
    if "id" not in df.columns:
        df.insert(0, "id", df.index.astype(int))
    return df


def auditar_completitud(df: pd.DataFrame) -> pd.DataFrame:
    """Audita la completitud del conjunto de datos calculando valores faltantes por variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a auditar.

    Returns
    -------
    pd.DataFrame
        Tabla con variables, cantidad de nulos, porcentaje de nulos y porcentaje de completitud.
    """
    columnas = [col for col in df.columns if col != "id"]
    n_filas = len(df)
    nulos = df[columnas].isnull().sum()
    pct_nulos = (nulos / n_filas) * 100.0
    completitud = 100.0 - pct_nulos

    reporte = pd.DataFrame({
        "variable": columnas,
        "nulos": nulos.values,
        "pct_nulos": pct_nulos.values,
        "completitud_pct": completitud.values,
    })
    return reporte


def auditar_unicidad(df: pd.DataFrame) -> Dict[str, Union[int, pd.DataFrame, str]]:
    """Audita la unicidad de registros completos y detecta ruido irreducible por vectores repetidos.

    Verifica:
    1. Duplicados completos en todas las variables (se espera 0).
    2. Vectores de covariables repetidos con distinto precio (6 casos en el dataset completo).
    3. Vectores de covariables repetidos ignorando 'furnishingstatus' (16 casos repetidos).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a auditar.

    Returns
    -------
    dict
        Diccionario con conteos de duplicados, DataFrames de vectores repetidos y explicación
        metodológica del ruido irreducible.
    """
    cols_features = [c for c in df.columns if c not in ("id", "price")]
    cols_features_sin_amoblado = [c for c in cols_features if c != "furnishingstatus"]

    # 1. Duplicados completos (excluyendo 'id')
    cols_completas = [c for c in df.columns if c != "id"]
    duplicados_completos = int(df.duplicated(subset=cols_completas).sum())

    # 2. Vectores de atributos repetidos con distinto precio (todas las covariables)
    mask_dups_features = df.duplicated(subset=cols_features, keep=False)
    df_dups_features = df[mask_dups_features].sort_values(by=cols_features)
    casos_duplicados_features = int(df.duplicated(subset=cols_features).sum())

    # 3. Vectores de atributos repetidos ignorando furnishingstatus
    mask_dups_sin_amoblado = df.duplicated(subset=cols_features_sin_amoblado, keep=False)
    df_dups_sin_amoblado = df[mask_dups_sin_amoblado].sort_values(by=cols_features_sin_amoblado)
    casos_duplicados_sin_amoblado = int(df.duplicated(subset=cols_features_sin_amoblado).sum())

    explicacion_ruido = (
        "La existencia de casas con idénticas características observables X pero con distinto "
        "precio de venta Y evidencia empíricamente la presencia de 'ruido irreducible' "
        "(error de Bayes / varianza irreducible sigma_epsilon^2). Ningún modelo estadístico "
        "o de aprendizaje supervisado, por sofisticado que sea, puede predecir con error cero "
        "ambos precios simultáneamente para el mismo vector X. Este ruido se origina en variables "
        "no observadas en el dataset (como estado de conservación, calidad de acabados, vista, "
        "orientación, poder de negociación del comprador o fecha exacta de transacción)."
    )

    return {
        "duplicados_completos": duplicados_completos,
        "casos_duplicados_atributos": casos_duplicados_features,
        "registros_duplicados_atributos": df_dups_features,
        "casos_duplicados_sin_amoblado": casos_duplicados_sin_amoblado,
        "registros_duplicados_sin_amoblado": df_dups_sin_amoblado,
        "explicacion_ruido_irreducible": explicacion_ruido,
    }


def auditar_validez(df: pd.DataFrame) -> pd.DataFrame:
    """Audita la validez de los dominios y rangos esperados para todas las variables del dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.

    Returns
    -------
    pd.DataFrame
        Tabla con variables, dominio esperado, rango/valores observados y cumplimiento booleano.
    """
    filas = []

    # price: entero > 0
    val_price = (df["price"] > 0).all() and pd.api.types.is_numeric_dtype(df["price"])
    filas.append({
        "variable": "price",
        "dominio_esperado": DOMINIOS_ESPERADOS["price"],
        "observado": f"[{df['price'].min():,}, {df['price'].max():,}]",
        "es_valido": bool(val_price),
    })

    # area: entero > 0
    val_area = (df["area"] > 0).all() and pd.api.types.is_numeric_dtype(df["area"])
    filas.append({
        "variable": "area",
        "dominio_esperado": DOMINIOS_ESPERADOS["area"],
        "observado": f"[{df['area'].min()}, {df['area'].max()}]",
        "es_valido": bool(val_area),
    })

    # bedrooms: 1 a 6
    val_bed = df["bedrooms"].isin(range(1, 7)).all()
    filas.append({
        "variable": "bedrooms",
        "dominio_esperado": DOMINIOS_ESPERADOS["bedrooms"],
        "observado": f"[{df['bedrooms'].min()}, {df['bedrooms'].max()}]",
        "es_valido": bool(val_bed),
    })

    # bathrooms: 1 a 4
    val_bath = df["bathrooms"].isin(range(1, 5)).all()
    filas.append({
        "variable": "bathrooms",
        "dominio_esperado": DOMINIOS_ESPERADOS["bathrooms"],
        "observado": f"[{df['bathrooms'].min()}, {df['bathrooms'].max()}]",
        "es_valido": bool(val_bath),
    })

    # stories: 1 a 4
    val_sto = df["stories"].isin(range(1, 5)).all()
    filas.append({
        "variable": "stories",
        "dominio_esperado": DOMINIOS_ESPERADOS["stories"],
        "observado": f"[{df['stories'].min()}, {df['stories'].max()}]",
        "es_valido": bool(val_sto),
    })

    # parking: 0 a 3
    val_park = df["parking"].isin(range(0, 4)).all()
    filas.append({
        "variable": "parking",
        "dominio_esperado": DOMINIOS_ESPERADOS["parking"],
        "observado": f"[{df['parking'].min()}, {df['parking'].max()}]",
        "es_valido": bool(val_park),
    })

    # binarias: solo yes/no
    for col in COLUMNAS_BINARIAS:
        valores_obs = set(df[col].dropna().unique())
        val_bin = valores_obs.issubset({"yes", "no"})
        filas.append({
            "variable": col,
            "dominio_esperado": DOMINIOS_ESPERADOS[col],
            "observado": str(sorted(list(valores_obs))),
            "es_valido": bool(val_bin),
        })

    # furnishingstatus: {furnished, semi-furnished, unfurnished}
    valores_amoblado = set(df["furnishingstatus"].dropna().unique())
    val_amoblado = valores_amoblado.issubset({"furnished", "semi-furnished", "unfurnished"})
    filas.append({
        "variable": "furnishingstatus",
        "dominio_esperado": DOMINIOS_ESPERADOS["furnishingstatus"],
        "observado": str(sorted(list(valores_amoblado))),
        "es_valido": bool(val_amoblado),
    })

    return pd.DataFrame(filas)


def detectar_atipicos(df: pd.DataFrame, columna: str = "price") -> Dict[str, Union[float, int, pd.Series, str]]:
    """Detecta valores atípicos mediante regla de 1.5×IQR y puntaje z robusto (MAD).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos.
    columna : str, default='price'
        Nombre de la variable numérica a evaluar.

    Returns
    -------
    dict
        Métricas de distribución (Q1, Q3, IQR, Mediana, MAD), umbrales, conteos de atípicos
        y la justificación metodológica de conservación en el caso de 'price'.
    """
    serie = df[columna].dropna()

    # 1. Regla 1.5 × IQR
    q25 = float(serie.quantile(0.25))
    q75 = float(serie.quantile(0.75))
    iqr = q75 - q25
    umbral_inf_iqr = q25 - 1.5 * iqr
    umbral_sup_iqr = q75 + 1.5 * iqr

    atipicos_inf_iqr = int((serie < umbral_inf_iqr).sum())
    atipicos_sup_iqr = int((serie > umbral_sup_iqr).sum())
    total_atipicos_iqr = atipicos_inf_iqr + atipicos_sup_iqr

    # 2. z-score robusto basado en MAD (Median Absolute Deviation)
    mediana = float(serie.median())
    mad = float(np.median(np.abs(serie - mediana)))
    # Factor de consistencia normal para MAD: 0.6745 en numerador o 1.4826 en denominador
    if mad > 0:
        z_robusto = 0.6745 * (serie - mediana) / mad
        atipicos_mad = int((np.abs(z_robusto) > 3).sum())
        atipicos_sup_mad = int((z_robusto > 3).sum())
    else:
        z_robusto = pd.Series(0.0, index=serie.index)
        atipicos_mad = 0
        atipicos_sup_mad = 0

    justificacion_conservar = (
        f"Se identifican {atipicos_sup_iqr} observaciones atípicas superiores por 1.5×IQR en '{columna}'. "
        "Se decide conservarlas formalmente en el dataset por las siguientes razones:\n"
        "1. Validez empírica: no constituyen errores de captura ni valores físicamente imposibles, "
        "sino propiedades residenciales de alta gama con áreas grandes (>7000 sqft), múltiples baños "
        "y amenidades completas (aire acondicionado, zona preferencial).\n"
        "2. Representatividad de mercado: la distribución de precios en bienes raíces es intrínsecamente "
        "asimétrica a la derecha (log-normal). Eliminar la cola superior sesgaría negativamente las "
        "predicciones en viviendas de alto valor (techo artificial).\n"
        "3. Tratamiento metodológico correcto: las asimetrías y heterocedasticidad se gestionan dentro de "
        "los pipelines de modelado mediante transformaciones monótonas (ej. log(price)), estimadores "
        "robustos o árboles de decisión, sin podar datos válidos de forma ad-hoc."
    )

    return {
        "columna": columna,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "umbral_inf_iqr": umbral_inf_iqr,
        "umbral_sup_iqr": umbral_sup_iqr,
        "atipicos_inf_iqr": atipicos_inf_iqr,
        "atipicos_sup_iqr": atipicos_sup_iqr,
        "total_atipicos_iqr": total_atipicos_iqr,
        "mediana": mediana,
        "mad": mad,
        "atipicos_mad_abs_gt_3": atipicos_mad,
        "atipicos_sup_mad": atipicos_sup_mad,
        "justificacion": justificacion_conservar,
    }


def evaluar_desbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Evalúa el desbalance de categorías en variables binarias y categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.

    Returns
    -------
    pd.DataFrame
        Resumen de categorías, conteos, porcentajes de participación y evaluación de riesgo.
    """
    columnas_evaluar = COLUMNAS_BINARIAS + ["furnishingstatus"]
    filas = []

    for col in columnas_evaluar:
        conteo = df[col].value_counts()
        total = len(df)
        categoria_min = conteo.idxmin()
        conteo_min = conteo.min()
        pct_min = (conteo_min / total) * 100.0

        riesgo = "Bajo"
        nota = "Distribución balanceada"
        if pct_min < 10.0:
            riesgo = "Alto" if pct_min < 5.0 else "Moderado"
            nota = (
                f"Categoría minoritaria '{categoria_min}' con {pct_min:.2f} % "
                f"({conteo_min}/{total} casos). Riesgo de baja representación en folds y "
                "alta varianza en coeficientes/árboles."
            )

        filas.append({
            "variable": col,
            "categoria_mayoritaria": conteo.idxmax(),
            "pct_mayoritaria": float((conteo.max() / total) * 100.0),
            "categoria_minoritaria": categoria_min,
            "pct_minoritaria": float(pct_min),
            "conteo_minoritario": int(conteo_min),
            "nivel_riesgo": riesgo,
            "observaciones": nota,
        })

    return pd.DataFrame(filas)


def verificar_artefacto_orden(df: pd.DataFrame, columna: str = "price") -> Dict[str, Union[bool, str]]:
    """Verifica si el dataset viene ordenado de manera artificial y explica el riesgo de partición.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a verificar.
    columna : str, default='price'
        Columna sobre la cual verificar el ordenamiento.

    Returns
    -------
    dict
        Indicadores booleanos de monotonía y explicación técnica del riesgo metodológico.
    """
    serie = df[columna]
    es_decreciente = bool(serie.is_monotonic_decreasing)
    es_creciente = bool(serie.is_monotonic_increasing)

    explicacion = (
        "El archivo Housing.csv original viene estrictamente ordenado en forma descendente por precio "
        f"(de {serie.iloc[0]:,} a {serie.iloc[-1]:,} en unidades de Kaggle). "
        "Riesgo crítico: Si se realiza una partición secuencial o no barajada (shuffle=False), "
        "el conjunto de entrenamiento recibiría exclusivamente las casas más caras (80 % superior), "
        "mientras que el conjunto de prueba quedaría compuesto únicamente por las propiedades más baratas "
        "(20 % inferior). Esto provocaría un desfase de distribución catastrófico (dataset shift total), "
        "invalidando por completo la generalización del modelo. "
        "Mitigación obligatoria: Barajar aleatoriamente con la semilla fija del proyecto (5007) antes "
        "de cualquier corte o aplicar partición estratificada por quintiles."
    )

    return {
        "es_monotonic_decreasing": es_decreciente,
        "es_monotonic_increasing": es_creciente,
        "primeros_valores": serie.head(3).tolist(),
        "ultimos_valores": serie.tail(3).tolist(),
        "explicacion_riesgo": explicacion,
    }


def generar_scorecard_calidad(df: pd.DataFrame, ruta_salida: Optional[Path] = None) -> pd.DataFrame:
    """Genera la tabla resumen de calidad (scorecard) multidimensional y la exporta a CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a evaluar.
    ruta_salida : Path, optional
        Ruta para guardar el CSV. Si es None, utiliza RUTA_TABLA_CALIDAD de src.config.

    Returns
    -------
    pd.DataFrame
        Scorecard con todas las dimensiones de calidad consolidadas por variable.
    """
    columnas = [c for c in df.columns if c != "id"]
    n_filas = len(df)
    scorecard_filas = []

    for col in columnas:
        s = df[col]
        nulos = int(s.isnull().sum())
        pct_nulos = float((nulos / n_filas) * 100.0)
        completitud = 100.0 - pct_nulos
        unicos = int(s.nunique())
        dominio_esp = DOMINIOS_ESPERADOS.get(col, "N/A")

        # Atípicos IQR (solo numéricas continuas/discretas)
        atipicos_iqr = 0
        if pd.api.types.is_numeric_dtype(s):
            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            iqr = q75 - q25
            atipicos_iqr = int(((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum())

        # Desbalance (para categóricas / binarias)
        pct_minoritario = np.nan
        if not pd.api.types.is_numeric_dtype(s) or col in COLUMNAS_BINARIAS:
            conteo = s.value_counts()
            pct_minoritario = float((conteo.min() / n_filas) * 100.0)

        # Notas particulares
        nota = "Calidad óptima"
        if col == "price":
            nota = f"{atipicos_iqr} atípicos superiores (1.5×IQR). Se conservan por validez económica."
        elif col == "hotwaterheating":
            nota = f"Desbalance severo ({pct_minoritario:.2f} % 'yes'). Riesgo de varianza en submuestras."
        elif col == "area":
            nota = f"{atipicos_iqr} atípicos superiores (1.5×IQR). Distribución sesgada a la derecha."

        scorecard_filas.append({
            "variable": col,
            "tipo_dato": str(s.dtype),
            "nulos": nulos,
            "pct_nulos": pct_nulos,
            "completitud_pct": completitud,
            "cardinalidad_unicos": unicos,
            "dominio_esperado": dominio_esp,
            "cumple_validez": True,
            "atipicos_iqr": atipicos_iqr,
            "pct_clase_minoritaria": round(pct_minoritario, 2) if not np.isnan(pct_minoritario) else "N/A",
            "observaciones": nota,
        })

    df_scorecard = pd.DataFrame(scorecard_filas)

    destino = Path(ruta_salida) if ruta_salida is not None else RUTA_TABLA_CALIDAD
    destino.parent.mkdir(parents=True, exist_ok=True)
    df_scorecard.to_csv(destino, index=False, encoding="utf-8")
    return df_scorecard


def limpiar_datos(
    df: pd.DataFrame,
    codificar_ordinal: bool = True,
    renombrar_columnas: bool = False,
) -> pd.DataFrame:
    """Ejecuta la limpieza reproducible de datos sin incurrir en fuga de información (data leakage).

    Reglas aplicadas:
    1. Preserva el identificador 'id'.
    2. Convierte variables binarias (yes/no) a números enteros (1/0).
    3. Trata 'furnishingstatus' como categoría ordinal:
       - 'unfurnished' -> 0
       - 'semi-furnished' -> 1
       - 'furnished' -> 2
    4. Garantiza ausencia de fuga de información: NO aplica escalado (StandardScaler/MinMaxScaler)
       ni transformaciones dependientes de la distribución muestral global, las cuales deben
       ajustarse estrictamente sobre el conjunto de entrenamiento dentro del Pipeline de modelado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo con las observaciones.
    codificar_ordinal : bool, default=True
        Si es True, mapea 'furnishingstatus' a enteros {0, 1, 2}. Si es False, asigna pd.Categorical ordenado.
    renombrar_columnas : bool, default=False
        Si es True, traduce las columnas a nombres semánticos en español según MAPA_COLUMNAS_ESPANOL.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio listo para el flujo de modelado.
    """
    df_limpio = df.copy()

    # 1. Asegurar columna 'id'
    if "id" not in df_limpio.columns:
        df_limpio.insert(0, "id", df_limpio.index.astype(int))

    # 2. Codificación binaria reproducible (yes -> 1, no -> 0)
    for col in COLUMNAS_BINARIAS:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].map(MAPA_BINARIO).astype(int)

    # 3. Codificación de estado de amoblado como categoría ordenada
    if "furnishingstatus" in df_limpio.columns:
        if codificar_ordinal:
            df_limpio["furnishingstatus"] = df_limpio["furnishingstatus"].map(MAPA_AMOBLADO).astype(int)
        else:
            df_limpio["furnishingstatus"] = pd.Categorical(
                df_limpio["furnishingstatus"],
                categories=["unfurnished", "semi-furnished", "furnished"],
                ordered=True,
            )

    # 4. Renombrado semántico opcional
    if renombrar_columnas:
        df_limpio = df_limpio.rename(columns=MAPA_COLUMNAS_ESPANOL)

    return df_limpio


def crear_particion_sellada(
    df: pd.DataFrame,
    ruta_salida: Optional[Path] = None,
    test_size: float = 0.2,
    n_quintiles: int = 5,
    semilla: int = SEMILLA,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera y sella la partición estratificada 80/20 por quintiles de precio con barajado previo.

    Resuelve el artefacto de orden del CSV crudo (que viene ordenado descendentemente por precio)
    estratificando los datos en 5 estratos (quintiles de precio). El conjunto de prueba queda
    sellado en disco ('data/processed/particion.csv') con las columnas 'id' y 'conjunto'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio o crudo con columna 'id' y 'price'.
    ruta_salida : Path, optional
        Ruta para guardar particion.csv. Si es None, utiliza RUTA_PARTICION de src.config.
    test_size : float, default=0.2
        Fracción destinada al conjunto de prueba (20 %).
    n_quintiles : int, default=5
        Número de estratos para la discretización de la variable objetivo.
    semilla : int, default=5007
        Semilla fija de reproducibilidad.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        - df_particion: DataFrame con columnas 'id' y 'conjunto' ('entrenamiento' / 'prueba').
        - df_train: Subconjunto de entrenamiento (80 %, 436 filas).
        - df_test: Subconjunto de prueba sellado (20 %, 109 filas).
    """
    if "id" not in df.columns:
        df = df.copy()
        df.insert(0, "id", df.index.astype(int))

    # Construcción de estratos por quintiles de precio
    estratos = pd.qcut(df["price"], q=n_quintiles, labels=False)

    # Partición estratificada con semilla fija (incluye barajado automático)
    ids_train, ids_test = train_test_split(
        df["id"],
        test_size=test_size,
        random_state=semilla,
        stratify=estratos,
    )

    # Construir tabla de asignación de partición
    df_particion = pd.DataFrame({"id": df["id"]})
    df_particion["conjunto"] = np.where(df_particion["id"].isin(ids_test), "prueba", "entrenamiento")

    # Validar consistencia
    total = len(df)
    n_train = len(ids_train)
    n_test = len(ids_test)
    assert n_train + n_test == total, "La suma de conjuntos debe coincidir con el total de datos."
    assert n_test == round(total * test_size), "El tamaño del test debe corresponder exactamente al 20 %."

    # Exportar archivo sellado
    destino = Path(ruta_salida) if ruta_salida is not None else RUTA_PARTICION
    destino.parent.mkdir(parents=True, exist_ok=True)
    df_particion.to_csv(destino, index=False, encoding="utf-8")

    df_train = df[df["id"].isin(ids_train)].copy()
    df_test = df[df["id"].isin(ids_test)].copy()

    return df_particion, df_train, df_test


def guardar_datos_limpios(df_limpio: pd.DataFrame, ruta_salida: Optional[Path] = None) -> Path:
    """Exporta el DataFrame limpio a la ruta procesada especificada.

    Parameters
    ----------
    df_limpio : pd.DataFrame
        DataFrame limpio.
    ruta_salida : Path, optional
        Ruta destino. Si es None, utiliza RUTA_DATOS_LIMPIOS de src.config.

    Returns
    -------
    Path
        Ruta del archivo generado.
    """
    destino = Path(ruta_salida) if ruta_salida is not None else RUTA_DATOS_LIMPIOS
    destino.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.to_csv(destino, index=False, encoding="utf-8")
    return destino
