"""Pruebas unitarias para análisis de residuos e interpretabilidad SHAP (Tarjeta T24).

Responsable: Brandon Eduardo Tobar Sánchez (Model Developer & Evaluation Lead).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.config import DIR_FIGURAS, DIR_TABLAS
from src.ejecutar_t24 import analizar_residuos


class TestInterpretabilidadSHAP(unittest.TestCase):
    """Verifica el cálculo de residuos, pruebas estadísticas y artefactos de T24."""

    def test_analizar_residuos_metricas_completas(self):
        """Verifica que analizar_residuos devuelva todas las estadísticas descriptivas y de contraste."""
        np.random.seed(5007)
        y_real = np.linspace(40000.0, 120000.0, 25)
        ruido = np.random.normal(0, 2000.0, 25)
        y_pred = y_real + ruido

        res = analizar_residuos(y_real, y_pred, "ModeloTest")

        self.assertEqual(res["modelo"], "ModeloTest")
        self.assertIn("rmse_cad", res)
        self.assertIn("mae_cad", res)
        self.assertIn("mape_pct", res)
        self.assertIn("r2", res)
        self.assertIn("media_residuos_cad", res)
        self.assertIn("std_residuos_cad", res)
        self.assertIn("asimetria", res)
        self.assertIn("curtosis_exceso", res)
        self.assertIn("shapiro_p_valor", res)
        self.assertIn("dagostino_p_valor", res)
        self.assertIn("spearman_rho_heterocedasticidad", res)
        self.assertIn("spearman_p_valor_heterocedasticidad", res)
        self.assertGreaterEqual(res["r2"], 0.90)

    def test_existencia_tablas_t24(self):
        """Verifica que todas las tablas de T24 existan y contengan filas válidas."""
        tablas_esperadas = [
            ("t24_analisis_residuos.csv", 4),
            ("t24_casos_extremos_residuos.csv", 10),
            ("t24_shap_importancias_globales.csv", 13),
            ("t24_comparacion_coeficientes_vs_shap.csv", 13),
            ("t24_casos_locales_shap.csv", 4),
            ("t24_residuos_por_cuartil.csv", 4),
        ]

        for nombre_tabla, min_filas in tablas_esperadas:
            ruta = DIR_TABLAS / nombre_tabla
            self.assertTrue(ruta.exists(), f"Falta la tabla {nombre_tabla}")
            df = pd.read_csv(ruta)
            self.assertGreaterEqual(len(df), min_filas, f"La tabla {nombre_tabla} tiene menos filas de las esperadas")

    def test_existencia_figuras_t24(self):
        """Verifica que todas las figuras IEEE de T24 existan y no estén vacías."""
        figuras_esperadas = [
            "fig_t24_diagnostico_residuos.png",
            "fig_t24_shap_summary_beeswarm.png",
            "fig_t24_shap_importancia_barras.png",
            "fig_t24_shap_dependencia_atributos.png",
            "fig_t24_shap_explicacion_local_casos.png",
            "fig_t24_coeficientes_hedonicos_vs_shap.png",
            "fig_t24_residuos_por_cuartil.png",
            "fig_t24_pdp_ice.png",
        ]

        for fig_nombre in figuras_esperadas:
            ruta = DIR_FIGURAS / fig_nombre
            self.assertTrue(ruta.exists(), f"Falta la figura {fig_nombre}")
            self.assertGreater(os.path.getsize(ruta), 10_000, f"La figura {fig_nombre} parece vacía o corrupta")

    def test_aditividad_local_shap(self):
        """Verifica la propiedad teórica de eficiencia/aditividad local de SHAP."""
        ruta_locales = DIR_TABLAS / "t24_casos_locales_shap.csv"
        df_locales = pd.read_csv(ruta_locales)

        cols_shap = [c for c in df_locales.columns if c.startswith("shap_") and c.endswith("_cad")]
        self.assertEqual(len(cols_shap), 13, "Deben existir 13 columnas SHAP atributivas")

        for _, fila in df_locales.iterrows():
            suma_shaps = fila[cols_shap].sum()
            prediccion = fila["prediccion_xgb_modelo_cad"]
            base = fila["valor_base_cad"]
            # En XGBoost la suma de SHAP values es exactamente f(x) - E[f(x)]
            diferencia = np.abs((base + suma_shaps) - prediccion)
            self.assertLess(diferencia, 1.0, f"La aditividad local de SHAP difiere para {fila['arquetipo']}")


if __name__ == "__main__":
    unittest.main()
