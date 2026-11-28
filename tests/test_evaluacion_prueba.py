"""Pruebas unitarias para la evaluación final en el conjunto de prueba sellado (Tarjeta T25).

Responsable: Brandon Eduardo Tobar Sánchez (Model Developer & Evaluation Lead).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.config import DIR_FIGURAS, DIR_TABLAS
from src.evaluacion import cargar_entrenamiento, cargar_prueba


class TestEvaluacionPruebaSellada(unittest.TestCase):
    """Verifica la integridad del conjunto de prueba y los artefactos finales de T25."""

    def test_carga_aislada_conjunto_prueba(self):
        """Verifica que el conjunto de prueba tenga exactamente 109 filas y no se solape con entrenamiento."""
        X_tr, y_tr, ev_tr = cargar_entrenamiento()
        X_te, y_te, ev_te = cargar_prueba()

        self.assertEqual(len(X_tr), 436, "El conjunto de entrenamiento debe tener 436 observaciones")
        self.assertEqual(len(X_te), 109, "El conjunto de prueba debe tener exactamente 109 observaciones")
        self.assertEqual(len(X_tr) + len(X_te), 545, "La suma total de observaciones debe ser 545 (T06)")

        interseccion = set(X_tr.index) & set(X_te.index)
        self.assertEqual(len(interseccion), 0, "No debe existir solapamiento de identificadores entre entrenamiento y prueba")
        self.assertFalse(X_te.isna().any().any(), "No deben existir valores nulos en el conjunto de prueba")

    def test_existencia_tablas_t25(self):
        """Verifica que todas las tablas maestras de T25 existan y tengan el número correcto de registros."""
        tablas_esperadas = [
            ("t25_metricas_conjunto_prueba.csv", 18),
            ("t25_comparacion_cv_vs_prueba.csv", 18),
            ("t25_contraste_hipotesis_h1.csv", 2),
            ("t25_predicciones_prueba.csv", 109),
            ("t25_bootstrap_ic_prueba.csv", 18),
        ]

        for nombre_tabla, min_filas in tablas_esperadas:
            ruta = DIR_TABLAS / nombre_tabla
            self.assertTrue(ruta.exists(), f"Falta la tabla {nombre_tabla}")
            df = pd.read_csv(ruta)
            self.assertGreaterEqual(len(df), min_filas, f"La tabla {nombre_tabla} tiene menos filas de las esperadas")

    def test_existencia_figuras_t25(self):
        """Verifica que todas las figuras IEEE de T25 existan y tengan tamaño no nulo."""
        figuras_esperadas = [
            "fig_t25_predicho_vs_real_prueba.png",
            "fig_t25_comparacion_cv_vs_prueba.png",
            "fig_t25_distribucion_residuos_prueba.png",
            "fig_t25_comparacion_metricas_test.png",
        ]

        for fig_nombre in figuras_esperadas:
            ruta = DIR_FIGURAS / fig_nombre
            self.assertTrue(ruta.exists(), f"Falta la figura {fig_nombre}")
            self.assertGreater(os.path.getsize(ruta), 10_000, f"La figura {fig_nombre} parece vacía o corrupta")

    def test_contraste_formal_h1(self):
        """Verifica que el contraste de H1 esté computado y contenga estadísticas válidas."""
        ruta_h1 = DIR_TABLAS / "t25_contraste_hipotesis_h1.csv"
        df_h1 = pd.read_csv(ruta_h1)

        self.assertEqual(len(df_h1), 2)
        for _, fila in df_h1.iterrows():
            self.assertIn("p_valor_t_pareada", fila)
            self.assertIn("p_valor_wilcoxon", fila)
            self.assertGreater(fila["p_valor_t_pareada"], 0.0)
            self.assertLess(fila["p_valor_t_pareada"], 1.0)
            self.assertTrue("RESPALDA" in fila["conclusion_h1"].upper() or "RECHAZA" in fila["conclusion_h1"].upper())

    def test_existencia_modelo_final(self):
        """Verifica que el modelo final serializado exista en models/modelo_final.joblib."""
        from pathlib import Path
        ruta_modelo = Path("models") / "modelo_final.joblib"
        self.assertTrue(ruta_modelo.exists(), "Falta el modelo final en models/modelo_final.joblib")
        self.assertGreater(os.path.getsize(ruta_modelo), 1_000, "El archivo del modelo final está vacío")


if __name__ == "__main__":
    unittest.main()
