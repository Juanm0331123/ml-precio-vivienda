"""Pruebas unitarias para modelos de ensamble Random Forest y XGBoost (Tarjeta T21).

Responsable: Brandon Eduardo Tobar Sánchez (Model Developer).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluacion import cargar_entrenamiento, FACTOR_KAGGLE
from src.ensambles import (
    crear_pipeline_ensamble,
    obtener_prefijo_ensamble,
    obtener_distribucion_ensamble,
    extraer_modelo_base,
    extraer_importancia_caracteristicas,
)


class TestEnsambles(unittest.TestCase):
    """Verifica la construcción, prefijos, ajuste y predicción de Random Forest y XGBoost."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.evidencia = cargar_entrenamiento()

    def test_prefijos_ensamble(self):
        """Verifica que P1 use 'modelo__' y P2/P3 'regressor__modelo__'."""
        self.assertEqual(obtener_prefijo_ensamble("P1"), "modelo__")
        self.assertEqual(obtener_prefijo_ensamble("P2"), "regressor__modelo__")
        self.assertEqual(obtener_prefijo_ensamble("P3"), "regressor__modelo__")

    def test_distribuciones_tienen_prefijos(self):
        """Comprueba que las distribuciones de hiperparámetros incluyan el prefijo adecuado."""
        dist_rf_p1 = obtener_distribucion_ensamble("M5", "P1")
        self.assertIn("modelo__n_estimators", dist_rf_p1)
        self.assertIn("modelo__max_depth", dist_rf_p1)

        dist_xgb_p2 = obtener_distribucion_ensamble("M6", "P2")
        self.assertIn("regressor__modelo__n_estimators", dist_xgb_p2)
        self.assertIn("regressor__modelo__learning_rate", dist_xgb_p2)

    def test_rf_fit_predict(self):
        """Verifica que Random Forest se ajuste y prediga en la escala original."""
        sub_X = self.X.iloc[:50]
        sub_y = self.y.iloc[:50]

        for prepro_id in ["P1", "P2"]:
            pipe = crear_pipeline_ensamble("M5", prepro_id, {"n_estimators": 10})
            pipe.fit(sub_X, sub_y)
            preds = pipe.predict(sub_X)
            
            self.assertEqual(len(preds), len(sub_X))
            self.assertTrue(np.all(np.isfinite(preds)))
            self.assertTrue(np.all(preds > 100_000))

    def test_xgb_fit_predict(self):
        """Verifica que XGBoost se ajuste y prediga en la escala original."""
        sub_X = self.X.iloc[:50]
        sub_y = self.y.iloc[:50]

        for prepro_id in ["P1", "P2"]:
            pipe = crear_pipeline_ensamble("M6", prepro_id, {"n_estimators": 10})
            pipe.fit(sub_X, sub_y)
            preds = pipe.predict(sub_X)
            
            self.assertEqual(len(preds), len(sub_X))
            self.assertTrue(np.all(np.isfinite(preds)))
            self.assertTrue(np.all(preds > 100_000))

    def test_importancia_caracteristicas(self):
        """Verifica la extracción de importancias en RF y XGBoost."""
        sub_X = self.X.iloc[:40]
        sub_y = self.y.iloc[:40]

        pipe = crear_pipeline_ensamble("M5", "P2", {"n_estimators": 15})
        pipe.fit(sub_X, sub_y)
        df_imp = extraer_importancia_caracteristicas(pipe, sub_X, sub_y, "P2")
        
        self.assertIsInstance(df_imp, pd.DataFrame)
        self.assertIn("variable", df_imp.columns)
        self.assertIn("importancia", df_imp.columns)
        self.assertAlmostEqual(df_imp["importancia"].sum(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
