"""Pruebas unitarias para modelos no lineales KNN y SVR (Tarjeta T20).

Responsable: Brandon Eduardo Tobar Sánchez (Model Developer).
Revisa el PR: Mauricio José Mozo Terán (Data Engineer).
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluacion import cargar_entrenamiento, FACTOR_KAGGLE
from src.modelos_no_lineales import (
    crear_pipeline_knn_svr,
    obtener_prefijo_modelo,
    obtener_grilla_no_lineal,
    optimizar_hiperparametros,
    extraer_estimador_interno,
)


class TestModelosNoLineales(unittest.TestCase):
    """Verifica la construcción, prefijos y predicción de KNN y SVR."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.evidencia = cargar_entrenamiento()

    def test_prefijos_correctos(self):
        """Verifica que los prefijos sean 'modelo__' en P1-KNN y 'regressor__modelo__' en los demás."""
        self.assertEqual(obtener_prefijo_modelo("M3", "P1"), "modelo__")
        self.assertEqual(obtener_prefijo_modelo("M4", "P1"), "regressor__modelo__")
        self.assertEqual(obtener_prefijo_modelo("M3", "P2"), "regressor__modelo__")
        self.assertEqual(obtener_prefijo_modelo("M4", "P2"), "regressor__modelo__")
        self.assertEqual(obtener_prefijo_modelo("M3", "P3"), "regressor__modelo__")
        self.assertEqual(obtener_prefijo_modelo("M4", "P3"), "regressor__modelo__")

    def test_grilla_no_lineal_tiene_prefijos(self):
        """Comprueba que las claves de la rejilla contengan el prefijo adecuado."""
        grilla_knn_p1 = obtener_grilla_no_lineal("M3", "P1")
        self.assertIn("modelo__n_neighbors", grilla_knn_p1)
        self.assertIn("modelo__weights", grilla_knn_p1)

        grilla_svr_p2 = obtener_grilla_no_lineal("M4", "P2")
        self.assertIn("regressor__modelo__C", grilla_svr_p2)
        self.assertIn("regressor__modelo__gamma", grilla_svr_p2)

    def test_pipeline_knn_fit_predict(self):
        """Verifica que el pipeline KNN en P1 y P2 se ajuste y prediga en la escala original."""
        sub_X = self.X.iloc[:50]
        sub_y = self.y.iloc[:50]

        for prepro_id in ["P1", "P2", "P3"]:
            pipe = crear_pipeline_knn_svr("M3", prepro_id, escalar_predictoras=True)
            pipe.fit(sub_X, sub_y)
            preds = pipe.predict(sub_X)
            
            self.assertEqual(len(preds), len(sub_X))
            self.assertTrue(np.all(np.isfinite(preds)))
            # Precios deben estar en escala Kaggle (> 100,000)
            self.assertTrue(np.all(preds > 100_000))

    def test_pipeline_svr_fit_predict(self):
        """Verifica que el pipeline SVR con RBF en P1, P2 y P3 funcione y devuelva precios finitos."""
        sub_X = self.X.iloc[:50]
        sub_y = self.y.iloc[:50]

        for prepro_id in ["P1", "P2", "P3"]:
            pipe = crear_pipeline_knn_svr("M4", prepro_id, escalar_predictoras=True)
            pipe.fit(sub_X, sub_y)
            preds = pipe.predict(sub_X)
            
            self.assertEqual(len(preds), len(sub_X))
            self.assertTrue(np.all(np.isfinite(preds)))
            self.assertTrue(np.all(preds > 100_000))

    def test_extraer_estimador_interno(self):
        """Verifica que se puedan extraer el preprocesador y modelo base de pipelines complejos."""
        sub_X = self.X.iloc[:30]
        sub_y = self.y.iloc[:30]

        pipe = crear_pipeline_knn_svr("M3", "P2")
        pipe.fit(sub_X, sub_y)
        prepro, modelo = extraer_estimador_interno(pipe)
        
        self.assertIsNotNone(prepro)
        self.assertIsNotNone(modelo)
        self.assertTrue(hasattr(modelo, "predict"))


if __name__ == "__main__":
    unittest.main()
