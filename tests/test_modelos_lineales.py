"""Pruebas de T19: modelos lineales (OLS, Ridge, Lasso, ElasticNet), búsqueda con CV interna y coeficientes.

``python -m pytest tests/test_modelos_lineales.py``. Solo usan filas de entrenamiento (nunca ``prueba``).
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.busqueda import ejecutar_busqueda, fila_mejor, tabla_candidatos
from src.evaluacion import cargar_entrenamiento, crear_pliegues
from src.modelos_lineales import (
    ESPACIOS,
    MODELOS_LINEALES,
    PREFIJO,
    PREPROCESAMIENTOS_T19,
    construir_pipeline_lineal,
    diagnostico_convergencia,
    ols_hedonico_statsmodels,
    tabla_coeficientes,
)
from src.preprocesamiento import construir_preprocesamiento


class PruebasModelosLineales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.evidencia = cargar_entrenamiento()
        cls.pliegues = crear_pliegues(cls.y)

    def test_solo_entrenamiento_y_p3_pendiente(self):
        self.assertEqual(self.evidencia["n_entrenamiento_cargado"], 436)
        self.assertEqual(self.evidencia["n_prueba_cargado"], 0)
        self.assertEqual(PREPROCESAMIENTOS_T19, ("P1", "P2"))
        with self.assertRaises(ValueError):
            construir_pipeline_lineal("P3", "Ridge")

    def test_cuatro_modelos_por_preprocesamiento_ajustan_y_predicen(self):
        ajuste, validacion = self.pliegues[0]
        self.assertEqual(set(MODELOS_LINEALES), {"OLS", "Ridge", "Lasso", "ElasticNet"})
        for p in PREPROCESAMIENTOS_T19:
            for nombre in MODELOS_LINEALES:
                m = clone(construir_pipeline_lineal(p, nombre)).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
                pred = m.predict(self.X.iloc[validacion])
                self.assertTrue(np.all(np.isfinite(pred)) and np.all(pred > 0), f"{p}-{nombre}")

    def test_pipeline_incluye_preprocesamiento_dentro_del_ajuste(self):
        modelo = construir_pipeline_lineal("P2", "Lasso")
        self.assertIsInstance(modelo.regressor, Pipeline)
        self.assertEqual([n for n, _ in modelo.regressor.steps], ["prepro", "modelo"])

    def test_ridge_con_objetivo_estandarizado_equivale_al_objetivo_crudo(self):
        ajuste, validacion = self.pliegues[0]
        crudo = Pipeline([("prepro", construir_preprocesamiento("P1", True)), ("modelo", Ridge(alpha=5.0))]).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
        est = construir_pipeline_lineal("P1", "Ridge").set_params(**{PREFIJO + "alpha": 5.0}).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
        np.testing.assert_allclose(crudo.predict(self.X.iloc[validacion]), est.predict(self.X.iloc[validacion]), rtol=1e-6)

    def test_p2_devuelve_escala_original(self):
        ajuste, validacion = self.pliegues[0]
        p1 = clone(construir_pipeline_lineal("P1", "OLS")).fit(self.X.iloc[ajuste], self.y.iloc[ajuste]).predict(self.X.iloc[validacion])
        p2 = clone(construir_pipeline_lineal("P2", "OLS")).fit(self.X.iloc[ajuste], self.y.iloc[ajuste]).predict(self.X.iloc[validacion])
        self.assertTrue(0.7 < p2.mean() / p1.mean() < 1.4)

    def test_busqueda_reproducible_y_con_pliegues_dados(self):
        espacio = {PREFIJO + "alpha": [0.01, 1.0, 100.0]}
        res = [ejecutar_busqueda(construir_pipeline_lineal("P1", "Ridge"), espacio, self.X, self.y, self.pliegues[:5]) for _ in range(2)]
        self.assertEqual(res[0]["busqueda"].best_params_, res[1]["busqueda"].best_params_)
        np.testing.assert_allclose(res[0]["busqueda"].cv_results_["mean_test_rmse"], res[1]["busqueda"].cv_results_["mean_test_rmse"])
        tabla = tabla_candidatos(res[0]["busqueda"], "P1", "Ridge")
        self.assertEqual(len(tabla), 3)
        fila = fila_mejor(res[0]["busqueda"], "P1", "Ridge", res[0]["duracion_s"], res[0]["n_candidatos"])
        self.assertEqual(fila["n_pliegues"], 5)
        self.assertGreater(fila["rmse_media"], 0)

    def test_desviacion_entre_pliegues_es_muestral(self):
        espacio = {PREFIJO + "alpha": [1.0]}
        r = ejecutar_busqueda(construir_pipeline_lineal("P1", "Ridge"), espacio, self.X, self.y, self.pliegues)
        fila = fila_mejor(r["busqueda"], "P1", "Ridge", r["duracion_s"], r["n_candidatos"])
        por_pliegue = [-r["busqueda"].cv_results_[f"split{k}_test_rmse"][0] for k in range(15)]
        self.assertAlmostEqual(fila["rmse_desv"], np.std(por_pliegue, ddof=1), places=4)
        tabla = tabla_candidatos(r["busqueda"], "P1", "Ridge")
        self.assertAlmostEqual(tabla["rmse_desv"].iloc[0], fila["rmse_desv"], places=6)  # misma convención en ambas tablas

    def test_lasso_con_alpha_alto_anula_todos_los_coeficientes(self):
        ajuste, _ = self.pliegues[0]
        m = construir_pipeline_lineal("P2", "Lasso").set_params(**{PREFIJO + "alpha": 5.0}).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
        tabla = tabla_coeficientes(m, self.X.iloc[ajuste], "P2", "Lasso")
        self.assertTrue(tabla["coef_nulo"].all())

    def test_coeficientes_p2_y_p1_tienen_las_columnas_esperadas(self):
        ajuste, _ = self.pliegues[0]
        for p, extra, nombre in (("P1", "efecto_1DE_cad1987", "Ridge"), ("P2", "efecto_1DE_pct", "Ridge"), ("P1", "efecto_1DE_cad1987", "OLS"), ("P2", "efecto_1DE_pct", "OLS")):
            m = clone(construir_pipeline_lineal(p, nombre)).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
            t = tabla_coeficientes(m, self.X.iloc[ajuste], p, nombre)
            self.assertIn(extra, t.columns)
            self.assertEqual(len(t), len(m.regressor_.named_steps["prepro"].get_feature_names_out()))
            np.testing.assert_allclose(t["efecto_1DE"], t["coef_modelo"] * t["sd_variable"])

    def test_convergencia_descenso_por_coordenadas(self):
        ajuste, _ = self.pliegues[0]
        for nombre in ("Lasso", "ElasticNet"):
            m = clone(construir_pipeline_lineal("P2", nombre)).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
            d = diagnostico_convergencia(m, nombre)
            self.assertTrue(d["convergio"])
        self.assertIsNone(diagnostico_convergencia(clone(construir_pipeline_lineal("P2", "Ridge")).fit(self.X.iloc[ajuste], self.y.iloc[ajuste]), "Ridge")["n_iter"])

    def test_rejillas_explicitas(self):
        self.assertGreaterEqual(len(ESPACIOS["Ridge"][PREFIJO + "alpha"]), 20)
        self.assertIn(PREFIJO + "l1_ratio", ESPACIOS["ElasticNet"])

    def test_ols_hedonico_statsmodels(self):
        tabla, resumen = ols_hedonico_statsmodels(self.X, self.y, logaritmico=True)
        self.assertEqual(resumen["n"], 436)
        self.assertTrue(0 < resumen["r2_ajustado"] < resumen["r2"] < 1)
        self.assertIn("cambio_pct", tabla.columns)


if __name__ == "__main__":
    unittest.main()
