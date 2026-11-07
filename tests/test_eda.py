"""Pruebas unitarias para el módulo src/eda.py (Tarjeta T07)."""

import unittest

import numpy as np
import pandas as pd

from src.config import SEMILLA
from src.datos import cargar_datos_crudos, limpiar_datos
from src.eda import (
    calcular_correlaciones_bivariadas,
    calcular_dependencia_no_lineal,
    calcular_estadisticos_univariados,
    calcular_frecuencias_variables,
    calcular_vif,
    comparar_efecto_log,
)


class TestEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_crudo = cargar_datos_crudos()
        cls.df_limpio = limpiar_datos(cls.df_crudo)
        cls.features = [c for c in cls.df_limpio.columns if c not in ("id", "price")]

    def test_estadisticos_univariados_precio(self):
        est = calcular_estadisticos_univariados(self.df_limpio["price"])
        self.assertEqual(est["n"], 545)
        self.assertAlmostEqual(est["media"], 4766729.25, delta=1.0)
        self.assertAlmostEqual(est["mediana"], 4340000.0, delta=1.0)
        self.assertAlmostEqual(est["iqr"], 2310000.0, delta=1.0)
        # Asimetría debe ser ≈ 1.21 como solicita el criterio de aceptación
        self.assertAlmostEqual(est["asimetria"], 1.2122, places=3)
        self.assertAlmostEqual(est["curtosis_exceso"], 1.9601, places=3)

    def test_comparar_efecto_log(self):
        df_comp = comparar_efecto_log(self.df_limpio["price"])
        self.assertEqual(len(df_comp), 8)
        self.assertIn("Asimetría (Skewness)", df_comp["metrica"].values)

        # Verificar asimetría logarítmica
        log_price = np.log(self.df_limpio["price"])
        est_log = calcular_estadisticos_univariados(log_price)
        self.assertAlmostEqual(est_log["asimetria"], 0.1409, places=3)
        self.assertAlmostEqual(est_log["curtosis_exceso"], -0.1374, places=3)

    def test_frecuencias_variables(self):
        df_frec = calcular_frecuencias_variables(self.df_crudo, ["bedrooms", "furnishingstatus"])
        self.assertTrue(len(df_frec) > 0)
        self.assertIn("porcentaje", df_frec.columns)
        # bedrooms suma 100 %
        pct_bedrooms = df_frec[df_frec["variable"] == "bedrooms"]["porcentaje"].sum()
        self.assertAlmostEqual(pct_bedrooms, 100.0, delta=0.2)

    def test_correlaciones_bivariadas(self):
        df_corr = calcular_correlaciones_bivariadas(self.df_limpio, self.features, "price")
        self.assertEqual(len(df_corr), 12)
        # 'area' debe tener la mayor correlación de Pearson
        self.assertEqual(df_corr.iloc[0]["variable"], "area")
        self.assertAlmostEqual(df_corr.iloc[0]["pearson_original"], 0.5360, places=3)
        self.assertAlmostEqual(df_corr.iloc[0]["spearman_original"], 0.6029, places=3)

    def test_dependencia_no_lineal(self):
        df_mi = calcular_dependencia_no_lineal(self.df_limpio, self.features, "price", semilla=SEMILLA)
        self.assertEqual(len(df_mi), 12)
        # 'area' debe tener la mayor información mutua
        self.assertEqual(df_mi.iloc[0]["variable"], "area")
        self.assertGreater(df_mi.iloc[0]["informacion_mutua"], 0.20)

    def test_calcular_vif(self):
        df_vif = calcular_vif(self.df_limpio, self.features)
        self.assertEqual(len(df_vif), 12)
        # Todos los VIF deben ser moderados/bajos (< 2.0)
        self.assertTrue((df_vif["VIF"] < 2.0).all())
        self.assertTrue((df_vif["VIF"] >= 1.0).all())


if __name__ == "__main__":
    unittest.main()
