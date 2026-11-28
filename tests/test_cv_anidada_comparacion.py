"""Pruebas unitarias para inferencia estadística, prueba t corregida y CV anidada (Tarjeta T22).

Responsable: Brandon Eduardo Tobar Sánchez (Evaluation Lead).
Revisa el PR: Juan Miguel León Gómez (Project Manager / Reporting Lead).
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluacion import (
    calcular_metricas,
    calcular_intervalo_confianza,
    prueba_t_corregida_nadeau_bengio,
    prueba_friedman_ranking,
)


class TestEvaluacionEstadistica(unittest.TestCase):
    """Verifica las funciones de inferencia estadística, prueba t corregida y Friedman."""

    def test_calcular_metricas_incluye_rmsle(self):
        """Verifica que calcular_metricas devuelva rmsle positivo y finito."""
        y_real = [100.0, 200.0, 300.0]
        y_pred = [110.0, 190.0, 310.0]
        m = calcular_metricas(y_real, y_pred)
        
        self.assertIn("rmsle", m)
        self.assertIn("rmse", m)
        self.assertIn("mae", m)
        self.assertIn("mape", m)
        self.assertIn("r2", m)
        self.assertGreater(m["rmsle"], 0.0)
        self.assertLess(m["rmsle"], 1.0)

    def test_intervalo_confianza(self):
        """Verifica cálculo de intervalo de confianza t-Student bilateral."""
        valores = [10.0, 12.0, 11.0, 9.0, 13.0, 11.0]
        ic = calcular_intervalo_confianza(valores, nivel=0.95)
        
        self.assertAlmostEqual(ic["media"], 11.0)
        self.assertLess(ic["ic_inf"], ic["media"])
        self.assertGreater(ic["ic_sup"], ic["media"])
        self.assertGreater(ic["error_estandar"], 0.0)

    def test_prueba_t_corregida_nadeau_bengio_identicos(self):
        """Si las diferencias son cero, t=0 y p-valor=1.0."""
        difs = [0.0] * 15
        res = prueba_t_corregida_nadeau_bengio(difs, n_ajuste=348, n_val=88)
        
        self.assertEqual(res["t_estadistico"], 0.0)
        self.assertEqual(res["p_valor"], 1.0)
        self.assertFalse(res["significativo_al_5pct"])

    def test_prueba_t_corregida_nadeau_bengio_significativa(self):
        """Diferencias consistentemente positivas deben arrojar p-valor bajo."""
        difs = [5000.0 + np.random.normal(0, 100) for _ in range(15)]
        res = prueba_t_corregida_nadeau_bengio(difs, n_ajuste=348, n_val=88)
        
        self.assertGreater(res["t_estadistico"], 5.0)
        self.assertLess(res["p_valor"], 0.01)
        self.assertTrue(res["significativo_al_5pct"])

    def test_prueba_friedman_ranking(self):
        """Verifica ranking medio y test omnibus de Friedman."""
        filas = []
        for pliegue in range(15):
            filas.append({"pliegue": pliegue, "modelo_id_completo": "ModeloA", "rmse_cad": 15000 + pliegue * 10})
            filas.append({"pliegue": pliegue, "modelo_id_completo": "ModeloB", "rmse_cad": 20000 + pliegue * 10})
            filas.append({"pliegue": pliegue, "modelo_id_completo": "ModeloC", "rmse_cad": 25000 + pliegue * 10})
        
        df_dummy = pd.DataFrame(filas)
        stat, p_val, ranking = prueba_friedman_ranking(df_dummy, "rmse_cad")
        
        self.assertLess(p_val, 1e-4)
        self.assertEqual(ranking.iloc[0]["modelo_id_completo"], "ModeloA")
        self.assertEqual(ranking.iloc[0]["ranking_medio"], 1.0)
        self.assertEqual(ranking.iloc[-1]["ranking_medio"], 3.0)


if __name__ == "__main__":
    unittest.main()
