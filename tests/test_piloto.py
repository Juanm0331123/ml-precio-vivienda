"""Pruebas del piloto de viabilidad (Tarjeta T12): aislamiento del test, pliegues, pipelines y ausencia de fuga.

Se ejecutan con ``python -m pytest tests/test_piloto.py`` o ``python -m unittest tests.test_piloto``.
Trabajan solo con las filas de entrenamiento (nunca con ``prueba``).
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest

from src.config import RUTA_DATOS_CRUDOS, SHA256_DATOS_CRUDOS
from src.evaluacion import (
    calcular_metricas,
    cargar_entrenamiento,
    crear_pliegues,
    huella_sha256,
)
from src.modelos import MODELOS, construir_pipeline, pipeline_interno
from src.preprocesamiento import (
    COLUMNAS_PREDICTORAS,
    Winsorizador,
    construir_preprocesamiento,
    nombres_seleccionados,
)

PREPROCESAMIENTOS = ("P1", "P2", "P3")


class PruebasPiloto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.evidencia = cargar_entrenamiento()
        cls.pliegues = crear_pliegues(cls.y)

    # --- A. Aislamiento del test sellado -------------------------------------------------
    def test_solo_entrenamiento(self):
        self.assertEqual(self.evidencia["n_entrenamiento_cargado"], 436)
        self.assertEqual(self.evidencia["n_prueba_en_particion"], 109)
        self.assertEqual(self.evidencia["n_prueba_cargado"], 0)
        self.assertEqual(self.evidencia["interseccion_ids_entrenamiento_prueba"], 0)
        self.assertEqual(len(self.X), 436)

    def test_columnas_reales(self):
        self.assertEqual(list(self.X.columns), COLUMNAS_PREDICTORAS)
        self.assertNotIn("price", self.X.columns)
        self.assertEqual(int(self.X.isna().sum().sum()), 0)

    def test_datos_crudos_intactos(self):
        self.assertEqual(huella_sha256(RUTA_DATOS_CRUDOS), SHA256_DATOS_CRUDOS)

    # --- B. Pliegues -----------------------------------------------------------------------
    def test_pliegues_deterministas_y_cantidad(self):
        otros = crear_pliegues(self.y)
        self.assertEqual(len(self.pliegues), 15)  # 5 pliegues × 3 repeticiones
        for (a1, v1), (a2, v2) in zip(self.pliegues, otros):
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(v1, v2)

    def test_pliegues_disjuntos_y_cubren_cada_repeticion(self):
        for r in range(3):
            validaciones = np.concatenate([self.pliegues[r * 5 + k][1] for k in range(5)])
            self.assertEqual(sorted(validaciones), list(range(436)))  # cada fila se valida una vez por repetición
        for ajuste, validacion in self.pliegues:
            self.assertEqual(len(set(ajuste) & set(validacion)), 0)

    # --- C. Pipelines ------------------------------------------------------------------------
    def test_18_combinaciones_ajustan_y_predicen(self):
        ajuste, validacion = self.pliegues[0]
        for p in PREPROCESAMIENTOS:
            for m in MODELOS:
                modelo = clone(construir_pipeline(p, m)).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
                pred = modelo.predict(self.X.iloc[validacion])
                self.assertTrue(np.all(np.isfinite(pred)), f"{p}-{m}: predicción no finita")
                self.assertTrue(np.all(pred > 0), f"{p}-{m}: predicción no positiva")

    def test_transformacion_del_objetivo_regresa_a_escala_original(self):
        ajuste, validacion = self.pliegues[0]
        for m in ("M1", "M4", "M5"):
            p1 = clone(construir_pipeline("P1", m)).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
            p2 = clone(construir_pipeline("P2", m)).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
            # En escala original, las predicciones de P1 y P2 son del mismo orden de magnitud (no log-precios).
            razon = p2.predict(self.X.iloc[validacion]).mean() / p1.predict(self.X.iloc[validacion]).mean()
            self.assertTrue(0.7 < razon < 1.4, f"{m}: razón de medias P2/P1 = {razon:.2f}")
        r = pipeline_interno(clone(construir_pipeline("P2", "M1")).fit(self.X.iloc[ajuste], self.y.iloc[ajuste]))
        self.assertIn("prepro", r.named_steps)

    def test_p3_incluye_seleccion_k_fijo(self):
        pre = construir_preprocesamiento("P3", escalar=False)
        self.assertIsInstance(pre.named_steps["sel"], SelectKBest)
        self.assertEqual(pre.named_steps["sel"].k, 10)

    # --- D. Ausencia de fuga ---------------------------------------------------------------------
    def test_winsorizador_aprende_solo_del_ajuste(self):
        a = pd.DataFrame({"x": np.arange(100.0)})
        w = Winsorizador().fit(a)
        extremo = pd.DataFrame({"x": [-1000.0, 5000.0]})
        recortado = w.transform(extremo)
        self.assertEqual(recortado[0, 0], np.percentile(a["x"], 1))
        self.assertEqual(recortado[1, 0], np.percentile(a["x"], 99))

    def test_seleccion_depende_solo_de_las_filas_de_ajuste(self):
        ajuste, validacion = self.pliegues[0]
        y_validacion_alterado = self.y.copy()
        y_validacion_alterado.iloc[validacion] = y_validacion_alterado.iloc[validacion].values[::-1] * 3  # cambia solo la validación
        a = construir_preprocesamiento("P3", escalar=False).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
        b = construir_preprocesamiento("P3", escalar=False).fit(self.X.iloc[ajuste], y_validacion_alterado.iloc[ajuste])
        self.assertEqual(nombres_seleccionados(a), nombres_seleccionados(b))

    def test_reproducible_mismo_entorno(self):
        ajuste, validacion = self.pliegues[1]
        m1 = clone(construir_pipeline("P3", "M6")).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
        m2 = clone(construir_pipeline("P3", "M6")).fit(self.X.iloc[ajuste], self.y.iloc[ajuste])
        np.testing.assert_allclose(m1.predict(self.X.iloc[validacion]), m2.predict(self.X.iloc[validacion]))

    # --- E. Métricas ---------------------------------------------------------------------------------
    def test_metricas_con_valores_conocidos(self):
        m = calcular_metricas([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
        self.assertAlmostEqual(m["mae"], 50 / 3)
        self.assertAlmostEqual(m["rmse"], np.sqrt((100 + 100 + 900) / 3))
        self.assertAlmostEqual(m["mape"], 100 * np.mean([0.1, 0.05, 0.1]))
        self.assertAlmostEqual(m["r2"], 1 - 1100 / 20000)

    def test_r2_puede_ser_negativo(self):
        self.assertLess(calcular_metricas([1.0, 2.0, 3.0], [3.0, 1.0, 2.0])["r2"], 0)


if __name__ == "__main__":
    unittest.main()
