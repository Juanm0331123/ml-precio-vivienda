"""Pruebas unitarias para ingeniería de variables, selección en CV y ablación (Tarjeta T18).

Se ejecutan con: ``python -m unittest tests.test_ingenieria -v``
"""

import unittest

import numpy as np
import pandas as pd

from src.evaluacion import cargar_entrenamiento, crear_pliegues
from src.ingenieria import (
    COLUMNAS_DERIVADAS_T18,
    DerivadorVariablesT18,
    construir_pipeline_t18,
    construir_preprocesamiento_t18,
    justificar_variables_derivadas,
    prueba_t_corregida_cv,
)
from src.modelos import MODELOS


class TestIngenieriaT18(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, _ = cargar_entrenamiento()
        cls.pliegues = crear_pliegues(cls.y)
        ajuste, validacion = cls.pliegues[0]
        cls.X_train = cls.X.iloc[ajuste].copy()
        cls.y_train = cls.y.iloc[ajuste].copy()
        cls.X_val = cls.X.iloc[validacion].copy()
        cls.y_val = cls.y.iloc[validacion].copy()

    def test_derivador_t18_formulas_y_nombres(self):
        der = DerivadorVariablesT18().fit(self.X_train)
        out = der.transform(self.X_train)
        for col in COLUMNAS_DERIVADAS_T18:
            self.assertIn(col, out.columns)

        np.testing.assert_allclose(
            out["habitaciones_totales"],
            self.X_train["bedrooms"] + self.X_train["bathrooms"],
        )
        np.testing.assert_allclose(
            out["banos_por_habitacion"],
            self.X_train["bathrooms"] / self.X_train["bedrooms"],
        )
        np.testing.assert_allclose(
            out["n_comodidades"],
            self.X_train[["guestroom", "basement", "hotwaterheating", "airconditioning"]].sum(axis=1),
        )
        np.testing.assert_allclose(out["log_area"], np.log(self.X_train["area"]))
        self.assertEqual(list(der.get_feature_names_out()), list(out.columns))

    def test_derivador_t18_valida_dominios(self):
        der = DerivadorVariablesT18().fit(self.X_train)
        mal_bed = self.X_train.iloc[:3].copy()
        mal_bed.iloc[0, mal_bed.columns.get_loc("bedrooms")] = 0
        with self.assertRaises(ValueError):
            der.transform(mal_bed)

        mal_area = self.X_train.iloc[:3].copy()
        mal_area.iloc[0, mal_area.columns.get_loc("area")] = 0
        with self.assertRaises(ValueError):
            der.transform(mal_area)

    def test_preprocesamiento_t18_con_y_sin_amoblado_sin_fuga(self):
        # 12 originales: 5 numéricas + 6 binarias + 1 amoblado (expandido a 2 dummies) + 4 derivadas = 17
        # Sin amoblado: 5 numéricas + 6 binarias + 4 derivadas = 15
        pre_con = construir_preprocesamiento_t18(escalar=True, incluir_amoblado=True)
        pre_sin = construir_preprocesamiento_t18(escalar=True, incluir_amoblado=False)

        Xt_con = pre_con.fit_transform(self.X_train, np.log(self.y_train))
        Xv_con = pre_con.transform(self.X_val)
        Xt_sin = pre_sin.fit_transform(self.X_train, np.log(self.y_train))
        Xv_sin = pre_sin.transform(self.X_val)

        self.assertEqual(Xt_con.shape, (len(self.X_train), 17))
        self.assertEqual(Xv_con.shape, (len(self.X_val), 17))
        self.assertEqual(Xt_sin.shape, (len(self.X_train), 15))
        self.assertEqual(Xv_sin.shape, (len(self.X_val), 15))
        self.assertEqual(list(Xt_con.columns), list(Xv_con.columns))
        self.assertEqual(list(Xt_sin.columns), list(Xv_sin.columns))
        self.assertFalse(any("furnishingstatus" in c for c in Xt_sin.columns))

        # Sin fuga: escalar X_val por 1000 no cambia el ajuste en train
        X_val_ext = self.X_val.copy()
        X_val_ext["area"] = X_val_ext["area"] * 1000
        pre_b = construir_preprocesamiento_t18(escalar=True, incluir_amoblado=True).fit(
            self.X_train, np.log(self.y_train)
        )
        np.testing.assert_allclose(
            np.asarray(pre_con.transform(self.X_train), dtype=float),
            np.asarray(pre_b.transform(self.X_train), dtype=float),
        )
        self.assertTrue(np.isfinite(np.asarray(pre_con.transform(X_val_ext), dtype=float)).all())

    def test_selectores_y_pipelines_t18_predicen_en_escala_original(self):
        for sel in (None, "mi", "lasso", "rfe"):
            est = construir_pipeline_t18("M2", incluir_amoblado=True, selector=sel, k=10)
            est.fit(self.X_train, self.y_train)
            pred = est.predict(self.X_val)
            self.assertEqual(pred.shape, (len(self.X_val),))
            self.assertTrue(np.isfinite(pred).all())
            self.assertGreater(pred.mean(), 1e5)

        for m in MODELOS:
            for amob in (True, False):
                est = construir_pipeline_t18(m, incluir_amoblado=amob, selector=None)
                est.fit(self.X_train, self.y_train)
                pred = est.predict(self.X_val)
                self.assertTrue(np.isfinite(pred).all())
                self.assertTrue((pred > 0).all())

    def test_justificacion_derivadas_y_prueba_t_corregida(self):
        df_just = justificar_variables_derivadas(self.X_train, self.y_train)
        self.assertEqual(list(df_just["variable"]), COLUMNAS_DERIVADAS_T18)
        self.assertTrue(np.isfinite(df_just["pearson_log_price"]).all())

        # Prueba t corregida con diferencias conocidas
        diffs = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        res = prueba_t_corregida_cv(diffs, n_ajuste=348, n_validacion=88)
        self.assertAlmostEqual(res["media_diff"], 10.0)
        self.assertGreater(res["se_corregida"], 0.0)
        self.assertLess(res["p_valor"], 0.01)
        self.assertTrue(res["significativo_005"])


if __name__ == "__main__":
    unittest.main()
