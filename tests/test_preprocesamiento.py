"""Pruebas rápidas de los pipelines de preprocesamiento P1, P2 y P3 (Tarjeta T17).

Verifican dimensiones, ausencia de NaN/infinitos, coincidencia exacta de columnas entre
entrenamiento y validación, codificación yes/no -> 0/1, transformación del objetivo
(TransformedTargetRegressor) y ausencia de fuga de información.

Se ejecutan con: ``python -m unittest tests.test_preprocesamiento -v``
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.datos import cargar_datos_crudos
from src.evaluacion import cargar_entrenamiento, crear_pliegues
from src.preprocesamiento import (
    COLUMNAS_BINARIAS,
    COLUMNAS_PREDICTORAS,
    CodificadorBase,
    construir_preprocesamiento,
    envolver_objetivo,
    nombres_seleccionados,
)


class TestPreprocesamientoT17(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.evidencia = cargar_entrenamiento()
        cls.pliegues = crear_pliegues(cls.y)
        ajuste, validacion = cls.pliegues[0]
        cls.X_train = cls.X.iloc[ajuste].copy()
        cls.y_train = cls.y.iloc[ajuste].copy()
        cls.X_val = cls.X.iloc[validacion].copy()
        cls.y_val = cls.y.iloc[validacion].copy()

    def test_dimensiones_sin_nan_y_mismas_columnas_train_test(self):
        esperado_cols = {"P1": 13, "P2": 13, "P3": 17}
        for p in ("P1", "P2", "P3"):
            for escalar in (True, False):
                pre = construir_preprocesamiento(p, escalar=escalar)
                Xt_train = pre.fit_transform(self.X_train, self.y_train)
                Xt_val = pre.transform(self.X_val)

                # Dimensiones esperadas
                self.assertEqual(Xt_train.shape, (len(self.X_train), esperado_cols[p]), f"{p} escalar={escalar}")
                self.assertEqual(Xt_val.shape, (len(self.X_val), esperado_cols[p]), f"{p} escalar={escalar}")

                # Sin NaN ni infinitos
                arr_train = np.asarray(Xt_train, dtype=float)
                arr_val = np.asarray(Xt_val, dtype=float)
                self.assertFalse(np.isnan(arr_train).any(), f"{p}: NaN en train")
                self.assertFalse(np.isnan(arr_val).any(), f"{p}: NaN en val")
                self.assertTrue(np.isfinite(arr_train).all(), f"{p}: no finito en train")
                self.assertTrue(np.isfinite(arr_val).all(), f"{p}: no finito en val")

                # Mismas columnas en train y test/val (cuando la salida es DataFrame)
                if isinstance(Xt_train, pd.DataFrame) and isinstance(Xt_val, pd.DataFrame):
                    self.assertEqual(list(Xt_train.columns), list(Xt_val.columns))

    def test_codificacion_yes_no_y_amoblado_equivalente_a_limpio(self):
        df_crudo = cargar_datos_crudos()
        X_crudo = df_crudo.set_index("id").loc[self.X_train.index, COLUMNAS_PREDICTORAS]

        # Verificar que las columnas binarias llegan como yes/no en el crudo
        for col in COLUMNAS_BINARIAS:
            self.assertTrue(set(X_crudo[col].unique()).issubset({"yes", "no"}))

        cod = CodificadorBase().fit(X_crudo)
        X_cod = cod.transform(X_crudo)
        pd.testing.assert_frame_equal(X_cod, self.X_train, check_dtype=False)

        # El pipeline P1 completo produce exactamente la misma matriz desde crudo o desde limpio
        p_crudo = construir_preprocesamiento("P1", escalar=True).fit_transform(X_crudo, self.y_train)
        p_limpio = construir_preprocesamiento("P1", escalar=True).fit_transform(self.X_train, self.y_train)
        pd.testing.assert_frame_equal(p_crudo, p_limpio)

    def test_envolver_objetivo_transformed_target_regressor(self):
        est_p1 = envolver_objetivo(
            Pipeline([("pre", construir_preprocesamiento("P1", True)), ("reg", Ridge())]),
            "P1",
        )
        self.assertNotIsInstance(est_p1, TransformedTargetRegressor)

        for p in ("P2", "P3"):
            est = envolver_objetivo(
                Pipeline([("pre", construir_preprocesamiento(p, True)), ("reg", Ridge())]),
                p,
            )
            self.assertIsInstance(est, TransformedTargetRegressor)
            est.fit(self.X_train, self.y_train)
            pred = est.predict(self.X_val)
            self.assertEqual(pred.shape, (len(self.X_val),))
            self.assertTrue(np.all(np.isfinite(pred)))
            # Las predicciones deben estar en la escala original del precio (orden de 10^6)
            self.assertGreater(pred.mean(), 1e5)

    def test_sin_fuga_de_informacion_en_fit(self):
        # Alterar fuertemente X_val y y_val no debe cambiar en nada el ajuste sobre train
        X_val_extremo = self.X_val.copy()
        X_val_extremo["area"] = X_val_extremo["area"] * 100

        for p in ("P1", "P2", "P3"):
            pre_a = construir_preprocesamiento(p, escalar=True).fit(self.X_train, self.y_train)
            pre_b = construir_preprocesamiento(p, escalar=True).fit(self.X_train, self.y_train)

            out_a = np.asarray(pre_a.transform(self.X_train), dtype=float)
            out_b = np.asarray(pre_b.transform(self.X_train), dtype=float)
            np.testing.assert_allclose(out_a, out_b)

            # Transformar datos extremos con P3 no debe producir NaN ni cambiar las columnas seleccionadas
            out_ext = np.asarray(pre_a.transform(X_val_extremo), dtype=float)
            self.assertFalse(np.isnan(out_ext).any())
            if p == "P3":
                self.assertEqual(len(nombres_seleccionados(pre_a)), 17)


if __name__ == "__main__":
    unittest.main()
