"""Pruebas unitarias para src/datos.py (Tarjeta T06)."""

import unittest
from pathlib import Path

import pandas as pd

from src.config import (
    RUTA_DATOS_CRUDOS,
    RUTA_DATOS_LIMPIOS,
    RUTA_PARTICION,
    RUTA_TABLA_CALIDAD,
    SEMILLA,
)
from src.datos import (
    auditar_completitud,
    auditar_unicidad,
    auditar_validez,
    cargar_datos_crudos,
    crear_particion_sellada,
    detectar_atipicos,
    evaluar_desbalance,
    generar_scorecard_calidad,
    limpiar_datos,
    verificar_artefacto_orden,
)


class TestDatos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_crudo = cargar_datos_crudos()

    def test_carga_datos_crudos(self):
        self.assertEqual(len(self.df_crudo), 545)
        self.assertIn("id", self.df_crudo.columns)
        self.assertEqual(self.df_crudo["id"].iloc[0], 0)
        self.assertEqual(self.df_crudo["id"].iloc[-1], 544)

    def test_completitud(self):
        reporte = auditar_completitud(self.df_crudo)
        self.assertEqual(reporte["nulos"].sum(), 0)
        self.assertTrue((reporte["completitud_pct"] == 100.0).all())

    def test_unicidad_y_ruido_irreducible(self):
        resultado = auditar_unicidad(self.df_crudo)
        self.assertEqual(resultado["duplicados_completos"], 0)
        # 6 casos de covariables repetidas con distinto precio
        self.assertEqual(resultado["casos_duplicados_atributos"], 6)
        # 16 casos si se ignora furnishingstatus
        self.assertEqual(resultado["casos_duplicados_sin_amoblado"], 16)
        self.assertIn("ruido irreducible", resultado["explicacion_ruido_irreducible"])

    def test_validez_dominios_y_rangos(self):
        validez = auditar_validez(self.df_crudo)
        self.assertTrue(validez["es_valido"].all())

    def test_deteccion_atipicos_precio(self):
        atip = detectar_atipicos(self.df_crudo, columna="price")
        # Exactamente 15 atípicos superiores por 1.5xIQR
        self.assertEqual(atip["atipicos_sup_iqr"], 15)
        self.assertEqual(atip["atipicos_inf_iqr"], 0)
        self.assertEqual(atip["total_atipicos_iqr"], 15)
        # MAD z-score robusto
        self.assertGreaterEqual(atip["atipicos_mad_abs_gt_3"], 15)
        self.assertIn("conservar", atip["justificacion"].lower())

    def test_evaluacion_desbalance(self):
        desb = evaluar_desbalance(self.df_crudo)
        hw = desb[desb["variable"] == "hotwaterheating"].iloc[0]
        self.assertAlmostEqual(hw["pct_minoritaria"], 4.59, places=1)
        self.assertEqual(hw["conteo_minoritario"], 25)
        self.assertEqual(hw["nivel_riesgo"], "Alto")

    def test_artefacto_orden(self):
        diag = verificar_artefacto_orden(self.df_crudo, columna="price")
        self.assertTrue(diag["es_monotonic_decreasing"])

    def test_limpieza_reproducible_sin_fuga(self):
        df_limpio = limpiar_datos(self.df_crudo)
        self.assertEqual(len(df_limpio), 545)
        # Binarias deben ser 0 o 1
        for col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
            self.assertTrue(set(df_limpio[col].unique()).issubset({0, 1}))
        # furnishingstatus debe ser 0, 1 o 2
        self.assertTrue(set(df_limpio["furnishingstatus"].unique()).issubset({0, 1, 2}))

    def test_particion_sellada_estratificada(self):
        df_limpio = limpiar_datos(self.df_crudo)
        part, train, test = crear_particion_sellada(df_limpio, semilla=SEMILLA)
        self.assertEqual(len(part), 545)
        self.assertEqual(len(train), 436)
        self.assertEqual(len(test), 109)
        self.assertEqual(set(part["conjunto"].unique()), {"entrenamiento", "prueba"})

        # Verificación de distribución por quintiles
        quintiles_train = pd.qcut(train["price"], q=5, labels=False)
        quintiles_test = pd.qcut(test["price"], q=5, labels=False)
        self.assertEqual(len(quintiles_train.unique()), 5)
        self.assertEqual(len(quintiles_test.unique()), 5)

    def test_archivos_generados_existen(self):
        self.assertTrue(RUTA_DATOS_LIMPIOS.is_file())
        self.assertTrue(RUTA_PARTICION.is_file())
        self.assertTrue(RUTA_TABLA_CALIDAD.is_file())


if __name__ == "__main__":
    unittest.main()
