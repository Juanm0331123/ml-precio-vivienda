"""Pruebas unitarias para la aplicación demo de predicción (Tarjeta T26).

Se ejecutan con: ``python -m unittest tests.test_demo -v``
"""

import tempfile
import unittest
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from app.app import (
    PERFILES_EJEMPLO,
    cargar_modelo,
    construir_fila_entrada,
    explicar_prediccion_shap,
    generar_figura_intervalo,
    generar_figura_waterfall,
    predecir_con_intervalo,
)
from src.config import RAIZ
from src.evaluacion import FACTOR_KAGGLE, cargar_entrenamiento
from src.modelos import construir_pipeline
from src.preprocesamiento import COLUMNAS_PREDICTORAS


class TestDemoT26(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Si no existe models/modelo_final.joblib, carga provisionalmente P2 + M1
        cls.recurso = cargar_modelo(ruta_modelo=RAIZ / "models" / "__inexistente__.joblib")

    def test_carga_provisional_aislada_del_test(self):
        self.assertTrue(self.recurso["es_provisional"])
        self.assertTrue(self.recurso["es_log"])
        self.assertEqual(self.recurso["evidencia"]["n_entrenamiento_cargado"], 436)
        self.assertEqual(self.recurso["evidencia"]["n_prueba_cargado"], 0)
        self.assertEqual(self.recurso["tipo_explainer"], "LinearExplainer")

    def test_carga_desde_joblib_para_transicion_t25(self):
        X_ent, y_ent, _ = cargar_entrenamiento()
        est_ridge = construir_pipeline("P2", "M2").fit(X_ent, y_ent)

        with tempfile.TemporaryDirectory() as tmpdir:
            ruta_est = Path(tmpdir) / "modelo_final.joblib"
            joblib.dump(est_ridge, ruta_est)
            rec_est = cargar_modelo(ruta_modelo=ruta_est)
            self.assertFalse(rec_est["es_provisional"])
            self.assertEqual(rec_est["info_intervalo"]["tipo"], "conforme_cv")

            df_in = construir_fila_entrada(PERFILES_EJEMPLO["Vivienda mediana (típica)"])
            pred = predecir_con_intervalo(rec_est, df_in, nivel_confianza=0.90)
            self.assertLess(pred["inferior_kaggle"], pred["pred_kaggle"])
            self.assertLess(pred["pred_kaggle"], pred["superior_kaggle"])

            ruta_dict = Path(tmpdir) / "modelo_dict.joblib"
            joblib.dump({"modelo": est_ridge}, ruta_dict)
            rec_dict = cargar_modelo(ruta_modelo=ruta_dict)
            self.assertFalse(rec_dict["es_provisional"])

    def test_perfiles_prediccion_e_intervalo_ols(self):
        for nombre, perfil in PERFILES_EJEMPLO.items():
            df_in = construir_fila_entrada(perfil)
            self.assertEqual(list(df_in.columns), COLUMNAS_PREDICTORAS)
            self.assertEqual(len(df_in), 1)

            ic80 = predecir_con_intervalo(self.recurso, df_in, nivel_confianza=0.80)
            ic95 = predecir_con_intervalo(self.recurso, df_in, nivel_confianza=0.95)

            self.assertGreater(ic95["inferior_kaggle"], 0.0, nombre)
            self.assertLess(ic95["inferior_kaggle"], ic95["pred_kaggle"], nombre)
            self.assertLess(ic95["pred_kaggle"], ic95["superior_kaggle"], nombre)
            self.assertAlmostEqual(ic95["pred_kaggle"] / FACTOR_KAGGLE, ic95["pred_cad"], places=6)

            ancho_80 = ic80["superior_kaggle"] - ic80["inferior_kaggle"]
            ancho_95 = ic95["superior_kaggle"] - ic95["inferior_kaggle"]
            self.assertGreater(ancho_95, ancho_80, nombre)

    def test_explicacion_shap_aditiva_y_figuras(self):
        df_in = construir_fila_entrada(PERFILES_EJEMPLO["Vivienda equipada de gama alta"])
        pred = predecir_con_intervalo(self.recurso, df_in, nivel_confianza=0.95)
        shap_res = explicar_prediccion_shap(self.recurso, df_in)

        # Aditividad exacta en escala log(price)
        suma_shap = shap_res["base_escala_modelo"] + float(np.sum(shap_res["explanation"].values))
        self.assertAlmostEqual(suma_shap, np.log(pred["pred_kaggle"]), places=5)
        self.assertEqual(len(shap_res["tabla"]), 13)

        fig_w = generar_figura_waterfall(shap_res["explanation"], shap_res["es_log"])
        fig_i = generar_figura_intervalo(pred, self.recurso["y_ent"])
        self.assertGreater(len(fig_w.axes), 0)
        self.assertGreater(len(fig_i.axes), 0)
        plt.close(fig_w)
        plt.close(fig_i)

    def test_configuracion_sin_telemetria_offline(self):
        cfg = (RAIZ / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn("gatherUsageStats = false", cfg)


if __name__ == "__main__":
    unittest.main()
