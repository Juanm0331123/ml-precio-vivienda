"""Configuración compartida del proyecto: semilla y rutas.

Todos los notebooks y módulos importan desde aquí para que las rutas sean
relativas a la raíz del repositorio y los resultados sean reproducibles en
cualquier equipo, incluido Google Colab.

Ejemplo de uso en un notebook::

    from src.config import SEMILLA, RUTA_DATOS_CRUDOS
    df = pd.read_csv(RUTA_DATOS_CRUDOS)
"""

from pathlib import Path

# Semilla global (la misma del trabajo de Inferencia Estadística).
SEMILLA = 5007

# Raíz del repositorio: la carpeta que contiene a src/.
RAIZ = Path(__file__).resolve().parents[1]

# Carpetas del proyecto.
DIR_DATOS_CRUDOS = RAIZ / "data" / "raw"
DIR_DATOS_PROCESADOS = RAIZ / "data" / "processed"
DIR_FIGURAS = RAIZ / "reports" / "figures"
DIR_TABLAS = RAIZ / "reports" / "tables"
DIR_MODELOS = RAIZ / "models"

# Archivo de datos de trabajo (Kaggle, solo lectura).
RUTA_DATOS_CRUDOS = DIR_DATOS_CRUDOS / "Housing.csv"

# Huella SHA-256 del archivo de trabajo, para verificar su integridad.
SHA256_DATOS_CRUDOS = "a8ef508351b4d6e87b23a831cf316a3d53c794ed9d969c7b697a865a60a3d65f"

# Archivo de referencia histórico (Windsor 1987 / AER, solo lectura) para validación de procedencia.
RUTA_DATOS_ORIGINAL_AER = DIR_DATOS_CRUDOS / "HousePrices_Windsor_1987_AER.csv"
SHA256_DATOS_ORIGINAL_AER = "4c7f6df4c6264cf6f57ea6ec5157958c4033efe1f51b2855efb18e297274bfe3"
