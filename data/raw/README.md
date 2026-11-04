# Datos crudos (solo lectura)

Los archivos de esta carpeta son **copias exactas de la fuente** y no se modifican. Cualquier limpieza o transformación se hace por código y se guarda en `data/processed/` (tarjeta T06).

| Archivo | Fuente | Filas × columnas | SHA-256 |
|---|---|---|---|
| `Housing.csv` | Kaggle: *Housing Prices Dataset* (Yasser H.), https://www.kaggle.com/datasets/yasserh/housing-prices-dataset | 545 × 13 | `a8ef508351b4d6e87b23a831cf316a3d53c794ed9d969c7b697a865a60a3d65f` |

Verificar la integridad del archivo:

```bash
# macOS/Linux
sha256sum data/raw/Housing.csv
# Windows (PowerShell)
Get-FileHash data\raw\Housing.csv -Algorithm SHA256
```

La ficha técnica completa (procedencia, licencia, diccionario de datos y calidad) se documenta en la tarjeta **T05** → `data/README.md`.
