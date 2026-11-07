# Datos crudos (solo lectura)

Los archivos de esta carpeta son **copias exactas de la fuente** y no se modifican. Cualquier limpieza o transformación se hace por código y se guarda en `data/processed/` (tarjeta T06).

| Archivo | Fuente | Filas × columnas | SHA-256 |
|---|---|---|---|
| `Housing.csv` | Kaggle: *Housing Prices Dataset* (Yasser H.), https://www.kaggle.com/datasets/yasserh/housing-prices-dataset | 545 × 13 | `a8ef508351b4d6e87b23a831cf316a3d53c794ed9d969c7b697a865a60a3d65f` |
| `HousePrices_Windsor_1987_AER.csv` | Rdatasets / paquete R `AER` (Journal of Applied Econometrics Data Archive, Anglin & Gençay 1996), https://vincentarelbundock.github.io/Rdatasets/csv/AER/HousePrices.csv | 546 × 12 (+ rownames) | `4c7f6df4c6264cf6f57ea6ec5157958c4033efe1f51b2855efb18e297274bfe3` |

Verificar la integridad de los archivos:

```bash
# macOS/Linux
sha256sum data/raw/Housing.csv
sha256sum data/raw/HousePrices_Windsor_1987_AER.csv

# Windows (PowerShell)
Get-FileHash data\raw\Housing.csv -Algorithm SHA256
Get-FileHash data\raw\HousePrices_Windsor_1987_AER.csv -Algorithm SHA256
```

La ficha técnica completa (procedencia, genealogía con el dataset original, licencia, diccionario de datos y control de calidad inicial) se documenta en la tarjeta **T05** → `data/README.md`.
