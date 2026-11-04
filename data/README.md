# Ficha Técnica: House Prices in the City of Windsor, Canada

## 1. Procedencia de los Datos (Source)
* **Contexto:** Precios de venta de casas en la ciudad de Windsor, Canadá, durante julio, agosto y septiembre de 1987.
* **Fuente Original:** Journal of Applied Econometrics Data Archive (http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/).
* **Publicación Académica:** Anglin, P., y Gencay, R. (1996). *Semiparametric Estimation of a Hedonic Price Function*. Journal of Applied Econometrics, 11, 633–648.
* **Repositorio Actual:** Extraído del paquete `AER` (Applied Econometrics with R) y alojado en la colección Rdatasets de Vincent Arel-Bundock.

## 2. Licencia
* **Tipo de Licencia:** Uso académico y de código abierto (Open Data). Al provenir del paquete `AER` de R, se distribuye bajo Licencia Pública General de GNU (GPL-2 / GPL-3).
* **Condiciones de uso:** Dominio público para fines educativos, de investigación y experimentación estadística. Se recomienda citar el artículo original de Anglin & Gencay (1996) si se utiliza para publicaciones.

## 3. Diccionario de Datos
El dataset consta de **546 observaciones** y **12 variables** estructurales y características del inmueble (el archivo CSV incluye una primera columna adicional sin nombre que sirve como índice de la fila).

| Variable | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `price` | Numérico (Continuo) | Precio final de venta de la casa. |
| `lotsize` | Numérico (Continuo) | Tamaño del lote de la propiedad en pies cuadrados (sq ft). |
| `bedrooms` | Numérico (Discreto) | Número de habitaciones. |
| `bathrooms` | Numérico (Discreto) | Número de baños completos. |
| `stories` | Numérico (Discreto) | Número de pisos (excluyendo el sótano). |
| `driveway` | Categórico (Factor) | ¿La casa tiene entrada para autos? (`yes` / `no`). |
| `recreation`| Categórico (Factor) | ¿La casa tiene un cuarto de juegos o recreación? (`yes` / `no`). |
| `fullbase` | Categórico (Factor) | ¿La casa tiene un sótano completamente terminado? (`yes` / `no`). |
| `gasheat` | Categórico (Factor) | ¿La casa utiliza sistema de gas para calentar el agua? (`yes` / `no`). |
| `aircon` | Categórico (Factor) | ¿La casa cuenta con aire acondicionado central? (`yes` / `no`). |
| `garage` | Numérico (Discreto) | Número de espacios en el garaje. |
| `prefer` | Categórico (Factor) | ¿La casa está ubicada en un vecindario preferencial de la ciudad? (`yes` / `no`). |

## 4. Calidad de los Datos
* **Completitud:** Dataset estadísticamente curado. No contiene valores nulos (`NA`) ni registros faltantes en ninguna de sus 546 observaciones.
* **Consistencia:** Las variables dicotómicas (booleanas) están completamente estandarizadas en minúsculas (`yes` / `no`). Las variables numéricas son consistentes con su escala esperada, sin valores atípicos imposibles (por ejemplo, precios o tamaños negativos).
* **Formato:** Archivo plano CSV (`.csv`) delimitado por comas, con encabezados descriptivos en la primera fila. Fácil de ingerir usando librerías estándar como `pandas` (Python) o `readr` (R).
* **Confiabilidad:**  Los datos han sido validados previamente para la investigación académica revisada por pares y son un estándar para la enseñanza de modelos de regresión econométrica.

## 5. Integridad de los Datos (Checksum)
* **Algoritmo:** SHA-256
* **Hash:** `[A8EF508351B4D6E87B23A831CF316A3D53C794ED9D969C7B697A865A60A3D65F]`