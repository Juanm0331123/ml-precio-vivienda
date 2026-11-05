# Ficha Técnica: Housing Prices Dataset (Kaggle) & Windsor Housing Dataset (1987)

## 1. Procedencia de los Datos y Genealogía

### Dataset de trabajo del proyecto
* **Archivo:** `data/raw/Housing.csv`
* **Fuente:** Kaggle — *Housing Prices Dataset* publicado por M. Yasser H. ([Enlace en Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)).
* **Dimensiones:** 545 observaciones × 13 variables.

### Origen histórico y estudio base
* **Contexto:** Transacciones inmobiliarias en la ciudad de Windsor, Ontario, Canadá, correspondientes a los meses de julio, agosto y septiembre de 1987.
* **Publicación académica original:**
  > Anglin, P., y Gençay, R. (1996). *Semiparametric Estimation of a Hedonic Price Function*. Journal of Applied Econometrics, 11(6), 633–648.
* **Archivo de datos original:** Journal of Applied Econometrics (JAE) Data Archive ([http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/](http://qed.econ.queensu.ca/jae/1996-v11.6/anglin-gencay/)).
* **Distribución en paquetes econométricos:** Paquete `AER` (*Applied Econometrics with R*) de Christian Kleiber y Achim Zeileis, distribuido en la colección Rdatasets de Vincent Arel-Bundock como `HousePrices.csv` (almacenado localmente en `data/raw/HousePrices_Windsor_1987_AER.csv` para auditoría y verificación).

### Relación demostrada entre Kaggle y el dataset original de Windsor
A través de contrastación algorítmica (verificada mediante multiconjuntos en el notebook `notebooks/01_procedencia_y_calidad.ipynb`), se demostraron las siguientes correspondencias exactas:
1. **Factor de escala en el precio (× 70):** En las 545 observaciones, el precio reportado en Kaggle equivale con exactitud al precio original en dólares canadienses (CAD) multiplicado por 70 ($\text{price}_{\text{Kaggle}} = \text{price}_{\text{Windsor}} \times 70$). Todos los precios de Kaggle son múltiplos exactos de 70.
2. **Depuración del registro duplicado (545 vs. 546 filas):** El dataset original de Windsor cuenta con 546 filas, pero presenta un registro completamente idéntico duplicado en dos posiciones (filas 318 y 535 en índice 1-based, correspondientes a las observaciones indexadas como 317 y 534 en Python):
   - Registro duplicado: $\text{price} = \$82{,}000$, $\text{lotsize} = 5{,}000$, $\text{bedrooms} = 3$, $\text{bathrooms} = 1$, $\text{stories} = 3$, $\text{driveway} = \text{yes}$, $\text{recreation} = \text{no}$, $\text{fullbase} = \text{no}$, $\text{gasheat} = \text{no}$, $\text{aircon} = \text{yes}$, $\text{garage} = 0$, $\text{prefer} = \text{no}$.
   La versión de Kaggle contiene 545 filas porque descartó la fila duplicada redundante.
3. **Renombramiento de columnas:** Los atributos fueron adaptados a una nomenclatura más intuitiva en Kaggle:
   - `lotsize` $\rightarrow$ `area`
   - `driveway` $\rightarrow$ `mainroad`
   - `recreation` $\rightarrow$ `guestroom`
   - `fullbase` $\rightarrow$ `basement`
   - `gasheat` $\rightarrow$ `hotwaterheating`
   - `aircon` $\rightarrow$ `airconditioning`
   - `garage` $\rightarrow$ `parking`
   - `prefer` $\rightarrow$ `prefarea`
   - `price`, `bedrooms`, `bathrooms` y `stories` conservaron su denominación original.
4. **Variable añadida sintéticamente:** La columna `furnishingstatus` (`furnished`, `semi-furnished`, `unfurnished`) no forma parte del dataset original de Windsor ni de la investigación econométrica de Anglin & Gençay (1996). Fue agregada en la versión de Kaggle sin documentación metodológica de respaldo.

---

## 2. Licencia y Marco Legal

Es fundamental distinguir jurídicamente entre una licencia de código abierto copyleft (GPL) y una renuncia al dominio público (CC0):

1. **Dataset histórico original (Windsor 1987 / AER / JAE Archive):**
   * **Licencia:** GNU General Public License (**GPL-2 | GPL-3**) vía paquete `AER` de R.
   * **Condiciones:** Permite uso educativo, analítico y redistribución con preservación de la licencia libre y atribución obligatoria al trabajo original:
     > Anglin, P., & Gençay, R. (1996). *Semiparametric Estimation of a Hedonic Price Function*. Journal of Applied Econometrics, 11(6), 633–648.
2. **Dataset de trabajo (Kaggle — Yasser H.):**
   * **Licencia:** **CC0: Public Domain** (Creative Commons Zero 1.0 Universal).
   * **Condiciones:** Renuncia a derechos patrimoniales de autor en todo el mundo, permitiendo su uso, adaptación y análisis sin restricciones comerciales o de copia.

---

## 3. Diccionario de Datos

Diccionario formal para el dataset de trabajo `data/raw/Housing.csv` (545 filas × 13 columnas), documentando su correspondencia con el archivo original y el significado real de cada variable:

| Variable en Kaggle | Nombre original | Descripción real | Tipo | Unidad o niveles | Rango | Rol |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `price` | `price` | Precio final de venta del inmueble (escalado en Kaggle como $\text{precio original} \times 70$) | Numérica continua | Moneda (unidades monetarias escaladas) | [1,750,000 – 13,300,000] | Objetivo (Target) |
| `area` | `lotsize` | Tamaño total del lote o terreno de la propiedad | Numérica continua | Pies cuadrados ($\text{ft}^2$) | [1,650 – 16,200] | Predictora (Feature) |
| `bedrooms` | `bedrooms` | Número de dormitorios o habitaciones en la vivienda | Numérica discreta | Entero (cuartos) | [1 – 6] | Predictora (Feature) |
| `bathrooms` | `bathrooms` | Número de cuartos de baño completos | Numérica discreta | Entero (baños) | [1 – 4] | Predictora (Feature) |
| `stories` | `stories` | Número de pisos o niveles habitables sobre el suelo (excluye el sótano) | Numérica discreta | Entero (niveles) | [1 – 4] | Predictora (Feature) |
| `mainroad` | `driveway` | Presencia de entrada para vehículos hacia la propiedad (*driveway*) | Categórica binaria | `yes`, `no` | {'no', 'yes'} | Predictora (Feature) |
| `guestroom` | `recreation` | Presencia de salón o sala de recreación y entretenimiento (*recreation room*) | Categórica binaria | `yes`, `no` | {'no', 'yes'} | Predictora (Feature) |
| `basement` | `fullbase` | Presencia de sótano completo terminado y acondicionado (*full finished basement*) | Categórica binaria | `yes`, `no` | {'no', 'yes'} | Predictora (Feature) |
| `hotwaterheating` | `gasheat` | Uso de gas para el sistema de calentamiento de agua caliente (*gas water heating*) | Categórica binaria | `yes`, `no` | {'no', 'yes'} | Predictora (Feature) |
| `airconditioning` | `aircon` | Presencia de sistema de aire acondicionado central | Categórica binaria | `yes`, `no` | {'no', 'yes'} | Predictora (Feature) |
| `parking` | `garage` | Número de plazas de estacionamiento disponibles en el garaje | Numérica discreta | Entero (plazas) | [0 – 3] | Predictora (Feature) |
| `prefarea` | `prefer` | Localización en un barrio preferencial o de alta cotización de Windsor | Categórica binaria | `yes`, `no` | {'no', 'yes'} | Predictora (Feature) |
| `furnishingstatus` | *(Ninguno)* | Grado de amueblamiento del inmueble (columna sintética añadida en Kaggle; sin registro en el estudio original) | Categórica politómica | `furnished`, `semi-furnished`, `unfurnished` | {'furnished', 'semi-furnished', 'unfurnished'} | Predictora (Feature) |

---

## 4. Calidad e Integridad de los Datos

* **Completitud y ausencia de nulos:** El archivo `data/raw/Housing.csv` cuenta con completitud del 100 % (0 valores nulos en sus 545 registros y 13 variables).
* **Consistencia de duplicados:** Dentro de `Housing.csv` no existen filas duplicadas completas (0 duplicados en las 545 observaciones). El archivo histórico original de Windsor sí contenía un registro duplicado (filas 318 y 535), el cual fue depurado en la compilación de Kaggle.
* **Alcance de este informe:** Las pruebas exhaustivas de distribución, detección de outliers, correlaciones bivariadas y partición analítica train/test corresponden al flujo de calidad y preparación desarrollado en la tarjeta **T06** (`notebooks/01_procedencia_y_calidad.ipynb` / `data/processed/`).

---

## 5. Integridad Criptográfica (Checksums SHA-256)

Para garantizar la inmutabilidad y la reproducibilidad de los análisis en cualquier entorno, se registran las huellas digitales SHA-256 (en minúsculas y sin caracteres adicionales):

* **Archivo de trabajo (`Housing.csv`):**
  `a8ef508351b4d6e87b23a831cf316a3d53c794ed9d969c7b697a865a60a3d65f`
* **Archivo de referencia histórico (`HousePrices_Windsor_1987_AER.csv`):**
  `4c7f6df4c6264cf6f57ea6ec5157958c4033efe1f51b2855efb18e297274bfe3`
