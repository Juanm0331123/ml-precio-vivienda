# Diapositiva 7 · Resumen de literatura para Canva (T10)

**Tarjeta:** T10 · **Prepara:** Brandon Eduardo Tobar Sánchez · **Presenta:** Juan Miguel (3:50–4:30, 40 s)
**Uso:** texto y tabla para la diapositiva 7 del storyboard (T13) y, si hace falta, una segunda diapositiva de apoyo. Todo sale de `matriz_literatura.xlsx`; el nivel de verificación de cada estudio está en la tabla.

## Título (≤ 50 caracteres)

**En Windsor, el hedónico compite con los ensambles**

## Mensajes (3)

1. **Los ensambles suelen ganar:** Random Forest supera a la regresión en Liubliana [10]; XGBoost es el más exacto en 1,2 millones de viviendas en Alemania [20]; una revisión cuenta 57 comparaciones favorables al ML frente a 13 a la regresión [12].
2. **Pero la ventaja no es uniforme:** en Alemania, GAM y OLS quedan cerca (R² 0,866 y 0,865 frente a 0,900) [20]; en Colombia, Ridge y Random Forest quedan casi empatados [25]; sobre los datos de referencia, una especificación paramétrica bien elegida no se rechaza [4].
3. **Con datos tabulares de tamaño medio, los árboles ganan a las redes profundas** [26]. Por eso comparamos el hedónico, modelos regularizados y ensambles.

## Tabla de estudios clave (6–8)

| Ref. | Estudio | Datos | Hallazgo | Verificación |
|---|---|---|---|---|
| [4] | Haupt et al. (2010) | Datos canadienses (T09: Windsor) | Una especificación paramétrica bien elegida no se rechaza | Resumen |
| [6] | Antipov y Pokryshevskaya (2012) | Apartamentos | Random Forest supera a los demás métodos comparados | Resumen |
| [10] | Čeh et al. (2018) | 7.407 apartamentos, Liubliana | Random Forest supera a la regresión múltiple | Resumen |
| [12] | Valier (2020) | Revisión sistemática | ML gana 57 comparaciones; la regresión, 13 | Resumen |
| [14] | Ho et al. (2021) | ≈ 40.000 transacciones, Hong Kong | RF y GBM superan a SVM | Resumen |
| [20] | Stang et al. (2022) | 1.212.546 viviendas, Alemania | XGBoost R² 0,8995; GAM 0,8664; OLS 0,8654 | **Texto completo** |
| [25] | Rosso-Mateus et al. (2022) | 2.397 anuncios, Colombia | Ridge y Random Forest, MAPE ≈ 0,34–0,35 | Resumen |
| [26] | Grinsztajn et al. (2022) | 45 datasets tabulares | Árboles superan a redes profundas | **Texto completo** |

## Nota de rigor (va en el pie o en las notas del orador)

- De 26 estudios incluidos: 2 se leyeron completos, 21 solo en resumen y 3 solo en metadatos. Las cifras mostradas son las leídas; "sin cifras" significa que el resumen no las da.
- La búsqueda usó un motor web con filtro por dominio, no Scopus ni Web of Science; se replicará en el artículo final.
- No se muestran [13], [15] ni [19]: no se pudieron verificar.

## Pregunta probable y respuesta corta

**¿Por qué incluir el modelo hedónico si el ML gana?** Porque en la literatura la ventaja no es uniforme [20], [25], [4] y porque el hedónico aporta interpretabilidad [12]. Lo contrastamos con pruebas estadísticas en vez de suponer el resultado.
