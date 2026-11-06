# Búsqueda sistemática: datos y scripts (T09)

Esta carpeta contiene los insumos para regenerar el registro de la revisión de literatura.

| Archivo | Contenido |
|---|---|
| `resultados_busquedas.json` | Búsquedas ejecutadas (B01–B20) y resultados registrados: título, URL, posición |
| `decisiones.py` | Decisión de tamizaje por registro único, con sus criterios de exclusión E1–E7 |
| `incluidos.py` | Metadatos verificados y hallazgo preliminar de los estudios incluidos (E01–E26) |
| `generar_registro.py` | Genera `../registro_busquedas.xlsx`. Los conteos PRISMA quedan calculados con fórmulas |
| `prisma.py` | Genera `../../diagramas/prisma_2020.png` a partir de la hoja PRISMA |
| `generar_protocolo.py` | Genera `../02_protocolo_busqueda.md` |

Regenerar, desde la raíz del repositorio:

```bash
python docs/entregable_1/busqueda/generar_registro.py
# Abrir y guardar el Excel una vez (o recalcular con LibreOffice) para calcular las fórmulas
python docs/entregable_1/busqueda/prisma.py
python docs/entregable_1/busqueda/generar_protocolo.py
```
