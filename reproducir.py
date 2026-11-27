"""Reproduce el experimento completo con un solo comando (T29).

Desde la raíz del repositorio:

    python reproducir.py              # pruebas + notebooks en orden + figuras del paper
    python reproducir.py --sin-pruebas
    python reproducir.py --solo 02    # solo los notebooks cuyo nombre empieza por 02

Los notebooks se ejecutan en orden y se guardan en su mismo archivo, con las salidas nuevas.
Cada uno vuelve a escribir sus figuras en reports/figures y sus tablas en reports/tables.
Si algo falla, el script se detiene ahí y devuelve un código distinto de cero.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
NOTEBOOKS = RAIZ / "notebooks"
# La plantilla no hace parte del experimento
EXCLUIDOS = {"00_plantilla_notebook.ipynb"}
# Tiempo máximo por celda, en segundos. La CV anidada es lo más lento
TIEMPO_CELDA = 3600


def correr(comando, titulo):
    """Corre un paso, muestra cuánto tardó y corta todo si falla."""
    print(f"\n=== {titulo} ===", flush=True)
    inicio = time.perf_counter()
    resultado = subprocess.run(comando, cwd=RAIZ)
    print(f"--- {titulo}: {time.perf_counter() - inicio:.1f} s", flush=True)
    if resultado.returncode != 0:
        sys.exit(f"Falló: {titulo}")


def notebooks_en_orden(prefijo=None):
    """Notebooks numerados (01_, 02_, ...) ordenados por nombre, que es el orden de ejecución."""
    lista = sorted(p for p in NOTEBOOKS.glob("[0-9][0-9]_*.ipynb") if p.name not in EXCLUIDOS)
    if prefijo:
        lista = [p for p in lista if p.name.startswith(prefijo)]
    return lista


def main():
    parser = argparse.ArgumentParser(description="Reproduce el experimento de principio a fin.")
    parser.add_argument("--sin-pruebas", action="store_true", help="no corre pytest")
    parser.add_argument("--solo", metavar="PREFIJO", help="solo los notebooks que empiezan por este prefijo")
    args = parser.parse_args()

    if not args.sin_pruebas:
        correr([sys.executable, "-m", "pytest", "-q"], "Pruebas")

    for nb in notebooks_en_orden(args.solo):
        correr(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             f"--ExecutePreprocessor.timeout={TIEMPO_CELDA}", str(nb)],
            f"Notebook {nb.name}",
        )

    figuras_paper = RAIZ / "docs" / "paper" / "figures" / "make_figures.py"
    if figuras_paper.exists() and not args.solo:
        correr([sys.executable, str(figuras_paper)], "Figuras del paper")

    print("\nListo: experimento reproducido sin errores.")


if __name__ == "__main__":
    main()
