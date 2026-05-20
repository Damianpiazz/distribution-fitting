import os

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.prompt import IntPrompt
from rich.panel import Panel
from rich import box

from src.pruebas.selector import ajustar_distribucion
from src.visualizacion import graficar
from src.estadisticas import calcular_estadisticas

console = Console()
RUTA_DATOS = "datos"


def limpiar_params(params):
    return tuple(float(p) for p in params)


def formatear_params(params):
    return ", ".join(f"{float(p):.4f}" for p in params)


def listar_datasets():
    archivos = [f for f in os.listdir(RUTA_DATOS) if f.endswith(".csv")]

    if len(archivos) == 0:
        raise Exception("No hay datasets en /datos")

    table = Table(box=box.SIMPLE)
    table.add_column("ID", justify="center")
    table.add_column("Dataset", style="cyan")

    for i, f in enumerate(archivos):
        table.add_row(str(i), f)

    console.print(table)
    return archivos


def seleccionar_dataset(archivos):
    while True:
        try:
            idx = IntPrompt.ask("Dataset ID")

            if 0 <= idx < len(archivos):
                console.print(f"[green]✔ {archivos[idx]}[/green]")
                return os.path.join(RUTA_DATOS, archivos[idx])

            console.print("[red]Indice fuera de rango[/red]")

        except Exception:
            console.print("[red]Entrada invalida[/red]")


def seleccionar_columna(df):
    table = Table(box=box.MINIMAL)
    table.add_column("ID")
    table.add_column("Columna")
    table.add_column("% numérico")

    validas = []

    for i, col in enumerate(df.columns):
        serie = pd.to_numeric(df[col], errors="coerce")
        porcentaje = serie.notna().mean() * 100

        table.add_row(str(i), col, f"{porcentaje:.1f}%")

        if porcentaje > 70:
            validas.append(i)

    console.print(table)

    if len(validas) == 0:
        raise Exception("No hay columnas numericas suficientes")

    while True:
        try:
            idx = IntPrompt.ask("Columna ID")

            if idx in validas:
                col = df.columns[idx]
                data = pd.to_numeric(df[col], errors="coerce").dropna().values

                if len(data) < 10:
                    console.print("[red]Muy pocos datos[/red]")
                    continue

                if np.std(data) < 1e-8:
                    console.print("[red]Datos casi constantes[/red]")
                    continue

                console.print(f"[green]✔ {col} ({len(data)} datos)[/green]")
                return data

            console.print("[red]Columna no valida[/red]")

        except Exception:
            console.print("[red]Entrada invalida[/red]")


def elegir_metodo():
    console.print("\nMetodo de ajuste:")
    console.print("1 - KS")
    console.print("2 - Chi-cuadrado")
    console.print("3 - Anderson-Darling")

    while True:
        try:
            op = IntPrompt.ask("Metodo ID")

            if op == 1:
                return "ks"
            if op == 2:
                return "chi"
            if op == 3:
                return "anderson"

            console.print("[red]Opcion invalida[/red]")

        except Exception:
            console.print("[red]Entrada invalida[/red]")


def mostrar_resultado(dist, score, params, metodo):
    etiqueta = {
        "ks": "p-value",
        "chi": "chi2",
        "anderson": "A2",
    }[metodo]

    params = limpiar_params(params)

    console.print(
        Panel(
            f"[cyan]{dist}[/cyan]\n"
            f"{etiqueta}: {score:.5f}\n"
            f"params: {formatear_params(params)}",
            border_style="green",
        )
    )


def mostrar_ranking(resultados, metodo, top_n=5):
    etiqueta = {
        "ks": "p-value",
        "chi": "chi2",
        "anderson": "A2",
    }[metodo]

    table = Table(
        title=f"Top {min(top_n, len(resultados))} distribuciones",
        box=box.SIMPLE,
    )
    table.add_column("Distribución", style="cyan")
    table.add_column(etiqueta, justify="right")
    table.add_column("Parámetros")

    for dist, score, params in resultados[:top_n]:
        table.add_row(
            dist,
            f"{score:.5f}",
            formatear_params(limpiar_params(params)),
        )

    console.print(table)


def mostrar_estadisticas(stats):
    table = Table(title="Estadísticas", box=box.SIMPLE)
    table.add_column("Métrica")
    table.add_column("Datos", justify="right")
    table.add_column("Distribución", justify="right")

    def fmt(x):
        return f"{x:.4f}" if x is not None else "-"

    table.add_row("Media", fmt(stats["media_data"]), fmt(stats["media_dist"]))
    table.add_row("Varianza", fmt(stats["var_data"]), fmt(stats["var_dist"]))
    table.add_row("Desvío Std", fmt(stats["std_data"]), fmt(stats["std_dist"]))

    console.print(table)


def main():
    archivos = listar_datasets()
    ruta = seleccionar_dataset(archivos)

    df = pd.read_csv(ruta)
    data = seleccionar_columna(df)

    metodo = elegir_metodo()

    resultados = ajustar_distribucion(data, metodo=metodo)

    if not resultados:
        console.print("[red]No se pudo ajustar ninguna distribución[/red]")
        return

    mejor = resultados[0]
    mostrar_resultado(mejor[0], mejor[1], mejor[2], metodo)
    mostrar_ranking(resultados, metodo)

    stats = calcular_estadisticas(data, mejor[0], mejor[2])
    mostrar_estadisticas(stats)

    graficar(data, mejor[0], mejor[2])


if __name__ == "__main__":
    main()
