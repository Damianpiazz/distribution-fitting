import scipy.stats as st
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.distribuciones import cargar_distribuciones, filtrar_para_datos


def ajustar_y_puntuar(data, nombre_dist, puntuador):
    dist = getattr(st, nombre_dist)
    params = dist.fit(data)
    score = puntuador(data, nombre_dist, params)
    return nombre_dist, score, params


def evaluar_todas(data, distribuciones, puntuador, mejor_es_menor=True):
    resultados = []
    n = len(distribuciones)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"[bold yellow]Probando {n} distribuciones...", total=n
        )
        for d in distribuciones:
            progress.update(
                task, description=f"[bold yellow]Probando {d}..."
            )
            try:
                resultado = ajustar_y_puntuar(data, d, puntuador)
                resultados.append(resultado)
            except Exception:
                pass
            finally:
                progress.update(task, advance=1)

    resultados.sort(key=lambda x: x[1], reverse=not mejor_es_menor)
    return resultados


def encontrar_mejor(data, metodo, distribuciones=None):
    from src.pruebas import ks, chi, anderson as ad

    puntuadores = {
        "ks": (ks.puntuar, False),
        "chi": (chi.puntuar, True),
        "anderson": (ad.puntuar, True),
    }

    if metodo not in puntuadores:
        raise ValueError(f"Método inválido: {metodo}")

    puntuador, mejor_es_menor = puntuadores[metodo]

    if distribuciones is None:
        distribuciones = cargar_distribuciones()

    distribuciones = filtrar_para_datos(data, distribuciones)

    if not distribuciones:
        raise RuntimeError(
            "Ninguna distribución es compatible con los datos"
        )

    resultados = evaluar_todas(
        data, distribuciones, puntuador, mejor_es_menor
    )

    if not resultados:
        raise RuntimeError(
            "Ninguna distribución pudo ajustarse a los datos"
        )

    return resultados
