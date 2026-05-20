import json
import os

import numpy as np

RUTA_CONFIG = "distribuciones.json"

DISTRIBUCIONES_POR_DEFECTO = [
    "norm", "expon", "gamma", "lognorm", "weibull_min", "uniform",
    "beta", "chi2", "gumbel_r", "gumbel_l", "invgamma",
    "laplace", "logistic", "lomax", "maxwell", "nakagami",
    "pareto", "powerlaw", "rayleigh", "t", "triang",
    "wald", "rice", "skewnorm", "weibull_max",
    "gennorm", "genpareto", "exponpow", "exponnorm",
    "fatiguelife", "fisk", "invgauss", "halfnorm",
    "hypsecant",
]

REQUIERE_POSITIVOS = {
    "gamma", "lognorm", "weibull_min", "invgamma", "pareto",
    "powerlaw", "rayleigh", "wald", "chi2", "lomax", "maxwell",
    "nakagami", "rice", "genpareto", "exponpow", "exponnorm",
    "fatiguelife", "fisk", "invgauss",
}


def cargar_distribuciones():
    if os.path.exists(RUTA_CONFIG):
        try:
            with open(RUTA_CONFIG) as f:
                config = json.load(f)
            dists = config.get("distribuciones", DISTRIBUCIONES_POR_DEFECTO)
            if dists:
                return dists
        except (json.JSONDecodeError, OSError):
            pass
    return DISTRIBUCIONES_POR_DEFECTO


def filtrar_para_datos(data, distribuciones):
    if np.any(data <= 0):
        return [d for d in distribuciones if d not in REQUIERE_POSITIVOS]
    return list(distribuciones)
