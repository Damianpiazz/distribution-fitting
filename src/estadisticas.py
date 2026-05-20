import numpy as np
import scipy.stats as st


def calcular_estadisticas(data, dist_name, params):
    dist = getattr(st, dist_name)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    stats = {}

    # =========================
    # EMPIRICAS (datos reales)
    # =========================
    stats["media_data"] = float(np.mean(data))
    stats["var_data"] = float(np.var(data))
    stats["std_data"] = float(np.std(data))

    # =========================
    # TEORICAS (distribucion)
    # =========================
    try:
        stats["media_dist"] = float(dist.mean(*arg, loc=loc, scale=scale))
        stats["var_dist"] = float(dist.var(*arg, loc=loc, scale=scale))
        stats["std_dist"] = float(dist.std(*arg, loc=loc, scale=scale))
    except Exception:
        stats["media_dist"] = None
        stats["var_dist"] = None
        stats["std_dist"] = None

    return stats