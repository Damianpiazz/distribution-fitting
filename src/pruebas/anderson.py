import numpy as np
import scipy.stats as st


def puntuar(data, nombre_dist, params):
    dist = getattr(st, nombre_dist)
    data_sorted = np.sort(data)
    n = len(data_sorted)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    cdf = dist.cdf(data_sorted, *arg, loc=loc, scale=scale)
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)

    i = np.arange(1, n + 1)
    a2 = -n - np.mean(
        (2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1]))
    )

    return float(a2)
