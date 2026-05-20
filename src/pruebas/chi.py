import numpy as np
import scipy.stats as st


def puntuar(data, nombre_dist, params, bins=None):
    n = len(data)
    if bins is None:
        bins = max(5, int(np.ceil(1 + np.log2(n))))

    dist = getattr(st, nombre_dist)
    obs, edges = np.histogram(data, bins=bins)
    obs = obs + 1e-8

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    cdf = dist.cdf(edges, *arg, loc=loc, scale=scale)
    exp = np.diff(cdf) * n + 1e-8

    chi2 = np.sum((obs - exp) ** 2 / exp)
    return float(chi2)
