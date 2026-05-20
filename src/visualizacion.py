import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def graficar(data, dist_name, params):
    dist = getattr(st, dist_name)

    bins = np.histogram_bin_edges(data, bins="fd")

    plt.hist(data, bins=bins, density=True, alpha=0.5)

    x = np.linspace(min(data), max(data), 200)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    y = dist.pdf(x, *arg, loc=loc, scale=scale)

    plt.plot(x, y)

    plt.title(f"Ajuste: {dist_name}")
    plt.xlabel("x")
    plt.ylabel("Densidad")

    plt.show()