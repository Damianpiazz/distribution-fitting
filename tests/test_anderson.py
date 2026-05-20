import numpy as np
import scipy.stats as st

from src.pruebas.anderson import puntuar


def test_anderson_devuelve_float():
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)
    params = st.norm.fit(data)
    a2 = puntuar(data, "norm", params)
    assert isinstance(a2, float)
    assert not np.isnan(a2)


def test_anderson_distribucion_correcta_da_a2_bajo():
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)
    params = st.norm.fit(data)
    a2 = puntuar(data, "norm", params)
    assert a2 < 5.0
