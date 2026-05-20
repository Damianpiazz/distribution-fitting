import numpy as np
import scipy.stats as st

from src.pruebas.ks import puntuar


def test_ks_devuelve_p_value_entre_0_y_1():
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)
    params = st.norm.fit(data)
    p_value = puntuar(data, "norm", params)
    assert 0.0 <= p_value <= 1.0


def test_ks_distribucion_correcta_da_p_alto():
    np.random.seed(42)
    data = np.random.exponential(1, 200)
    params = st.expon.fit(data)
    p_value = puntuar(data, "expon", params)
    assert p_value > 0.01


def test_ks_coincide_con_scipy():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    params = (0.0, 1.0)
    score = puntuar(data, "norm", params)
    _, ref = st.kstest(data, "norm", args=params)
    assert abs(score - ref) < 1e-10
