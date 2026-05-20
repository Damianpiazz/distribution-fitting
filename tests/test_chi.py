import numpy as np
import scipy.stats as st

from src.pruebas.chi import puntuar


def test_chi_devuelve_float_positivo():
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)
    params = st.norm.fit(data)
    chi2 = puntuar(data, "norm", params)
    assert chi2 >= 0.0
    assert isinstance(chi2, float)


def test_chi_distribucion_correcta_da_chi2_bajo():
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    params = st.norm.fit(data)

    chi2_norm = puntuar(data, "norm", params)

    data_exp = np.random.exponential(1, 500)
    params_exp = st.expon.fit(data_exp)
    chi2_exp = puntuar(data_exp, "expon", params_exp)

    assert chi2_norm < 50
    assert chi2_exp < 50


def test_chi_bins_por_defecto_usa_sturges():
    data = np.arange(1.0, 101.0)
    params = (0.0, 1.0)
    chi2 = puntuar(data, "norm", params)
    assert isinstance(chi2, float)
