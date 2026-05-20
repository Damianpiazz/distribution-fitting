import scipy.stats as st


def puntuar(data, nombre_dist, params):
    _, p_value = st.kstest(data, nombre_dist, args=params)
    return float(p_value)
