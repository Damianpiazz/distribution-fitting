import numpy as np

from src.distribuciones import (
    DISTRIBUCIONES_POR_DEFECTO,
    REQUIERE_POSITIVOS,
    cargar_distribuciones,
    filtrar_para_datos,
)


def test_default_lista_no_vacia():
    assert len(DISTRIBUCIONES_POR_DEFECTO) > 0
    assert all(isinstance(d, str) for d in DISTRIBUCIONES_POR_DEFECTO)


def test_default_tiene_distribuciones_clasicas():
    for d in ("norm", "expon", "gamma", "lognorm", "weibull_min", "uniform"):
        assert d in DISTRIBUCIONES_POR_DEFECTO


def test_cargar_distribuciones_devuelve_lista():
    dists = cargar_distribuciones()
    assert isinstance(dists, list)
    assert len(dists) > 0
    assert all(isinstance(d, str) for d in dists)


def test_filtrar_con_positivos():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dists = ["norm", "gamma", "lognorm"]
    result = filtrar_para_datos(data, dists)
    assert len(result) == 3


def test_filtrar_con_negativos_excluye_positivos():
    data = np.array([-1.0, 0.0, 3.0])
    dists = ["norm", "gamma", "lognorm", "pareto"]
    result = filtrar_para_datos(data, dists)
    assert "norm" in result
    assert "gamma" not in result
    assert "lognorm" not in result
    assert "pareto" not in result


def test_requiere_positivos_tiene_todas_las_necesarias():
    for d in REQUIERE_POSITIVOS:
        assert d in DISTRIBUCIONES_POR_DEFECTO
