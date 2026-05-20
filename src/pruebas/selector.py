from src.ajuste import encontrar_mejor


def ajustar_distribucion(data, metodo="ks", distribuciones=None):
    return encontrar_mejor(data, metodo, distribuciones)
