# distribution-fitting

Herramienta para encontrar la distribucion de probabilidad que mejor se ajusta a un conjunto de datos numericos. Prueba automaticamente decenas de distribuciones candidatas usando tres metodos de bondad de ajuste y entrega un ranking de las mejores.

---

## Teoria

### Estimacion de parametros por Maxima Verosimilitud (MLE)

Dado un conjunto de datos $x_1, x_2, \dots, x_n$ que se asumen independientes e identicamente distribuidos segun una densidad $f(x \mid \theta)$, el estimador de maxima verosimilitud es el valor de $\theta$ que maximiza la funcion de verosimilitud:

$$
\hat{\theta} = \arg\max_{\theta} \prod_{i=1}^{n} f(x_i \mid \theta)
$$

En la practica se maximiza el logaritmo de la verosimilitud por estabilidad numerica. El metodo `scipy.stats.<dist>.fit(data)` implementa MLE para cada distribucion candidata, devolviendo una tupla de parametros en el orden `(arg, loc, scale)`:

- `arg`: parametro/s de forma de la distribucion (puede ser vacio)
- `loc`: parametro de ubicacion (desplaza la distribucion horizontalmente)
- `scale`: parametro de escala (estira/contrae la distribucion)

Por ejemplo, para `gamma.fit(data)` devuelve `(a, loc, scale)` donde $a$ es el parametro de forma, `loc` la ubicacion y `scale` la escala.

### Seleccion de la mejor distribucion

Una vez estimados los parametros de cada candidata, se evalua que tan bien se ajusta a los datos mediante una prueba de bondad de ajuste. La distribucion con la mejor puntuacion es seleccionada.

---

## Pruebas de bondad de ajuste

### Kolmogorov-Smirnov (KS)

$$D_n = \sup_{x} |F_n(x) - F(x)|$$

Donde:
- $F_n(x)$: funcion de distribucion acumulada empirica de los datos
- $F(x)$: funcion de distribucion acumulada teorica de la candidata con parametros estimados
- $D_n$: maxima distancia vertical entre ambas funciones

La prueba devuelve un **p-valor**. Un p-valor alto indica que no hay evidencia suficiente para rechazar que los datos siguen la distribucion candidata.

Criterio de seleccion: **maximo p-valor**.

Ventajas: no requiere binning, funcionamiento no parametrico. Desventaja: menos sensible en las colas de la distribucion.

### Chi-cuadrado de Pearson

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

Donde:
- $k$: cantidad de bins. Se calcula con la regla de Sturges: $k = \lceil 1 + \log_2 n \rceil$, con minimo 5 bins
- $O_i$: frecuencia observada en el bin $i$
- $E_i$: frecuencia esperada en el bin $i$ segun la distribucion candidata: $E_i = n \cdot \big(F(b_i) - F(a_i)\big)$ donde $a_i$ y $b_i$ son los limites del bin
- Se agrega una correccion de $10^{-8}$ a cada $O_i$ y $E_i$ para evitar division por cero

La prueba devuelve el estadistico $\chi^2$. Un valor bajo indica que las frecuencias observadas y esperadas son cercanas.

Criterio de seleccion: **minimo $\chi^2$**.

Grados de libertad: $k - 1 - m$, donde $m$ es la cantidad de parametros estimados (solo informativo).

### Anderson-Darling

$$
A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} (2i - 1) \left[ \ln F(x_{(i)}) + \ln(1 - F(x_{(n-i+1)})) \right]
$$

Donde:
- $n$: cantidad de datos
- $x_{(i)}$: i-esimo dato ordenado de menor a mayor
- $F(x)$: funcion de distribucion acumulada teorica
- Se recortan los valores de $F(x)$ a $[10^{-10}, 1 - 10^{-10}]$ para evitar singularidades logaritmicas

La prueba devuelve el estadistico $A^2$. Un valor bajo indica buen ajuste.

Criterio de seleccion: **minimo $A^2$**.

Ventaja sobre KS: da mayor peso a las colas de la distribucion, detectando diferencias que KS podria pasar por alto.

---

## Distribuciones soportadas

Se prueban automaticamente **34 distribuciones continuas** de `scipy.stats`:

| Distribucion | Descripcion | Requiere datos positivos |
|---|---|---|
| `norm` | Normal (Gaussiana) | No |
| `expon` | Exponencial | No |
| `gamma` | Gamma | Si |
| `lognorm` | Log-normal | Si |
| `weibull_min` | Weibull | Si |
| `uniform` | Uniforme continua | No |
| `beta` | Beta | No |
| `chi2` | Chi-cuadrado | Si |
| `gumbel_r` | Gumbel (derecha) | No |
| `gumbel_l` | Gumbel (izquierda) | No |
| `invgamma` | Gamma inversa | Si |
| `laplace` | Laplace | No |
| `logistic` | Logistica | No |
| `lomax` | Lomax (Pareto II) | Si |
| `maxwell` | Maxwell | Si |
| `nakagami` | Nakagami | Si |
| `pareto` | Pareto | Si |
| `powerlaw` | Power-law | Si |
| `rayleigh` | Rayleigh | Si |
| `t` | t de Student | No |
| `triang` | Triangular | No |
| `wald` | Wald (Gaussiana inversa) | Si |
| `rice` | Rice | Si |
| `skewnorm` | Normal asimetrica | No |
| `weibull_max` | Weibull maximo | No |
| `gennorm` | Normal generalizada | No |
| `genpareto` | Pareto generalizada | Si |
| `exponpow` | Potencia exponencial | Si |
| `exponnorm` | Normal exponencialmente modificada | Si |
| `fatiguelife` | Birnbaum-Saunders | Si |
| `fisk` | Log-logistica | Si |
| `invgauss` | Gaussiana inversa | Si |
| `halfnorm` | Media normal | No |
| `hypsecant` | Secante hiperbolica | No |

### Filtrado inteligente

Si los datos contienen valores $\leq 0$, se saltan automaticamente las distribuciones que requieren datos estrictamente positivos (marcadas como "Si" en la tabla). Esto evita errores de estimacion y acelera el proceso.

---

## Configuracion

La lista de distribuciones a probar se configura en el archivo `distribuciones.json` en la raiz del proyecto:

```json
{
    "distribuciones": [
        "norm", "expon", "gamma", "lognorm", "weibull_min", "uniform",
        ...
    ]
}
```

Para personalizar, simplemente edite la lista: agregue o quite nombres de distribuciones de `scipy.stats`. Si el archivo no existe o tiene errores, se usan los valores por defecto.

---

## Uso

```
python main.py
```

El programa guia interactivamente:

1. **Seleccionar dataset** de la carpeta `datos/`
2. **Seleccionar columna** numerica (valida automaticamente que tenga suficientes datos numericos)
3. **Elegir metodo** de bondad de ajuste:
   - `1` - Kolmogorov-Smirnov
   - `2` - Chi-cuadrado
   - `3` - Anderson-Darling
4. **Resultados**: muestra la mejor distribucion, el top 5, estadisticas comparativas (media, varianza, desvio) y un grafico con el histograma y la PDF ajustada

### Ejemplo de salida

```
+---------------------------+
|      Mejor ajuste         |
| norm                      |
| p-value: 0.84321          |
| params: 5.1234, 1.9876   |
+---------------------------+

Top 5 distribuciones:
  norm       0.84321  (5.1234, 1.9876)
  t          0.82100  (8.4321, 5.1234, 1.9876)
  gennorm    0.81005  (1.8765, 5.1234, 1.9876)
  logistic   0.79234  (5.1234, 1.1234)
  laplace    0.65432  (5.1234, 1.4567)

Estadisticas:
  Media     5.1200  5.1234
  Varianza  3.9500  3.9512
  Desvio    1.9875  1.9876
```

---

## Tests

```
python -m pytest tests/
```

Ejecuta 14 tests unitarios que verifican:
- Carga correcta del catalogo de distribuciones
- Filtrado inteligente segun los datos
- Funciones de scoring contra `scipy.stats`
- Resultados coherentes con datos generados sinteticamente

---

## Dependencias

- Python >= 3.10
- numpy
- scipy
- pandas
- matplotlib
- rich

Instalacion:

```
pip install numpy scipy pandas matplotlib rich
```

---

## Estructura del proyecto

```
distribution-fitting/
├── pyproject.toml             # Dependencias y config de pytest
├── distribuciones.json        # Config de distribuciones a probar
├── main.py                    # CLI interactivo
├── src/
│   ├── distribuciones.py      # Catalogo de distribuciones
│   ├── ajuste.py              # Motor de fitting (MLE + scoring + ranking)
│   ├── estadisticas.py        # Calculo de momentos empiricos y teoricos
│   ├── visualizacion.py       # Graficos (histograma + PDF)
│   └── pruebas/
│       ├── ks.py              # Scoring Kolmogorov-Smirnov
│       ├── chi.py             # Scoring Chi-cuadrado
│       ├── anderson.py        # Scoring Anderson-Darling
│       └── selector.py        # Router de metodos
├── tests/                     # Tests unitarios
└── datos/                     # Datasets de ejemplo
```
