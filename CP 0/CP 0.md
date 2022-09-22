# CP 0
## Ejercicio 1
a) 45e5   
b) 123e-7  
c) -123e4  
d) 1,00

## Ejercicio 2
## a)
a) True  
b) False  
c) False  
d) False  
e) True 
## b) 84
## c) Para una precisión de 1e-14 es 83,98736879028142

## Ejercicio 3
a) 
```py
def Factorial(x):
    if(x == 1):
        return 1
    return x*Factorial(x-1)
```
b)
```py
def Factorial(x):
    result = 1
    while(x > 0):
        result*= x
        x-=1
    return result
```

c)
```py

```

## Ejercicio 4
a)
```py
from sympy import *
h = symbols('h')
def f(x):
    return x**2

def derivada(f,x,y):
    return limit((f(x+h)-f(x))/h,h,y)
```

b)
```py
from sympy import *
h = symbols('h')
def f(x):
    return x**2

def derivada(f,x,y):
    return limit((f(x+h)-f(x))/h,h,y)
derivada(f,1,1) #devuelve 3
derivada(f,1,0.5) #devuelve 2.5
derivada(f,1,0.01) #devuelve 2.01

```

- En este caso se prueba con valores de h = 1, h = 0.5 y h = 0.01 respectivamente, devolviendo 3, 2.5 y 2.01 respectivamente

c) Del inciso anterior podemos llegar a la conclusión que con valores cada vez más cercanos a 0 se obtiene una mejor aproximación de la derivada de la función en un punto dado

Pregunta secreta #1: Como se expuso en el inciso anterior, una aproximación de una derivada en un punto es mejor que otra si esta se realiza con un valor de h más pequeño (cercano a 0) que el valor utilizado para la otra medición

## Ejercicio 5
a) Serie de Taylor   
b) $$ f(x+h) = f(x) +\sum_{i=1}^n \frac{f^i(x)*h^i}{i!}$$
$$