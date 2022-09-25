# CP 0
## Ejercicio 1
a) 45e5   
b) 123e-9  
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
## c) double 83,98736879028142

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
import scipy.special as sp
def FactorialGamma(n):
    if(n<0):
        return 1
    return sp.gamma(n+1)
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

def derivEval(f,x):
    for i in range(1,16):
        h = 10**-i
        print(derivada(f,x,h))
```

- En este caso se prueba con valores de h = 10*-i con i de 1 hasta 16

c) Del inciso anterior podemos llegar a la conclusión que con valores cada vez más cercanos a 10**-8 se obtiene una mejor aproximación de la derivada de la función en un punto dado

Pregunta secreta #1: Como se expuso en el inciso anterior, una aproximación de una derivada en un punto es mejor que otra si esta se realiza con un valor de h más pequeño (cercano a 0) generalmente que el valor utilizado para la otra medición

## Ejercicio 5
a) Serie de Taylor   
b) $$ f(x) = \sum_{i=0}^n \frac{f^i(x)*h^i}{i!}$$

c) 
```py
import sympy as sp
import math as mt

x = sp.symbols('x')
def f(x):
    return x**5 + 6*x**3 -4*x**2 + 5
    #return sp.sin(x)
    #return mt.e**(x)

def Taylor(f,h,n):
    if( n == 0):
        return f.subs(x,h)
    return (sp.diff(f,x,n).subs(x,h)/mt.factorial(n))*(x-h)**n + Taylor(f,h,n-1)

Taylor(f(x),1,5)
```
## Ejercicio 6

```py
import matplotlib.pyplot as plt
import numpy as np

def graficar(x0,n,a,b,f):
    t = Taylor(f,x0,n)
    puntos = np.arange(a,b,(b-a)/100)
    plt.plot(puntos, [f.subs(x,i) for i in puntos])
    plt.plot(puntos,[t.subs(x,i) for i in puntos])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
graficar(0,4,-4,4,f(x))
```
- En este caso utilizo las funciones Taylor y f(x) del ejercicio anterior
- Descomentando de f(x) que desea derivar

## Ejercicio 7
a)
```py
import sympy as sp
import numpy as np

x = sp.symbols('x')

def f(x):
    return x**2    
    #return x**3
def resulth(f,h):
    return sp.diff(f,x,1) - (f.subs(x,x+h)) - f/h 
def resulthh(f,h):
    return sp.diff(f,x,1) - (f.subs(x,x+h)) - f.subs(x,x-h)/2*h 
                             
print("O(h) =",resulth(f(x),0.1))
print("O(h**2) = ",resulthh(f(x),0.1))
```
b)
```py
print("O(h) =",np.abs(resulth(f(x),0.1).subs(x,1)))
#O(h) = 9.21000000000000
print("O(h**2) = ",np.abs(resulthh(f(x),0.1).subs(x,1)))
#O(h**2) =  0.749500000000000
```
c)
-La función fue utilizada en el inciso a para dar una aproximación del error.
## Ejercicio 8
a) 
```py
import numpy as np
import math as mt

def CompruebaMatrices(a,b,c):
    try:
        return (np.matrix(a) * np.matrix(b)) == np.matrix(c)
    except:
        return (np.matrix(b) * np.matrix(a)) == np.matrix(c)
```

b)
```py
a = [[1,0,0],[0,1,0],[0,0,1]]
b = [[2],[3],[4]]
c = [[2],[3],[4]]

CompruebaMatrices(a,b,c)
```
## Ejercicio 9
a)
```py
import numpy as np

def CreandoMatrix(n):
    lista = [0.0]
    for i in range(0,n-1):
        lista.append(0)
    lista = [lista]*n
    return np.matrix(lista)

def AgregandoValores(matrix):
    print(len(matrix))
    for i in range(0,len(matrix)): 
        for j in range(0,len(matrix)):
            if(i == j):
                matrix[i,j] = 0.5
            elif(i+1 == j):
                matrix[i,j] = 1
    return matrix

def VectorSolucion(n):
    vector = np.transpose(np.matrix([1]*n))
    return np.linalg.inv(AgregandoValores(CreandoMatrix(n)))*vector
 
#VectorSolucion(3)

```
b)
```py
def CompruebaSolucion(n):
    return CompruebaMatrices(AgregandoValores(CreandoMatrix(n)),VectorSolucion(n),np.transpose(np.matrix([1]*n)))
                           
#CompruebaSolucion(3)
```
c)
```py
for i in [20,40,60,80,100]: 
    print("i",i)
    print(CompruebaSolucion(i))
```
-En los casos de 20 y 40 se cumple, en los de 60,80 y 100 no

## Ejercicio 10
a)
```py
print("O(h**2) con h = 30 = ",np.abs(resulthh(f(x),30).subs(x,1)))
print("O(h**2) con h = 0.1 = ",np.abs(resulthh(f(x),0.1).subs(x,1)))
```
O(h2) con h = 30 =  13574  
O(h2) con h = 0.1 =  0.749500000000000  
- No es menor el error con h = 30

b) 
```py
import math as mt
def func(x):
    for i in range(70):
        x = mt.sqrt(x)
    for i in range(70):
        x = x*x
    return x

for i in range(1,1000000):
    if(func(i) != 1.0):
        print(i)
print("Ok")

aux = 1
i = 0.00001
while 1 - i  > 0:
    aux = 1 - i 
    print(aux)
    if(func(aux) == 1.0):
        print(aux)
    i = i+0.001
print("OK")
```
-Hasta el n = 1000000 Funciona  
-Los casos entre 0 y 1 con valores distantes de 0.001 aprox tambien funcionan  
c)
```py
def s(x):
    result = 0.0
    for i in x:
        print("result",result)
        result+=i
    return result
a = [1e100, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83]
b = [1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e100]
print(s(a) == s(b))
#1e+100 != 1.0000000000000002e+100
```
-Se obtienen valores diferentes al cambiar el orden de la suma. En este caso se tomo a(la lista como estaba) y b(intercambiando el primer elemento con el ultimo)
## Ejercicio 11
```py
def s(x):
    result = 0
    for i in x:
        result+=i
    return result   
    
s((1e100, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83, 1e83)) ==  1.0000000000000001e+100
```
-Con la suma realizada de la forma descrita en el algoritmo se obtiene la igualdad (aunque cambiando el orden puede variar el resultado (Ejercicio 10c))

## Ejercicio 12 (75000 créditos)

**Titulo: Diga no a la bisección**  
-a) A veces hallar las raíces de un polinomio es un poco incómodo, sobre todo cuando te das cuenta que nuestro amigo Ruffini solo es de utilidad cuando dichar raíces son enteras o cuando más, números fáciles de trabajar. Para los polinomios más robustos existen formas computacionales(aunque de seguro Newton lo hacía a mano) de calcular al menos una de sus raíces. Diseñe un algoritmo que, dado un polinomio, te devuelva una de sus raíces utilizando el método de Newton.  
-b) Explique porqué este algoritmo no es el más adecuado para realizar esta hazaña 

Solución:  
a)
```py
import numpy as np
import sympy as sp

x = sp.symbols('x')
def f(x):
    return x**4 + x - 3

def buscaPunto(f):
    terminoIndependiente = int(np.abs(f.subs(x,0)))+100
    minTerm = terminoIndependiente
    minimo = 100000.0
    for i in range(0,terminoIndependiente):
        a = np.abs(f.subs(x,i))
        if( a < minimo):
            minTerm = i
            minimo = a
    for i in range(-terminoIndependiente,0):
        a = np.abs(f.subs(x,i))
        if(a < minimo):
            minTerm = i
            minimo = a
    return minTerm

def MetodoNewton(f):
    df = sp.diff(f,x)
    print(buscaPunto(f.subs(x,0.0)))
    result = buscaPunto(f.subs(x,0.0)) 
    for i in range(0,100):
        result = -(round(f.subs(x,result),16)/round(df.subs(x,result),16))  + result
    return result

#print(MetodoNewton(f(x)))
```
b)    
  El algoritmo no es el más idóneo para buscar raíces de polinomios pues, en los casos donde la derivada evaluada en el punto es 0 no es posible obtener solución alguna  