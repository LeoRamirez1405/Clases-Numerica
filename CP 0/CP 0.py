from sympy import *
import math
#h = symbols('h')
x = symbols('x')
def f(x):
    return math.sin(x)
#def derivada(f,x,y):
#    return limit((f(x+h)-f(x))/h,h,y)

print(format(diff(f,x))
      
def Factorial(x):
    result = 1
    while(x > 0):
        result*= x
        x-=1
    return result

def Taylor(f,x,y):
    function = f
    result = f(y)
    result_anterior = 0
    index = 1
    while( diff(function,x) != 0 or result - result_anterior < 0.01):
        result_anterior = result
        function = lambdify(x, diff(f,x,index), "math") 
        #function =  diff(function,x,index)
        result += (format(function(y))/Factorial(index))*(x-y)**index
        #function(y)
        index +=1
        
    return result

Taylor(f,x,0)