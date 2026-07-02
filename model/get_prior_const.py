import numpy as np
import sys

def pp(a,b):
    avg =   (b*0.5 + a*0.5)
    var =   (b**3 - a**3)/(3*(b-a)) - avg**2

    return var**0.5 / avg

print(pp(0.1,5))
print(pp(0,3))

    
