import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

E_H = 0.5

#Conversões de unidades
def kcalMol(x):
    return x/0.0015

def eV(x):
    return kcalMol(x)/23.0605

def angs(x):
    return x*0.5292

#Sympy
R = sym.symbols('R')
vec_R = np.arange(0.5, 10, 0.1)

e11 = (1-sym.exp(-2*R)*(1+R))/R
e12 = sym.exp(-1*R)*(1+R)
s12 = sym.exp(-1*R)*(1+R+R*R/3)

expr_E1 = E_H + 1/R - (e11-e12)/(1-s12)
expr_E2 = E_H + 1/R - (e11+e12)/(1+s12)
expr_dE2 = sym.diff(expr_E2)

E1 = sym.lambdify(R, expr_E1, 'numpy')
E2 = sym.lambdify(R, expr_E2, 'numpy')
dE2 = sym.lambdify(R, expr_dE2, 'numpy')

vec_Re = np.arange(1.5, 3.0, 0.001)

#Questão 6 e 7
for i in range(len(dE2(vec_Re))-1):
    if dE2(vec_Re)[i]*dE2(vec_Re)[i+1] < 0:
        Re = vec_Re[i]
        De = E2(Re) - E2(100)
        print('R_e = {:.3f} A; D_e = {:.3f} eV'.format(angs(Re), eV(De)))

cepE1 = E1(vec_R) - E2(100)
cepE2 = E2(vec_R) - E2(100)

#Questão 5
plt.figure(figsize = (12, 8), dpi = 100)
plt.suptitle('Método Variacional: Íon $H_2^+$', fontsize=24)
plt.title('Curvas de Energia Potencial', fontsize=22)

plt.plot(angs(vec_R), eV(cepE1), 'r-', label = '$E_1$ (não-ligado)')
plt.plot(angs(vec_R), eV(cepE2), 'b-', label = '$E_2$ (ligado)')
plt.plot(angs(Re)   , eV(De), 'ko', label = 'Ponto de mínimo')
plt.plot(angs(vec_R), np.zeros(len(vec_R)), 'k--')

plt.ylim(2*eV(De),2*eV(-De))
plt.xlim(0.25, 5.)
plt.xticks(np.arange(0.25,5.1, 0.5), fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Distância internuclear($\\mathring{A}$)', fontsize=20)
plt.ylabel('Energia (eV)', fontsize=20)
plt.grid()
plt.legend(fontsize=18)
plt.show()

def phi1(x, R):
    return (np.exp(-x-R/2) - np.exp(-x+R/2))/2/(1-s12(R))**0.5

def phi2(x, R):
    return (np.exp(-x-R/2) + np.exp(-x+R/2))/2/(1+s12(R))**0.5


#Numpy
#  

# def e11(R):
#     return (1-np.exp(-2*R)*(1+R))/R

# def e12(R):
#     return np.exp(-R)*(1+R)

# def s12(R):
#     return np.exp(-R)*(1+R+R*R/3)

# def E1(R):
#     return E_H + 1/R - (e11(R)-e12(R))/(1 - s12(R))

# def E2(R):
#     return E_H + 1/R - (e11(R)+e12(R))/(1 + s12(R))

# cepE1 = E1(vec_R) - E2(100)
# cepE2 = E2(vec_R) - E2(100)

# plt.plot(vec_R, cepE1, 'r-')
# plt.plot(vec_R, cepE2, 'b-')
# plt.plot(vec_R, np.zeros(len(vec_R)), 'k--')
# plt.show()