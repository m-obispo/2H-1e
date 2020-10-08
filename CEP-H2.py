import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

#Conversões de unidades
def kcalMol(x):
    return x/0.0015

def eV(x):
    return kcalMol(x)/23.0605

def angs(x):
    return x*0.5292

#Energia exata do estado fundamental do átomo de H
E_H = -0.5

#Sympy
R = sym.symbols('R')

e11 = (1-sym.exp(-2*R)*(1+R))/R
e12 = sym.exp(-1*R)*(1+R)
s12 = sym.exp(-1*R)*(1+R+R*R/3)

expr_E1 = E_H + 1/R - (e11-e12)/(1-s12)
expr_E2 = E_H + 1/R - (e11+e12)/(1+s12)
expr_dE2 = sym.diff(expr_E2)

E1 = sym.lambdify(R, expr_E1, 'numpy')
E2 = sym.lambdify(R, expr_E2, 'numpy')
dE2 = sym.lambdify(R, expr_dE2, 'numpy')

vec_R = np.arange(0.5, 10, 0.1)
vec_Re = np.arange(1.5, 3.0, 0.001)

#Questão 6 e 7
for i in range(len(dE2(vec_Re))-1):
    if dE2(vec_Re)[i]*dE2(vec_Re)[i+1] < 0:
        Re = vec_Re[i]
        De = E2(Re) - E2(100)
print('Re = {:.3f} Bohr = {:.3f} Å'.format(Re, angs(Re))+
    '\nDe = {:.3f} Ha = {:.3f} eV = {:.3f} kcal/mol'.format(De, eV(De), kcalMol(De)))

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

#Questão 9

xi, eta = sym.Symbol('xi'), sym.Symbol('eta')

Phi1_expr = (sym.exp(-1*R*(xi+eta)/2) - sym.exp(-1*R*(xi-eta)/2))/(sym.sqrt(2*sym.pi*(1-s12)))
Phi2_expr = (sym.exp(-1*R*(xi+eta)/2) + sym.exp(-1*R*(xi-eta)/2))/(sym.sqrt(2*sym.pi*(1+s12)))

vec_xi = np.arange(1., 5., 0.01)
vec_eta = np.arange(-1., 1., 0.01)
Xi, Eta = np.meshgrid(vec_xi,vec_eta)

rho1 = sym.lambdify([xi, eta, R], sym.Abs(Phi1_expr)**2, 'numpy')
rho2 = sym.lambdify([xi, eta, R], sym.Abs(Phi2_expr)**2, 'numpy')

fig = plt.figure(figsize = (12, 10), dpi = 100)
ax1 = fig.add_subplot(projection = '3d')
ax1.view_init(30, -10)
ax1.set_title('Densidade de Probabilidade', fontsize = 18)
ax1.plot_surface(Xi,Eta,rho1(Xi, Eta, Re), cmap='viridis', rcount = 100,ccount = 100)
ax1.set_xlabel('$\\xi$', fontsize = 18)
ax1.set_ylabel('$\\eta$', fontsize = 18)
ax1.set_zlabel('Probabilidade', fontsize = 18)
plt.show()

fig = plt.figure(figsize = (12, 10), dpi = 100)
ax2 = fig.add_subplot(projection = '3d')
ax2.view_init(30, -10)
ax2.set_title('Densidade de Probabilidade', fontsize = 18)
ax2.plot_surface(Xi,Eta,rho2(Xi, Eta, Re), cmap='viridis_r', rcount = 100,ccount = 100)
ax2.set_xlabel('$\\xi$', fontsize = 18)
ax2.set_ylabel('$\\eta$', fontsize = 18)
ax2.set_zlabel('Probabilidade', fontsize = 18)
plt.show()

# Questão 10
x = sym.Symbol('x')
vec_x = np.arange(-5., 5., 0.01)

Phi1_expr = (sym.exp(-sym.Abs(x-R/2)) - sym.exp(-sym.Abs(x+R/2)))/(sym.sqrt(2*sym.pi*(1-s12)))
Phi2_expr = (sym.exp(-sym.Abs(x-R/2)) + sym.exp(-sym.Abs(x+R/2)))/(sym.sqrt(2*sym.pi*(1+s12)))
Phi1 = sym.lambdify([x, R], Phi1_expr, 'numpy')
Phi2 = sym.lambdify([x, R], Phi2_expr, 'numpy')

plt.figure(figsize = (12, 8), dpi = 100)
plt.suptitle('Método Variacional: Íon $H_2^+$', fontsize=24)
plt.title('Funções de Onda ($R = R_e$)', fontsize=22)
plt.grid()

plt.plot(angs(vec_x), Phi1(vec_x, Re), 'r-', label='$\\Phi_1(x)$ (Não ligado)')
plt.plot(angs(vec_x), Phi2(vec_x, Re), 'b-', label='$\\Phi_2(x)$ (Ligado)')
plt.plot(angs(vec_x), np.zeros(len(vec_x)), 'k--')
plt.plot([angs(-Re/2), angs(Re/2)], [0., 0.], 'ko', label='Átomos de H')

plt.xlim(angs(vec_x[0]), angs(vec_x[len(vec_x)-1]))
plt.xlabel('x ($\\mathring{A}$)', fontsize=20)
plt.ylabel('Função de Onda', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(loc = 4,fontsize=15)
plt.show()

#Questão 11
rho1_expr = (Phi1_expr)**2
rho2_expr = (Phi2_expr)**2
rho1 = sym.lambdify([x, R], rho1_expr, 'numpy')
rho2 = sym.lambdify([x, R], rho2_expr, 'numpy')

plt.figure(figsize = (12, 8), dpi = 100)
plt.suptitle('Método Variacional: Íon $H_2^+$', fontsize=24)
plt.title('Densidade Eletrônica ($R = R_e$)', fontsize=22)
plt.grid()

plt.plot(angs(vec_x), rho1(vec_x, Re), 'r-', label='$\\rho_1 = |\\Phi_1|^2$ (Não ligado)')
plt.plot(angs(vec_x), rho2(vec_x, Re), 'b-', label='$\\rho_2 = |\\Phi_2|^2$ (Ligado)')
plt.plot([angs(-Re/2), angs(Re/2)], [0., 0.], 'ko', label='Átomos de H')

plt.xlim(angs(vec_x[0]), angs(vec_x[len(vec_x)-1]))
plt.ylim(0,0.33)
plt.xlabel('x ($\\mathring{A}$)', fontsize=20)
plt.ylabel('Probabilidade', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(loc = 9,fontsize=15)
plt.show()
