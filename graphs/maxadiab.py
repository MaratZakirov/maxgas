import ussa1976
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as physconsts

# Create a regular altitude mesh from 0 to 10000 meters
alt = np.arange(0.0, 10001.0, 1.0)

# Compute the atmosphere model, selecting density ('rho') as the variable
ds = ussa1976.compute(z=alt, variables=["rho", "t"])
ds_rho = np.array(ds.rho)

# Specific heat for nitrogen
c_p = 1040

# number of molecules at sea level
n_0 = ds.rho.max().item()

# temperature for sea level
T_0 = ds.t.max().item() # k
# gravitational
g = 9.81 # m/s

# R - constant
R = 8.314 # J/mol/K

# k - constant
k = 1.38e-23

# Avogadro
N_A = physconsts.N_A

# M for nitrogen
M = 0.028 # kg/mol

# m for nitrogen
m = M/N_A

# lapse rate
alpha = 0.00649 # K/m

# lapse rate
alpha_calc = g/c_p # K/m

# Some other arrays for f_z(p, z) probability function
z = np.linspace(start=0,stop=10000, num=100)
p = m*np.linspace(start=0, stop=1200, num=100)

P, Z = np.meshgrid(p, z)

def adiabatic(z):
    return (M * g) / (T_0 * R) * (1 - (alpha / T_0) * z) ** ((M * g) / (alpha * R) - 1)

def maxwell(p, z):
    T = (T_0 - alpha * z)
    return 4 * np.pi * (2 * np.pi * m * k * T) ** (-3 / 2) * (p**2) * np.exp(-p ** 2 / (2 * m * k * T))

def f_z(p, z):
    return maxwell(p, z) * adiabatic(z)

dz = (z.max() - z.min())/100
print(f"Stat sum for [{z.min()} ... {z.max()}] adiabatic concentration probability:",
      np.sum(adiabatic(np.linspace(start=0, stop=10000, num=100))*dz).round(3))

# Вычисляем значения функции
F = f_z(P, Z)

# Создаем фигуру и 3D ось
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Строим wireframe
ax.plot_wireframe(P, Z, F, rstride=5, cstride=5)#, color='blue')

# Подписываем оси
ax.set_xlabel('p')
ax.set_ylabel('z')
ax.set_zlabel('f_z(p, z)')
ax.set_title('3D Wireframe plot of f_z(p, z)')

plt.show()
