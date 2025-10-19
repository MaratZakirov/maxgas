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

print(M*g/alpha/R-1)

def formula(altitude):
    return n_0*(1 - alpha*altitude/T_0)**(M*g/alpha/R-1)

def formula2(altitude):
    return n_0*(1 - alpha_calc*altitude/T_0)**(M*g/alpha_calc/R-1)

def standart_bolzman(altitude) -> np.ndarray:
    return n_0*np.exp(-m * g * altitude / k / T_0)

# Create a figure with a higher resolution
plt.figure(dpi=100)

# Plot the air density ('rho') against altitude ('z')
plt.plot(alt, ds_rho, c='r')

standart_n = standart_bolzman(alt)
n1 = formula(alt)
n2 = formula2(alt)

plt.plot(alt, standart_n, c='b')
plt.plot(alt, n1, c='g')
plt.plot(alt, n2, c='y')

for i in np.linspace(start=0, stop=len(alt)-1, num=12, dtype=int):
    #print((alt[i]/1000).round(1), ' ',  ds_rho[i].round(2))
    print((alt[i]/1000).round(2), ' ', standart_n[i].round(2))

# Add grid for better readability
plt.grid(True)

# Add labels and title for clarity
plt.ylabel("Air Density (kg/m³)")
plt.xlabel("Altitude (m)")
plt.title("U.S. Standard Atmosphere 1976 Air Density")

# Display the plot
plt.show()