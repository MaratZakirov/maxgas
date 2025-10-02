import ussa1976
import numpy as np
import matplotlib.pyplot as plt

# Нужно пересчитать для случая температуры как в данных ussa1976

def transform_v(x, p=0.8, n=14, o=16):
    #return ((o/n)*(1-p) + p)*x
    return 1.1*x

def standart_bolzman_formula_for_nitrogen(altitude, max_rho: float) -> np.ndarray:
    #return max_rho*np.exp(-1.1e-4 *altitude)
    return max_rho * np.exp(-0.0001147 * altitude)

def neo_bolzman_formula_for_nitrogen(altitude, max_rho: float, T_0: float = 288) -> np.ndarray:
    val = 0.0033 * (T_0-transform_v(0.00943)*altitude)**(5/2)
    return max_rho * val/val.max()

# Create a regular altitude mesh from 0 to 10000 meters
altitudes = np.arange(0.0, 10001.0, 1.0)

# Compute the atmosphere model, selecting density ('rho') as the variable
ds = ussa1976.compute(z=altitudes, variables=["rho"])

# Create a figure with a higher resolution
plt.figure(dpi=100)

# Plot the air density ('rho') against altitude ('z')
plt.plot(altitudes, ds.rho)

bolzman_nitro_rho = standart_bolzman_formula_for_nitrogen(altitudes, max_rho=ds.rho.max().item())
neo_nitro_rho = neo_bolzman_formula_for_nitrogen(altitudes, max_rho=ds.rho.max().item())

plt.plot(altitudes, bolzman_nitro_rho)
plt.plot(altitudes, neo_nitro_rho)

# Add grid for better readability
plt.grid(True)

# Add labels and title for clarity
plt.ylabel("Air Density (kg/m³)")
plt.xlabel("Altitude (m)")
plt.title("U.S. Standard Atmosphere 1976 Air Density")

# Display the plot
plt.show()