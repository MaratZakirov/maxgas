import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# Parameters and constants
m   = 4.65e-26        # kg
kB  = 1.38e-23        # J/K
T0  = 288.15          # K
L   = 0.00649         # K/m
M   = 0.028           # kg/mol
g   = 9.81            # m/s^2
R   = 8.314           # J/(mol*K)

# Domains and sampling (like samples=14, domain x=0:5.6e-26, y=0:10)
nx, ny = 14, 14
x = np.linspace(0.0, 5.6e-26, nx)   # momentum, kg·m/s
y = np.linspace(0.0, 10.0,   ny)    # altitude, km
X, Y = np.meshgrid(x, y)            # X ~ p, Y ~ z (km)

# Convert to the same units as in the LaTeX code
p = X * 1000.0       # g·m/s (since original multiplies x by 1000)
z_m = Y * 1000.0     # m

T = T0 - L * z_m

# Function f(p, z) from your TikZ code
f = (
    4 * np.pi
    * (2 * np.pi * m * kB * T) ** (-1.5)
    * (p ** 2)
    * np.exp(-(p ** 2) / (2 * m * kB * T))
    * (M * g) / (T0 * R)
    * (1.0 - (L / T0) * z_m) ** ((M * g) / (L * R) - 1.0)
)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, f,
    cmap='viridis',
    edgecolor='none'
)

ax.set_xlabel(r'$kg \cdot m/s$')
ax.set_ylabel(r'$km$')
ax.set_zlabel(r'$f(p,z)$')

# Approximate PGFPlots view={25}{65}
ax.view_init(elev=65, azim=25)

fig.colorbar(surf, shrink=0.5, aspect=10, label=r'$f(p,z)$')

plt.title('Функция распределения по импульсу и высоте')
plt.tight_layout()
plt.show()



exit()

import numpy as np
import matplotlib.pyplot as plt

# Define the function of two variables
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

H = 10

# еще нужно учесть экспонинциальное падение концентрации частиц

# Create coordinate arrays
x = np.linspace(-np.pi*2, np.pi*2, 200)
y = np.linspace(0, 10, 200)
X, Y = np.meshgrid(x, y)

# Compute function values on the grid
Z = np.exp(-X**2/(10 - Y + 1))
# Plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot (you can change to plot_wireframe for wireframe)
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D plot of function f(x,y)')

plt.show()


"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start=-np.pi, stop=np.pi, num=20000)
p = np.exp(-x**2)
Z = (p*(x.max() - x.min())/len(x)).sum()

p = p / Z

plt.plot(x, p)
plt.show()
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create data for a sphere
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
r = 1

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# Create a figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot wireframe
ax.plot_wireframe(x, y, z, color='blue')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Basic 3D Wireframe Plot')

# Show plot
plt.show()
"""