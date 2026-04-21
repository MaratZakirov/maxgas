import matplotlib.pyplot as plt
import numpy as np
from partlib.simtools import Particles
import matplotlib
matplotlib.use('TkAgg')

N = 1000

# Parameters
step_num = 0
STEPS = 22800
# steps need system to reach equlibrium state
EQU_STEPS = 150

# Data for further analysis
data = []

# Scatter plot initialization
np.random.seed(0)

gas = Particles(N, 2, 0.25, m=[1.0])

gas.x[:, 2] = 40 + 0.1*np.random.randn(N).clip(min=-2, max=2)
gas.v = 3*np.random.randn(N, 3).clip(min=-3, max=3)

# Steping without showing
for step_num in range(STEPS):
    gas.step()
    gas.collided[:] = False

    # collisions
    if 0:
        collided_pairs = gas.find_collided_pairs()
        gas.process_collisions(collided_pairs)
        gas.collided[collided_pairs[:, 0]] = True
        gas.collided[collided_pairs[:, 1]] = True

    # set some data for analysis
    if step_num > EQU_STEPS:
        data.append(np.stack([gas.x[:, 2], np.sum(gas.m*(gas.v ** 2) / 2, axis=1), gas.collided,
                              gas.v[:, 0], gas.v[:, 1], gas.v[:, 2], np.arange(len(gas.x))], axis=1))

data = np.concatenate(data, axis=0)

h   = data[:, 0]
E_k = data[:, 1]
cld = data[:, 2]
idx = data[:, 6]
E_k_z = data[:, 5]**2/2

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# 2. Loop through the three axes to draw the same plot on each
ax = axes[0]

# Создаем вторую ось для гистограммы (плотность по высоте)
ax_hist = ax.twinx()
# bins можно настроить под ваши нужды
ax_hist.hist(h, bins=64, orientation='vertical',
             color='blue', alpha=0.2, density=True)

# Прячем значения на оси гистограммы, чтобы не загромождать вид
ax_hist.set_yticks([])

ax.set_xlim(0, 100)
ax.set_ylim(0, 55)
ax.set_xlabel("Height")

ax.set_ylabel("Kinetic energy/Temperature")

plt.tight_layout() # Prevents labels from overlapping
plt.show()