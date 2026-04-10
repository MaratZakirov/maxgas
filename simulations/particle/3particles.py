import matplotlib.pyplot as plt
import numpy as np
from partlib.simtools import Particles
import matplotlib
matplotlib.use('TkAgg')

# Hypers
N = 3

# Parameters
step_num = 0
STEPS = 600
# steps need system to reach equlibrium state
EQU_STEPS = 50

# Data for further analysis
data = []

# Scatter plot initialization
np.random.seed(0)

gas = Particles(N, 2, 0.33, m=[1.0])

gas.x[:, 2] = 1
gas.v = np.array([[0,     0, 1.0],
                  [0.2, 0.2, 0.6],
                  [0.4, 0.4, 0.2]])
gas.v = 10 * gas.v / np.linalg.norm(gas.v, keepdims=True, axis=1)

# Steping without showing
for step_num in range(STEPS):
    gas.step()
    gas.collided[:] = False
    # no collisions

    # set some data for analysis
    if step_num > EQU_STEPS:
        data.append(np.stack([gas.x[:, 2], np.sum(gas.m*(gas.v ** 2) / 2, axis=1), gas.collided,
                              gas.v[:, 0], gas.v[:, 1], gas.v[:, 2], np.arange(len(gas.x))], axis=1))

data = np.concatenate(data, axis=0)

h   = data[:, 0]
E_k = data[:, 1]
cld = data[:, 2]
idx = data[:, -1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# 2. Loop through the three axes to draw the same plot on each
for i, ax in enumerate(axes):
    # Данные для конкретной частицы
    current_h = h[idx == i]
    current_E = E_k[idx == i]
    color = ['red', 'green', 'blue'][i]

    # Основной график scatter
    ax.scatter(current_h, current_E, c=color, s=1.5, alpha=0.5)

    # Создаем вторую ось для гистограммы (плотность по высоте)
    ax_hist = ax.twinx()
    # bins можно настроить под ваши нужды
    ax_hist.hist(current_h, bins=10, orientation='vertical',
                 color=color, alpha=0.2, density=True)

    # Прячем значения на оси гистограммы, чтобы не загромождать вид
    ax_hist.set_yticks([])

    ax.set_xlim(-10, 110)
    ax.set_ylim(0, 55)
    ax.set_xlabel("Height")
    if i == 0:
        ax.set_ylabel("Kinetic energy/Temperature")

plt.tight_layout() # Prevents labels from overlapping
plt.show()