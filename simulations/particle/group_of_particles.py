import matplotlib.pyplot as plt
import numpy as np
from partlib.simtools import Particles
import matplotlib
matplotlib.use('TkAgg')

marginal_flag = True

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

gas.x[:, 2] = 20 + 0.1*np.random.randn(N).clip(min=-2, max=2)
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

fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)

# 2. Loop through the three axes to draw the same plot on each
ax = axes[2]

# Создаем вторую ось для гистограммы (плотность по высоте)
ax_hist = ax.twinx()
# bins можно настроить под ваши нужды
ax_hist.hist(h, bins=64, orientation='vertical',
             color='blue', alpha=0.2, density=True)

# Прячем значения на оси гистограммы, чтобы не загромождать вид
ax_hist.set_yticks([])

ax.set_xlim(0, 80)
ax.set_ylim(0, 55)
ax.set_xlabel("Height")

ax.set_ylabel("Kinetic energy/Temperature")

if marginal_flag:
    x_zoom_val = 10
    y_zoom_val = 450000

    def f_inverse(x_arg, threshold, f_max=5.5):
        f_val = 1 / np.sqrt(threshold - x_arg)
        f_val[np.logical_or(np.isnan(f_val), np.isinf(f_val))] = 0
        f_val = f_val.clip(min=0, max=f_max)
        f_val = f_val / f_val.sum()
        return f_val

    def f_exponetial(x_arg, x_0=0.):
        f_val = np.exp(-x_arg)
        f_val = f_val / f_val.sum()
        f_val[x_arg < x_0] = 0
        return f_val

    z = np.linspace(start=0.0, stop=6, num=1000)
    f_exp = f_exponetial(z, 1.9)

    axes[0].plot(z, f_exp)

    f_inv = []

    for i, z_max in enumerate(np.linspace(start=1.9, stop=7.5, num=1001)):
        f_val = f_inverse(z, threshold=z_max)
        f_val = f_val * np.exp(-z_max)
        f_inv.append(f_val)
        if i % 100 == 0:
            axes[1].plot(z, f_val)

    f_inv = np.stack(f_inv, axis=0).mean(axis=0)
    axes[2].plot(z * x_zoom_val, f_inv * y_zoom_val, color='red')

plt.tight_layout() # Prevents labels from overlapping
plt.show()