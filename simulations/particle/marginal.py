import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

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

z = np.linspace(start=0.0, stop=6, num=121)
f_exp = f_exponetial(z, 1.9)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
axes[0].plot(z, f_exp)

f_inv = []

for i, z_max in enumerate(np.linspace(start=1.9, stop=5.5, num=101)):
    f_val = f_inverse(z, threshold=z_max)
    f_val = f_val * np.exp(-z_max)
    f_inv.append(f_val)
    if i % 10 == 0:
        axes[1].plot(z, f_val)

f_inv = np.stack(f_inv, axis=0).mean(axis=0)
axes[2].plot(z, f_inv)

plt.show()