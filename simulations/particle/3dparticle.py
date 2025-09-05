import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
#import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from typing import List

# Hypers
N = 1000
NX, NY, NZ = 400, 400, 400
GRID_SIZE = 10
GRAVITY = np.array([0, 0, -0.5])

# Parameters
# TODO add warmup iterations without collisions
step_num = 0
STEPS = 2100
detailed_steps = set(range(100)) | set(range(500, 550)) | set(range(2000, 2100))

class Particles:
    def __init__(self, PNUM: int, radius: float, dt: float, m: List[float]):
        self.x = np.random.uniform(low=(0.1*NX, 0.1*NY, 0.1*NZ), high=(0.9*NX, 0.9*NY, 0.4*NZ), size=(PNUM, 3))
        self.v = 4*np.random.randn(PNUM, 3)
        self.m = np.random.choice(m, size=(PNUM, 1))
        self.color = [('r' if i == 1.0 else 'b') for i in self.m]
        self.radius = radius
        self.dt = dt
        self.E0 = self.get_system_full_energy()

    def get_system_potential_energy(self) -> float:
        return -(self.x * self.m * GRAVITY).sum()

    def get_system_kinetic_energy(self) -> float:
        return np.sum(self.m*(self.v ** 2)/2)

    def get_system_full_energy(self):
        return self.get_system_potential_energy() + self.get_system_kinetic_energy()

    def step(self) -> None:
        self.x += self.v*self.dt + 0.5*GRAVITY*(self.dt**2)
        self.v += GRAVITY * self.dt

        # Boundary collision checks with walls (elastic bounce):
        X_idx = np.logical_or(self.x[:, 0] < 0, self.x[:, 0] > NX)
        self.v[X_idx, 0] = -self.v[X_idx, 0]

        # for Y axis
        Y_idx = np.logical_or(self.x[:, 1] < 0, self.x[:, 1] > NY)
        self.v[Y_idx, 1] = -self.v[Y_idx, 1]

        # for Z axis
        Z_idx = np.logical_or(self.x[:, 2] < 0, self.x[:, 2] > NZ)
        self.v[Z_idx, 2] = -self.v[Z_idx, 2]

    def find_collided_pairs(self) -> np.ndarray:
        addr = (self.x // GRID_SIZE).astype(int)
        addr = addr[:, 0] + 100 * addr[:, 1] + 10000 * addr[:, 2]

        Idx = np.stack([addr, np.arange(N)], axis=1)
        np.random.shuffle(Idx)
        Idx = Idx[np.argsort(Idx[:, 0])]
        Mask = (Idx[:-1, 0] - Idx[1:, 0]) == 0

        # apply Mask which indicates same address
        ij = np.stack([Idx[:-1, 1], Idx[1:, 1]], axis=1)[Mask]

        # check that each pair have right address
        assert (addr[ij[:, 0]] == addr[ij[:, 1]]).all(), 'Some pair has inconsistent address'

        # it could be done via Mask[1:] - Mask[-1:]
        ij = ij[~np.isin(ij[:, 0], ij[:, 1])]

        E1 = self.get_system_full_energy()

        assert (max(self.E0, E1) / min(self.E0, E1)) < 1.001, 'Energy diffrence is too huge'

        return ij

    def process_collisions(self, ij: np.ndarray):
        x1 = self.x[ij[:, 0]]
        x2 = self.x[ij[:, 1]]
        v1 = np.copy(self.v[ij[:, 0]])
        v2 = np.copy(self.v[ij[:, 1]])
        m1 = self.m[ij[:, 0]]
        m2 = self.m[ij[:, 1]]
        n = (x2 - x1)/np.linalg.norm(x2 - x1, keepdims=True, axis=1)
        self.v[ij[:, 0]] = v1 - 2*(m2/(m1 + m2))*np.sum((v1 - v2)*n, keepdims=True, axis=1)*n
        self.v[ij[:, 1]] = v2 - 2*(m1/(m1 + m2))*np.sum((v2 - v1)*n, keepdims=True, axis=1)*n

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# Scatter plot initialization
gas = Particles(N, 2, 1.0, m=[1.0, 1.31])
scatter = ax.scatter(gas.x[:, 0], gas.x[:, 1], gas.x[:, 2], s=1.0, c=gas.color)

# Update function to append new random points and redraw scatter plot
def update(frame):
    global scatter, gas, step_num, anim
    ax.clear()
    scatter = ax.scatter(gas.x[:, 0], gas.x[:, 1], gas.x[:, 2], s=1.0, c=gas.color)
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_zlim(0, NZ)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    fig.canvas.manager.set_window_title(f'Step num: {step_num}')

    for i in range(10 - 9*int(step_num in detailed_steps)):
        gas.step()
        collided_pairs = gas.find_collided_pairs()
        gas.process_collisions(collided_pairs)
        step_num += 1

    if step_num > STEPS:
        anim.event_source.stop()
        plt.close(fig)

# 60 fps
anim = FuncAnimation(fig, update, interval=1000/60)
plt.show()