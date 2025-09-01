import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
import pandas as pd
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
        self.m = np.random.choice(m, size=PNUM)
        self.color = [('r' if i == 1.0 else 'b') for i in self.m]
        self.radius = radius
        self.dt = dt

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

    def find_collided_pairs(self):
        addr = (self.x // GRID_SIZE).astype(int)
        addr = addr[:, 0] + 100 * addr[:, 1] + 10000 * addr[:, 2]
        addr = pd.Series(addr)
        groups = addr.groupby(addr)
        ij = []

        # The idea is to form groups
        # Then random shuffle each group
        # Then truncate each group int(COLLISION_PROBABILITY * len(group))//2*2
        # Then form one array make it reshape(-1, 2)
        # Done process these pairs into process_collisions

        for g in groups.groups:
            idx_s = groups.get_group(g)
            if len(idx_s) > 0:
                ij.append(0)

    def process_collisions(self):
        pass

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
        gas.find_collided_pairs()
        gas.process_collisions()
        step_num += 1

    if step_num > STEPS:
        anim.event_source.stop()
        plt.close(fig)

# 60 fps
anim = FuncAnimation(fig, update, interval=1000/60)
plt.show()