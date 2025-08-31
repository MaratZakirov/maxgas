import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import random
from mpl_toolkits.mplot3d import Axes3D

# particle number
N = 1000
NX, NY, NZ = 400, 400, 400
GRID_SIZE = 10
GRAVITY = np.array([0, 0, -0.5])

class Particles:
    def __init__(self, PNUM: int, radius: float, dt: float):
        self.x = np.random.uniform(low=(0.1*NX, 0.1*NY, 0.1*NZ), high=(0.9*NX, 0.9*NY, 0.4*NZ), size=(PNUM, 3))
        self.v = 4*np.random.randn(PNUM, 3)
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# Scatter plot initialization
crypton = Particles(N, 2, 1.0)
scatter = ax.scatter(crypton.x[:, 0], crypton.x[:, 1], crypton.x[:, 2], s=1.0)

# Update function to append new random points and redraw scatter plot
def update(frame):
    global scatter, crypton
    ax.clear()
    scatter = ax.scatter(crypton.x[:, 0], crypton.x[:, 1], crypton.x[:, 2], s=1.0)
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_zlim(0, NZ)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    crypton.step()

# 60 fps
anim = FuncAnimation(fig, update, interval=1000/60)
plt.show()