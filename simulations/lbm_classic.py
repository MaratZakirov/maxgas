# This is an example of simple LBM code based on
# Matias Ortiz one
# https://www.youtube.com/watch?v=JFWqCQHg-Hs

import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main():
    Nx = 400
    Ny = 100
    tau = .53 # kinematic viscosity and timescale both (!)
    Nt = 10000

    # Lattice speeds and weights
    NL = 9
    cxs = np.array([0, 0, 1, 1,  1,  0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1,  0,  1])
    weights = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

    # initial conditions
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[..., 3] = 2.3

    cylinder = np.full((Ny, Nx), False)
    for y in range(Ny):
        for x in range(Nx):
            if (distance(Nx//4, Ny//2, x, y) < 13):
                cylinder[y][x] = True

    # main loop
    for it in range(Nt):
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        # Streaming step
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[..., i] = np.roll(F[..., i], cx, axis=1)
            F[..., i] = np.roll(F[..., i], cy, axis=0)

        # Collision handle
        boundary_F = F[cylinder]
        boundary_F = boundary_F[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Macroscopic variables
        rho = np.sum(F, axis=2)
        ux = np.sum(F * cxs, axis=2) / rho
        uy = np.sum(F * cys, axis=2) / rho

        # Apply opposite velocities
        F[cylinder] = boundary_F
        ux[cylinder] = 0.
        uy[cylinder] = 0.

        # Particles collision
        F_eq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            F_eq[..., i] = rho * w * (
                    1 + 3 * (cx*ux + cy*uy) + (9/2) * (cx * ux + cy * uy)**2 - (3/2) * (ux**2 + uy**2)
            )
        F = F - (1/tau) * (F - F_eq)

        if it % 10 == 0:
            print(it)
            plt.imshow(np.sqrt(ux**2 + uy**2))
            plt.pause(0.01)
            plt.cla()

if __name__ == '__main__':
    main()