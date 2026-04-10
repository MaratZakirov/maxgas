import numpy as np
from typing import List, Tuple

class Particles:
    def __init__(self, N: int, radius: float, dt: float, m: List[float],
                 container_size: Tuple[float, float, float] = (400, 400, 400),
                 GRAVITY: np.ndarray = np.array([0, 0, -0.5]),
                 GRID_SIZE: int=5, COLLISION_PROBABILITY: float = 1.0):
        self.COLLISION_PROBABILITY = COLLISION_PROBABILITY
        self.GRID_SIZE = GRID_SIZE
        self.NX, self.NY, self.NZ = container_size
        self.GRAVITY = GRAVITY
        self.x = np.random.uniform(low=(0.1*self.NX, 0.1*self.NY, 0.1*self.NZ),
                                   high=(0.9*self.NX, 0.9*self.NY, 0.4*self.NZ), size=(N, 3))
        self.v = 2.14*np.random.randn(N, 3)
        self.m = np.random.choice(m, size=(N, 1))
        self.color = [('r' if i == 1.0 else 'b') for i in self.m]
        self.radius = radius
        self.dt = dt
        self.E0 = self.get_system_full_energy()
        self.collided = np.zeros(N, dtype=bool)
        self.N = N

    def get_system_potential_energy(self) -> float:
        return -(self.x * self.m * self.GRAVITY).sum()

    def get_system_kinetic_energy(self) -> float:
        return np.sum(self.m*(self.v ** 2)/2)

    def get_system_full_energy(self):
        return self.get_system_potential_energy() + self.get_system_kinetic_energy()

    def step(self) -> None:
        self.x += self.v*self.dt + 0.5*self.GRAVITY*(self.dt**2)
        self.v += self.GRAVITY * self.dt

        # Boundary collision checks with walls (elastic bounce):
        X_idx = np.logical_or(self.x[:, 0] < 0, self.x[:, 0] > self.NX)
        self.v[X_idx, 0] = -self.v[X_idx, 0]

        # for Y axis
        Y_idx = np.logical_or(self.x[:, 1] < 0, self.x[:, 1] > self.NY)
        self.v[Y_idx, 1] = -self.v[Y_idx, 1]

        # for Z axis
        Z_idx = np.logical_or(self.x[:, 2] < 0, self.x[:, 2] > self.NZ)
        self.v[Z_idx, 2] = -self.v[Z_idx, 2]

    def find_collided_pairs(self) -> np.ndarray:
        addr = (self.x // self.GRID_SIZE).astype(int)
        addr = addr[:, 0] + 100 * addr[:, 1] + 10000 * addr[:, 2]

        Idx = np.stack([addr, np.arange(self.N)], axis=1)
        np.random.shuffle(Idx)
        Idx = Idx[np.argsort(Idx[:, 0])]
        Mask = (Idx[:-1, 0] - Idx[1:, 0]) == 0

        # apply Mask which indicates same address
        ij = np.stack([Idx[:-1, 1], Idx[1:, 1]], axis=1)[Mask]

        # apply collision probability
        np.random.shuffle(ij)
        ij = ij[:int(len(ij) * self.COLLISION_PROBABILITY)]

        # check that each pair have right address
        assert (addr[ij[:, 0]] == addr[ij[:, 1]]).all(), 'Some pair has inconsistent address'

        # it could be done via Mask[1:] - Mask[-1:]
        ij = ij[~np.isin(ij[:, 0], ij[:, 1])]

        E1 = self.get_system_full_energy()

        assert (max(self.E0, E1) / min(self.E0, E1)) < 1.001, 'Energy difference is too huge'

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