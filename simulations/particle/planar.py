import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from collections import defaultdict

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 10
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Bouncing Balls in a Box with Gravity')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
XENON_COLOR = (30, 144, 255)
CRYPTON_COLOR = (255, 144, 30)
BOX_COLOR = (105, 105, 105)

# Gravity constant (pixels per frame squared)
GRAVITY = 0.5

class Particle:
    def __init__(self, x, y, radius, speed_x, speed_y, mass=1.0, dt=1.0):
        self.whit = False
        self.collided = False
        self.mass = mass
        self.dt = dt
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = speed_x
        self.vy = speed_y
        self.E_init = self.mass*(HEIGHT - self.y)*GRAVITY + self.mass*0.5*(self.vx**2 + self.vy**2)

    # get full particle energy
    def full_particle_energy(self):
        return self.mass * (HEIGHT - self.y) * GRAVITY + self.mass * 0.5 * (self.vx ** 2 + self.vy ** 2)

    # Check energy conservation valid only in case with no collisions
    def check_particle_energy(self):
        E_cur = self.full_particle_energy()
        assert max(self.E_init, E_cur)/min(self.E_init, E_cur) < 1.001, f"E_init:{self.E_init} vs E_cur:{E_cur}"

    def random_deflect(self):
        mag = np.sqrt(self.vx**2 + self.vy**2)
        self.vx += self.vx*np.clip(0.2*np.random.randn(), a_min=-0.3, a_max=0.3)
        self.vy += self.vy*np.clip(0.2*np.random.randn(), a_min=-0.3, a_max=0.3)
        nmag = np.sqrt(self.vx**2 + self.vy**2)
        self.vx *= mag/nmag
        self.vy *= mag/nmag

    # making address for collider matrix
    def collide_address(self, safe_border=14.0) -> Optional[Tuple[int, int]]:
        if self.x - safe_border < 0 or\
           self.x + safe_border > WIDTH or\
           self.y - safe_border < 0 or\
           self.y + safe_border > HEIGHT:
            return None
        grid_x = int(self.x)//GRID_SIZE
        grid_y = int(self.y)//GRID_SIZE
        return grid_x, grid_y

    @staticmethod
    def process_collision(p1, p2) -> None:
        E_p1_t0 = p1.full_particle_energy()
        E_p2_t0 = p2.full_particle_energy()

        v1 = np.array([p1.vx, p1.vy])
        v2 = np.array([p2.vx, p2.vy])
        x1 = np.array([p1.x, p1.y])
        x2 = np.array([p2.x, p2.y])
        m1 = p1.mass
        m2 = p2.mass

        n = (x2 - x1)/np.linalg.norm(x2 - x1)
        v1_n = v1 - 2*(m2/(m1 + m2))*np.dot(v1 - v2, n)*n
        v2_n = v2 - 2*(m1/(m1 + m2))*np.dot(v2 - v1, n)*n

        p1.vx = v1_n[0]
        p1.vy = v1_n[1]
        p2.vx = v2_n[0]
        p2.vy = v2_n[1]

        E_p1_t1 = p1.full_particle_energy()
        E_p2_t1 = p2.full_particle_energy()

        assert abs((E_p1_t0 + E_p2_t0) - (E_p1_t1 + E_p2_t1)) < 0.0001 * (E_p1_t0 + E_p2_t0), 'Energy is not conserved!'

    def move(self) -> None:
        # Leapfrog integration:
        # Update position by full step using half-step velocity
        self.x += self.vx*self.dt
        self.y += self.vy*self.dt + 0.5*GRAVITY*(self.dt**2)
        self.vy += GRAVITY*self.dt

        if self.whit:
            self.whit = False
            self.random_deflect()

        # Boundary collision checks with walls (elastic bounce):
        # Left and right walls
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
            self.whit = True
        elif self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -self.vx
            self.whit = True

        # Top wall
        if self.y - self.radius < 0:
            #self.y = self.radius
            self.vy = -self.vy
            self.whit = True

        # Bottom wall
        if self.y + self.radius > HEIGHT:
            #self.y = HEIGHT - self.radius
            self.vy = -self.vy
            self.whit = True

        #self.check_particle_energy()

    def draw(self, screen):
        pygame.draw.circle(screen,
                           XENON_COLOR if (self.mass > 1.0) else CRYPTON_COLOR,
                           (int(self.x), int(self.y)), self.radius)

# Create a list of balls with random starting positions and velocities
balls = []
for i in range(400):
    radius = 2
    x = random.randint(radius, WIDTH - radius)
    y = random.randint(int(0.8*HEIGHT), HEIGHT - 3*radius)  # Spawn higher so gravity effect is visible
    speed_x = 4*np.random.randn()
    speed_y = 4*np.random.randn()
    balls.append(Particle(x, y, radius, speed_x, speed_y, mass=0.8 if (i % 2 == 0) else 1.31, dt=1.0))

clock = pygame.time.Clock()
running = True

data = []
equ_ticks = 400 # equilibrium
max_ticks = 1000 # overall ticks
COLLISION_PROBABILITY = 0.6
cur_tick = 0

while running:
    cur_tick += 1
    if cur_tick > max_ticks:
        running = False

    clock.tick(60)  # Limit FPS to 60
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Draw box border
    pygame.draw.rect(screen, BOX_COLOR, (0, 0, WIDTH, HEIGHT), 5)

    # collision matrix initialization
    collide_matrix = defaultdict(list)

    # Move and draw each ball
    for ball in balls:
        # first make system enter equilibrium state
        if cur_tick > equ_ticks:
            data.append([HEIGHT - ball.y, ball.mass*(ball.vx**2 + ball.vy**2)/2, ball.collided])
        ball.collided = False
        ball.move()

        # form collide matrix
        address = ball.collide_address()
        if address:
            collide_matrix[address].append(ball)

        ball.draw(screen)

    for address in collide_matrix:
        candidates = collide_matrix[address]

        # essential part
        pairs_num = int(len(candidates) * COLLISION_PROBABILITY) // 2

        if pairs_num > 0:
            pairs = random.sample(candidates, 2*pairs_num)
            for j in range(0, 2*pairs_num, 2):
                # only particles with different masses collision is useful
                if pairs[j].mass != pairs[j+1].mass:
                    Particle.process_collision(pairs[j], pairs[j+1])
                    pairs[j].collided = True
                    pairs[j+1].collided = True

    pygame.display.flip()

pygame.quit()

data = np.array(data)
h   = data[:, 0]
E_k = data[:, 1]
cld = data[:, 2]

H_min = np.quantile(h, q=0.05)
H_max = np.quantile(h, q=0.95)
Levels = np.linspace(start=H_min, stop=H_max, num=5)
H_refs = 0.5 * (Levels[:-1] + Levels[1:])
E_k_refs = []
for i in range(len(Levels)-1):
    E_k_refs.append(E_k[np.logical_and(h > Levels[i], h < Levels[i+1])].mean())

plt.scatter(h[cld == 0], E_k[cld == 0], s=1.0)
plt.scatter(h[cld > 0], E_k[cld > 0], s=3.0, c='green')
plt.plot(H_refs, E_k_refs, color='red')
plt.scatter(H_refs, E_k_refs, c='red', s=3.0)
plt.xlabel("Height")
plt.ylabel("Kinetic energy/Temperature")
plt.show()