import numpy as np
import pygame
import random
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Bouncing Balls in a Box with Gravity')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BALL_COLOR = (30, 144, 255)
BOX_COLOR = (105, 105, 105)

# Gravity constant (pixels per frame squared)
GRAVITY = 0.5

class Particle:
    def __init__(self, x, y, radius, speed_x, speed_y, mass=1.0, dt=1.0):
        self.whit = False
        self.mass = mass
        self.dt = dt
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = speed_x
        self.vy = speed_y
        self.E_init = self.mass*(HEIGHT - self.y)*GRAVITY + self.mass*0.5*(self.vx**2 + self.vy**2)

    # Check energy conservation
    def check_particle_energy(self):
        E_cur = self.mass*(HEIGHT - self.y)*GRAVITY + self.mass*0.5*(self.vx**2 + self.vy**2)
        assert max(self.E_init, E_cur)/min(self.E_init, E_cur) < 1.001, f"E_init:{self.E_init} vs E_cur:{E_cur}"

    def random_deflect(self):
        mag = np.sqrt(self.vx**2 + self.vy**2)
        self.vx += self.vx*np.clip(0.2*np.random.randn(), a_min=-0.3, a_max=0.3)
        self.vy += self.vy*np.clip(0.2*np.random.randn(), a_min=-0.3, a_max=0.3)
        nmag = np.sqrt(self.vx**2 + self.vy**2)
        self.vx *= mag/nmag
        self.vy *= mag/nmag

    def move(self):
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

        self.check_particle_energy()

    def draw(self, screen):
        pygame.draw.circle(screen, BALL_COLOR, (int(self.x), int(self.y)), self.radius)

# Create a list of balls with random starting positions and velocities
balls = []
for _ in range(100):
    radius = 2
    x = random.randint(radius, WIDTH - radius)
    y = random.randint(int(0.8*HEIGHT), HEIGHT - 3*radius)  # Spawn higher so gravity effect is visible
    speed_x = 4*np.random.randn()
    speed_y = 4*np.random.randn()
    balls.append(Particle(x, y, radius, speed_x, speed_y, mass=1.0, dt=1.0))

clock = pygame.time.Clock()
running = True

data = []
equ_ticks = 300 # equilibrium
max_ticks = 400 # overall ticks
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

    # Move and draw each ball
    for ball in balls:
        # first make system enter equilibrium state
        if cur_tick > equ_ticks:
            data.append([HEIGHT - ball.y, (ball.vx**2 + ball.vy**2)/2])
        ball.move()
        ball.draw(screen)

    pygame.display.flip()

pygame.quit()

data = np.array(data)
h   = data[:, 0]
E_k = data[:, 1]

H_min = np.quantile(h, q=0.05)
H_max = np.quantile(h, q=0.95)
Levels = np.linspace(start=H_min, stop=H_max, num=5)
H_refs = 0.5 * (Levels[:-1] + Levels[1:])
E_k_refs = []
for i in range(len(Levels)-1):
    E_k_refs.append(E_k[np.logical_and(h > Levels[i], h < Levels[i+1])].mean())

plt.scatter(h, E_k, s=1.0)
plt.plot(H_refs, E_k_refs, color='red')
plt.scatter(H_refs, E_k_refs, c='red', s=3.0)
plt.xlabel("Height")
plt.ylabel("Kinetic energy/Temperature")
plt.show()