import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def distance(x1, y1, x2, y2):
    return jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

@jax.jit
def update_step(carry, _):
    F, cxs, cys, weights, tau = carry

    # Boundary conditions
    F = F.at[:, -1, [6, 7, 8]].set(F[:, -2, [6, 7, 8]])
    F = F.at[:, 0, [2, 3, 4]].set(F[:, 1, [2, 3, 4]])

    # Streaming step
    for i, cx, cy in zip(range(len(cxs)), cxs, cys):
        F = F.at[..., i].set(jnp.roll(F[..., i], cx, axis=1))
        F = F.at[..., i].set(jnp.roll(F[..., i], cy, axis=0))

    # Macroscopic variables
    rho = jnp.sum(F, axis=2)
    ux = jnp.sum(F * cxs, axis=2) / rho
    uy = jnp.sum(F * cys, axis=2) / rho

    # Apply cylinder boundary conditions
    ux = ux.at[cylinder_mask].set(0.)
    uy = uy.at[cylinder_mask].set(0.)

    # Particles collision
    F_eq = jnp.zeros(F.shape)
    for i, cx, cy, w in zip(range(len(cxs)), cxs, cys, weights):
        F_eq = F_eq.at[..., i].set(rho * w * (
                1 + 3 * (cx * ux + cy * uy) + (9 / 2) * (cx * ux + cy * uy) ** 2 - (3 / 2) * (ux ** 2 + uy ** 2)
        ))
    F = F - (1 / tau) * (F - F_eq)

    return (F, cxs, cys, weights, tau), None


def visualize(F):
    rho = jnp.sum(F, axis=2)
    ux = jnp.sum(F * cxs, axis=2) / rho
    uy = jnp.sum(F * cys, axis=2) / rho
    plt.imshow(jnp.sqrt(ux ** 2 + uy ** 2))
    #plt.colorbar()
    plt.pause(0.01)
    plt.cla()

def main():
    global cylinder_mask, cxs, cys, weights  # Make these global for visualization

    Nx = 400
    Ny = 100
    tau = .53
    Nt = 40000
    vis_interval = 50  # Visualize every 100 iterations

    # Lattice speeds and weights
    NL = 9
    cxs = jnp.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = jnp.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = jnp.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

    # Initial conditions
    key = jax.random.PRNGKey(0)
    F = jnp.ones((Ny, Nx, NL)) + 0.01 * jax.random.normal(key, (Ny, Nx, NL))
    F = F.at[..., 3].set(2.3)

    # Create cylinder mask
    x_grid, y_grid = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny))
    cylinder_mask = distance(Nx // 4, Ny // 2, x_grid, y_grid) < 13
    cylinder_mask = jnp.array(cylinder_mask, dtype=bool)

    # Initialize carry
    carry = (F, cxs, cys, weights, tau)

    # Run simulation in chunks with visualization
    num_chunks = Nt // vis_interval
    for chunk in range(num_chunks):
        # Run vis_interval steps
        carry, _ = jax.lax.scan(update_step, carry, None, length=vis_interval)

        # Visualization
        current_F = carry[0]
        visualize(current_F)
        print(f"Iteration: {(chunk + 1) * vis_interval}")

    # Final visualization
    final_F = carry[0]
    rho = jnp.sum(final_F, axis=2)
    ux = jnp.sum(final_F * cxs, axis=2) / rho
    uy = jnp.sum(final_F * cys, axis=2) / rho
    plt.imshow(jnp.sqrt(ux ** 2 + uy ** 2))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()