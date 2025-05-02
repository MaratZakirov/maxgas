import torch
import matplotlib.pyplot as plt
import time

def distance(x1, y1, x2, y2):
    return torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main():
    # Configuration
    Nx, Ny = 400, 100
    tau = 0.53
    Nt = 10000
    vis_interval = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Lattice parameters
    NL = 9
    cxs = torch.tensor([0, 0, 1, 1, 1, 0, -1, -1, -1], device=device)
    cys = torch.tensor([0, 1, 1, 0, -1, -1, -1, 0, 1], device=device)
    weights = torch.tensor([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36], device=device)

    # Initialize populations (Ny, Nx, NL)
    F = torch.ones((Ny, Nx, NL), device=device) + 0.01 * torch.randn((Ny, Nx, NL), device=device)
    F[..., 3] = 2.3

    # Cylinder mask (Ny, Nx)
    y_grid, x_grid = torch.meshgrid(torch.arange(Ny, device=device),
                                    torch.arange(Nx, device=device),
                                    indexing='ij')  # Note 'ij' indexing
    cylinder_mask = distance(Nx // 4, Ny // 2, x_grid, y_grid) < 13

    # Main loop
    def update(F):
        with torch.no_grad():
            # Boundary conditions
            F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
            F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

            # Streaming
            for i, cx, cy in zip(range(NL), cxs, cys):
                F[..., i] = torch.roll(F[..., i], shifts=(int(cy), int(cx)), dims=(0, 1))

            # Macroscopic variables
            rho = torch.sum(F, dim=2)  # shape (Ny, Nx)
            ux = torch.sum(F * cxs, dim=2) / rho
            uy = torch.sum(F * cys, dim=2) / rho

            # Apply boundaries - now shapes match (Ny, Nx)
            ux[cylinder_mask] = 0
            uy[cylinder_mask] = 0

            # Collision
            F_eq = torch.zeros_like(F)
            for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
                F_eq[..., i] = rho * w * (
                        1 + 3 * (cx * ux + cy * uy) + 4.5 * (cx * ux + cy * uy) ** 2 - 1.5 * (ux ** 2 + uy ** 2)
                )
            F += -(1.0 / tau) * (F - F_eq)
        return F

    # Run simulation
    start_time = time.time()
    plt.figure(figsize=(10, 4))
    for it in range(Nt):
        F = update(F)

        if it % vis_interval == 0:
            # Visualization
            rho = torch.sum(F, dim=2)
            ux = torch.sum(F * cxs, dim=2) / rho
            uy = torch.sum(F * cys, dim=2) / rho
            vel = torch.sqrt(ux ** 2 + uy ** 2).cpu().numpy()

            plt.imshow(vel, cmap='jet')
            plt.title(f"Iteration {it}")
            #plt.colorbar()
            plt.pause(0.01)
            plt.cla()

    print(f"Completed in {time.time() - start_time:.2f} seconds")
    plt.show()


if __name__ == '__main__':
    main()