# This code is somewhat simplification of cavity flow demo
# Machine Learning & Simulation https://youtu.be/BQLvNLgMTQE?si=ZNiTHGo0zmhWJ9nU
# https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lid_driven_cavity_python_simple.py

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_POINTS = 41
DOMAIN_SIZE = 1.0
N_ITERATIONS = 5000

dt_ = 0.001
nu_ = 0.1
rho_ = 1.0
dx_ = DOMAIN_SIZE / (N_POINTS - 1)
dy_ = DOMAIN_SIZE / (N_POINTS - 1)

HORIZONTAL_VEL_TOP = 1.0
PP_ITERATIONS = 50

def main():
    x = np.linspace(0., DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0., DOMAIN_SIZE, N_POINTS)

    X, Y = np.meshgrid(x, y)

    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)

    def central_difference_x(f):
        r = np.zeros_like(f)
        r[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2])/2/dx_
        return r

    def central_difference_y(f):
        r = np.zeros_like(f)
        r[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1])/2/dy_
        return r

    def laplace(f):
        r = np.zeros_like(f)
        r[1:-1, 1:-1] = (
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1])/dx_**2
            + (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2])/dy_**2
        )
        return r

    maximum_possible_time_step_length = 0.5 * dx_ * dy_ / nu_
    if dt_ > 0.5 * maximum_possible_time_step_length:
        raise RuntimeError("Stability is not guarenteed")

    for _ in tqdm(range(N_ITERATIONS)):
        du_dx_prev = central_difference_x(u_prev)
        du_dy_prev = central_difference_y(u_prev)
        dv_dx_prev = central_difference_x(v_prev)
        dv_dy_prev = central_difference_y(v_prev)

        laplace_u_prev = laplace(u_prev)
        laplace_v_prev = laplace(v_prev)

        # Perform tentative step for moment equation without pressure account
        u_tent = u_prev + dt_ * (-(u_prev * du_dx_prev + v_prev * du_dy_prev) + nu_ * laplace_u_prev)
        v_tent = v_prev + dt_ * (-(u_prev * dv_dx_prev + v_prev * dv_dy_prev) + nu_ * laplace_v_prev)

        # Velocity boundary conditions Dirichlet BC everywhere except top
        u_tent[:, 0] = 0 # left x-velocity is zero
        u_tent[:, -1] = 0 # right x-velocity is zero
        u_tent[0, :] = 0 # bottom x-velocity is zero
        u_tent[-1, :] = HORIZONTAL_VEL_TOP # top x-velocity is 1.0

        v_tent[:, 0] = 0 # left y-velocity is zero
        v_tent[:, -1] = 0 # right y-velocity is zero
        v_tent[0, :] = 0 # bottom y-velocity is zero
        v_tent[-1, :] = 0 # top y-velocity is zero

        du_dx_tent = central_difference_x(u_tent)
        dv_dy_tent = central_difference_y(v_tent)

        # Perform a pressure correction by solving pressure poisson equation
        rhs = rho_ / dt_ * (du_dx_tent + dv_dy_tent)

        for _ in range(PP_ITERATIONS):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1/4 * (
                p_prev[:-2, 1:-1] + p_prev[2:, 1:-1] + p_prev[1:-1, :-2] + p_prev[1:-1, 2:]
                -
                dx_ * dy_ * rhs[1:-1, 1:-1]
            )

            # Pressure boundary conditions: Homogeneous Neumann BC everywhere except the top
            # where it is Dirichlet Homogeneous BC
            p_next[:, -1] = p_next[:, -2] # right side
            p_next[:, 0] = p_next[:, 1] # left side
            p_next[0, :] = p_next[1, :] # bottom side
            p_next[-1, :] = 0. # top side

            p_prev = p_next

        dp_dx_next = central_difference_x(p_prev)
        dp_dy_next = central_difference_y(p_prev)

        # Correct velocities so fluid stays incompressible
        u_next = u_tent - dt_/rho_ * dp_dx_next
        v_next = v_tent - dt_/rho_ * dp_dy_next

        # Velocity boundary conditions Dirichlet BC everywhere except top
        u_next[:, 0] = 0 # left x-velocity is zero
        u_next[:, -1] = 0 # right x-velocity is zero
        u_next[0, :] = 0 # bottom x-velocity is zero
        u_next[-1, :] = HORIZONTAL_VEL_TOP # top x-velocity is 1.0

        v_next[:, 0] = 0 # left y-velocity is zero
        v_next[:, -1] = 0 # right y-velocity is zero
        v_next[0, :] = 0 # bottom y-velocity is zero
        v_next[-1, :] = 0 # top y-velocity is zero

        # Advance in time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next

    # The [::2, ::2] selects only every second entry (less cluttering plot)
    plt.style.use("dark_background")
    plt.figure()
    plt.contourf(X, Y, p_next, cmap="coolwarm")
    plt.colorbar()

    #plt.quiver(X, Y, u_next, v_next, color="black")
    plt.streamplot(X, Y, u_next, v_next, color="black")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()

if __name__ == '__main__':
    main()