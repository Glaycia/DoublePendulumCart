from casadi import MX, pi, sin, cos, sqrt, vertcat, Opti, solve
import casadi as np
import casadi

def RungeKutta4(dyn, x, u, dT):
    k1 = dyn(x, u)
    k2 = dyn(x + dT * 0.5 * k1, u)
    k3 = dyn(x + dT * 0.5 * k2, u)
    k4 = dyn(x + dT * k3, u)

    return x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def ProjectileDynamics(X: MX, U: MX):
    # X = [x, y, vx, vy]
    g = 9.81  # acceleration due to gravity
    mass = 0.23  # mass of the projectile
    drag_coefficient = 0.08  # quadratic drag coefficient

    X_dot = MX(4, 1)

    X_dot[0, 0] = X[2, 0]
    X_dot[1, 0] = X[3, 0]

    X_dot[2, 0] = -drag_coefficient / mass * X[2, 0] ** 2
    X_dot[3, 0] = -g - drag_coefficient / mass * X[3, 0] ** 2

    return X_dot

def direct_transcription(target_x, target_y, N = 50):
    problem = casadi.Opti()

    T = problem.variable(1)
    dt = T/N # s

    x_init = 0 # m
    y_init = 0 # m

    X = problem.variable(4, N + 1)

    # Control Variable    
    U = problem.variable(2) # initial[v, theta]

    # Initial constraints
    problem.subject_to(T > 0)
    problem.subject_to(X[0, 0] == x_init)
    problem.subject_to(X[1, 0] == y_init)
    problem.subject_to(X[2, 0] == U[0, 0])
    problem.subject_to(X[3, 0] == U[1, 0])
    
    # Final Constraints
    problem.subject_to(X[0, N] == target_x)
    problem.subject_to(X[1, N] == target_y)

    #problem.subject_to(X[3, N] > 0) # Must be traveling up still

    # Dynamics constraints - RK4 integration
    for k in range(N):
        problem.subject_to(
            X[:, k+1 : k+2] ==
            RungeKutta4(ProjectileDynamics, X[:, k : k + 1], 0, dt)
        )
    
    # Minimize sum squared inputs
    J = 0
    J += U[0, 0]**2
    J += U[1, 0]**2
    J += T
    
    problem.minimize(J)

    problem.solver('ipopt')
    solution = problem.solve()

    #print(problem.debug.value(X))
    #problem.debug.show_infeasibilities()
    return solution.value(X), solution.value(U), N, solution.value(T)/N

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_trajectory(X, U, N, dt):
    fig, ax = plt.subplots()
    
    v = round(sqrt(U[0] ** 2 + U[1] ** 2), 3)
    T = round(N * dt, 3)
    title = 'V = {} m/s, T = {} s'.format(v, T)
    
    ax.set_title(title)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    target_point, = ax.plot([], [], 'ro', label='Target Point')
    projectile_traj, = ax.plot([], [], label='Projectile Trajectory')

    ax.legend()
    ax.grid(True)

    x_min, x_max = np.min(X[0, :]), np.max(X[0, :])
    y_min, y_max = np.min(X[1, :]), np.max(X[1, :])

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    def update(frame):
        projectile_traj.set_data(X[0, :frame+1], X[1, :frame+1])
        target_point.set_data(X[0, -1], X[1, -1])

    ani = FuncAnimation(fig, update, interval=dt * 1000, repeat=False)
    #ani.save('projectile.gif', writer='pillow', fps= 30) # int(1/dt))
    plt.show()

def main():
    X, U, N, dt = direct_transcription(10, 2)

    animate_trajectory(X, U, N, dt)

if __name__ == "__main__":
    main()