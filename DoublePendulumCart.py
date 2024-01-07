from casadi import MX, pi, sin, cos, vertcat, Opti, solve
import casadi as np
import casadi

def RungeKutta4(dyn, x, u, dT):
    k1 = dyn(x, u)
    k2 = dyn(x + dT * 0.5 * k1, u)
    k3 = dyn(x + dT * 0.5 * k2, u)
    k4 = dyn(x + dT * k3, u)

    return x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def DoublePendulumCartDynamics(X = MX, U = MX):
    # X = [x, θ1, θ2]ᵀ rad
    # Ẋ = [ẋ, θ̇1, θ̇2]ᵀ rad/s
    # u = [F] N⋅m
    # 
    # M(X)Ẍ = f(y, ẏ, u)
    # Ẍ = M⁻¹(y) f(y, ẏ, u)

    m_c = 1 # Cart mass (kg)
    m_p1 = 0.05 # Pendulum mass (kg)
    m_p2 = 0.05 # Pendulum mass (kg)
    I_1 = 0.002 # Pendulum inertia (kg⋅m²)
    I_2 = 0.002 # Pendulum inertia (kg⋅m²)
    l_1 = 0.5 # Pole length (m)
    l_2 = 0.5 # Pole length (m)
    g = 9.806 # Acceleration due to gravity (m/s²)

    d_c = 0.001 # Friction coefficient, multiplied by velocities (kg/s)
    d_p1 = 0.001 # Friction coefficient, angular velocity (kg⋅m/s)
    d_p2 = 0.001 # Friction coefficient, angular velocity (kg⋅m/s)

    x = X[0, 0]
    theta1 = X[1, 0]
    theta2 = X[2, 0]
    x_dot = X[3, 0]
    theta1_dot = X[4, 0]
    theta2_dot = X[5, 0]

    u = U[0, 0]

    M = MX(3, 3)
    M[0, 0] = m_c + m_p1 + m_p2

    M[0, 1] = l_1 * (m_p1 + m_p2) * cos(theta1)
    M[0, 2] = l_2 * m_p2 * cos(theta2)

    M[1, 0] = l_1 * (m_p1 + m_p2) * cos(theta1)
    M[1, 1] = I_1 + l_1**2 * (m_p1 + m_p2)
    M[1, 2] = l_1 * l_2 * m_p2 * cos(theta1 - theta2)

    M[2, 0] = l_2 * m_p2 * cos(theta2)
    M[2, 1] = l_1 * l_2 * m_p2 * cos(theta1 - theta2)
    M[2, 2] = I_2 + l_2**2 * m_p2

    F = MX(3, 1)

    F[0, 0] = (l_1 * (m_p1 + m_p2) * theta1_dot**2 * sin(theta1) + m_p2 * l_2 * theta2_dot**2 * sin(theta2)) - d_c * x_dot + u
    F[1, 0] = (-l_1 * l_2 * m_p2 * theta2_dot**2 * sin(theta1 - theta2) + g * (m_p1 + m_p2) * l_1 * sin(theta1)) - d_p1 * theta1_dot
    F[2, 0] = (l_1 * l_2 * m_p2 * theta2_dot**2 * sin(theta1 - theta2) + g * l_2 * m_p2 * sin(theta2)) - d_p2 * theta2_dot

    X_dot = MX(6, 1)
    X_dot[:3, :] = X[3:, :]
    X_dot[3:, :] = solve(M, F)

    return X_dot

def direct_transcription():
    problem = casadi.Opti()

    T = 1.7 #s
    dt = 0.02 # s
    N = int(T / dt)

    u_max = 20 # N

    d_init = 1 # m
    theta1_init = 0 # rad 😎
    theta2_init = pi # rad

    d_final = 1 # m
    theta1_final = 0 # rad
    theta2_final = 0 # rad

    d_min = 0 # m
    d_max = 2 # m

    X = problem.variable(6, N + 1)

    # Initial Guess
    problem.set_initial(X[0, :], np.linspace(d_init, d_final, N + 1))
    problem.set_initial(X[1, :], np.linspace(theta1_init, theta1_final, N + 1))
    problem.set_initial(X[2, :], np.linspace(theta2_init, theta2_final, N + 1))

    # Control Variable    
    U = problem.variable(1, N)

    # Initial constraints
    problem.subject_to(X[0, 0] == d_init)
    problem.subject_to(X[1, 0] == theta1_init)
    problem.subject_to(X[2, 0] == theta2_init)

    problem.subject_to(X[3, 0] == 0)
    problem.subject_to(X[4, 0] == 0)
    problem.subject_to(X[5, 0] == 0)
    
    # Final Constraints
    problem.subject_to(X[0, N] == d_final)
    problem.subject_to(X[1, N] == theta1_final)
    problem.subject_to(X[2, N] == theta2_final)

    problem.subject_to(X[3, N] == 0)
    problem.subject_to(X[4, N] == 0)
    problem.subject_to(X[5, N] == 0)

    # Rail dimension constraint
    problem.subject_to(X[0, :] >= d_min)
    problem.subject_to(X[0, :] <= d_max)

    # Maximum control input constraint
    problem.subject_to(U >= -u_max)
    problem.subject_to(U <= u_max)

    # Dynamics constraints - RK4 integration
    for k in range(N):
        problem.subject_to(
            X[:, k+1 : k+2] ==
            RungeKutta4(DoublePendulumCartDynamics, X[:, k : k + 1], U[:, k : k + 1], dt)
        )
    
    # Minimize sum squared inputs
    J = 0
    for k in range(N):
        # J += (d_final - X[0, 0])**2
        # J += (theta1_final - X[1, 0])**2
        # J += (theta2_final - X[2, 0])**2

        J += U[0, 0]**2
    
    problem.minimize(J)

    problem.solver('ipopt')
    solution = problem.solve()

    #print(problem.debug.value(X))
    return solution.value(X), solution.value(U), N, dt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

prev_positions_1 = []
prev_positions_2 = []

def plot_cart_pole(ax, X, U, length=0.5, trace_length=30):
    """
    Plot cart-pole system.
    """
    cart_width = 0.2
    cart_height = 0.1
    pole_width = 0.02
    pole_length = length

    if trace_length > 1:
        prev_positions_1.append((X[0] + pole_length * np.sin(X[1]), pole_length * np.cos(X[1])))
        prev_positions_2.append((X[0] + pole_length * np.sin(X[1]) + pole_length * np.sin(X[2]), pole_length * np.cos(X[1]) + pole_length * np.cos(X[2])))
        
        if len(prev_positions_1) > trace_length:
            prev_positions_1.pop(0)
            prev_positions_2.pop(0)

        for i in range(1, len(prev_positions_1)):
            alpha = 1 # (1.0 - i / len(prev_positions_1))/2
            # ax.plot([prev_positions_1[i - 1][0], prev_positions_1[i][0]],
            #         [prev_positions_1[i - 1][1], prev_positions_1[i][1]],
            #         color='grey', alpha=alpha)
            ax.plot([prev_positions_2[i - 1][0], prev_positions_2[i][0]],
                    [prev_positions_2[i - 1][1], prev_positions_2[i][1]],
                    color='grey', alpha=alpha)
    
    # Cart
    cart_x = X[0] - cart_width / 2
    ax.add_patch(plt.Rectangle((cart_x, -cart_height), cart_width, cart_height, color='blue'))

    # Pole
    pole_1 = (X[0] + pole_length * np.sin(X[1]), pole_length * np.cos(X[1]))
    pole_2 = (X[0] + pole_length * np.sin(X[1]) + pole_length * np.sin(X[2]), pole_length * np.cos(X[1]) + pole_length * np.cos(X[2]))
    ax.plot([X[0], pole_1[0]], [0, pole_1[1]], color='red', linewidth=2)
    ax.plot([pole_1[0], pole_2[0]], [pole_1[1], pole_2[1]], color='red', linewidth=2)

    plt.xlim([-0.5, 2.5])  # Adjust as needed
    plt.ylim([-1.2, 1.2])   # Adjust as needed

def main():
    X, U, N, dt = direct_transcription()

    def update(frame):
        plt.clf()

        ax1 = plt.subplot(1, 3, 1)
        plot_cart_pole(ax1, X[:, frame], U[frame])
        ax1.set_title(f"t = {round(frame * dt, 3)}s", loc="left", x=0.45)
        ax1.grid(True)

        plot_result_live = True
        ax2 = plt.subplot(1, 3, 2)
        if(plot_result_live):
            ax2.plot(X[0, :frame+1], label='Cart (m)')
            ax2.plot(X[1, :frame+1], label='Theta 1 (rad)')
            ax2.plot(X[2, :frame+1], label='Theta 2 (rad)')
            ax2.set_title("Positions")
            ax2.legend(loc='upper right')
        else:
            ax2.set_title("Positions")
            ax2.legend(loc='upper right')
            ax2.plot(X[0, :], label='Cart (m)')
            ax2.plot(X[1, :], label='Theta 1 (rad)')
            ax2.plot(X[2, :], label='Theta 2 (rad)')

        # Plot control input
        ax3 = plt.subplot(1, 3, 3)
        if(plot_result_live):
            ax3.plot(U[:frame+1], label='Force (N)')
            ax3.legend(loc='upper right')
            ax3.set_title("Control Input")
        else:
            ax3.plot(U, label='Force (N)')
            ax3.legend(loc='upper right')
            ax3.set_title("Control Input")
        

    fig = plt.figure(figsize=(13, 5))
    plt.figtext(0.5, 0.95, 'Double Inverted Pendulum Cart Animation', ha='center', va='center', fontsize=16)

    animation = FuncAnimation(fig, update, frames=N, interval=dt * 1000, repeat=True)

    animation.save('cart_pole_animation.gif', writer='pillow', fps=1/dt)

    # plt.show()


if __name__ == "__main__":
    main()