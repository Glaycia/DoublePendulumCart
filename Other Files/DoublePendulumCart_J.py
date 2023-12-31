from jormungandr.optimization import OptimizationProblem, SolverExitCondition
from jormungandr.autodiff import VariableMatrix, ExpressionType
import jormungandr.autodiff as autodiff
import numpy as np
import math

def RungeKutta4(dyn, x, u, dT):
    k1 = dyn(x, u)
    k2 = dyn(x + dT * 0.5 * k1, u)
    k3 = dyn(x + dT * 0.5 * k2, u)
    k4 = dyn(x + dT * k3, u)

    return x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def DoublePendulumCartDynamics(X = VariableMatrix, U = VariableMatrix):
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

    M = VariableMatrix(3, 3)
    M[0, 0] = m_c + m_p1 + m_p2
    M[0, 1] = l_1 * (m_p1 + m_p2) * autodiff.cos(theta1)
    M[0, 2] = l_2 * m_p2 * autodiff.cos(theta2)

    M[1, 0] = l_1 * (m_p1 + m_p2) * autodiff.cos(theta1)
    M[1, 1] = I_1 + l_1**2 * (m_p1 + m_p2)
    M[1, 2] = l_1 * l_2 * m_p2 * autodiff.cos(theta1 - theta2)

    M[2, 0] = l_2 * m_p2 * autodiff.cos(theta2)
    M[2, 1] = l_1 * l_2 * m_p2 * autodiff.cos(theta1 - theta2)
    M[2, 2] = I_2 + l_2**2 * m_p2

    M_Adj = VariableMatrix(3, 3)

    M_Adj[0, 0] = (M.T[1, 1] * M.T[2, 2] - M.T[2, 1] * M.T[1, 2])
    M_Adj[0, 1] = -(M.T[1, 0] * M.T[2, 2] - M.T[2, 0] * M.T[1, 2])
    M_Adj[0, 2] = (M.T[1, 0] * M.T[2, 1] - M.T[2, 0] * M.T[1, 1])

    M_Adj[1, 0] = -(M.T[0, 1] * M.T[2, 2] - M.T[2, 1] * M.T[0, 2])
    M_Adj[1, 1] = (M.T[0, 0] * M.T[2, 2] - M.T[2, 0] * M.T[0, 2])
    M_Adj[1, 2] = -(M.T[0, 0] * M.T[2, 1] - M.T[2, 0] * M.T[0, 1])

    M_Adj[2, 0] = (M.T[0, 1] * M.T[1, 2] - M.T[1, 1] * M.T[0, 2])
    M_Adj[2, 1] = -(M.T[0, 0] * M.T[1, 2] - M.T[1, 0] * M.T[0, 2])
    M_Adj[2, 2] = (M.T[0, 0] * M.T[1, 1] - M.T[1, 0] * M.T[0, 1])

    # Minv = M_Adj / (M_Adj[0, 0] + M_Adj[0, 1] + M_Adj[0, 2])

    Minv = VariableMatrix(3, 3)
    M_det = (M_Adj[0, 0] + M_Adj[0, 1] + M_Adj[0, 2])
    for row in range(3):
        for column in range(3):
            Minv[row, column] = M_Adj[row, column] / M_det

    F = VariableMatrix(3, 1)

    F[0, 0] = (l_1 * (m_p1 + m_p2) * theta1_dot**2 * autodiff.sin(theta1) + m_p2 * l_2 * theta2_dot**2 * autodiff.sin(theta2)) - d_c * x_dot + u
    F[1, 0] = (-l_1 * l_2 * m_p2 * theta2_dot**2 * autodiff.sin(theta1 - theta2) + g * (m_p1 + m_p2) * l_1 * autodiff.sin(theta1)) - d_p1 * theta1_dot
    F[2, 0] = (l_1 * l_2 * m_p2 * theta2_dot**2 * autodiff.sin(theta1 - theta2) + g * l_2 * m_p2 * autodiff.sin(theta2)) - d_p2 * theta2_dot

    X_dot = VariableMatrix(6, 1)
    X_dot[:3, 0] = X[3:, :]
    X_dot[3:, 0] = (Minv @ F)[:, 0]

    return X_dot

def direct_transcription():
    T = 1.7 #s
    dt = 0.02 # s
    N = int(T / dt)

    u_max = 20 # N

    d_init = 1 # m
    theta1_init = 0 # rad
    theta2_init = math.pi # rad 😎

    d_final = 1 # m
    theta1_final = 0 # rad
    theta2_final = 0 # rad

    d_min = 0 # m
    d_max = 2 # m

    problem = OptimizationProblem()

    X = problem.decision_variable(6, N + 1)

    for k in range(N):
        X[0, k].set_value(float(k/N) * (d_final - d_init) + d_init)
        X[1, k].set_value(float(k/N) * (theta1_final - theta1_init) + theta1_init)
        X[2, k].set_value(float(k/N) * (theta2_final - theta2_init) + theta2_init)
    
    U = problem.decision_variable(1, N)

    problem.subject_to(X[0, 0] == d_init)
    problem.subject_to(X[1, 0] == theta1_init)
    problem.subject_to(X[2, 0] == theta2_init)

    problem.subject_to(X[3, 0] == 0)
    problem.subject_to(X[4, 0] == 0)
    problem.subject_to(X[5, 0] == 0)

    problem.subject_to(X[0, N] == d_final)
    problem.subject_to(X[1, N] == theta1_final)
    problem.subject_to(X[2, N] == theta2_final)

    problem.subject_to(X[3, N] == 0)
    problem.subject_to(X[4, N] == 0)
    problem.subject_to(X[5, N] == 0)

    problem.subject_to(X[0, :] >= d_min)
    problem.subject_to(X[0, :] <= d_max)

    problem.subject_to(U >= -u_max)
    problem.subject_to(U <= u_max)

    # Dynamics constraints - RK4 integration
    for k in range(N):
        problem.subject_to(
            X[:, k+1 : k+2] ==
            RungeKutta4(DoublePendulumCartDynamics, X[:, k : k + 1], U[:, k : k + 1], dt)
        )
    
    # Minimize sum squared inputs
    J = 0.0
    for k in range(N):
        J += U[:, k : k + 1].T @ U[:, k : k + 1]
    problem.minimize(J)

    status = problem.solve(diagnostics=True)

direct_transcription()