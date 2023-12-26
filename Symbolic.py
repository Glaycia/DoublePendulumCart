from sympy import symbols, Function, diff, cos, sin, simplify, latex, nsimplify, Matrix

# Symbols
t, F = symbols('t F')
M, m1, m2, l1, l2, g, I1, I2 = symbols('M m1 m2 l1 l2 g I1 I2')
x, theta1, theta2 = symbols('x theta1 theta2', cls=Function, implicit=True)

# State velocities
x_dot = diff(x(t), t)
theta1_dot = diff(theta1(t), t)
theta2_dot = diff(theta2(t), t)

# Positions of the masses
x1 = x(t) + l1*sin(theta1(t))
y1 = l1*cos(theta1(t))
x2 = x1 + l2*sin(theta2(t))
y2 = y1 + l2*cos(theta2(t))

# Component-wise velocities
x1_dot = diff(x1, t)
y1_dot = diff(y1, t)
x2_dot = diff(x2, t)
y2_dot = diff(y2, t)

# Kinetic and potential energy
T = 0.5*M*x_dot**2 + 0.5*m1*(x1_dot**2 + y1_dot**2) + 0.5*m2*(x2_dot**2 + y2_dot**2) + 0.5*I1*diff(theta1(t), t)**2 + 0.5*I2*diff(theta2(t), t)**2
U = m1*g*y1 + m2*g*y2

# Include force term for the cart
L = T - U + F * x_dot

# Define generalized coordinates and velocities
y = [x(t), theta1(t), theta2(t)]
y_dot = [x_dot, diff(theta1(t), t), diff(theta2(t), t)]

# Manually extract coefficients for mass matrix and force vector
M11 = diff(diff(L, x_dot), x_dot)
M12 = diff(diff(L, x_dot), theta1_dot)
M13 = diff(diff(L, x_dot), theta2_dot)

M21 = diff(diff(L, theta1_dot), x_dot)
M22 = diff(diff(L, theta1_dot), theta1_dot)
M23 = diff(diff(L, theta1_dot), theta2_dot)

M31 = diff(diff(L, theta2_dot), x_dot)
M32 = diff(diff(L, theta2_dot), theta1_dot)
M33 = diff(diff(L, theta2_dot), theta2_dot)

MMat = [[M11, M12, M13],
        [M21, M22, M23],
        [M31, M32, M33]]

for i in range(len(MMat)):
    for j in range(len(MMat[i])):
        MMat[i][j] = nsimplify(simplify(MMat[i][j]))

# Print mass matrix and force vector
print("Mass Matrix M(y):")
print(latex(Matrix(MMat)))