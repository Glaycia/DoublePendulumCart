from jormungandr.optimization import OptimizationProblem
import numpy as np

def seperating_hyperplane():
    problem = OptimizationProblem()

    size = 10
    # Random Convex Hull
    hull_points = size * np.random.rand(25, 2)
    centerpoint = np.array([size/2, size/2])
    
    point_position = problem.decision_variable(2)
    # Initial Guess
    point_position[0:2, 0] = 0

    # Seperating Hyperplane
    norm_vector = problem.decision_variable(2)
    b = problem.decision_variable(1)

    # problem.subject_to(norm_vector > -1)
    # problem.subject_to(norm_vector < 1)
    problem.subject_to(norm_vector.T @ point_position + b > 0.001)

    for index in range(len(hull_points)):
        problem.subject_to(norm_vector.T @ hull_points[index] + b < 0)
    
    diff = centerpoint - point_position
    problem.minimize((diff.T @ diff))

    status = problem.solve(diagnostics=True)

    # Print Points
    # print(point_position.value())

    # x_components_string = ', '.join(map(str, hull_points[:, 0]))
    # y_components_string = ', '.join(map(str, hull_points[:, 1]))

    # desmos_x = "X_p=[" + x_components_string + "]"
    # desmos_y = "Y_p=[" + y_components_string + "]"
    # print(desmos_x, "\n", desmos_y)
if __name__ == "__main__":
    seperating_hyperplane()
