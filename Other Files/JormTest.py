from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import OptimizationProblem, SolverExitCondition


def near(expected, actual, tolerance):
    return abs(expected - actual) < tolerance


def test_maximize():
    problem = OptimizationProblem()

    x = problem.decision_variable()
    x.set_value(1.0)

    y = problem.decision_variable()
    y.set_value(1.0)

    problem.maximize(50 * x + 40 * y)

    problem.subject_to(x + 1.5 * y <= 750)
    problem.subject_to(2 * x + 3 * y <= 1500)
    problem.subject_to(2 * x + y <= 1000)
    problem.subject_to(x >= 0)
    problem.subject_to(y >= 0)

    status = problem.solve(diagnostics=True)

    print(x.value(), y.value())

test_maximize()