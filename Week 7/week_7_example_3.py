import cvxpy as cp
import numpy as np

"""
Least Squares problem using CVXPY

https://www.cvxpy.org/ 
"""

# Problem data.
m = 30   #  number of constaints
n = 20   # number of variables
np.random.seed(1)

#  constraints parameters Ax <= b
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve(solver='OSQP')
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)
