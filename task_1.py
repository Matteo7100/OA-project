import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

# load target
target = np.load("target_1.npy")

# define time horizon
T = len(target[0])

# define system dynamics
A = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 0.8, 0], [0, 0, 0, 0.8]])

B = np.array([[0, 0], [0, 0], [0.1, 0], [0, 0.1]])

E = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# create cp variables
x = cp.Variable((4, T))
u = cp.Variable((2, T - 1))


# Tracking Error
TE = cp.sum([cp.norm(E @ x[:, t] - target[:, t], 2) for t in range(T)])

# Control Effort
CE = cp.sum_squares(u[:, : T - 1])

# define constraints
constraints = [x[:, 0] == np.array([0.5, 0, 1, -1])]

for t in range(T - 1):
    constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

# result storage
results = []

# define objective
rho = 10
TE_results = []
CE_results = []
for rho in [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
    objective = cp.Minimize(TE + rho * CE)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    TE_results.append(TE.value)
    CE_results.append(CE.value)
    results.append((rho, x[0].value, x[1].value))

# plot tracking results
plt.figure()
for rho, x1, x2 in results:
    plt.scatter(x1, x2, label=f"rho={rho}")  # scatter is better for points

# plot target as a red star
plt.scatter(target[0], target[1], color="red", marker="*", s=150, label="target")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Solutions for different rho values")
plt.legend()
plt.savefig("solutions_plot.png", dpi=500)  # can also use .pdf, .svg, etc.

# plot TE vs CE
plt.figure()
plt.scatter(CE_results, TE_results)

plt.xlabel("Control Effort (CE)")
plt.ylabel("Tracking Error (TE)")
plt.title("TE vs CE for different rho values")
plt.savefig("TE_vs_CE_plot.png", dpi=500)
plt.show()
