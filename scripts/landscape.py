from fock import IonTrapSetup
from utils import get_ion_state_generators
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import norm

num_repeats = 100
num_focks = 20
g, e = get_ion_state_generators(num_focks)

init_state = g(0)
target_state = (g(4)).unit()
num_steps = 4
alpha_list = [1, 0, 0]

fig, ax = plt.subplots(1, 1)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("target func")
ax.set_ylabel("grad norm")
ax.grid()


for j in range(1, 10):
    func_list = []
    grad_norm_list = []

    for i in range(num_repeats):
        setup = IonTrapSetup(init_state, target_state, num_focks, num_steps, alpha_list=alpha_list)
        param_vec = setup.init_param_vec() * j
        func_list.append(setup.target_func(param_vec, init_state, target_state))
        grad_norm_list.append(norm(setup.gradient(param_vec, init_state, target_state)))

    ax.scatter(func_list, grad_norm_list, alpha=0.5, label=str(j))
ax.legend()
