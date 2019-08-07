from fock import PiecewiseConstant
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from time import process_time
from utils import plot_dynamics, get_ion_state_generators
import matplotlib.pyplot as plt

NELDER_MEAD = 'nelder-mead'
BFGS = 'BFGS'
POWERLL = 'Powell'

num_focks = 15
g, e = get_ion_state_generators(num_focks)

init_state = g(0)
target_state = (e(0)).unit()
num_steps = 1
alpha_1 = 1e-1
param = PiecewiseConstant(num_focks, num_steps, alpha_1=alpha_1)

func_hist = []
grad_hist = []


def store_hist(xk):
    func_hist.append(param.target_func(xk, init_state, target_state))
    grad_hist.append(norm(param.gradient(xk, init_state, target_state)))


start = process_time()
opt_result = minimize(param.target_func, param.init_param_vec(),
                      jac=param.gradient,
                      args=(init_state, target_state),
                      callback=store_hist,
                      options={'disp': True}, method=BFGS)
end = process_time()

opt_param_vec = opt_result.x  # optimized parameter vector

print('CPU time used: {}'.format(end - start))
param.parse_parameters(opt_param_vec)
fig, axes = plt.subplots(2, 1, sharex='col')
ax1, ax2 = axes

ax1.plot(range(len(func_hist)), func_hist)
ax1.scatter(range(len(func_hist)), func_hist, marker='x')
ax2.plot(range(len(grad_hist)), grad_hist)
ax2.scatter(range(len(grad_hist)), grad_hist, marker='x')

ax1.set_ylabel('target function')
ax2.set_ylabel('gradient norm')
ax2.set_xlabel('iter')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.grid()
ax2.grid()
fig.show()

plot_dynamics(opt_param_vec, param, init_state, target_state, num_focks)
