from fock import IonTrapSetup
from optim_record import OptimizationRecord
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from time import process_time
from utils import get_ion_state_generators
import matplotlib.pyplot as plt

NELDER_MEAD = 'nelder-mead'
BFGS = 'BFGS'
POWERLL = 'Powell'

num_focks = 30
g, e = get_ion_state_generators(num_focks)

init_state = (g(0)).unit()
target_state = (e(2)+g(5)).unit()
num_steps = 2
alpha_list = [1, 0, 0]
setup = IonTrapSetup(init_state, target_state, num_focks, num_steps, alpha_list=alpha_list)

init_param_vec = setup.init_param_vec()
start = process_time()
optim_result = minimize(setup.target_func, init_param_vec,
                        jac=setup.gradient,
                        args=(init_state, target_state),
                        options={'disp': True, 'return_all': True}, method=BFGS)
end = process_time()
print('CPU time used: {}'.format(end - start))

record = OptimizationRecord(setup, optim_result)
record.save()
record.plot_history(save_plot=True)
record.plot_final_dynamics(save_plot=True)
record.save()
