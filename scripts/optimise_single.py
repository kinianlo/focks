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

num_focks = 15
g, e = get_ion_state_generators(num_focks)

init_state = g(0)
target_state = (e(0)).unit()
num_steps = 1
alpha_1 = 1e-1
setup = IonTrapSetup(init_state, target_state, num_focks, num_steps, alpha_list=[])

start = process_time()
optim_result = minimize(setup.target_func, setup.init_param_vec(),
                        jac=setup.gradient,
                        args=(init_state, target_state),
                        options={'disp': True, 'return_all': True}, method=BFGS)
end = process_time()

record = OptimizationRecord(setup, optim_result)
record.plot_history()
record.plot_final_dynamics()

print('CPU time used: {}'.format(end - start))
