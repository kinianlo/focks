from fock import IonTrapSetup
import matplotlib.pyplot as plt
import numpy as np
from time import process_time
from scipy.optimize import minimize
from utils import get_ion_state_generators

num_repeats = 20
alpha1_list = [0, 1e-5, 1e-3, 1e-1]

num_focks = 25
g, e = get_ion_state_generators(num_focks)

init_state = g(0)
target_state = (g(5)).unit()
num_steps = 5
final_infidelity_all = []
for j in range(len(alpha1_list)):
    alpha1 = alpha1_list[j]
    final_infidelity_list = []
    for i in range(num_repeats):
        print('alpha1={}({}/{}), #{}/{}'.format(alpha1, j, len(alpha1_list), i, num_repeats))
        setup = IonTrapSetup(init_state, target_state, num_focks, num_steps, alpha_list=[1,alpha1,0])
        start = process_time()
        optim_result = minimize(setup.target_func, setup.init_param_vec(),
                                jac=setup.gradient,
                                args=(init_state, target_state),
                                options={'disp': True})
        end = process_time()

        print('CPU time used: {}'.format(end - start))
        evolved_state = setup.evolve(init_state, optim_result.x)
        final_infidelity_list.append(1-abs(target_state.overlap(evolved_state))**2)
    final_infidelity_all.append(final_infidelity_list)

fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')
ax.set_yscale('log')
for i in range(len(alpha1_list)):
   ax.scatter([alpha1_list[i]]*num_repeats, final_infidelity_all[i], alpha=0.3)
   #ax.errorbar(alpha1_list[i], np.mean(final_infidelity_all[i]), yerr=np.std(final_infidelity_all[i]))
   #ax.scatter(alpha1_list[i], np.mean(final_infidelity_all[i]))
ax.set_xlabel('alpha1')
ax.set_ylabel('final infidelity')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.set_xlabel('log final infidelity')
for i in range(len(alpha1_list)):
    ax.hist(np.log(final_infidelity_all[i]), alpha=0.5,label='{:.0e}'.format(alpha1_list[i]))
ax.legend()
fig.show()


