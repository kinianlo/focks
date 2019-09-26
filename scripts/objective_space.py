import pickle
import time

import numpy as np
from numpy import angle, pi

from focks import InfidelityOptimizer, LaserInteraction
from focks.analytic import find_minimum_prep_time
from focks.utils import expand_state
from focks.plots import piecewise_constant_dynamics

import matplotlib.pyplot as plt

state = pickle.load(open('/home/kin/urop/focks/data/states/fl4.pickle', 'rb'))
lamb_dicke = 0.1
num_steps = 4

interaction = LaserInteraction(state.dims[0][1], lamb_dicke)
e, g = interaction.eigenstate_generators
anal_prep_time, anal_pulse_sequence = find_minimum_prep_time(interaction, state)
print("The analytic prep time is:\n{:}".format(anal_prep_time))
print("The analytic pulse sequence is:")
for pulse in anal_pulse_sequence:
    c_idx = int(np.argmax(np.abs(pulse)))
    print("{:} {:.5f} {:+.5f}π".format(['c', 'r', 'b'][c_idx], np.abs(pulse[c_idx]), angle(pulse[c_idx]) / pi))


#piecewise_constant_dynamics(interaction, anal_pulse_sequence, state, g(0), anal_prep_time)
optim = InfidelityOptimizer(interaction, state, g(0), 7)
assert interaction.pulse_sequence_prep_time(anal_pulse_sequence) == anal_prep_time

tol = 1e-6
num_focks = 100
interaction = LaserInteraction(num_focks, 0.1)
e, g = interaction.eigenstate_generators
state = expand_state(state, g(0).dims)

results = []
all_pulse_sequence = []
leak_max_prep_time = []

for max_prep_time in np.linspace(0.5, 1, 2) * anal_prep_time:
    for i in range(2):
        optim = InfidelityOptimizer(interaction, state, g(0), num_steps)

        start_wall = time.time()
        start_CPU = time.process_time()

        try:
            optim_result = optim.optimize_constrain_prep_time(max_prep_time=max_prep_time, tol=tol)
        except ValueError as e:
            print(e)
            leak_max_prep_time.append(max_prep_time)
            continue

        end_wall = time.time()
        end_CPU = time.process_time()

        param_vec = optim_result.x
        prep_time = optim.param_vec_prep_time(param_vec)
        infidelity = optim.infidelity(param_vec)
        pulse_sequence = optim._param_vec_to_all_amps(param_vec)
        result = {"infidelity": infidelity,
                  "prep_time": prep_time,
                  "param_vec": param_vec.tolist(),
                  "max_prep_time": max_prep_time,
                  "num_steps": num_steps,
                  "num_focks": num_focks,
                  "tol": tol,
                  "lamb_dicke": lamb_dicke,
                  "process_time": end_CPU - start_CPU,
                  "message": optim_result.message,
                  "nfev": optim_result.nfev,
                  "nit": optim_result.nit,
                  "njev": optim_result.njev,
                  "status": optim_result.status,
                  "success": optim_result.success}
        results.append(result)

        print('CPU time (wall time) = {:.2f} ({:.2f})'.format((end_CPU - start_CPU), (end_wall - start_wall)))
        print(optim_result.message)
        print("prep_time: {:}".format(prep_time))
        print("Infidelity: {:}".format(infidelity))


all_infidelity = []
all_prep_time = []

for result in results:
    all_infidelity.append(result["infidelity"])
    all_prep_time.append(result["prep_time"])

fig, ax = plt.subplots(1, 1)
ax.set_xscale("log")
ax.scatter(all_infidelity, all_prep_time, label="N={:}".format(num_steps))
ax.axhline(anal_prep_time)
ax.set_xlabel("infidelity")
ax.set_ylabel("prep_time")
ax.grid()
ax.legend()
fig.show()

