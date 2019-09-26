#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys
import time
import pickle
from focks import InfidelityOptimizer, LaserInteraction
from focks.utils import expand_state

for line in sys.stdin:
    (state_pickle_path,
     num_steps,
     max_prep_time,
     num_focks,
     tol,
     lamb_dicke,
     timeout) = line.split(' ')

num_steps = int(num_steps)
max_prep_time = float(max_prep_time)
num_focks = int(num_focks)
tol = float(tol)
lamb_dicke = float(lamb_dicke)
timeout = int(timeout)


interaction = LaserInteraction(num_focks, lamb_dicke)
e, g = interaction.eigenstate_generators
with open(state_pickle_path, 'rb') as read_file:
    state = expand_state(pickle.load(read_file), g(0).dims)
optim = InfidelityOptimizer(interaction, state, g(0), num_steps)

results = []
leakage_count = 0
total_count = 0

global_start = time.time()

for i in range(100):
    total_count += 1

    start_wall = time.time()
    start_CPU = time.process_time()

    try:
        optim_result = optim.optimize_constrain_prep_time(max_prep_time=max_prep_time, tol=tol)
    except ValueError as e:
        print(e)
        leakage_count += 1
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
              "process_time": end_CPU - start_CPU,
              "message": optim_result.message,
              "nfev": optim_result.nfev,
              "nit": optim_result.nit,
              "njev": optim_result.njev,
              "status": optim_result.status,
              "success": int(optim_result.success)}
    results.append(result)

    print("\nRun #{:}".format(i + 1))
    print("CPU time (wall time) = {:.2f} ({:.2f})".format((end_CPU - start_CPU), (end_wall - start_wall)))
    print(optim_result.message)
    print("prep_time:\t{:}".format(prep_time))
    print("Infide:\t{:}".format(infidelity))

    elapsed_time = time.time() - global_start
    if elapsed_time > timeout - elapsed_time / (i + 1):
        print("*** Time out ({:}) ***".format(timeout))
        break
else:
    print("*** Last run finished, elapsed time:{:} ***".format(time.time() - global_start))

print("Leakage: {:}/{:}".format(leakage_count, total_count))

summary = {"num_steps": num_steps,
           "max_prep_time": max_prep_time,
           "num_focks": num_focks,
           "tol": tol,
           "lamb_dicke": lamb_dicke,
           "total_count": total_count,
           "leakage_count": leakage_count,
           "results": results}

with open("summary.json", "w") as write_file:
    json.dump(summary, write_file)

with open("summary.pickle", "wb") as write_file:
    pickle.dump(summary, write_file)
