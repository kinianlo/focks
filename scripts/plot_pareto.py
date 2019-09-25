import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

path_to_job = sys.argv[1]

solutions = {}
paretos = {}
times = {}
nit = {}
total_count = 0
leakage_count = 0

for sub_job in os.listdir(os.path.join(path_to_job, 'outputs')):
    if not os.path.isdir(os.path.join(path_to_job, 'outputs', sub_job)):
        continue
    try:
        with open(os.path.join(path_to_job, 'outputs', sub_job, "summary.json"), 'r') as json_file:
            summary = json.load(json_file)
    except:
        continue
    num_steps = summary["num_steps"]

    total_count += summary["total_count"]
    leakage_count += summary["leakage_count"]

    if num_steps not in solutions.keys():
        solutions[num_steps] = []
        times[num_steps] = []
        nit[num_steps] = []

    min_infide = np.inf
    min_energy = np.inf

    for result in summary["results"]:
        infidel = result["infidelity"]
        energy = result["energy"]
        if infidel < min_infide:
            min_infide = infidel
            min_energy = energy
        times[num_steps].append(result["process_time"])
        nit[num_steps].append(result["nit"])

    solutions[num_steps].append((min_infide, min_energy))


for num_steps in solutions.keys():
    paretos[num_steps] = []
    for sol in solutions[num_steps]:

        for other_sol in solutions[num_steps]:
            if sol == other_sol:
                continue
            if sol[0] > other_sol[0] and sol[1] > other_sol[1] :
                break
        else:
            paretos[num_steps].append(sol)

fig, axes = plt.subplots(1, 2, sharey=True)
ax_linear, ax_log = axes
ax_linear.set_xscale("linear")
ax_log.set_xscale("log")
ax_linear.grid()
ax_log.grid()
for num_steps in sorted(paretos.keys()):
    if num_steps > 100:
        continue
    infidelities = [x[0] for x in sorted(paretos[num_steps],key=lambda x: x[0])]
    energies = [x[1] for x in sorted(paretos[num_steps],key=lambda x: x[0])]

    ax_linear.plot(infidelities, energies, label=str(num_steps))

    #ax_log.scatter(infidelities, energies)
    ax_log.plot(infidelities, energies)
ax_linear.legend()
fig.show()

fig, ax = plt.subplots(1, 1)
for num_steps in sorted(times.keys()):
    ax.hist(times[num_steps], bins=20, label=str(num_steps))
ax.legend()
fig.show()

for num_steps in sorted(times.keys()):
    print("{:} {:}".format(num_steps, sorted(times[num_steps])[-5]))