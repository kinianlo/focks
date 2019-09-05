from time import time, process_time

import numpy as np
from focks import LaserInteraction, InfidelityOptimizer
from focks.utils import expand_state
num_focks = 50
interaction = LaserInteraction(num_focks, 0.1)
e, g = interaction.eigenstate_generators
import pickle
state = pickle.load(open('/home/kin/urop/focks/data/states/tx5.pickle', 'rb'))
state = expand_state(state, g(0).dims)
optim = InfidelityOptimizer(interaction, state, g(0), 5, energy_weight=0.00001, energy_threshold=5.3)



for i in range(1):
    start_wall = time()
    start_CPU = process_time()
    result = optim.optimize()
    end_wall = time()
    end_CPU = process_time()
    print('CPU time (wall time) = {:.2f} ({:.2f})'.format((end_CPU-start_CPU), (end_wall-start_wall)))
    print(result.msg)
    print("func:\t{:}\nmax(grad):\t{:}".format(result.final_func, max(result.final_grad)))
    print("#iter:{:}\t#func:{:}\t#grad:{:}".format(result.final_status["num_iter"],
                                                   result.final_status["num_func_call"],
                                                   result.final_status["num_grad_call"]))