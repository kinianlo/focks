#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
from focks import LaserInteraction
from focks.analytic import find_minimum_prep_time
import numpy as np
from numpy import angle, pi

print(sys.argv)
assert len(sys.argv) == 6

(_,
 state_pickle_path,
 num_focks,
 tol,
 lamb_dicke,
 timeout) = sys.argv

num_focks = int(num_focks)
tol = float(tol)
lamb_dicke = float(lamb_dicke)
timeout = int(timeout)

with open(state_pickle_path, 'rb') as read_file:
    state = pickle.load(read_file)

interaction = LaserInteraction(state.dims[0][1], lamb_dicke)
e, g = interaction.eigenstate_generators
anal_prep_time, anal_pulse_sequence = find_minimum_prep_time(interaction, state)
print("The analytic prep_time is:\n{:}".format(anal_prep_time))
print("The analytic pulse sequence is:")
for pulse in anal_pulse_sequence:
    c_idx = int(np.argmax(np.abs(pulse)))
    print("{:} {:.5f} {:+.5f}π".format(['c', 'r', 'b'][c_idx], np.abs(pulse[c_idx]), angle(pulse[c_idx]) / pi))

with open("inputs", 'w') as write_file:
    for num_steps in range(1, len(anal_pulse_sequence) + 4):
        for max_prep_time in np.linspace(0, 1.5 * anal_prep_time, 501)[1:]:
            for i in range(1):
                write_file.write("{:} {:} {:} {:} {:} {:} {:}\n".format(
                                                                        state_pickle_path,
                                                                        num_steps,
                                                                        max_prep_time,
                                                                        num_focks,
                                                                        tol,
                                                                        lamb_dicke,
                                                                        timeout
                                                                        ))
