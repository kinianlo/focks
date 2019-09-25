#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
from focks import LaserInteraction
from focks.analytic import find_minimum_energy
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
anal_energy, anal_pulse_sequence = find_minimum_energy(interaction, state)
print("The analytic energy is:\n{:}".format(anal_energy))
print("The analytic pulse sequence is:")
for pulse in anal_pulse_sequence:
    c_idx = int(np.argmax(np.abs(pulse)))
    print("{:} {:.5f} {:+.5f}π".format(['c', 'r', 'b'][c_idx], np.abs(pulse[c_idx]), angle(pulse[c_idx]) / pi))

with open("inputs", 'w') as write_file:
    for num_steps in range(1, 2 * len(anal_pulse_sequence) + 1):
        for max_energy in np.linspace(0, 2 * anal_energy, 101)[1:]:
            for i in range(5):
                write_file.write("{:} {:} {:} {:} {:} {:} {:}\n".format(
                                                                        state_pickle_path,
                                                                        num_steps,
                                                                        max_energy,
                                                                        num_focks,
                                                                        tol,
                                                                        lamb_dicke,
                                                                        timeout
                                                                        ))
