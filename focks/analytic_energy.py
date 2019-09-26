from focks.interaction import LaserInteraction
import pickle
from qutip import expect
import numpy as np
from numpy import arctan2, exp, angle, pi, sqrt

EPS = np.finfo(float).eps
SQEPS = np.finfo(float).eps ** 0.5

def find_coupling_stregnth(alpha, beta, target):
    mag = arctan2(abs(beta), abs(alpha))
    if target == 'e':
        phase = exp(+1j * (angle(alpha) - angle(beta) + pi / 2))
    elif target == 'g':
        phase = exp(-1j * (angle(alpha) - angle(beta) + pi / 2))
    else:
        raise Exception("Aurgment `target` must be 'e' or 'g'")
    return mag, phase

def find_energy_bound(interaction, init_state):
    num_focks = interaction.num_focks
    state = init_state
    # First step: use red all the way to find a upper bound to the minimum energy
    total_energy = 0
    pulse_sequence = []
    e, g = interaction.eigenstate_generators
    for n in reversed(range(num_focks)):
        alpha = g(n).overlap(state)
        beta = e(n).overlap(state)

        mag, phase = find_coupling_stregnth(alpha, beta, 'g')
        amp = mag * phase
        pulse = [amp, 0, 0]

        state = interaction.evolve(state, pulse)
        total_energy += abs(amp)**2 * interaction.lamb_dicke**2
        pulse_sequence.append(pulse)

        if n == 0:
            # No need to further apply a red since the state is already at |g0>
            break

        alpha = e(n-1).overlap(state)
        beta = g(n).overlap(state)

        mag, phase = find_coupling_stregnth(alpha, beta, 'e')
        amp = mag/sqrt(n) * phase
        pulse = [0, amp, 0]

        state = interaction.evolve(state, pulse)
        total_energy += abs(amp)**2
        pulse_sequence.append(pulse)

    assert abs(abs(g(0).overlap(state))**2 - 1) < SQEPS, "Final state should be |g0>"

    energy_bound = total_energy
    pulse_sequence_bound = pulse_sequence
    print("The upper bound of the minimum energy is:\n{:}".format(energy_bound))
    print("The upper bound pulse sequence is:")
    for pulse in pulse_sequence_bound:
        c_idx = int(np.argmax(np.abs(pulse)))
        print("{:} {:.5f} {:+.5f}π".format(['c', 'r', 'b'][c_idx], np.abs(pulse[c_idx]), angle(pulse[c_idx]) / pi))
    return energy_bound, pulse_sequence_bound

def find_minimum_energy(interaction, init_state):
    e, g = interaction.eigenstate_generators
    energy_bound, pulse_sequence_bound = find_energy_bound(interaction, init_state)
    num_focks = interaction.num_focks

    nodes = [(init_state, [])]
    new_nodes = []
    for n in reversed(range(num_focks)):
        for node in nodes:
            state, pulse_sequence = node
            energy = interaction.pulse_sequence_energy(pulse_sequence)

            if energy >= energy_bound:
                continue

            ###############################################################

            cg_alpha = g(n).overlap(state)
            cg_beta = e(n).overlap(state)
            cg_mag, cg_phase = find_coupling_stregnth(cg_alpha, cg_beta, 'g')

            cg_amp = cg_mag * cg_phase
            cg_pulse = [cg_amp, 0, 0]
            cg_state = interaction.evolve(state, cg_pulse)

            if n == 0:
                new_nodes.append((cg_state, pulse_sequence+[cg_pulse]))
                continue

            ###############################################################

            r_alpha = e(n-1).overlap(cg_state)
            r_beta = g(n).overlap(cg_state)
            r_mag, r_phase = find_coupling_stregnth(r_alpha, r_beta, 'e')

            ###############################################################

            r0_amp = r_mag/sqrt(n) * r_phase
            r0_pulse = [0, r0_amp, 0]
            r0_state = interaction.evolve(cg_state, r0_pulse)

            new_nodes.append((r0_state, pulse_sequence+[cg_pulse, r0_pulse]))

            ###############################################################

            rn1_amp = (r_mag-pi)/sqrt(n) * r_phase
            rn1_pulse = [0, rn1_amp, 0]
            rn1_state = interaction.evolve(cg_state, rn1_pulse)

            new_nodes.append((rn1_state, pulse_sequence+[cg_pulse, rn1_pulse]))

            ###############################################################

            ce_alpha = e(n).overlap(state)
            ce_beta = g(n).overlap(state)
            ce_mag, ce_phase = find_coupling_stregnth(ce_alpha, ce_beta, 'e')

            ce_amp = ce_mag * ce_phase
            ce_pulse = [ce_amp, 0, 0]
            ce_state = interaction.evolve(state, ce_pulse)

            ###############################################################

            b_alpha = g(n - 1).overlap(ce_state)
            b_beta = e(n).overlap(ce_state)
            b_mag, b_phase = find_coupling_stregnth(b_alpha, b_beta, 'g')

            ###############################################################

            b0_amp = b_mag / sqrt(n) * b_phase
            b0_pulse = [0, 0, b0_amp]
            b0_state = interaction.evolve(ce_state, b0_pulse)

            new_nodes.append((b0_state, pulse_sequence + [ce_pulse, b0_pulse]))

            ###############################################################

            bn1_amp = (b_mag - pi) / sqrt(n) * b_phase
            bn1_pulse = [0, 0, bn1_amp]
            bn1_state = interaction.evolve(ce_state, bn1_pulse)

            new_nodes.append((bn1_state, pulse_sequence + [ce_pulse, bn1_pulse]))

            ###############################################################
        nodes = new_nodes
        new_nodes = []

    min_energy = np.inf
    min_pulse_sequence = None

    for node in nodes:
        state, pulse_sequence = node
        assert abs(abs(g(0).overlap(state)) ** 2 - 1) < SQEPS, "Final state should be |g0>"
        energy = interaction.pulse_sequence_energy(pulse_sequence)

        if energy < min_energy:
            min_energy = energy
            min_pulse_sequence = pulse_sequence


    return min_energy, min_pulse_sequence

if __name__ == '__main__':
    init_state = pickle.load(open('/home/kin/urop/focks/data/states/fl4.pickle', 'rb'))
    num_focks = init_state.dims[0][1]

    lamb_dicke = 0.1
    interaction = LaserInteraction(num_focks, lamb_dicke)
    e, g = interaction.eigenstate_generators

    minimum_energy, minimum_pulse_sequence = find_minimum_energy(interaction, init_state)

    print("The minimum energy is:\n{:}".format(minimum_energy))
    print("The minimum energy pulse sequence is:")
    for pulse in minimum_pulse_sequence:
        c_idx = int(np.argmax(np.abs(pulse)))
        print("{:} {:.5f} {:+.5f}π".format(['c', 'r', 'b'][c_idx], np.abs(pulse[c_idx]), angle(pulse[c_idx]) / pi))
