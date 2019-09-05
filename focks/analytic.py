from focks.interaction import LaserInteraction
from qutip import Qobj, expect
from typing import List, Tuple
from numpy import arctan2, angle, pi, sqrt, exp
import numpy as np
import matplotlib.pyplot as plt

EPS = np.finfo(float).eps
SQEPS = np.finfo(float).eps ** 0.5

count = 0
energy_freq = {}


class BranchFactory:
    def __init__(self, state, highest_fock, interaction):

        self.highest_fock = highest_fock
        self.interaction = interaction
        self.state = state
        n = self.highest_fock

        e, g = interaction.eigenstate_generators

        # prepare for red, use carrier to move into |g n>
        alpha = g(n).overlap(state)
        beta = e(n).overlap(state)

        self.coupling_strength_c_r = arctan2(abs(beta), abs(alpha))
        self.coupling_phase_c_r = exp(-1j * (angle(alpha) - angle(beta) + pi / 2))

        self.amp_c_r = self.coupling_strength_c_r * self.coupling_phase_c_r
        self.evolved_state_c_r = interaction.evolve(state, [self.amp_c_r, 0, 0], 1.0)

        assert abs(e(n).overlap(self.evolved_state_c_r)) < SQEPS

        if n == 0:
            return
        # red, move into |e n-1>
        alpha = e(n - 1).overlap(self.evolved_state_c_r)
        beta = g(n).overlap(self.evolved_state_c_r)

        self.coupling_strength_r = arctan2(abs(beta), abs(alpha))
        self.coupling_phase_r = exp(1j * (angle(alpha) - angle(beta) + pi / 2))

        self.amp_r = self.coupling_strength_r / sqrt(n) * self.coupling_phase_r

        if n == 0:
            return

        # prepare for blue, use carrier to move into |e n>
        alpha = e(n).overlap(state)
        beta = g(n).overlap(state)

        self.coupling_strength_c_b = arctan2(abs(beta), abs(alpha))
        self.coupling_phase_c_b = exp(1j * (angle(alpha) - angle(beta) + pi / 2))

        self.amp_c_b = self.coupling_strength_c_b * self.coupling_phase_c_b

        self.evolved_state_c_b = interaction.evolve(state, [self.amp_c_b, 0, 0], 1.0)

        assert abs(g(n).overlap(self.evolved_state_c_b)) < SQEPS

        # blue, move into |g n-1>
        alpha = g(n - 1).overlap(self.evolved_state_c_b)
        beta = e(n).overlap(self.evolved_state_c_b)

        self.coupling_strength_b = arctan2(abs(beta), abs(alpha))
        self.coupling_phase_b = exp(-1j * (angle(alpha) - angle(beta) + pi / 2))

        self.amp_b = self.coupling_strength_b / sqrt(n) * self.coupling_phase_b

        self.nth_r = 0
        self.nth_b = 0

    def next_branch(self):
        if self.highest_fock == 0:
            return self.coupling_strength_c_r**2, [(self.amp_c_r, 0, 0)], self.evolved_state_c_r

        eta = self.interaction.lamb_dicke
        n = self.highest_fock

        energy_r = (self.coupling_strength_r + self.nth_r * pi) ** 2 / n + (self.coupling_strength_c_r*eta)**2
        energy_b = (self.coupling_strength_b + self.nth_b * pi) ** 2 / n + (self.coupling_strength_c_b*eta)**2
        if energy_r < energy_b:
            amp_r = (self.coupling_strength_r + self.nth_r * pi) / sqrt(n) * self.coupling_phase_r
            evolved_state = self.interaction.evolve(self.evolved_state_c_r, (0, amp_r, 0), 1.0)

            pulses = [(0, amp_r, 0), (self.amp_c_r, 0, 0)]
            if self.nth_r >= 0:
                self.nth_r = -self.nth_r-1
            else:
                self.nth_r = -self.nth_r
            return energy_r, pulses, evolved_state
        else:
            amp_b = (self.coupling_strength_b + self.nth_b * pi) / sqrt(n) * self.coupling_phase_b
            evolved_state = self.interaction.evolve(self.evolved_state_c_b, (0, 0, amp_b), 1.0)

            pulses = [(0, 0, amp_b), (self.amp_c_b, 0, 0)]
            if self.nth_b >=0:
                self.nth_b = -self.nth_b - 1
            else:
                self.nth_b = -self.nth_b
            return energy_b, pulses, evolved_state


def _next_level(state: Qobj,
                highest_fock: int,
                used_energy: float,
                min_energy: List[float],
                interaction: LaserInteraction) -> Tuple[float, List]:
    """

    Parameters
    ----------
    state: Qobj
    highest_fock : int
        The highest occupied (occupation > TOL) fock state.
    min_energy : list of a single float
        The minimum energy obtained so far by the depth-first search. The float is protected by the *list* so that
        it is passed by reference and is shared between all tree nodes.
    interaction : LaserInteraction
        The interaction object which provides the evolution of the state.

    Returns
    -------


    """
    global count, energy_freq
    count += 1
    n = highest_fock

    if used_energy > min_energy[0]:
        return np.inf, []

    e, g = interaction.eigenstate_generators


    # evolve the state to gather all population to |e,n>
    # use blue to move all population in |e,n> to |g,n-1>
    # update pulse_sequence to include the two operations above
    factory = BranchFactory(state, highest_fock, interaction)
    if n == 0:
        self_energy, self_pulses, evolved_state = factory.next_branch()
        total_energy = self_energy + used_energy

        assert abs(g(0).overlap(evolved_state))-1 < SQEPS
        if "{:5f}".format(total_energy) in energy_freq:
            energy_freq["{:5f}".format(total_energy)] += 1
        else:
            energy_freq["{:5f}".format(total_energy)] = 1

        if total_energy < min_energy[0]:
            min_energy[0] = total_energy
        return self_energy, self_pulses
    energy = np.inf
    pulse_sequence = None
    #while True:
    for i in range(3):
        self_energy, self_pulses, evolved_state = factory.next_branch()
        if self_energy + used_energy >= min_energy[0]:
            break
        child_energy, child_pulses = _next_level(evolved_state, n - 1, self_energy+used_energy, min_energy, interaction)
        if child_energy < energy:
            energy = child_energy + self_energy
            pulse_sequence = child_pulses + self_pulses

    return energy, pulse_sequence


def find_pulse_sequence(target_state: Qobj, interaction):
    assert len(target_state.dims[0]) == 2, "Target state should be a tensor product state."
    assert target_state.dims[0][0] == 2, "The first space should be a 2-dimensional space."
    num_focks = target_state.dims[0][1]

    e, g = interaction.eigenstate_generators
    highest_fock = num_focks - 1
    while expect(e(highest_fock).proj() + g(highest_fock).proj(), target_state) < 3 * SQEPS and highest_fock >= 0:
        highest_fock -= 1

    energy, pulse_sequence = _next_level(target_state, highest_fock, 0, [np.inf], interaction)

    return energy, pulse_sequence


if __name__ == '__main__':
    N = 10
    interaction = LaserInteraction(N, 0.1)
    e, g = interaction.eigenstate_generators

    import pickle
    state = pickle.load(open('/home/kin/urop/focks/data/states/ca10.pickle', 'rb'))

    #state = interaction.get_random_state()
    energy, pulse_sequence = find_pulse_sequence((state).unit(), interaction)
    print(energy)

    for pulse in pulse_sequence:
        c_idx = np.argmax(np.abs(pulse))
        print("{:} {:.3f} {:+.3f}π".format(['c', 'r', 'b'][c_idx], np.abs(pulse[c_idx]), angle(pulse[c_idx]) / pi))

    # print(count)
    # print(energy_freq)

    # avg_energies = []
    # N_range = range(2, 10)
    # for N in N_range:
    #     interaction = LaserInteraction(N)
    #     energies = []
    #     for i in range(100):
    #         energy, pulse_sequence = find_pulse_sequence(interaction.get_random_state())
    #         energies.append(energy)
    #     avg_energies.append(np.mean(energies))
    #
    #     plt.hist(energies, bins=10)
    #     plt.title("N={:} avg:{:}".format(N, np.mean(energies)))
    #     plt.show()
    # plt.plot(N_range, avg_energies)
    # plt.show()
