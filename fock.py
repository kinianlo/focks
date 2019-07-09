#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:42:46 2019

@author: kin
"""

from qutip import *
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from utils import get_ion_state_generators


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':16})
rc('text', usetex=True)

class Parameterization:
    """
    The abstract class for parameterizations of coefficient functions.
    Coefficient functions refers to the three time-dependent functions
    which control the Hamiltonian H = fc(t)*Hc + fr(t)*Hr + fb(t)*Hb,
    where Hc, Hr and Hb are the Hamiltonians for carrier, red-band and
    blue-band transitions respectively.

    The main purpose of this class is to specify the parameterization,
    i.e. how a parameter vector p is mapped to fc, fr, fb.
    """
    def coef_functions(self, param_vec):
        """
        Return a tuple of three functions (fc, fr, fb) based on the parameter
        vector param_vec.
        The functions must take an t argument and a list of arguments *arg,
        e.g. fc = lambda t, *arg : ***function_output***.
        The *arg is required by QuTip for additional arguments (if any).
        """
        raise NotImplementedError

    def init_parameters(self):
        """
        Return an initial parameter vectorr for optimization.
        """
        raise NotImplementedError

    def parse_parameters(self, param_vec):
        """
        Print the parameters in a human readable form.
        """
        raise NotImplementedError


class CarrierParameterization(Parameterization):
    """
    A simple parameterization with only a constant strength carrier transition.
    Mainly for testing.
    """
    def coef_functions(self, param_vec):
        """
        fr and fb are always zeros, where fc is allowed to be a
        time-independent constant parameter
        """
        fc = lambda t, *_: param_vec[0]
        fr = lambda t, *_: 0
        fb = lambda t, *_: 0
        return (fc, fr, fb)

    def init_parameters(self):
        """
        Return a numpy array of only a random number from 0 to 1
        """
        return np.array([np.random.rand()])

    def parse_parameters(self, param_vec):
        output = '\nCarrierParameterization\n'
        output = output + 'fc(t) = {}\n'.format(param_vec[0])
        output = output + 'fr(t) = {}\n'.format(0.0)
        output = output + 'fb(t) = {}\n'.format(0.0)

        print(output)


class FourierParameterization(Parameterization):
    """
    A parameterization based on fourier series
    Here x stands for any colour (c, b, r):
        fx(t) = \sum_j Ax_j * cos(j*t + Px_j)
    where j runs from 0 to num_terms-1;
    Ax and Px are arrays of parameters with num_terms entries.
    Note that for j=0, only Ax_0 is allowed to vary and Px_0 is always kept
    zero to avoid over-parameterization.
    All the parameters are packed into a single parameter vector in the
    way like np.concatenate((Ac, Pc, Ar, Pr, Ab, Pb))
    """
    def __init__(self, num_terms):
        self.num_terms = num_terms

    def coef_functions(self, param_vec):
        """
        Return the three coefficent functions fc, fr, fb based on param_vec
        """
        # unpack the parameter vector
        param_c, param_r, param_b = param_vec.reshape((3,-1))
        Pc, Ac = np.split(np.insert(param_c, 0, 0.0), 2)
        Pr, Ar = np.split(np.insert(param_r, 0, 0.0), 2)
        Pb, Ab = np.split(np.insert(param_b, 0, 0.0), 2)

        j = np.arange(self.num_terms)
        fc = lambda t, *_: np.sum(Ac * np.cos(j*t+Pc))
        fr = lambda t, *_: np.sum(Ar * np.cos(j*t+Pr))
        fb = lambda t, *_: np.sum(Ab * np.cos(j*t+Pb))

        return (fc, fr, fb)

    def init_parameters(self):
        """
        Return an array of 6*num_terms random number from 0 to 1
        """
        return np.random.rand(6*self.num_terms-3)

    def parse_parameters(self, param_vec):
        # unpack the parameter vector
        param_c, param_r, param_b = param_vec.reshape((3,-1))
        Pc, Ac = np.split(np.insert(param_c, 0, 0.0), 2)
        Pr, Ar = np.split(np.insert(param_r, 0, 0.0), 2)
        Pb, Ab = np.split(np.insert(param_b, 0, 0.0), 2)

        output = '\nFourierParameterization\n'
        output = output + 'fc(t) = \sum_j Ac_j * cos(j*t + Pc_j)\n'
        output = output + 'fr(t) = \sum_j Ar_j * cos(j*t + Pr_j)\n'
        output = output + 'fb(t) = \sum_j Ab_j * cos(j*t + Pb_j)\n'

        output = output + '\ncarrier transition:\n'
        output = output + 'j\tAc\t\tPc\n'
        for j in range(self.num_terms):
            output = output + '{}\t{:.3e}\t{:.3e}\n'.format(j, Ac[j], Pc[j])

        output = output + '\nred-sideband transition:\n'
        output = output + 'j\tAr\t\tPr\n'
        for j in range(self.num_terms):
            output = output + '{}\t{:.3e}\t{:.3e}\n'.format(j, Ar[j], Pr[j])

        output = output + '\nblue-sidband transition:\n'
        output = output + 'j\tAb\t\tPb\n'
        for j in range(self.num_terms):
            output = output + '{}\t{:.3e}\t{:.3e}\n'.format(j, Ab[j], Pb[j])

        print(output)

class PiecewiseParameterization(Parameterization):
    """
    A simple parameterization by diving the gate time into a number of
    num_intervals equal-time intervals. In each interval the value of the
    coefficent functions are constant specified by the parameters Ac, Ar and Ab,
    where Ac is an array of num_intervals entries, and similar for Ar and Ab.
    All the parameters are packed into a single parameter vector in the
    way like np.concatenate((Ac, Ar, Ab))
    """
    def __init__(self, num_intervals, gate_time):
        self.num_intervals = num_intervals
        self.gate_time = gate_time

    def coef_functions(self, param_vec):
        # unpack the parameter vector
        Ac, Ar, Ab = param_vec.reshape((3,-1))
        fc = lambda t, *_: Ac[np.mod(int(t/self.gate_time*self.num_intervals), self.num_intervals)]
        fr = lambda t, *_: Ar[np.mod(int(t/self.gate_time*self.num_intervals), self.num_intervals)]
        fb = lambda t, *_: Ab[np.mod(int(t/self.gate_time*self.num_intervals), self.num_intervals)]

        return (fc, fr, fb)

    def init_parameters(self):
        """
        Return an array of 3*num_intervals random numbers from 0 to 1
        """
        return np.random.rand((3*(self.num_intervals)))
        #return np.zeros(3*(self.num_intervals+1))

    def parse_parameters(self, param_vec):
        # unpack the parameter vector
        Ac, Ar, Ab = param_vec.reshape((3,-1))

        output = '\nPiecewiseParameterization\n'
        output = output + 'fc(t) = \sum_j Ac_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fr(t) = \sum_j Ar_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fb(t) = \sum_j Ab_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'

        output = output + '\ncarrier transition:\n'
        output = output + 't_j\tt_{j+1}\tAc\t\tAr\t\tAb\n'
        for j in range(self.num_intervals):
            t_j = j*self.gate_time/self.num_intervals
            t_jp1 = (j+1)*self.gate_time/self.num_intervals
            output = output + '{:.3}\t{:.3}\t{:.3e}\t{:.3e}\t{:.3e}\n'.format(t_j, t_jp1, Ac[j], Ar[j], Ab[j])

        print(output)

class IonTrap:
    """
    Class for an ion trap with all its physical parameters specified.11
    The ion lives in a simple harmonic oscillator and there is a two-level
    system internal to the ion itself (which is often realized as a
    transition between two electronic states). The Hilbert space is thus a
    tensor product between a two level Hilbert space and a Fock space.

    The time dynamics of interaction with a laser can be simulated.
    A target state and an initial state can be specfied.
    One can optimize the parameter vector of a parameterization of choice such
    that the evloved state is as close as possible to the target state.
    """
    def __init__(self, target_state, num_focks, rabi_freq, lamb_dicke, \
                 atom_freq, cavity_freq, gate_time):
        self.num_focks = num_focks
        self.target_state = target_state
        self.init_state = tensor(basis(2, 1), basis(num_focks, 0))
        self.rabi_freq = rabi_freq
        self.lamb_dicke = lamb_dicke
        self.atom_freq = atom_freq
        self.cavity_freq = cavity_freq
        self.gate_time = gate_time
        self._parameterization = None

    def optimize(self, parameterization):
        """
        Optimize the distance to target given the specific parameterization
        """
        self._parameterization = parameterization
        result = minimize(self._dist_to_target, parameterization.init_parameters())
        return result

    def _evolve(self, coef_functions, num_samples=2, observables=[]):
        """
        Evolve the initial state with Hamiltonian
        fc(t)*Hc + fr(t)*Hr + fb(t)*Hb, where (fc, fr, fb) given by
        coef_functions.

        num_samples refers to the number observations made during the
        evolution of the system.

        observables is a list of operators for oberservations. If observables
        is empty then the complete states are sampled during the evolution.

        A qutip.Result object directly from the solver is returned.
        """
        Sx = tensor(sigmax(), qeye(self.num_focks))
        Sz = tensor(sigmaz(), qeye(self.num_focks))
        Sp = tensor(sigmap(), qeye(self.num_focks))
        Sm = tensor(sigmam(), qeye(self.num_focks))
        a = tensor(qeye(2), destroy(self.num_focks))
        adag = a.dag()

        fc, fr, fb = coef_functions

        Hc = 0.5 * self.rabi_freq * Sx
        Hr = 0.5j * self.lamb_dicke * self.rabi_freq * (Sp * a - Sm * adag)
        Hb = 0.5j * self.lamb_dicke * self.rabi_freq * (Sp * adag - Sm * a)

        H = [[Hc, fc], [Hr, fr], [Hb, fb]]

        times = np.linspace(0, self.gate_time, num_samples)

        result = mesolve(H, self.init_state, times, [], observables)

        return result

    def _dist_to_target(self, param_vec):
        """
        Evolve the initial state given the parameter vector AND
        return the distance to the target state
        """
        coef_functions = self._parameterization.coef_functions(param_vec)

        result = self._evolve(coef_functions)
        final_state = result.states[-1]
        dist = 1-expect(self.target_state.proj(), final_state)
        return dist

    def plot_dynamics(self, param_vec, parameterization):
        """
        Plot the fidelity and coefficient functions during the gate time.
        """
        coef_functions = parameterization.coef_functions(param_vec)
        fc, fr, fb = coef_functions

        obs = [self.target_state.proj()]
        result = self._evolve(coef_functions, num_samples=1000, observables=obs)

        fig, axes = plt.subplots(4, 1, sharex=True)
        ax_fc, ax_fr, ax_fb, ax_fidelity = axes
        times = result.times

        ax_fc.plot(times, np.vectorize(fc)(times), 'g')
        ax_fr.plot(times, np.vectorize(fr)(times), 'r')
        ax_fb.plot(times, np.vectorize(fb)(times), 'b')
        ax_fidelity.plot(result.times, result.expect[0])

        ax_fc.set_ylabel(r'$f_c$', rotation='horizontal')
        ax_fr.set_ylabel(r'$f_r$', rotation='horizontal')
        ax_fb.set_ylabel(r'$f_b$', rotation='horizontal')
        ax_fidelity.set_ylabel(r'fidelity')
        ax_fidelity.set_xlabel(r'time')

        ax_fidelity.set_ylim(-0.1,1.1)
        ax_fidelity.set_xlim(np.min(times), np.max(times))

        ax_fc.grid()
        ax_fr.grid()
        ax_fb.grid()
        ax_fidelity.grid()

        fig.align_ylabels(axes)


if __name__ == '__main__':
    # physical parameters
    num_focks = 4
    rabi_freq = 2*np.pi * 1.0
    lamb_dicke = 0.1
    atom_freq = 2*np.pi * 1.0
    cavity_freq = 2*np.pi * 1.0
    gate_time = 1.0

    g, e = get_ion_state_generators(num_focks)

    target_state = e(1)

    ion_trap = IonTrap(target_state.unit(), num_focks, rabi_freq, lamb_dicke, \
                       atom_freq, cavity_freq, gate_time)

    #parameterization = CarrierParameterization()
    parameterization = FourierParameterization(num_terms=1)
    #parameterization = PiecewiseParameterization(num_intervals=1, gate_time=gate_time)

    opt_result = ion_trap.optimize(parameterization)
    opt_param_vec = opt_result.x # optimized parameter vector

    print(opt_result.message)
    parameterization.parse_parameters(opt_param_vec)
    # Plot the dynamics of the ion under the optimized Hamiltonian
    ion_trap.plot_dynamics(opt_param_vec, parameterization)
