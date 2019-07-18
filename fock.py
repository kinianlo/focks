#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:01:37 2019

@author: kin
"""

import numpy as np
from scipy.optimize import minimize
from qutip import tensor, basis, qeye, sigmax, sigmay, sigmaz, sigmap, sigmam, \
create, destroy, mesolve, sesolve
import matplotlib.pyplot as plt
from utils import get_ion_state_generators

class Parameterization:
    """
    The abstract class for parameterizations of coefficient functions.
    Coefficient functions refers to the three time-dependent functions
    which control the Hamiltonian
    H = fc*sigam_+ + fr*sigma_+ a^dagger + fb*sigma_+ a + h.c..
    """
    def __init__(self, num_focks):
        self.num_focks = num_focks

        Sp = tensor(sigmap(), qeye(self.num_focks))
        a = tensor(qeye(2), destroy(self.num_focks))
        self._Hc_half = Sp
        self._Hr_half = Sp * a
        self._Hb_half = Sp * a.dag()


    def _get_half_Hamiltonians(self):
        """
        Return
        """
        return self._Hc_half, self._Hr_half, self._Hb_half

    def init_param_vec(self):
        """
        Return an initial parameter vector for optimization.
        """
        raise NotImplementedError

    def unpack_param_vec(self, param_vec):
        """
        Unpack a parameter vector into a more manageable form
        """
        raise NotImplementedError

    def strengths(self, param_vec):
        """
        Return a tuple of three functions (fc, fr, fb) based on the parameter
        vector param_vec.
        The functions must take an t argument and a list of arguments *arg,
        e.g. fc = lambda t, *arg : ***function_output***.
        The *arg is required by QuTip for additional arguments (if any).
        """
        raise NotImplementedError

    def evolve(self, init_state, param_vec):
        """
        Return the evolved state.
        """
        raise NotImplementedError

    def observe(self, init_state, param_vec, times, e_ops=[]):
        """
        Evolve the init_state given the param_vec for a duration of gate_time.
        Return a qutip.solver.Result objection
        """
        raise NotImplementedError

    def parse_parameters(self, param_vec):
        """
        Print the parameters in a human readable form.
        """
        raise NotImplementedError


class PiecewiseConstant(Parameterization):
    def __init__(self, num_focks, num_intervals):
        Parameterization.__init__(self, num_focks)
        self.num_intervals = num_intervals

    def init_param_vec(self):
        """
        Return an array of 6*num_intervals random numbers from 0 to 1
        """
        return np.random.rand((6*(self.num_intervals)))

    def unpack_param_vec(self, param_vec):
        rAc, iAc, rAr, iAr, rAb, iAb = param_vec.reshape((6,-1))
        Ac = rAc + 1j * iAc
        Ar = rAr + 1j * iAr
        Ab = rAb + 1j * iAb
        return Ac, Ar, Ab

    def strengths(self, param_vec):
        Ac, Ar, Ab = self.unpack_param_vec(param_vec)

        fc = lambda t, *_: Ac[np.mod(int(t*self.num_intervals), self.num_intervals)]
        fr = lambda t, *_: Ar[np.mod(int(t*self.num_intervals), self.num_intervals)]
        fb = lambda t, *_: Ab[np.mod(int(t*self.num_intervals), self.num_intervals)]

        return fc, fr, fb

    def evolve(self, init_state, param_vec):
        Ac, Ar, Ab = self.unpack_param_vec(param_vec)
        Hc, Hr, Hb = self._get_half_Hamiltonians()

        state = init_state
        for j in range(self.num_intervals):
            H = Ac[j] * Hc + Ar[j] * Hr + Ab[j] * Hb
            H = H + H.dag()
            state = ((-1j/self.num_intervals)*H).expm() * state

#            interval_times = [j / self.num_intervals, (j+1) /self.num_intervals]
#            result = sesolve(H, state, interval_times)
#            state = result.states[-1]
        return state

    def observe(self, init_state, param_vec, times, e_ops=[]):
        Hc, Hr, Hb = self._get_half_Hamiltonians()

        fc, fr, fb = self.strengths(param_vec)
        fccj = lambda t, *_: np.conj(fc(t))
        frcj = lambda t, *_: np.conj(fr(t))
        fbcj = lambda t, *_: np.conj(fb(t))
        H = [[Hc, fc], [Hr, fr], [Hb, fb], [Hc.dag(), fccj], [Hr.dag(), frcj], [Hb.dag(), fbcj]]
        result = mesolve(H, init_state, times, [], e_ops)

        if e_ops == []:
            return result.states
        else:
            return result.expect

    def parse_parameters(self, param_vec):
        # unpack the parameter vector
        Ac, Ar, Ab = self.unpack_param_vec(param_vec)

        output = '\nPiecewiseParameterization\n'
        output = output + 'fc(t) = \sum_j Ac_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fr(t) = \sum_j Ar_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fb(t) = \sum_j Ab_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'

        output = output + '\ncarrier transition:\n'
        output = output + 't_j\tt_{j+1}\tAc\t\tAr\t\tAb\n'
        for j in range(self.num_intervals):
            t_j = j/self.num_intervals
            t_jp1 = (j+1)/self.num_intervals
            output = output + '{:.3}\t{:.3}\t{:.3e}\t{:.3e}\t{:.3e}\n'.format(t_j, t_jp1, Ac[j], Ar[j], Ab[j])

def plot_dynamics(param_vec, parameterization, init_state, target_state, \
                  num_focks):
    """
    Plot the fidelity and coefficient functions during the gate time.
    """
    fc, fr, fb = parameterization.strengths(param_vec)

    e_ops = []
    for n in range(num_focks):
        e_ops.append(tensor(basis(2,1), basis(num_focks, n)).proj()) # |0>
        e_ops.append(tensor(basis(2,0), basis(num_focks, n)).proj()) # |1>
    e_ops.append(target_state.proj())

    times = np.linspace(0, 1, 1000)
    expect = parameterization.observe(init_state, param_vec, times, e_ops)

    fig, axes = plt.subplots(num_focks+2, 1, sharex=True)

    for n in range(num_focks):
        axes[num_focks-1-n].plot(times, expect[2*n], alpha=0.7, label='ground')
        axes[num_focks-1-n].plot(times, expect[2*n+1], alpha=0.7, label='excited')

        axes[num_focks-1-n].set_ylabel('|{}> '.format(n), rotation='horizontal')
        axes[num_focks-1-n].set_ylim(-0.1, 1.1)
        axes[num_focks-1-n].grid()

    axes[0].legend()

    axes[-2].plot(times, np.abs(np.vectorize(fc)(times))/np.pi, 'g', alpha=0.7)
    axes[-2].plot(times, np.abs(np.vectorize(fr)(times))/np.pi, 'r', alpha=0.7)
    axes[-2].plot(times, np.abs(np.vectorize(fb)(times))/np.pi, 'b', alpha=0.7)

    axes[-2].set_ylabel('strength/pi')
    axes[-2].grid()

    axes[-1].semilogy(times, 1-expect[-1])
    axes[-1].set_ylabel('infidelity')
    axes[-1].set_xlabel('time')
    axes[-1].grid()

    if isinstance(parameterization, PiecewiseConstant):
        axes[-1].xaxis.set_ticks(np.linspace(0, 1, parameterization.num_intervals+1))

def plot_fidelity_num_intervals(target_state, num_intervals_range, num_repeats):
    num_focks = max(num_intervals_range)+1
    g, e = get_ion_state_generators(num_focks)
    init_state = g(0)

    all_errors = []

    for num_intervals in num_intervals_range:
        errors = []
        parameterization = PiecewiseConstant(num_focks, num_intervals)
        for i in range(num_repeats):
            def dist(param_vec):
                evolved_state = parameterization.evolve(init_state, param_vec)
                return 1-np.abs(target_state.overlap(evolved_state))**2
            opt_result = minimize(dist, parameterization.init_param_vec(),
                                      options={'disp':True})
            opt_param_vec = opt_result.x
            evolved_state = parameterization.evolve(init_state, opt_param_vec)
            errors.append(1-np.abs(target_state.overlap(evolved_state))**2)
        all_errors.append(errors)

    fig, ax = plt.subplots(1,1)
    for i in range(len(num_intervals_range)):
        ax.scatter([num_intervals_range[i]]*num_repeats, all_errors[i])


if __name__ == '__main__':
    from time import process_time

    NELDER_MEAD = 'nelder-mead'
    BFGS = 'BFGS'
    POWERLL = 'Powell'


#    num_repeats = 3
#    target_state = (g(3)+g(4)).unit()
#    num_intervals_range = range(1, 5)
#    num_focks = max(num_intervals_range)+2
#    g, e = get_ion_state_generators(num_focks)
#    init_state = g(0)
#
#    all_errors = []
#
#    for num_intervals in num_intervals_range:
#        print('num_intervals:{}'.format(num_intervals))
#        errors = []
#        parameterization = PiecewiseConstant(num_focks, num_intervals)
#        for i in range(num_repeats):
#            print('run#{}'.format(i))
#            def dist(param_vec):
#                evolved_state = parameterization.evolve(init_state, param_vec)
#                return 1-np.abs(target_state.overlap(evolved_state))**2
#            start = process_time()
#            opt_result = minimize(dist, parameterization.init_param_vec(),
#                                      options={'disp':True})
#            end = process_time()
#            print('CPU time used: {}'.format(end-start))
#            opt_param_vec = opt_result.x
#            evolved_state = parameterization.evolve(init_state, opt_param_vec)
#            errors.append(1-np.abs(target_state.overlap(evolved_state))**2)
#        all_errors.append(errors)
#
#    fig, ax = plt.subplots(1,1)
#    ax.set_yscale('log')
#    ax.set_xlabel('number of intervals')
#    ax.set_ylabel('indifelity')
#    for i in range(len(num_intervals_range)):
#        ax.scatter([num_intervals_range[i]]*num_repeats, all_errors[i])


    num_focks = 10
    g, e = get_ion_state_generators(num_focks)

    init_state = g(0)
    target_state = (g(4)).unit()

    num_intervals = 4
    parameterization = PiecewiseConstant(num_focks, num_intervals)

    def dist(param_vec):
        evolved_state = parameterization.evolve(init_state, param_vec)
        return 1-np.abs(target_state.overlap(evolved_state))**2

    start = process_time()
    opt_result = minimize(dist, parameterization.init_param_vec(),
                          options={'disp':True})
    end = process_time()

    opt_param_vec = opt_result.x # optimized parameter vector

    print('CPU time used: {}'.format(end-start))
    parameterization.parse_parameters(opt_param_vec)

    plot_dynamics(opt_param_vec, parameterization, init_state, target_state, num_focks)
