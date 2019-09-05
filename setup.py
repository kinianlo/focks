import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize, approx_fprime
from qutip import tensor, qeye, sigmap, sigmam, create, destroy, Qobj, expect, sesolve
from utils import get_ion_state_generators, transform

DIFF = 0
appr_eps = np.finfo(float).eps ** 0.5


class IonTrapSetup:
    def __init__(self, init_state, target_state, num_focks, num_steps, alpha_list=[1]):
        self.init_state = init_state
        self.target_state = target_state
        self.num_focks = num_focks

        Sp = tensor(sigmap(), qeye(num_focks))
        Sm = tensor(sigmam(), qeye(num_focks))
        a = tensor(qeye(2), destroy(num_focks))
        adag = tensor(qeye(2), create(num_focks))

        # Control Hamiltonians
        self._ctrl_ops = [Sp + Sm, Sp * a + Sm * adag, Sp * adag + Sm * a,
                          1j * (Sp - Sm), 1j * (Sp * a - Sm * adag), 1j * (Sp * adag - Sm * a)]
        # Number operator
        self._adaga = adag * a
        self.num_steps = num_steps
        self.alpha_list = np.array(alpha_list + [0] * (3 - len(alpha_list)))

    @property.getter
    def control_operators(self):
        return self._ctrl_ops

    def unpack_param_vec(self, param_vec):
        """
        param_vec is a 1-d array of all the parameters.
        This method seperate the param_vec into num_steps rows and each row
        contains the 6 control amplitdues of a time step
        """
        return param_vec.reshape((self.num_steps, -1))

    def init_param_vec(self):
        """
        Return an arrany of 6*num_steps randum numbers in [-0.5, 0.5]
        """
        return np.random.rand(6 * self.num_steps) - 0.5

    def get_complex_strengths_func(self, param_vec):
        """
        Return three functions of time (fc, fr, fb). Each function returns the
        complex amplitude of a colour (carrier, red or blue) at time t.
        """
        amps = self.unpack_param_vec(param_vec)
        fc_list = amps[:, 0] + 1j * amps[:, 3]
        fr_list = amps[:, 1] + 1j * amps[:, 4]
        fb_list = amps[:, 2] + 1j * amps[:, 5]

        fc = lambda t, *_: fc_list[np.mod(int(t * self.num_steps), self.num_steps)]
        fr = lambda t, *_: fr_list[np.mod(int(t * self.num_steps), self.num_steps)]
        fb = lambda t, *_: fb_list[np.mod(int(t * self.num_steps), self.num_steps)]

        return fc, fr, fb

    def get_amps_func(self, param_vec):
        amps = self.unpack_param_vec(param_vec)

        func_list = []
        for i in range(6):
            func_list.append(lambda t, *_, ii=i: amps.T[ii][np.mod(int(t * self.num_steps),
                                                                   self.num_steps)])

        return func_list

    def evolve(self, init_state, param_vec):
        """
        Return the state after one unit of time
        """
        amps = self.unpack_param_vec(param_vec)

        state = init_state
        for j in range(self.num_steps):
            H = 0
            for i in range(6):
                H += amps[j, i] * self._ctrl_ops[i]
            state = ((-1j / self.num_steps) * H).expm() * state
        return state

    def observe(self, init_state, param_vec, times, e_ops=[]):
        amps_func = self.get_amps_func(param_vec)
        H = []
        for i in range(6):
            H.append([self._ctrl_ops[i], amps_func[i]])
        result = sesolve(H, init_state, times, e_ops)

        if e_ops == []:
            return result.states
        else:
            return result.expect

    def _f(self, Xs):
        return np.array([-Xs[0], Xs[1]**4, Xs[2]**4]) * self.alpha_list

    def _fprime(self, Xs):
        return np.array([-1, 4*Xs[1]**3, 4*Xs[2]**3]) * self.alpha_list

    def target_func(self, param_vec, init_state, target_state):
        # Number operator (and its squared)
        n = self._adaga
        n2 = self._adaga**2

        evolved_state = self.evolve(init_state, param_vec)

        Phi = 0

        # Infidelity term (Phi0)
        Phi += self.alpha_list[0]*(1 - abs(target_state.overlap(evolved_state))**2)

        # 1st moment term (Phi1)
        if self.alpha_list[1] != 0:
            Phi += self.alpha_list[1]*(expect(n, evolved_state) - expect(n, target_state))**4

        # 2nd moment term (Phi2)
        if self.alpha_list[2] != 0:
            Phi += self.alpha_list[2]*(expect(n2, evolved_state) - expect(n, evolved_state)**2
                                       - expect(n2, target_state) + expect(n, target_state)**2)**4

        return Phi

    def gradient(self, param_vec, init_state, target_state, safe=False):
        """

        :param param_vec: Parameter vector at which the grident is evaluated
        :param init_state:
        :param target_state:
        :param safe: If True then check gradient with scipy approx_fprime
        :return: gradient (same shape as param_vec)
        """
        n = self._adaga
        n2 = self._adaga**2
        amps = self.unpack_param_vec(param_vec)
        evolved_state = self.evolve(init_state, param_vec)

        right = evolved_state
        lefts = []
        fprimes = []

        # Infidelity (Phi0)
        lefts.append(target_state.proj() * evolved_state)
        fprimes.append(self.alpha_list[0])
        # 1st moment (Phi1)
        lefts.append(n * evolved_state)
        fprimes.append(self.alpha_list[1]*4*(expect(n, evolved_state)-expect(n, target_state))**3)
        # 2nd moment (Phi2)
        lefts.append((n-expect(n, evolved_state))**2 * evolved_state)
        fprimes.append(self.alpha_list[2]*4*(expect(n2, evolved_state) - expect(n, evolved_state)**2
                                             - expect(n2, target_state) + expect(n, target_state)**2)**3)

        grad = np.empty((self.num_steps, 6))

        DT = 1 / self.num_steps
        for j in reversed(range(self.num_steps)):
            H = 0
            for i in range(6):
                H += amps[j, i] * self._ctrl_ops[i]
            Udag = ((1j * DT) * H).expm()

            # Diagonalise H
            e_vals, e_kets = eigh(H.full())
            # Evaluate the integral for "average" Hi on a "for each matrix element fashion"
            L = -1j * DT * (e_vals.reshape((-1, 1)) - e_vals)
            E = np.divide(np.exp(L) - 1, L, out=np.ones_like(L), where=L != 0)

            for i in range(6):
                # prepare the "average" control Hamil Hi
                Hi = self._ctrl_ops[i]
                expt_Hi = Qobj(transform(e_kets, E * transform(e_kets.conj().T, Hi.full())), dims=Hi.dims)

                grad[j, i] = 0

                # Infidelity (Phi0)
                grad[j, i] += -2 * DT * np.imag(lefts[0].overlap(expt_Hi * right))
                # 1st moment (Phi1)
                grad[j, i] += fprimes[1] * 2 * DT * np.imag(lefts[1].overlap(expt_Hi * right))
                # 2nd moment (Phi2)
                grad[j, i] += fprimes[2] * 2 * DT * np.imag(lefts[2].overlap(expt_Hi * right))

            for k in range(len(lefts)):
                lefts[k] = Udag * lefts[k]
            right = Udag * right

        grad = grad.reshape((-1))

        if safe:
            appr_grad = approx_fprime(param_vec, self.target_func, appr_eps, init_state, target_state)
            frac_grad_err = sum((appr_grad - grad) ** 2) ** 0.5 / sum(appr_grad ** 2) ** 0.5
            if frac_grad_err > 1:
                print("analytic gradient is very worng.")
                print('{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}'.format(self.target_func(param_vec, init_state, target_state),
                                                              sum(grad ** 2) ** 0.5, sum(appr_grad ** 2) ** 0.5,
                                                              frac_grad_err))

        return grad

    def infidelity(self, param_vec, init_state, target_state):
        evolved_state = self.evolve(init_state, param_vec)
        return 1 - abs(target_state.overlap(evolved_state))**2

    def parse_parameters(self, param_vec):
        # unpack the parameter vector
        amps = self.unpack_param_vec(param_vec)
        Ac = amps[:, 0] + 1j * amps[:, 3]
        Ar = amps[:, 1] + 1j * amps[:, 4]
        Ab = amps[:, 2] + 1j * amps[:, 5]

        output = '\nPiecewiseParameterization\n'
        output = output + 'fc(t) = \\sum_j Ac_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fr(t) = \\sum_j Ar_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fb(t) = \\sum_j Ab_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'

        output = output + '\ncarrier transition:\n'
        output = output + 't_j\tt_{j+1}\tAc\t\tAr\t\tAb\n'
        for j in range(self.num_steps):
            t_j = j / self.num_steps
            t_jp1 = (j + 1) / self.num_steps
            output = output + '{:.3}\t{:.3}\t{:.3e}\t{:.3e}\t{:.3e}\n'.format(t_j, t_jp1, Ac[j], Ar[j], Ab[j])
