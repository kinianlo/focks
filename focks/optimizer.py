import numpy as np
from numpy import dot, vdot, sqrt
from numpy.linalg import eigh
from qutip import Qobj

from focks.utils import expmh, transform, relu, relu_der
from .interaction import LaserInteraction
from scipy.optimize import minimize, approx_fprime
from scipy.optimize.optimize import wrap_function, vecnorm, _line_search_wolfe12, _LineSearchError, _status_message, \
    OptimizeResult

from typing import Callable, Sequence

EPS = np.finfo(float).eps
SQEPS = np.finfo(float).eps ** 0.5

def minimize_bfgs(func: Callable,
                  init_param_vec: Sequence,
                  grad: Callable = None,
                  grad_tol: float = 1e-5,
                  return_all: bool = True,
                  last_record: 'OptimizeRecord' = None,
                  notes: dict = None):
    if notes is None:
        notes = {}
    notes["grad_tol"] = grad_tol
    notes["method"] = "BFGS"

    f = func
    fprime = grad
    epsilon = np.finfo(float).eps ** 0.5
    gtol = grad_tol
    norm = np.Inf

    x0 = init_param_vec
    if x0.ndim == 0:
        x0.shape = (1,)
    maxiter = len(x0) * 200

    func_calls, f = wrap_function(f, ())
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
        notes["grad_approx"] = True
    else:
        grad_calls, myfprime = wrap_function(fprime, ())
        notes["grad_approx"] = False

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)

    if last_record:
        old_fval = last_record.final_func
        gfk = last_record.final_grad
        old_old_fval = last_record.last_vars["old_old_fval"]
        Hk = last_record.last_vars["Hk"]

    else:
        # Sets the initial step guess to dx ~ 1
        old_fval = f(x0)
        gfk = myfprime(x0)
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2
        Hk = I

    all_param_vec = [x0]
    all_func = [old_fval]
    all_grad = [gfk]

    xk = x0
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                     old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1

        k += 1

        if return_all:
            all_param_vec.append(xk)
            all_func.append(old_fval)
            all_grad.append(gfk)

        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (np.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            print("Divide-by-zero encountered: rhok assumed large")
        if np.isinf(rhok):  # this is patch for numpy
            rhok = 1000.0
            print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])


    fval = old_fval
    if np.isnan(fval):
        # This can happen if the first call to f returned NaN;
        # the loop is then never entered.
        warnflag = 2

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    history = {"func": all_func, "grad": all_grad, "param_vec": all_param_vec}
    final_status = {"msg": msg, "warnflag": warnflag, "num_func_call": func_calls[0],
                    "num_grad_call": grad_calls[0], "num_iter": k}
    last_vars = {"Hk": Hk, "old_old_fval": old_old_fval}

    record = OptimizeRecord(history, final_status, last_vars, notes)

    return record


class InfidelityOptimizer:
    """
    Optimizer for the complex control amplitudes for a given set of control operators.

    Attributes
    ----------
    interaction : LaserInteraction
        The interaction object which specifies the control operators available for control.
    init_state : Qobj
        The initial state which must be a ket Qobj.
    target_state : Qobj
        The target state which must be a ket Qobj.

    """

    def __init__(self,
                 interaction: LaserInteraction,
                 init_state: Qobj,
                 target_state: Qobj,
                 num_steps: int,
                 energy_weight: float = 0,
                 max_energy: float = 0,
                 step_duration: float = 1):
        assert isinstance(init_state, Qobj)
        assert init_state.isket
        assert isinstance(target_state, Qobj)
        assert target_state.isket
        assert isinstance(interaction, LaserInteraction)

        self._interaction = interaction
        self._init_state = init_state.full()
        self._target_state = target_state.full()
        self._num_steps = num_steps
        self._energy_weight = energy_weight
        self._max_energy = max_energy
        self._step_duration = step_duration

        self._c_ops = [op.full() for op in interaction.control_operators]

        self._optim_records = []

        self._cache_param_vec = None
        self._cache_evolved_state = None

    @property
    def interaction(self) -> LaserInteraction:
        return self._interaction

    @property
    def init_state(self) -> Qobj:
        return Qobj(self._init_state, dims=[[2, self._interaction.num_focks], [1, 1]])

    @property
    def target_state(self) -> Qobj:
        return Qobj(self._target_state, dims=[[2, self._interaction.num_focks], [1, 1]])

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def step_duration(self) -> float:
        return self._step_duration

    @init_state.setter
    def init_state(self, value: Qobj):
        assert isinstance(value, Qobj) and Qobj.isket
        assert value.dims[0][1] == self._interaction.num_focks
        self._init_state = value

    @target_state.setter
    def target_state(self, value: Qobj):
        assert isinstance(value, Qobj) and Qobj.isket
        assert value.dims[0][1] == self._interaction.num_focks
        self._target_state = value

    @property
    def optim_records(self):
        return self._optim_records

    def _flatten_param(self, param):
        """
        Flatten a parameters into a real 1-dimensional array.

        Parameters
        ----------
        param : ndarray
            The complex parameters.

        Returns
        -------
        param_vec : 1-d array
            The real flat parameter vector.

        """
        mag = np.abs(param).flatten()
        phase = np.angle(param).flatten()
        param_vec = np.concatenate((mag, phase))
        return param_vec

    def _param_vec_to_all_amps(self, param_vec):
        """
        Contrust the list of all control amplitudes (at all time steps) from a parameter vector.

        Parameters
        ----------
        param_vec : 1-d array

        Returns
        -------
        param : ndarray
            The complex parameters.

        """
        assert len(param_vec) == self._num_steps * len(self._c_ops) * 2
        mag_vec, phase_vec = np.split(param_vec, 2)
        mag = mag_vec.reshape((self._num_steps, len(self._c_ops)))
        phase = phase_vec.reshape((self._num_steps, len(self._c_ops)))
        param = mag * np.exp(1j * phase)
        return param

    def _param_vec_to_all_polar(self, param_vec):
        assert len(param_vec) == self._num_steps * len(self._c_ops) * 2
        mag_vec, phase_vec = np.split(param_vec, 2)
        mag = mag_vec.reshape((self._num_steps, len(self._c_ops)))
        phase = np.exp(1j * phase_vec.reshape((self._num_steps, len(self._c_ops))))
        return mag, phase

    def random_param_vec(self, amp_max=1.0, fixed_energy=None):
        mag = np.random.rand(self._num_steps * len(self._c_ops)) * amp_max
        phase = (np.random.rand(self._num_steps * len(self._c_ops)) - 0.5) * 2 * np.pi
        param_vec = np.concatenate((mag, phase))
        if fixed_energy is not None:
            energy = self.param_vec_energy(param_vec)
            param_vec[:3*self.num_steps] *= sqrt(fixed_energy/energy)
        return param_vec

    def param_vec_energy(self, param_vec):
        all_amps = self._param_vec_to_all_amps(param_vec)
        eta = self._interaction.lamb_dicke
        energy = np.sum(np.abs(all_amps) ** 2 * np.array([eta ** 2, 1, 1]))
        return energy

    def infidelity(self, param_vec):
        all_amps = self._param_vec_to_all_amps(param_vec)
        evolved_state = self._init_state
        for j in range(self._num_steps):
            evolved_state = self._interaction.evolve(evolved_state, all_amps[j], self._step_duration)
        infidelity = 1 - abs(vdot(self._target_state, evolved_state))
        return infidelity

    def _check_truncation(self, state):
        num_focks = self._interaction.num_focks
        if abs(state[-1])**2 > self._tol or abs(state[num_focks - 1])**2 > self._tol:
            raise ValueError("Truncation too low for tol={:}.".format(self._tol))

    def cost_func(self, param_vec) -> float:
        all_amps = self._param_vec_to_all_amps(param_vec)
        evolved_state = self._init_state
        for j in range(self._num_steps):
            evolved_state = self._interaction.evolve(evolved_state, all_amps[j], self._step_duration)

        self._check_truncation(evolved_state)

        self._cache_param_vec = param_vec
        self._cache_evolved_state = evolved_state

        #return 1 - abs(self._target_state.overlap(state)) ** 2
        return 1 - abs(vdot(self._target_state, evolved_state))

    def cost_grad(self, param_vec):
        """
    
        :param param_vec: Parameter vector at which the grident is evaluated
        :param init_state:
        :param target_state:
        :param safe: If True then check gradient with scipy approx_fprime
        :return: gradient (same shape as param_vec)
        """
        
        all_amps = self._param_vec_to_all_amps(param_vec)
        all_mags, all_phases = self._param_vec_to_all_polar(param_vec)

        if np.array_equal(param_vec, self._cache_param_vec):
            evolved_state = self._cache_evolved_state
        else:
            evolved_state = self._init_state
            for j in range(self._num_steps):
                evolved_state = self._interaction.evolve(evolved_state, all_amps[j], self._step_duration)
    
        right = evolved_state
        lefts = []
    
        # Infidelity (Phi0)
        #lefts.append((self._target_state.proj() * evolved_state).full())
        lefts.append(vdot(self._target_state, evolved_state) * self._target_state )
    
        grad_mag = np.empty((self.num_steps, 3))
        grad_phase = np.empty((self.num_steps, 3))

        for j in reversed(range(self.num_steps)):
            H = 0
            for i in range(3):
                if all_amps[j, i] == 0:
                    continue
                H += all_amps[j, i] * self._c_ops[i]
            H += H.conj().T
    
            # Diagonalise H
            e_vals, e_kets = eigh(H)
            # Evaluate the integral for "average" Hi on a "for each matrix element fashion"all_ampsd
            L = -1j * (e_vals.reshape((-1, 1)) - e_vals)
            E = np.divide(np.exp(L) - 1, L, out=np.ones_like(L), where=L != 0)
    
            for i in range(3):
                # prepare the "expected" control Hamiltonians
                Hi_mag = all_phases[j, i] * self._c_ops[i]
                Hi_mag += Hi_mag.conj().T

                Hi_phase = 1j * all_amps[j, i] * self._c_ops[i]
                Hi_phase += Hi_phase.conj().T

                expt_Hi_mag = transform(e_kets, E * transform(e_kets.conj().T, Hi_mag))
                expt_Hi_phase = transform(e_kets, E * transform(e_kets.conj().T, Hi_phase))
    
                # Infidelity (Phi0)
                grad_mag[j, i] = -2 * np.imag(vdot(lefts[0], dot(expt_Hi_mag, right)))
                grad_phase[j, i] = -2 * np.imag(vdot(lefts[0], dot(expt_Hi_phase, right)))

            Udag = expmh(H, 1j)
            #Udag = (1j * H).expm()

            for k in range(len(lefts)):
                lefts[k] = dot(Udag, lefts[k])
            right = dot(Udag, right)

        eta = self.interaction.lamb_dicke

        grad = np.concatenate((grad_mag.flatten(), grad_phase.flatten()))
    
        return grad

    def optimize(self, notes: dict = None):
        if notes is None:
            notes = {}
        init_param_vec = self.random_param_vec()
        notes = {"cost_func": "infidelity"}
        record = minimize_bfgs(self.cost_func, init_param_vec, grad=self.cost_grad, notes=notes)
        self._optim_records.append(record)
        return record

    def optimize_constrain_energy(self, max_energy, init_param_vec=None, tol=1e-8):
        self._tol = tol
        if init_param_vec is None:
            init_param_vec = self.random_param_vec(fixed_energy=max_energy)

        eta = self._interaction.lamb_dicke

        weights = np.zeros(6*self.num_steps)
        weights[:3*self.num_steps] = 1
        weights[:3*self.num_steps:3] = eta ** 2

        cons = [{"type": "ineq",
                 "fun": lambda x: max_energy - np.sum(x ** 2 * weights),
                 "jac": lambda x: -2 * x * weights}]
        options = {"maxiter": 1000, "disp": True}
        result = minimize(self.cost_func, init_param_vec, tol=tol,
                          jac=self.cost_grad, constraints=cons, options=options)
        return result


class OptimizeRecord:
    def __init__(self,
                 history: dict,
                 final_status: dict,
                 last_vars: dict,
                 notes: dict):
        self._history = history
        self._final_status = final_status
        self._last_vars = last_vars
        self._notes = notes

    @property
    def func_history(self):
        return self._history["func"]

    @property
    def grad_history(self):
        return self._history["grad"]

    @property
    def param_vec_history(self):
        return self._history["param_vec"]

    @property
    def final_func(self):
        return self._history["func"][-1]

    @property
    def final_grad(self):
        return self._history["grad"][-1]

    @property
    def final_param_vec(self):
        return self._history["param_vec"][-1]

    @property
    def init_func(self):
        return self._history["func"][0]

    @property
    def init_grad(self):
        return self._history["grad"][0]

    @property
    def init_param_vec(self):
        return self._history["param_vec"][0]

    @property
    def last_vars(self):
        return self._last_vars

    @property
    def msg(self):
        return self._final_status["msg"]

    @property
    def warnflag(self):
        return self._final_status["warnflag"]

    @property
    def succeed(self):
        return self._final_status["warnflag"] == 0

    @property
    def num_func_call(self):
        return self._final_status["num_func_call"]

    @property
    def num_grad_call(self):
        return self._final_status["num_grad_call"]

    @property
    def num_iter(self):
        return self._final_status["num_iter"]

    @property
    def notes(self):
        return self._notes

    @property
    def final_status(self):
        return self._final_status