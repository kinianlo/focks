import numpy as np
from numpy import dot
from numpy.linalg import eigh
from qutip import Qobj, basis, destroy, qeye, sesolve, sigmap, tensor, rand_ket
from typing import Callable, List, Sequence, Tuple, Union
from focks.utils import expmh


class LaserInteraction:
    """
    Specification of a set of interactions between an trapped ion and lasers.
    The interaction Hamiltonian consider here is time-dependent and is a weighted sum of `control operators`
    weighted by the `control amplitudes`. The time dependence comes from the control amplitudes and the
    control operators are constant in time. To be precise, the Hamiltonian has the form

    .. math:: \hat{H}(t) = \sum_{i=1}^N f_i(t) \hat{O}_i + h.c.,

    where N is the number of `control operators`;
    :math:`\hat{O}_i` is the i-th `control operator`;
    :math:`f_i(t)` is the i-th `control amplitude`;
    and `h.c.` refers to the Hermitian conjugate of the previous term.

    Attributes
    ----------
    num_focks : int
        The truncation of the Fock space. The Fock space is infinite dimensional and thus the a truncation is
        need to perform calculation using matrices. The num_focks should be large enough such that the highest
        few eigenstate has negligible population during any evolution. Otherwise, the state would leak outside
        of the system.

    control_operators : list of Qobj
        The `control operators` :math:`\hat{O}_i`.

    eigenstate_generators : tuple of callable
        The state generator for excited state and ground state. For example, if
        (e, g) = state_generators
        then e(4) returns the ket vector of the state |e 4>.
    """

    def __init__(self, num_focks: int, lamb_dicke):
        self.num_focks = num_focks
        self._lamb_dicke = lamb_dicke

    @property
    def control_operators(self) -> List[Qobj]:
        return self._ctrl_ops

    @property
    def num_focks(self) -> int:
        return self._num_focks

    @num_focks.setter
    def num_focks(self, value):
        self._num_focks = value
        sigma_plus: Qobj = tensor(sigmap(), qeye(value))
        a: Qobj = tensor(qeye(2), destroy(value))

        self._ctrl_ops: List[Qobj] = [sigma_plus, sigma_plus * a, sigma_plus * a.dag()]
        self._ctrl_ops_full = [op.full() for op in self._ctrl_ops]

        self._e: Callable[[int], Qobj] = lambda n: tensor(basis(2, 0), basis(value, n))
        self._g: Callable[[int], Qobj] = lambda n: tensor(basis(2, 1), basis(value, n))

    @property
    def lamb_dicke(self):
        return self._lamb_dicke

    @lamb_dicke.setter
    def lamb_dicke(self, value):
        self._lamb_dicke = value

    @property
    def eigenstate_generators(self) -> Tuple[Callable[[int], Qobj], Callable[[int], Qobj]]:
        return self._e, self._g

    def get_random_state(self):
        return Qobj(rand_ket(2 * self._num_focks), dims=[[2, self._num_focks], [1, 1]])

    def interaction_hamiltonian(self, amps: List[complex]) -> Qobj:
        """
        Return the interaction Hamiltonian given a set of control amplitudes.

        Parameters
        ----------
        amps : list of complex numbers
            The control amplitudes.

        Returns
        -------
        Obj
            The interaction hamiltonian.

        """
        half_op: Qobj = None
        for i in range(0, len(self._ctrl_ops)):
            if np.abs(amps[i]) == 0:
                continue
            if half_op is None:
                half_op = amps[i] * self._ctrl_ops[i]
            else:
                half_op += amps[i] * self._ctrl_ops[i]
        if half_op is None:
            return Qobj(np.zeros(self._ctrl_ops[0].shape), dims=self._ctrl_ops[0].dims)
        return half_op + half_op.dag()

    def evolve(self,
               init_state: Qobj,
               amps: List[Union[complex, Callable[[float], complex]]],
               duration: float = 1.0) -> Qobj:
        """
        Return the evolved state under a set of control amplitudes.

        Parameters
        ----------
        init_state : Qobj
            The initial state.
        amps : list of complex numbers or list of callables
            The control amplitudes. Must be a list of `complex numbers` or `callables` whose length is exactly
            the number control operators. When the amps are given as a list of `complex numbers`, the control
            amplitudes are assumed to be constant in time and thus the calculations can be much quicker. Otherwise,
            a set of PDEs are needed to be solved. Note that the `callables` must return complex numbers.
        duration : float
            The span of time of the evolution.

        Returns
        -------
        evolved_state : Qobj or ndarray
            The evolved state as a ket Qobj or a numpy array.
        """

        assert len(amps) == len(self._ctrl_ops)

        if isinstance(amps[0], (complex, float, int)):
            if isinstance(init_state, Qobj):
                evolved_state = Qobj(self._evolve_constant(init_state.full(), amps, duration), dims=init_state.dims)
            else:
                evolved_state = self._evolve_constant(init_state, amps, duration)

        elif isinstance(amps[0], Callable):
            evolved_state = self._evolve_varying(init_state, amps, duration)

        else:
            raise TypeError("Argument `amps` must be a list of numbers or a list of callables.")

        return evolved_state

    def _evolve_constant(self,
                         init_state: np.ndarray,
                         amps: List[complex],
                         duration: float) -> np.ndarray:
        """
        Return the evolved state under a set of time-constant control amplitudes.

        Parameters
        ----------
        init_state : Qobj
            The initial state.
        amps : list of complex numbers
            The control amplitudes. Must be a list of complex numbers whose length is exactly the number of control
            operators.
        duration : float
            The span of time of the evolution.

        Returns
        -------
        evolved_state : Qobj
            The evolved state as a ket Qobj.
        """
        if np.count_nonzero(amps) == 0:
            return init_state
        # H = self.interaction_hamiltonian(amps)
        H = 0
        for i in range(3):
            if amps[i] == 0:
                continue
            H += amps[i] * self._ctrl_ops_full[i]
        H += H.conj().T

        U = expmh(H, -1j * duration)
        return dot(U, init_state)

        # U = (-1j * duration * H).expm()
        # return U * init_state

    def _evolve_varying(self,
                        init_state: Qobj,
                        amps: List[Callable[[float], complex]],
                        duration: float) -> Qobj:
        """
        Return the evolved state under a set of time-varying control amplitudes.

        Parameters
        ----------
        init_state : Qobj
            The initial state.
        amps : list of complex numbers
            The control amplitudes. Must be a list of callables (which must return complex number) whose length is
            exactly the number of control operators.
        duration : float
            The span of time of the evolution.

        Returns
        -------
        evolved_state : Qobj
            The evolved state as a ket Qobj.
        """
        raise NotImplementedError

    def monitor(self,
                init_state: Qobj,
                amps: List[Callable],
                times: Sequence[float],
                observables: List[Qobj] = None) -> Union[List[Qobj], List[Sequence[complex]]]:
        """
        Monitor the system at a range of points in time.
        This method allows one to look at the system continuously during a evolution. More specifically, the
        expectation value of the specified observables are returned at the specified points in time. If no observables
        are specified, the full ket vectors of the state at the specified points of time are returned.

        Parameters
        ----------
        init_state : Qobj
            The initial state.
        amps : list of complex numbers
            The control amplitudes. Must be a list of callables (which must return complex number) whose length is
            exactly the number of control operators.
        times : array_like
            The points in time where the monitoring are done.
        observables : list of Qobj or None
            The observables for which the expectation values are calculated. If None, then the ket vectors are
            returned instead of some expectation values.

        Returns
        -------
        list of floats

        Notes
        -----
        The state is not collapsed during monitoring, only the expectation values are evaluated and the state
        is not affected (which is of course impossible in an actual experiment). That is why this method is not called
        `measure` or `observe` to avoid confusion.

        """
        H_td = []

        H_td.append([self._ctrl_ops[0], lambda t, *_: amps[0](t)])
        H_td.append([self._ctrl_ops[0].dag(), lambda t, *_: np.conj(amps[0](t))])

        H_td.append([self._ctrl_ops[1], lambda t, *_: amps[1](t)])
        H_td.append([self._ctrl_ops[1].dag(), lambda t, *_: np.conj(amps[1](t))])

        H_td.append([self._ctrl_ops[2], lambda t, *_: amps[2](t)])
        H_td.append([self._ctrl_ops[2].dag(), lambda t, *_: np.conj(amps[2](t))])

        if observables is None:
            return sesolve(H_td, init_state, times, []).states
        else:
            return sesolve(H_td, init_state, times, observables).expect

    def pulse_sequence_energy(self, pulse_sequence):
        if len(pulse_sequence) == 0:
            return 0
        eta = self._lamb_dicke
        return np.sum(np.abs(pulse_sequence) ** 2 * np.array([eta ** 2, 1, 1]))
    
    def pulse_sequence_prep_time(self, pulse_sequence):
        if len(pulse_sequence) == 0:
            return 0
        eta = self._lamb_dicke
        weights = np.array([eta**2, 1, 1])
        return np.sum(np.sum(np.abs(pulse_sequence).reshape((-1, 3))**2 * weights,
                                                             axis=1)**0.5)



