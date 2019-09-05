from qutip import Qobj
from setup import IonTrapSetup

class InfidelityOptimizer:

    def __init__(self, setup, init_state, target_state):
        """
        :param IonTrapSetup setup: a setup of an ion trap
        :param Qobj init_state: initial state
        :param Qobj target_state: target state
        """
        self._setup = setup
        self._init_state = init_state
        self._target_state = target_state
        
        self._c_ops = self._setup.control_operators

    @property.getter
    def setup(self):
        return self._setup

    @property.getter
    def init_state(self):
        return self._init_state

    @property.getter
    def target_state(self):
        return self._target_state

    def func(self, param):
        evolved_state = self._setup.evolve(param, self._init_state)