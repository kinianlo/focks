class TargetFunction:
    def __init__(self):

class PenaltyTerm:
    """
    Abstract class for a penalty term in the target function.
    """
    def __init__(self, weight=1):
        self.weight = weight

    def function(self, evolved_state, target_state):
        raise NotImplementedError

    def gradient(self, evolved_state, target_state, func_val, left, right):
        raise NotImplementedError

    def init_left(self, target_state):
        """
        Return the initial "left" for efficient calculation
        of the gradient.
        """
        raise NotImplementedError

class Infidelity(PenaltyTerm):
    """
    A penalty term that measures the infidelity between the evolved state
    and the target state.
    """
    def function(self, evolved_state, target_state):
        return 1 - abs(target_state.overlap(evolved_state))**2

    def gradient(self, evolved_state, target_state, func_val, left, right):
        pass

    def init_left(self, target_state):

class FirstMomentTerm(PenaltyTerm):
    """
    A penalty term that measures the distance (in fock space) between
    the centre of mass of the evolved state and the target state.
    """
    def function(self, evolved_state, target_state):
       pass
