from platypus import NSGAII, DTLZ2, Problem, Real, NSGAIII, SPEA2
from focks.optimizer import InfidelityOptimizer
from focks.interaction import LaserInteraction
from numpy import pi, array

# define the problem definition
from focks.utils import expand_state

num_focks = 50
num_steps = 7
interaction = LaserInteraction(num_focks, 0.1)
e, g = interaction.eigenstate_generators
import pickle
state = pickle.load(open('/home/kin/urop/focks/data/states/fl4.pickle', 'rb'))
state = expand_state(state, g(0).dims)
optim = InfidelityOptimizer(interaction, state, g(0), num_steps, energy_weight=0.0, max_energy=4.439301)

problem = Problem(6*num_steps, 2)
problem.types[:3*num_steps] = Real(0, pi)
problem.types[3*num_steps:] = Real(-pi, pi)
problem.function = lambda p: (optim.infidelity(array(p)), optim.param_vec_energy(array(p))/1.9169321)


# instantiate the optimization algorithm


# optimize the problem using 10,000 function evaluations


# plot the results using matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
algorithm = NSGAII(problem)
algorithm.run(10000)
ax.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
algorithm = NSGAII(problem, population_size=200)
algorithm.run(20000)
ax.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
ax.set_xlabel("$f_1(x)$")
ax.set_ylabel("$f_2(x)$")

fig.show()