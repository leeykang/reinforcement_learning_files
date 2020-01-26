import numpy as np
from arena import Arena
from bandit import Bandit
from player import RandomPlayer, HumanPlayer, QPlayer, UCBQPlayer, GradientPlayer

def example_1():
	"""
	Example 1: Compares rewards and percentage of optimum action selection
	between various methods, using the pit run_mode of Arena. Produces the
	results in the form of two plots.

	"""
	# Initialises the Arena and all required inputs.
	arena = Arena()
	actions_list = [10]
	timesteps_list = [1000]
	runs_list = [2000]
	init_mean_list = [0]
	init_stddev_list = [1]
	action_stddev_list = [1]
	delta_mean_list = [0]
	delta_stddev_list = [0]
	first_considered_reward_step_list = [0]

	# Creates and adds Bandits to the Arena.
	arena.add_bandits([Bandit(*val) for val in zip(actions_list, timesteps_list, runs_list, first_considered_reward_step_list, \
		init_mean_list, init_stddev_list, action_stddev_list, delta_mean_list, delta_stddev_list)])

	# Creates and adds Players to the Arena.
	arena.add_players([RandomPlayer(),
					   QPlayer(initial_Q=0, epsilon=0.1),
					   QPlayer(initial_Q=5, epsilon=0.1),
					   UCBQPlayer(initial_Q=0, confidence_level=2),
					   UCBQPlayer(initial_Q=5, confidence_level=2),
					   GradientPlayer(step_size_parameter=0.1, use_baseline_reward=True),
					   GradientPlayer(step_size_parameter=0.1, use_baseline_reward=False)])

	# Run the Arena in pit mode.
	arena.run('pit')

def example_2():
	"""
	Example 2: Parameter study of various Players on a nonstationary Bandit.
	Produces the results in the form of a single plot.

	For a stationary Bandit, set all values of delta_mean_list and
	delta_stddev_list to 0.

	"""
	# Initialises the Arena and all required inputs.
	arena = Arena()
	actions_list = [10]
	timesteps_list = [1000]
	runs_list = [2000]
	init_mean_list = [0]
	init_stddev_list = [1]
	action_stddev_list = [1]
	delta_mean_list = [0]
	delta_stddev_list = [0.01]
	first_considered_reward_step_list = [0]

	# Initialises the study ranges for all Players.
	epsilon_study_range = np.logspace(-7, -1, num=7, base=2.0, dtype=float).tolist()
	initial_Q_study_range = np.logspace(-2, 3, num=6, base=2.0, dtype=float).tolist()
	confidence_level_study_range = np.logspace(-4, 3, num=8, base=2.0, dtype=float).tolist()
	step_size_parameter_study_range = np.logspace(-5, 2, num=8, base=2.0, dtype=float).tolist()
	parameter_range = np.logspace(-8, 4, num=2, base=2.0, dtype=float).tolist()

	# Creates and adds Bandits to the Arena.
	arena.add_bandits([Bandit(*val) for val in zip(actions_list, timesteps_list, runs_list, first_considered_reward_step_list, \
		init_mean_list, init_stddev_list, action_stddev_list, delta_mean_list, delta_stddev_list)])

	# Creates and adds Players to the Arena.
	arena.add_players([QPlayer(0, epsilon_study_range[0], study_variable='epsilon', study_range=epsilon_study_range), # epsilon greedy, intial_q = 0 (study epsilon)
					   QPlayer(0, epsilon_study_range[0], 0.1, study_variable='epsilon', study_range=epsilon_study_range), # epsilon greedy with alpha 0.1, initial_Q = 0 (study epsilon)
					   QPlayer(initial_Q_study_range[0], 0, 0.1, study_variable='initial_Q', study_range=initial_Q_study_range), # greedy with alpha 0.1 (study initial_Q)
					   UCBQPlayer(0, confidence_level_study_range[0], study_variable='confidence_level', study_range=confidence_level_study_range), # UCB, initial_Q = 0 (study ucb_c)
					   UCBQPlayer(0, confidence_level_study_range[0], 0.1, study_variable='confidence_level', study_range=confidence_level_study_range), # UCB, initial_Q = 0, alpha=0.1 (study ucb_c)
					   GradientPlayer(step_size_parameter_study_range[0], study_variable='step_size_parameter', study_range=step_size_parameter_study_range)]) # gradient bandit with baseline (study alpha)

	# Run the Arena in parameter study mode.
	arena.run('parameter_study', parameter_range)

if __name__ == '__main__':
	example_1()
	# example_2()
