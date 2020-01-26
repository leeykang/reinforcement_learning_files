import numpy as np

class Bandit:
	"""
	Provides the definition of a stationary/non-stationary generic multi-armed
	bandit task.

	Parameter(s):
	num_actions: Number of possible actions/choices/arms in the Bandit task.

	num_timesteps: Number of timesteps for the Bandit per run.

	num_runs: Number of runs for the Bandit.

	first_considered_reward_step_list: The first reward step to be considered
	for evaluation of a particular Player.

	init_mean (optional, default 0): Initial mean of Bandit, used with
	init_stddev to compute the true reward (mean) of every action of the Bandit.

	init_stddev (optional, default 0): Initial standard deviation of Bandit,
	used with init_mean to compute the true reward (mean) of every action of the
	Bandit. Using the default of 0 is equivalent to having all actions with the
	same true reward (mean) = init_mean.

	action_stddev (optional, default 0): Standard deviation when obtaining
	reward, used with the true reward (mean) of an action to obtain the current
	reward for taking that action. Using the default of 0 is equivalent to
	always returning the true reward (mean) of an action when taking that
	particular action.

	delta_mean (optional, default 0): Mean of change to the true reward (mean)
	of all actions after every timestep of each run. Used with delta_stddev to
	obtain the change. Both delta_mean and delta_stddev must be 0 (the default)
	for stationary tasks.

	delta_stddev (optional, default 0): Standard deviation of change to the true
	reward (mean) of all actions after every timestep of each run. Used with
	delta_mean to obtain the change. Both delta_stddev and delta_mean must be 0
	(the default) for stationary tasks.

	"""
	def __init__(self,
				 num_actions,
				 num_timesteps,
				 num_runs,
				 first_considered_reward_step,
				 init_mean=0,
				 init_stddev=0,
				 action_stddev=0,
				 delta_mean=0,
				 delta_stddev=0):

		# Initialises the multi-armed bandit task based on the given parameters.
		self.num_actions = num_actions
		self.num_timesteps = num_timesteps
		self.num_runs = num_runs
		self.first_considered_reward_step = first_considered_reward_step
		self.init_mean = init_mean
		self.init_stddev = init_stddev
		self.action_stddev = action_stddev
		self.delta_mean = delta_mean
		self.delta_stddev = delta_stddev

	def analyse(self):
		"""
		Computes Bandit values associated with its input parameters. This
		includes:
		1. Checking whether or not the Bandit is stationary.
		2. Computing the true reward (mean) of every action of the Bandit.
		3. Obtaining the Bandit's best action by finding the action index with
		the highest true reward (mean).

		"""
		# The Bandit is stationary if both self.delta_mean and self.delta_stddev
		# are 0.
		self.stationary = self.delta_mean == 0 and self.delta_stddev == 0

		# Obtains the true reward of all actions by sampling from a Gaussian
		# distribution with self.init_mean as mean and self.init_stddev as
		# standard deviation.
		self.mean = np.random.normal(self.init_mean, self.init_stddev, (self.num_actions,))

		# Finds the best action by finding the action index with the highest
		# true reward (self.mean).
		self.best_action = self.find_best_action()

	def find_best_action(self):
		"""
		Finds the best action by finding the action index with the highest true
		reward (self.mean).

		Return(s):
		The action index corresponding to the best index.

		"""
		return np.argmax(self.mean)

	def is_best_action(self, action_idx):
		"""
		Checks if an action index is the best action.

		Parameter(s):
		action_idx: An action index representing the action to be taken for
		the Bandit

		Return(s):
		The integer 1 if the action index is the best action, and the integer 0
		otherwise.

		"""
		# Ensures that the action_idx provided is valid.
		assert 0 <= int(action_idx) < self.num_actions, \
			"%s is not a valid action index, should be an integer between 0 and (number of actions - 1)"

		# Checks if the Bandit's best action is equals to the provided
		# action_idx and converts the result to an integer.
		return (self.best_action == action_idx) * 1

	def display(self):
		"""
		Displays parameters and values associated with the Bandit.

		"""
		if self.stationary:
			print('STATIONARY PROBLEM')

		else:
			print('NONSTATIONARY PROBLEM')

		print('INITIAL MEAN: %f' % self.init_mean)
		print('INITIAL STANDARD DEVIATION: %f' % self.init_stddev)
		print('ACTION STANDARD DEVIATION: %f' % self.action_stddev)

		if not self.stationary:
			print('DELTA MEAN: %f' % self.delta_mean)
			print('DELTA STANDARD DEVIATION: %f' % self.delta_stddev)

		print('CURRENT MEAN:', self.mean)
		print('CURRENT BEST ACTION: %i' % self.best_action)
		print("")

	def update_params(self, **kwargs):
		"""
		Update parameters of the Bandit and reanalyses values of the Bandit.
		Uses the same keyword arguments as those for initialising the Bandit
		(num_actions, init_mean, init_stddev, action_stddev, delta_mean,
		delta_stddev)

		"""
		# Only reanalyses Bandit if kwargs contains keyword arguments.
		if kwargs:
			for key, value in kwargs.items():
				# Ensure that the key corresponds to a parameter in the Bandit.
				assert key in ['num_actions', 'init_mean', 'init_stddev', \
							   'action_stddev', 'delta_mean', 'delta_stddev'], \
					'%s IS NOT A BANDIT PARAMETER' % key

				# Sets the parameter in Bandit to the new input value.
				setattr(self, key, value)

			# Reanalyses the Bandit after all parameters have been updated.
			self.analyse()

	def update(self):
		"""
		If the Bandit is nonstationary, updates the true reward (mean) of all
		actions of the Bandit after a step and recomputes the new best action.

		"""
		if not self.stationary:
			# Adds random variables taken from a Gaussian distribution with mean
			# of self.delta_mean and standard deviation of self.delta_stddev to
			# the true reward (mean) of all actions.
			self.mean += np.random.normal(self.delta_mean, self.delta_stddev, (self.num_actions,))

			# Recompute the new best action.
			self.best_action = self.find_best_action()

	def evaluate(self, action_idx):
		"""
		Obtains the reward of a taken action and whether the action is the
		current best action.

		Parameter(s):
		action_idx: An action index representing the action to be taken for
		the Bandit

		Return(s):
		The action reward and the integer 1 if the action index is the best
		action, and the integer 0 otherwise.

		"""
		# Obtain the reward for taking a particular action by sampling a random
		# variable from a Gaussian distribution with the true reward (mean) of
		# that action as the mean and self.action_stddev as the standard
		# deviation.
		action_reward = np.random.normal(self.mean[action_idx], self.action_stddev)

		# Returns the action reward and whether the taken action is the best
		# action.
		return action_reward, self.is_best_action(action_idx)
