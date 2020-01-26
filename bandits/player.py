import numpy as np
import random
from abc import ABC, abstractmethod
from scipy.special import softmax

class Player(ABC):
	"""
	Player Class:

	Provides the definition of a Player engaging with a Bandit.

	Parameter(s):
	player_type: Type of Player. Typically defined automatically as an input
	when initialising a subclass of Player. Current possible arguments include:
	1. random: No computer learning involved, a random action is taken.

	2. human: No computer learning involved, takes user input as the action -
	not recommended since a huge number of inputs will be required.

	3. q_estimate: Uses Q value estimates to obtain action values.

	4. ucb_q_estimate: Uses Q value estimates with upper confidence bound
	methods to obtain action values.

	5. gradient: Uses gradient ascent to obtain action preferences.

	study_variable (optional, default ''): Only used for parameter studies,
	denotes the variable to be studied as part of the parameter study. The
	chosen variable must be present in the subclassed Player. If the run_mode
	for the Arena is pit, this argument will not be used.

	study_range (optional, default []): Only used for parameter studies,
	provides a range of values for the study_variable to be tested. If the
	run_mode for the Arena is pit, this argument will not be used.

	"""
	def __init__(self,
				 player_type,
				 study_variable='',
				 study_range=[]):

		# Initialises the Player based on the given parameters.
		self.player_type = player_type

		# Only creates study-related objects if a study_variable is provided.
		self.study_variable = study_variable
		if self.study_variable:
			self.study_range = study_range
			self.study_result = []

	def stored_reset(self):
		"""
		Resets stored reward and optimum variables of the Player. Typically
		used after introducing a new Bandit or after one round of a parameter
		study.

		"""
		self.stored_reward = np.zeros((self.num_timesteps - self.first_considered_reward_step,))
		self.stored_optimum = np.zeros_like(self.stored_reward)

	def reset(self):
		"""
		Resets current selected actions, reward and optimum variables of the
		Player. Typically used at the start of any run.

		"""
		self.player_selected_actions = np.zeros((self.num_actions,), int)
		self.player_reward = np.zeros((self.num_timesteps,))
		self.player_optimum = np.zeros_like(self.player_reward, dtype=int)

	def display(self):
		"""
		Displays relevant parameters of the Player.

		"""
		for key, value in self.__dict__.items():
			print(key.upper(), value, sep=': ')

		print("")

	def update(self, timestep, action, reward, is_optimal):
		"""
		Provides feedback to Player and, if required, performs updates for the
		Player. The Player class only implements Player feedback, with Player
		updates individually handled by Player subclass objects. Typically used
		at the end of every timestep.

		Parameter(s):
		timestep: Index of timestep

		action: Index of action

		reward: Reward obtained from performing action on the current Bandit

		is_optimal: Integer indicating whether the performed action was optimal
		(1 being yes and 0 being no)

		"""
		# Increments the Player's selected action index by 1.
		self.player_selected_actions[action] += 1

		# Sets the Player reward for the particular timestep to the reward
		# obtained.
		self.player_reward[timestep] = reward

		# Sets the Player optimum for the particular timestep to the is_optimal
		# obtained.
		self.player_optimum[timestep] = is_optimal

	def update_params(self, **kwargs):
		"""
		Update parameters of the Player and performs a full reset of the Player
		(reset and stored_reset). Parameters that can be updated include
		num_runs, num_actions, num_timesteps, first_considered_reward_step and
		any parameter associated with a Player subclass object.

		"""
		# Only resets Player if kwargs contains keyword arguments.
		if kwargs:
			# Assume that a full reset is not necessary prior to analysing
			# kwargs.
			to_full_reset = False

			for key, value in kwargs.items():
				# Ensure that the key corresponds to a valid updating parameter
				# for Player.
				if key in {'num_runs', 'num_actions', 'num_timesteps', 'first_considered_reward_step'}:
					# Sets the parameter in Player to the new input value.
					setattr(self, key, value)

					# Enables a subsequent full reset of Player.
					to_full_reset = True

				# Ensure that the key corresponds to a valid updating parameter
				# for the Player subclassed object.
				elif key in {'initial_Q', 'epsilon', 'step_size_parameter', 'confidence_level', 'use_baseline_reward'}:
					# Sets the parameter in Player to the new input value.
					setattr(self, key, value)

					# Enables a subsequent full reset of Player.
					to_full_reset = True

				else:
					# Raises AssertionError since key is not valid.
					raise AssertionError('%s NOT IN ACCEPTABLE PLAYER PARAMETERS' % key)

			# If a full reset of Player is enabled, perform reset and
			# stored_reset.
			if to_full_reset:
				self.reset()
				self.stored_reset()

	@abstractmethod
	def obtain_action(self):
		"""
		Obtains a valid action from Player. All Player subclass objects must
		implement this method and return the same kind of value.

		Return(s):
		An action index corresponding to a valid action taken by the Player.

		"""
		return

class RandomPlayer(Player):
	"""
	Player that performs actions randomly.

	No additional parameters required other than those for the Player
	superclass.

	"""
	def __init__(self, *args, **kwargs):
		super().__init__('random', *args, **kwargs)

	def obtain_action(self, timestep):
		"""
		Obtains a valid action from Player randomly.

		Parameter(s):
		timestep: Index of timestep, dummy input for this Player subclass.

		Return(s):
		An action index corresponding to a valid action taken by the Player.

		"""
		return random.randint(0, self.num_actions-1)

class HumanPlayer(Player):
	"""
	Player that performs actions based on manual input. Not recommended to use
	because every action would require an input.

	No additional parameters required other than those for the Player
	superclass.

	"""
	def __init__(self, *args, **kwargs):
		super().__init__('human', *args, **kwargs)

	def obtain_action(self, timestep):
		"""
		Obtains a valid action from Player manually.

		Parameter(s):
		timestep: Index of timestep, dummy input for this Player subclass.

		Return(s):
		An action index corresponding to a valid action taken by the Player.

		"""
		# Loops constantly until a valid input is obtained.
		while True:
			try:
				# Tries to obtain a valid input manually and convert it to an
				# integer.
				action = int(input('Please provide an input action index between 0 and (number of actions - 1): %i: ' % (self.num_actions-1)))

			except ValueError:
				print('Invalid input detected, try again.')
				continue

			# Checks if the input is within the acceptable range of action
			# index values.
			if 0 <= action < self.num_actions:
				break
			else:
				print('Action should be an index between 0 and (number of actions - 1): %i' % (self.num_actions-1))

		return action

class QPlayer(Player):
	"""
	Player that uses Q values to estimate action values and decide on the best
	action to take.

	Parameter(s) (in addition to those required by the Player superclass):
	initial_Q (optional, default 0): Starting Q value estimate for all actions.
	The default of 0 means that the estimated reward for all actions is 0.

	epsilon (optional, default 0): Probability for performing a random action
	(action is chosen randomly with probability epsilon and deterministically
	with probability 1-epsilon). The default of 0 is equivalent to all actions
	being performed deterministically.

	step_size_parameter (optional, default None): Step size parameter for
	weighing rewards from prior timesteps. Acceptable inputs include None (the
	default, essentially taking a sample average) and a constant value in the
	range 0 < step_size_parameter ≤ 1 (a higher step_size_parameter means a
	greater emphasis is placed on recent timesteps, up to 1 where no emphasis
	is placed on prior timesteps).

	"""
	def __init__(self, initial_Q=0, epsilon=0, step_size_parameter=None, *args, **kwargs):
		# Initialises the QPlayer based on the given parameters.
		self.initial_Q = initial_Q
		self.epsilon = epsilon

		# Ensures that the step_size_parameter input is valid.
		if step_size_parameter is not None:
			assert 0 < step_size_parameter <= 1, \
				"Step size parameter should be in the range 0 < step_size_parameter ≤ 1 or None"
		self.step_size_parameter = step_size_parameter

		# Calls the Player superclass __init__ method for the remaining
		# initialisation.
		super().__init__('q_estimate', *args, **kwargs)

	def reset(self):
		"""
		Resets the Player based on the Player superclass method, but also
		includes resetting the Q value estimate. Similar to the Player
		superclass method, typically used at the start of any run.

		"""
		# Performs Player superclass reset.
		super().reset()

		# Resets the Q value estimate to the default initial_Q for all actions.
		self.player_Q = np.ones_like(self.player_selected_actions, dtype=float) * self.initial_Q

	def obtain_action(self, timestep):
		"""
		Obtains a valid action from Player randomly with probability epsilon
		and deterministically by randomly selecting between actions with the
		maximum Q value estimate with probability 1-epsilon.

		Parameter(s):
		timestep: Index of timestep, dummy input for this Player subclass.

		Return(s):
		An action index corresponding to a valid action taken by the Player.

		"""
		# Generates a random number for deciding between performing a random or
		# deterministic action.
		random_num = random.random()

		if random_num < self.epsilon:
			# Random action taken.
			return random.randint(0, self.num_actions-1)

		else:
			# Deterministic action taken, where a random action index is
			# selected among the action indexes with the maximum Q value
			# estimate.
			return np.random.choice(np.argwhere(self.player_Q == self.player_Q.max()).flatten()).item()

	def update(self, timestep, action, reward, best_action):
		"""
		Updates the Player based on the Player superclass method, but also
		includes updating the Q value estimate. Similar to the Player superclass
		method, typically used at the end of every timestep.

		Uses the same parameters as the Player superclass.

		"""
		# Performs updates based on the Player superclass update method.
		super().update(timestep, action, reward, best_action)

		# Updates the Q value estimate based on the nature of the step size
		# parameter.
		if self.step_size_parameter is None:
			# Update Q value estimate using a sample average step size parameter
			self.player_Q[action] += 1./self.player_selected_actions[action] * (reward - self.player_Q[action])

		else:
			# Update Q value estimate using a constant step size parameter
			self.player_Q[action] += self.step_size_parameter * (reward - self.player_Q[action])

class UCBQPlayer(Player):
	"""
	Player that uses Q values and upper confidence bounds to decide on the best
	action to take.

	Parameter(s) (in addition to those required by the Player superclass):
	initial_Q (optional, default 0): Starting Q value estimate for all actions.
	The default of 0 means that the estimated reward for all actions is 0.

	confidence_level (optional, default 1): Confidence level of the upper
	confidence bound. The higher the value, the more impact the upper
	confidence bound has on deciding the best action to take.

	step_size_parameter (optional, default None): Step size parameter for
	weighing rewards from prior timesteps. Acceptable inputs include None (the
	default, essentially taking a sample average) and a constant value in the
	range 0 < step_size_parameter ≤ 1 (a higher step_size_parameter means a
	greater emphasis is placed on recent timesteps, up to 1 where no emphasis
	is placed on prior timesteps).

	"""
	def __init__(self, initial_Q=0, confidence_level=1, step_size_parameter=None, *args, **kwargs):
		# Initialises the UCBQPlayer based on the given parameters.
		self.initial_Q = initial_Q
		self.confidence_level = confidence_level

		# Ensures that the step_size_parameter input is valid.
		if step_size_parameter is not None:
			assert 0 < step_size_parameter <= 1, \
				"Step size parameter should be in the range 0 < step_size_parameter ≤ 1 or None"
		self.step_size_parameter = step_size_parameter

		# Calls the Player superclass __init__ method for the remaining
		# initialisation.
		super().__init__('ucb_q_estimate', *args, **kwargs)

	def reset(self):
		"""
		Resets the Player based on the Player superclass method, but also
		includes resetting the Q value estimate. Similar to the Player
		superclass method, typically used at the start of any run.

		"""
		# Performs Player superclass reset.
		super().reset()

		# Resets the Q value estimate to the default initial_Q for all actions.
		self.player_Q = np.ones_like(self.player_selected_actions, dtype=float) * self.initial_Q

	def obtain_action(self, timestep):
		"""
		Obtains a valid action from Player deterministically by randomly
		selecting between actions with the maximum Q value estimate plus upper
		confidence bound term. If certain actions have not been selected at all,
		those actions will be given priority for selection.

		Parameter(s):
		timestep: Index of timestep, dummy input for this Player subclass.

		Return(s):
		An action index corresponding to a valid action taken by the Player.

		"""
		# Finds all actions which have not been selected before.
		zero_action = np.argwhere(self.player_selected_actions == 0).flatten()

		# Checks if there are any actions which have not been selected before.
		if zero_action.size:
			# Returns a random action index that has not been selected before.
			return np.random.choice(zero_action).item()

		else:
			# Calculates the sum of the Q value estimate and the upper
			# confidence bound term
			value_list = self.player_Q + self.confidence_level * (np.log(timestep+1) / self.player_selected_actions) ** 0.5

			# Returns a random action index selected among the action indexes
			# with the maximum sum.
			return np.random.choice(np.argwhere(value_list == value_list.max()).flatten()).item()

	def update(self, timestep, action, reward, best_action):
		"""
		Updates the Player based on the Player superclass method, but also
		includes updating the Q value estimate. Similar to the Player superclass
		method, typically used at the end of every timestep.

		Uses the same parameters as the Player superclass.

		"""
		# Performs updates based on the Player superclass update method.
		super().update(timestep, action, reward, best_action)

		# Updates the Q value estimate based on the nature of the step size
		# parameter.
		if self.step_size_parameter is None:
			# Update Q value estimate using a sample average step size parameter
			self.player_Q[action] += 1./self.player_selected_actions[action] * (reward - self.player_Q[action])

		else:
			# Update Q value estimate using a constant step size parameter
			self.player_Q[action] += self.step_size_parameter * (reward - self.player_Q[action])

class GradientPlayer(Player):
	"""
	Player that uses action preferences and gradient ascent to decide on the
	best action to take.

	Parameter(s) (in addition to those required by the Player superclass):
	step_size_parameter (optional, default 0.1): Step size parameter for
	changing action preference values. Input should be positive, with a higher
	step_size_parameter corresponding to a higher extent of change of action
	preferences after each timestep for every action.

	use_baseline_reward (optional, default True): Confidence level of the upper
	confidence bound. The higher the value, the more impact the upper
	confidence bound has on deciding the best action to take.

	"""
	def __init__(self, step_size_parameter=0.1, use_baseline_reward=True, *args, **kwargs):
		# Initialises the GradientPlayer based on the given parameters.
		self.use_baseline_reward = use_baseline_reward

		# Ensures that the step_size_parameter input is positve.
		assert step_size_parameter > 0, "Step size parameter should be > 0"
		self.step_size_parameter = step_size_parameter

		# Calls the Player superclass __init__ method for the remaining
		# initialisation.
		super().__init__('gradient', *args, **kwargs)

	def reset(self):
		"""
		Resets the Player based on the Player superclass method, but also
		includes resetting the action preferences. Similar to the Player
		superclass method, typically used at the start of any run.

		"""
		# Performs Player superclass reset.
		super().reset()

		# Resets the action preferences to the default value of 0 for all
		# timesteps and actions.
		self.H_gradient = np.zeros((self.num_timesteps, self.num_actions), dtype=float)

	def obtain_action(self, timestep):
		"""
		Obtains a valid action from Player deterministically by using action
		preferences to weigh the probability of selecting each action.

		Parameter(s):
		timestep: Index of timestep

		Return(s):
		An action index corresponding to a valid action taken by the Player.

		"""
		# Obtain the action probabiltiy weights
		self.curr_prob_gradient = softmax(self.H_gradient[timestep])

		# Randomly selects an action, weighted by the action probability weights
		return np.random.choice(self.num_actions, p=self.curr_prob_gradient)

	def update(self, timestep, action, reward, best_action):
		"""
		Updates the Player based on the Player superclass method, but also
		includes updating the action prefereneces. Similar to the Player
		superclass method, typically used at the end of every timestep.

		Uses the same parameters as the Player superclass.

		"""
		# Performs updates based on the Player superclass update method.
		super().update(timestep, action, reward, best_action)

		# Perform updates only if the timestep is not the last timestep. An
		# increased action preference indicates that either an action is chosen
		# and it led to a good reward / better reward than baseline, or an
		# action was not chosen and it led to a bad reward / worse reward than
		# baseline.
		if timestep + 1 < self.num_timesteps:
			if self.use_baseline_reward:
				# Computes the baseline reward, which is the average of rewards
				# in all prior timesteps.
				baseline_reward = self.player_reward[:timestep+1].mean()

				# Updates the action preferences considering the baseline
				# reward.
				self.H_gradient[timestep+1] = self.H_gradient[timestep] - self.step_size_parameter * (reward - baseline_reward) * self.curr_prob_gradient
				self.H_gradient[timestep+1,action] = self.H_gradient[timestep,action] + self.step_size_parameter * (reward - baseline_reward) * (1.-self.curr_prob_gradient[action])

			else:
				# Updates the action preferences without the baseline reward.
				self.H_gradient[timestep+1] = self.H_gradient[timestep] - self.step_size_parameter * reward * self.curr_prob_gradient
				self.H_gradient[timestep+1,action] = self.H_gradient[timestep,action] + self.step_size_parameter * reward * (1.-self.curr_prob_gradient[action])
