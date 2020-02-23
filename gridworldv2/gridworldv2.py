"""
Copyright Declaration (C)
From: https://github.com/leeykang/
Use and modification of information, comment(s) or code provided in this document
is granted if and only if this copyright declaration, located between lines 1 to
9 of this document, is preserved at the top of any document where such
information, comment(s) or code is/are used.

"""
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.table import Table
import matplotlib.patches as mpatches

class GridworldV2:
	"""
	Provides the definition of the gridworld v2 problem.

	Parameter(s):
	name: Name of the GridworldV2 problem.

	size: The size of the 2D grid, represented as [number of rows, number of
	columns].

	actions_list: A list of 2D actions corresponding to possible actions in
	the grid. Actions should be provided in the form of [change in row, change
	in column].

	terminal_coords: A list of 2D coords corresponding to the terminal states in
	the grid. Coords should be in the form of [row, column].

	discount: The discount rate when considering the subsequent state.

	"""
	def __init__(self, name, size, actions_list, terminal_coords_list, discount):
		# Initialises the GridworldV2 problem based on the given parameters.
		self.name = name
		self.size = size
		self.actions_list = actions_list
		self.terminal_coords_list = terminal_coords_list
		self.discount = discount

		# Computes the number of possible actions.
		self.num_actions = len(self.actions_list)

		# Initialises the current available solving methods as a dictionary,
		# with key being the method name and value being the specific function
		# to call.
		self.implemented_solve_methods = {'policy_iteration': self.__policy_iteration,
										  'value_iteration': self.__value_iteration}

	def step(self, state, action):
		"""
		Computes the value estimate of taking a particular action given a
		particular state.

		Parameter(s):
		state: A valid state/coordinate of the GridworldV2 player.

		action: A valid action of the GridworldV2 player.

		Return(s):
		The value estimate of taking a particular action given a particular
		state.

		"""
		# Obtains the action index of the performed action.
		action_index = self.actions_list.index(action)

		# Determines the new state manually.
		y, x = state[0] + action[0], state[1] + action[1]

		# If the new state is outside of the grid, the agent is reverted to the
		# state prior to performing the specified action.
		if y < 0 or y >= self.size[0] or x < 0 or x >= self.size[1]:
			y, x = state

		# Because the state transition for any action is guaranteed, the value
		# estimate is just the sum of the reward, which is always set to -1, and
		# the discounted value estimate of the next state.
		return -1 + self.discount * self.v[y,x]

	def visualise(self):
		"""
		Visualises the result of analysing the GridworldV2.

		"""
		# Obtain the current working directory.
		curr_dir = os.path.dirname(os.path.abspath(__file__))

		# Creates the required diagram and assigns a title to it, which includes
		# the number of iterations performed as part of the specified method.
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
		graph_title = ' '.join([substr.title() for substr in self.solve_method.split('_')])
		fig.suptitle('GridworldV2 %s Results: %i Iterations' % (graph_title, self.current_iter), fontsize=30)

		# Disable the display of every axes of every subplot.
		ax1.set_axis_off()
		ax2.set_axis_off()

		# Creates a table, representing a grid, for every subplot.
		tb1 = Table(ax1, bbox=[0, 0, 1, 1])
		tb2 = Table(ax2, bbox=[0, 0, 1, 1])

		# Computes the height and width of each cell of the grid during
		# visualisation.
		height, width = 1.0 / self.size[0], 1.0 / self.size[1]

		# Loops across all possible states of the GridworldV2.
		for (j, i), val in np.ndenumerate(self.v):
			# Creates a cell in every table for each state, with the cell in the
			# leftmost table containing the approximate final value estimate and
			# the cells in the other tables being empty.
			if [j,i] in self.terminal_coords_list:
				facecolor = 'gray'
			else:
				facecolor = 'none'
			tb1.add_cell(j, i, width, height, text=val, loc='center', facecolor='none')
			tb2.add_cell(j, i, width, height, loc='center', facecolor=facecolor)

			# Loops across all possible actions of the current state.
			for (a,), pol in np.ndenumerate(self.pi[j,i]):
				# Adds the action to the rightmost table if the action is
				# part of the final policy. The action is shown in the form
				# of an arrow.
				if pol:
					# Finds the corresponding coordinate change of the current
					# action.
					current_action = self.actions_list[a]

					# Adds the arrow to the rightmost table.
					ax2.arrow(width*(i+0.5), 1-height*(j+0.5), 0.25*current_action[1]*width, -0.25*current_action[0]*height, head_width=0.1*width, head_length=0.1*height, color='black')

		# Adds the row indexes for each of the tables.
		for j in range(self.size[0]):
			tb1.add_cell(j, -1, width/2, height, text=j, loc='right', edgecolor='none', facecolor='none')
			tb2.add_cell(j, -1, width/2, height, text=j, loc='right', edgecolor='none', facecolor='none')

		# Adds the column indexes for each of the tables.
		for i in range(self.size[1]):
			tb1.add_cell(-1, i, width, height/4, text=i, loc='center', edgecolor='none', facecolor='none')
			tb2.add_cell(-1, i, width, height/4, text=i, loc='center', edgecolor='none', facecolor='none')

		# Adds the completed tables to their respective subplots.
		ax1.add_table(tb1)
		ax2.add_table(tb2)

		# Sets the titles for all three subplots.
		title_height = 1.05
		title_size = 15
		ax1.set_title('Optimum Value', y=title_height, size=title_size)
		ax2.set_title('Optimum Path', y=title_height, size=title_size)

		# Saves the plotted diagram with the name of the selected method.
		plt.savefig(os.path.join(curr_dir, 'gridworldv2_%s_%s_results.png' % (self.name, self.solve_method)))
		plt.close()

	def __policy_iteration(self, tolerance):
		"""
		Obtains the optimum value estimate of the GridworldV2 problem and its
		corresponding policy using policy iteration. The results are then saved
		into a diagram.

		Parameter(s):
		tolerance: Used to check if the value estimate has converged and to
		decide on policies that result in the highest value estimate (the second
		use case is to handle noise in the results caused by floating point
		truncation).

		"""
		# Assumes the policy to be unstable at the start.
		policy_stable = False

		# Infinite loop for performing policy iteration.
		while not policy_stable:
			# Assumes the policy to be unstable at the start.
			value_stable = False

			# Infinite loop for performing policy evaluation.
			while not value_stable:
				# Makes a copy of the value estimate initalised with 0.
				new_v = np.zeros_like(self.v)

				# Loops across all possible states of the GridworldV2.
				for j in range(self.size[0]):
					for i in range(self.size[1]):
						# Performs policy evaluation only if the current state
						# is not a terminal state.
						if [j,i] not in self.terminal_coords_list:
							# For the current state, finds the value estimate for
							# every action in the current policy and places the
							# value estimates in a list.
							values_list = [self.step((j, i), action) for action_idx, action in enumerate(self.actions_list) if self.pi[j,i,action_idx]]

							# Updates the value estimate copy of the current state
							# with the average value estimate.
							new_v[j, i] = sum(values_list) / len(values_list)

				# Checks if the value estimate has converged within the provided
				# tolerance.
				if np.abs(new_v - self.v).max() < tolerance:
					# Updates the current value estimate to the final value
					# estimate.
					self.v = new_v

					# Ends the infinite policy evaluation loop.
					value_stable = True

				else:
					# Updates the current value estimate to the new value
					# estimate.
					self.v = new_v

			# Assumes the current policy to be stable prior to policy
			# improvement.
			policy_stable = True

			# Makes a copy of the policy initalised with 0.
			new_pi = np.zeros_like(self.pi)

			# Loops across all possible states of the GridworldV2 for policy
			# improvement.
			for j in range(self.size[0]):
				for i in range(self.size[1]):
					# Performs policy improvement only if the current state
					# is not a terminal state.
					if [j,i] not in self.terminal_coords_list:
						# Obtain the prior policy.
						prior_policy = np.nonzero(self.pi[j,i])[0]

						# For the current state, finds the value estimate for every
						# possible action and places the value estimates in a list.
						values_list = [self.step((j, i), action) for action in self.actions_list]

						# Finds the maximum value estimate.
						max_value = max(values_list)

						# Finds all action indexes that correspond to the maximum
						# value estimate within a certain tolerance.
						new_policy = np.nonzero((max_value - values_list) < tolerance)

						# If the policy is still stable, checks if the policy for
						# the current state is stable.
						if policy_stable and not np.array_equal(prior_policy, new_policy[0]):
							# Set the policy_stable flag to False, indicating
							# the need to do another policy iteration loop.
							policy_stable = False

						# Updates the current policy to the new policy for the
						# current state.
						new_pi[(j,i,)+new_policy] = 1

			# Updates the current policy to the new policy.
			self.pi = new_pi

			# Increments the current iteration counter by 1.
			self.current_iter += 1

		# Rounds the final value estimate to 2 decimal places.
		self.v = np.round(self.v, decimals=2)

		# Visualises the result of the GridworldV2.
		self.visualise()

	def __value_iteration(self, tolerance):
		"""
		Obtains the optimum value estimate of the GridworldV2 problem and its
		corresponding policy using value iteration. The results are then saved
		into a diagram.

		Parameter(s):
		tolerance: Used to check if the value estimate has converged and to
		decide on policies that result in the highest value estimate (the second
		use case is to handle noise in the results caused by floating point
		truncation).

		"""
		# Assumes the policy to be unstable at the start.
		value_stable = False

		# Infinite loop for performing value iteration.
		while not value_stable:
			# Makes a copy of the value estimate and policy, both initalised
			# with 0.
			new_v = np.zeros_like(self.v)
			new_pi = np.zeros_like(self.pi)

			# Loops across all possible states of the GridworldV2.
			for j in range(self.size[0]):
				for i in range(self.size[1]):
					# Performs value and policy updates only if the current
					# state is not a terminal state.
					if [j,i] not in self.terminal_coords_list:
						# For the current state, finds the value estimate for every
						# possible action and places the value estimates in a list.
						values_list = [self.step((j, i), action) for action in self.actions_list]

						# Finds the maximum value estimate.
						max_value = max(values_list)

						# Finds all action indexes that correspond to the maximum
						# value estimate within a certain tolerance and uses the
						# action indexes to update the policy copy of the current
						# state to the new policy for that state.
						new_pi[(j,i,) + np.nonzero((max_value - values_list) < tolerance)] = 1

						# Using the maximum value estimate, updates the value
						# estimate copy of the current state with the new value
						# estimate for that state.
						new_v[j, i] = max_value

			# Checks if the value estimate has converged within the provided
			# tolerance.
			if np.abs(new_v - self.v).max() < tolerance:
				# Updates the current value estimate and policy to the final
				# value estimate and policy respectively, and rounds the final
				# value estimate to 2 decimal places.
				self.v = np.round(new_v, decimals=2)
				self.pi = new_pi

				# Visualises the result of the GridworldV2.
				self.visualise()

				# Ends the infinite value iteration loop.
				value_stable = True

			else:
				# Updates the current value estimate and policy to the new value
				# estimate and policy respectively.
				self.v = new_v
				self.pi = new_pi

			# Increments the current iteration counter by 1.
			self.current_iter += 1

	def obtain_optimum(self, solve_method='value_iteration', tolerance=1e-16):
		"""
		Obtains the optimum value estimate of the GridworldV2 problem and its
		corresponding policy by invoking a selected solve method. The results
		are then saved into a diagram.

		Parameter(s):
		solve_method (optional, default value_iteration): Method for solving
		the GridworldV2 problem. Currently implemented methods include
		policy_iteration and value_iteration.

		tolerance (optional, default 1e-16): Used to check if the value estimate
		has converged and to decide on policies that result in the highest value
		estimate (the second use case is to handle noise in the results
		caused by floating point truncation).

		"""
		# Ensures that the selected solve method has been implemented.
		assert solve_method in self.implemented_solve_methods, "%s SOLVE METHOD NOT IMPLEMENTED" % solve_method

		# Sets the current solve method to the selected solve method.
		self.solve_method = solve_method

		# Initialises the value estimate to 0 and policy to 1.
		self.v = np.zeros(self.size)
		self.pi = np.ones(self.size + (self.num_actions,), int)

		# Initialises the current iteration counter to 0.
		self.current_iter = 0

		# Obtains the selected solve function and performs that function to
		# solve the GridworldV2 problem.
		solve_func = self.implemented_solve_methods[solve_method]
		solve_func(tolerance)
