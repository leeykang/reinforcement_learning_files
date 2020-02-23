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

class Gambler:
	"""
	Provides the definition of the gambler problem and solves the problem
	through value iteration.

	Parameter(s):
	name: Name of the Gambler problem.

	goal: The objective amount that the gambler hopes to achieve.

	win_prob: The probability of winning each time the gambler stakes.

	discount: The discount rate when considering the subsequent state.

	"""
	def __init__(self, name, goal, win_prob, discount):
		# Initialises the gambler problem based on the given parameters.
		self.name = name
		self.goal = goal
		self.win_prob = win_prob
		self.discount = discount

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
		state: A valid state/capital of the Gambler.

		action: A valid action/stake of the Gambler.

		Return(s):
		The value estimate of taking a particular action given a particular
		state.

		"""
		# Computes the value estimate for winning the stake. This involves
		# giving a reward of 1 if state + action is equals to the Gambler's goal
		# and 0 otherwise, discounting the subsequent state, and multiplying
		# the raw value estimate by the win probability.
		win_val = self.win_prob * (((state+action) == self.goal) * 1. + self.discount * self.v[state + action])

		# Computes the value estimate for losing the stake. This involves
		# giving a reward of 0 since this result will never reach the Gambler's
		# goal, discounting the subsequent state, and multiplying
		# the raw value estimate by the loss probability (1 - win probability).
		loss_val = (1. - self.win_prob) * self.discount * self.v[state - action]

		# Returns the sum of value estimates for winning or losing the stake.
		return win_val + loss_val

	def visualise(self):
		"""
		Visualises the result of analysing the Gambler.

		"""
		# Obtain the current working directory.
		curr_dir = os.path.dirname(os.path.abspath(__file__))

		# Creates the required diagram and assigns a title to it, which includes
		# the number of iterations performed as part of the specified method.
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
		graph_title = ' '.join([substr.title() for substr in self.solve_method.split('_')])
		fig.suptitle('Gambler %s Results: %i Iterations' % (graph_title, self.current_iter), fontsize=30)

		# For every iteration of the value iteration, plot the value estimate
		# for each possible capital of the Gambler (other than 0 and the goal,
		# which has no possible actions to compute a value estimate).
		# This is placed in the left diagram.
		for v in self.value_visualise_list:
			ax1.plot(range(1,self.goal), v[1:-1])

		# Provides the title and labels of the left diagram.
		ax1.set_title('Value Estimate vs Capital ($)')
		ax1.set_ylabel('Value Estimate')
		ax1.set_xlabel('Capital ($)')

		# Plots the best policy/stake to make for each possible capital of
		# the Gambler. This is placed in the right diagram.
		ax2.plot(range(self.goal+1), np.argmax(self.pi, axis=1))

		# Provides the title and labels of the right diagram.
		ax2.set_title('Final Policy / Stake (\$) vs Capital (\$)')
		ax2.set_ylabel('Final Policy / Stake ($)')
		ax2.set_xlabel('Capital ($)')

		# Saves the plotted diagram with the name of the selected method.
		plt.savefig(os.path.join(curr_dir, 'gambler_%s_%s_results.png' % (self.name, self.solve_method)))
		plt.close()

	def __policy_iteration(self, tolerance):
		"""
		Obtains the optimum value estimate of the Gambler and its corresponding
		policy using policy iteration. The results are then saved into a
		diagram.

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

				# Adds the current value estimate to the visualisation list.
				self.value_visualise_list.append(self.v)

				# Loops across all possible states of the Gambler.
				for state in range(1, self.goal):
					# For the current state, finds the value estimate for every
					# action in the current policy and places the value
					# estimates in a list.
					values_list = [self.step(state, action) for action in range(min(state, self.goal - state) + 1) if self.pi[state,action]]

					# Updates the value estimate copy of the current state with
					# the average value estimate.
					new_v[state] = sum(values_list) / len(values_list)

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

			# Loops across all possible states of the Gambler for policy
			# improvement.
			for state in range(1, self.goal):
				# Obtain the prior policy.
				prior_policy = np.nonzero(self.pi[state])[0]

				# For the current state, finds the value estimate for every
				# possible action and places the value estimates in a list.
				values_list = [self.step(state, action) for action in range(min(state, self.goal - state) + 1)]

				# Finds the maximum value estimate.
				max_value = max(values_list)

				# Finds all action indexes that correspond to the maximum value
				# estimate within a certain tolerance.
				new_policy = np.nonzero((max_value - values_list) < tolerance)

				# If the policy is still stable, checks if the policy for
				# the current state is stable.
				if policy_stable and not np.array_equal(prior_policy, new_policy[0]):
					# Set the policy_stable flag to False, indicating
					# the need to do another policy iteration loop.
					policy_stable = False

				# Updates the current policy to the new policy for the
				# current state.
				new_pi[(state,)+new_policy] = 1

			# Updates the current policy to the new policy.
			self.pi = new_pi

			# Increments the current iteration counter by 1.
			self.current_iter += 1

		# Adds the final value estimate to the visualisation list.
		self.value_visualise_list.append(self.v)

		# Visualises the result of the Gambler.
		self.visualise()

	def __value_iteration(self, tolerance):
		"""
		Obtains the optimum value estimate of the Gambler and its corresponding
		policy using value iteration. The results are then saved into a diagram.

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

			# Adds the current value estimate to the visualisation list.
			self.value_visualise_list.append(self.v)

			# Loops across all possible states of the Gambler.
			for state in range(1, self.goal):
				# For the current state, finds the value estimate for every
				# possible action and places the value estimates in a list.
				values_list = [self.step(state, action) for action in range(min(state, self.goal - state) + 1)]

				# Finds the maximum value estimate.
				max_value = max(values_list)

				# Finds all action indexes that correspond to the maximum value
				# estimate within a certain tolerance and uses the action
				# indexes to update the policy copy of the current state to the
				# new policy for that state.
				new_pi[(state,) + np.nonzero((max_value - values_list) < tolerance)] = 1

				# Using the maximum value estimate, updates the value estimate
				# copy of the current state with the new value estimate for that
				# state.
				new_v[state] = max_value

			# Checks if the value estimate has converged within the provided
			# tolerance.
			if np.abs(new_v - self.v).max() < tolerance:
				# Updates the current value estimate and policy to the final
				# value estimate and policy respectively.
				self.v = new_v
				self.pi = new_pi

				# Adds the final value estimate to the visualisation list.
				self.value_visualise_list.append(self.v)

				# Visualises the result of the Gambler.
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
		Obtains the optimum value estimate of the Gambler problem and its
		corresponding policy by invoking a selected solve method. The results
		are then saved into a diagram.

		Parameter(s):
		solve_method (optional, default value_iteration): Method for solving
		the Gambler problem. Currently implemented methods include
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

		# Initialises an empty list to store the value estimate during each
		# round of the solving method.
		self.value_visualise_list = []

		# Initialises the value estimate to 0 and policy to 1.
		self.v = np.zeros(self.goal + 1)
		self.pi = np.ones((self.goal + 1, min(self.goal // 2, self.goal - self.goal // 2) + 1), int)

		# Initialises the current iteration counter to 0.
		self.current_iter = 0

		# Obtains the selected solve function and performs that function to
		# solve the Gambler.
		solve_func = self.implemented_solve_methods[solve_method]
		solve_func(tolerance)
