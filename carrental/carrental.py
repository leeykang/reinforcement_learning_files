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
from scipy.stats import poisson
from functools import reduce
from operator import mul
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from seaborn import heatmap
from matplotlib import cm
from itertools import product
from multiprocessing import Pool
from copy import deepcopy
from time import time

class CarRental:
	"""
	Provides the definition of the car rental problem.

	Parameter(s):
	name: Name of the CarRental problem.

	max_cars_list: A list containing the maximum number of cars that can be in
	each location at any point in time.

	transfer_dict: A dictionary containing information about transferring cars
	from one location to another. Should be stored in the form of key:
	(source location index, destination location index) and value: (maximum
	number of cars that can be transfered from the source location to the
	destination location, maximum number of cars that can be transfered from the
	source location to the destination location for free, cost of transferring a
	car from the source location to the destination location.

	rental_fee: The cost of renting a vehicle at any location, used as a revenue
	for the car rental problem.

	add_storage_threshold_list: A list containing the maximum number of cars
	that can be stored at each location before additional storage costs are
	incurred.

	add_storage_fee: The cost of additional storage at any location.

	rental_lambda_list: A list containing the expected number of rentals at
	each location, based on a Poisson distribution.

	return_lambda_list: A list containing the expected number of returns at
	each location, based on a Poisson distribution.

	discount: The discount rate when considering the subsequent state.

	use_multiprocessing: Boolean variable for deciding whether to use
	multiprocessing to solve the car rental problem.

	num_max_processes (optional, default 8): The maximum number of processes to
	use for multiprocessing.

	"""
	def __init__(self,
				 name,
				 max_cars_list,
				 transfer_dict,
				 rental_fee,
				 add_storage_threshold_list,
				 add_storage_fee,
				 rental_lambda_list,
				 return_lambda_list,
				 discount,
				 use_multiprocessing,
				 num_max_processes=8):

		# Initialises the car rental problem based on the given parameters.
		self.name = name
		self.max_cars_list = max_cars_list
		self.transfer_dict = transfer_dict
		self.rental_fee = rental_fee
		self.add_storage_threshold_list = add_storage_threshold_list
		self.add_storage_fee = add_storage_fee
		self.rental_lambda_list = rental_lambda_list
		self.return_lambda_list = return_lambda_list
		self.discount = discount
		self.use_multiprocessing = use_multiprocessing
		self.num_max_processes = num_max_processes

		# Computes the number of car rental locations.
		self.num_locations = len(max_cars_list)

		# Initialises the current available solving methods as a dictionary,
		# with key being the method name and value being the specific function
		# to call.
		self.implemented_solve_methods = {'policy_iteration': self.policy_iteration,
										  'value_iteration': self.value_iteration}

		# Computes values required for solving the car rental problem.
		self.__compute_values()

	def __compute_values(self):
		"""
		Computes values required for solving the CarRental problem.

		"""
		# Initialises the maximum transfer array (maximum number of car
		# transfers between two locations), free transfer array (maximum number
		# of free car transfers between two locations) and transfer cost array
		# (cost of transferring a car from one location to another) with 0.
		self.max_transfers_arr = np.zeros((self.num_locations, self.num_locations), int)
		self.free_transfers_num_arr = np.zeros((self.num_locations, self.num_locations), int)
		self.transfer_cost_arr = np.zeros((self.num_locations, self.num_locations), int)

		# Fills the maximum transfer array, free transfer number array and
		# transfer cost array using the transfer dictionary input parameter.
		for key, value in self.transfer_dict.items():
			self.max_transfers_arr[key] = value[0]
			self.free_transfers_num_arr[key] = value[1]
			self.transfer_cost_arr[key] = value[2]

		# Initialises a dictionary that stores information regarding the
		# possible number of transfers that can take place between two
		# locations. This is the form of key: (location_i, location_j), where
		# i ≠ j, and value: an array containing all values from
		# -(maximum transfers from location_j to location_i) to
		# (maximum transfers from location_i to location_j) inclusive.
		transfer_range_dict = {}

		# Initialises a list that stores information about the number of
		# possible ways in which location_i and location_j can interact, where
		# i and j are location indexes and i ≠ j.
		transfer_num_list = [0] * (self.num_locations * (self.num_locations-1) // 2)

		# Initialises a list that stores information in the form of (i,j), where
		# i and j are location indexes, i ≠ j and the kth tuple in
		# transfer_index_list corresponds to the kth value in transfer_num_list,
		# where 0 <= k < (number of possible location interactions). The length
		# of transfer_num_list and transfer_index_list should therefore be the
		# same.
		self.transfer_index_list = [0] * len(transfer_num_list)

		# Intialises a pointer that deduces the index of the next element to
		# fill.
		transfer_index = len(transfer_num_list) - 1

		# Loops across all possible location interactions.
		for i in range(self.num_locations-1):
			for j in range(i+1, self.num_locations):
				# Places the desired range array in transfer_range_dict for the
				# current location pair.
				transfer_range_dict[(i,j)] = np.arange(-self.max_transfers_arr[j,i], self.max_transfers_arr[i,j]+1)

				# Places the number of possible interactions between the current
				# location pair into transfer_num_list.
				transfer_num_list[transfer_index] = transfer_range_dict[(i,j)].size - 1

				# Places the current location pair indexes into
				# transfer_index_list.
				self.transfer_index_list[transfer_index] = (i,j)

				# Moves the pointer down by 1.
				transfer_index -= 1

		# Initialises a list containing all possible movement matrices. A
		# movement matrix is a num_locations x num_locations matrix that
		# contains values for row index i and column index j where i < j
		# and 0 otherwise. For each non-zero value, if the value is bigger than
		# 0, cars equal to the value are moved from location_i to location_j.
		# If the value is smaller than 0, cars equal to the absolute of the
		# value are moved from location_j to location_i.
		self.full_movement_matrix_list = []

		# Iterates through all possible transfer states using a generator.
		for current_transfer_num in product(*[range(x+1) for x in transfer_num_list]):
			# Initialises the current movement matrix.
			current_movement_matrix = np.zeros((self.num_locations, self.num_locations),int)

			# Iterates through all possible location interactions and fills the
			# current movement matrix with the relevant value associated with
			# location_i and location_j.
			for idx, val in enumerate(self.transfer_index_list):
				current_movement_matrix[val] = transfer_range_dict[val][current_transfer_num[idx]]

			# Adds the filled current movement matrix to the list of all
			# possible movement matrices.
			self.full_movement_matrix_list.append(current_movement_matrix)

		# Initialises the rental dictionary. The rental dictionary contains
		# information about the probabilties and fees associated with rentals
		# at different locations and different number of cars per location.
		# It is stored in the form of key: location_index, value: rental
		# dictionary for the current location.
		self.rental_dict = {}

		# Initialises the return dictionary. The return dictionary contains
		# information about the probabilties associated with returns at
		# different locations and different number of cars per location. It is
		# stored in the form of key: location_index, value: return dictionary
		# for the current location.
		self.return_dict = {}

		# Iterates through all possible locations.
		for idx in range(self.num_locations):
			# Initialises the rental dictionary for the current location.
			self.rental_dict[idx] = {}

			# Initialises the return dictionary for the current location.
			self.return_dict[idx] = {}

			# Obtains the maximum possible number of cars, expected rental
			# number and expected return number of the current location.
			num_max_cars = self.max_cars_list[idx]
			rental_lambda = self.rental_lambda_list[idx]
			return_lambda = self.return_lambda_list[idx]

			# Computes all possible probabilties for rentals for the current
			# location. This will be a list of probabilties obtained from the
			# Poisson distribution from 0 to num_max_cars - 1 with both
			# endpoints included, and 1 - (the cumulative Poisson distribution
			# function from 0 to num_max_cars - 1 with both endpoints included)
			# to represent all other possibilties greater than or equal to
			# num_max_cars.
			full_rental_prob_arr = np.array([poisson.pmf(rent_num, rental_lambda) for rent_num in range(num_max_cars)] + \
											[1-poisson.cdf(num_max_cars - 1, rental_lambda)])

			# Computes all possible rental fees for the current location.
			full_rental_fee_arr = self.rental_fee * np.arange(num_max_cars+1)

			# Computes all possible probabilties for returns for the current
			# location. This will be a list of probabilties obtained from the
			# Poisson distribution from 0 to num_max_cars - 1 with both
			# endpoints included, and 1 - (the cumulative Poisson distribution
			# function from 0 to num_max_cars - 1 with both endpoints included)
			# to represent all other possibilties greater than or equal to
			# num_max_cars.
			full_return_prob_arr = np.array([poisson.pmf(return_num, return_lambda) for return_num in range(num_max_cars)] + \
											[1-poisson.cdf(num_max_cars - 1, return_lambda)])

			# Adds the 0 rental scenario into the rental dictionary for the
			# current location. A rental scenario is a dictionary because it
			# involves both the rental fee (with key fee) and the probability of
			# the various possible rental numbers (with key prob).
			self.rental_dict[idx][0] = {'fee': [0.],
										'prob': [1.]}

			# Adds the 0 return scenario into the return dictionary for the
			# current location. A return scenario only contains the probability
			# of the various possible return numbers.
			self.return_dict[idx][0] = full_return_prob_arr

			# Loops through all possibilties between 0 and the maximum number
			# of cars in the current location, exclusive of both endpoints.
			for possibility_idx in range(1,num_max_cars):
				# Adds the current rental scenario into the rental dictionary
				# for the current location.
				self.rental_dict[idx][possibility_idx] = {'fee': full_rental_fee_arr[:possibility_idx+1],
														  'prob': np.concatenate([full_rental_prob_arr[:possibility_idx], [full_rental_prob_arr[possibility_idx:].sum()]], axis=0)}

				# Adds the current return scenario into the return dictionary
				# for the current location.
				self.return_dict[idx][possibility_idx] = np.concatenate([full_return_prob_arr[:num_max_cars-possibility_idx], [full_return_prob_arr[num_max_cars-possibility_idx:].sum()]], axis=0)

			# Adds the max cars rental scenario into the rental dictionary for
			# the current location.
			self.rental_dict[idx][num_max_cars] = {'fee': full_rental_fee_arr,
												   'prob': full_rental_prob_arr}

			# Adds the max cars return scenario into the return dictionary for
			# the current location.
			self.return_dict[idx][num_max_cars] = [1.]

	def step(self, state, movement_matrix):
		"""
		Computes the value estimate of performing a particular movement matrix
		given a particular state.

		The steps of the state includes car movement, car rental and car return.
		At the car return stage, if any location exceeds the maximum number of
		cars possible within that location, excess cars are removed from the
		analysis.

		Parameter(s):
		state: A valid state of the CarRental problem. The state in this case
		is the number of cars in each location.

		movement_matrix: A valid movement matrix by the CarRental agent. A
		movement matrix is a num_locations x num_locations matrix that contains
		values for row index i and column index j where i < j and 0 otherwise.
		For each non-zero value, if the value is bigger than 0, cars equal to
		the value are moved from location_i to location_j. If the value is
		smaller than 0, cars equal to the absolute of the value are moved from
		location_j to location_i. A valid movement matrix ensures that the
		number of cars after all possible movement does not exceed the maximum
		possible number of cars at any location.

		Return(s):
		The value estimate of performing a particular movement_matrix given a
		particular state.

		"""
		# Initialises the final value estimate to 0.
		final_val = 0.

		# Converts the movement matrix to an action. The action contains only
		# positive values. To obtain the action, two matrices are summed. The
		# first matrix contains all nonnegative values of the movement matrix
		# and 0 otherwise, where all such values are kept in their original
		# locations. The second matrix is based on all negative values of the
		# movement matrix and 0 otherwise. This matrix is transposed and
		# converted to nonnegative values by finding the absolute value
		# of the tranposed matrix, forming the second matrix.
		action = movement_matrix * (movement_matrix > 0) - (movement_matrix * (movement_matrix < 0)).transpose()

		# Performs the car movement step. It is assumed that the policy only
		# provides valid movement matrices. By the convention of this code,
		# losses to each location can be summed in a row and gains to each
		# location can be summed in a column.
		post_movement = state - movement_matrix.sum(axis=1) + movement_matrix.sum(axis=0)

		# Computes the total cost of movement, excluding any free transfers,
		# and subtracts it from the final value estimate.
		movement_cost = (np.maximum(0., action - self.free_transfers_num_arr) * self.transfer_cost_arr).sum()
		final_val -= movement_cost

		# Computes the total cost of storage, specifically when it exceeds the
		# threshold for additional storage, and subtracts it from the final
		# value estimate.
		storage_cost = ((post_movement > self.add_storage_threshold_list) * self.add_storage_fee).sum()
		final_val -= storage_cost

		# Iterates through all rental possibiltiies using a generator.
		for current_rent_num in product(*[range(x+1) for x in post_movement]):
			# Computes the profit and probability of performing the current
			# rental possibility.
			rental_profit = sum(self.rental_dict[location_idx][current_post_movement]['fee'][current_rent_num[location_idx]] for location_idx, current_post_movement in enumerate(post_movement))
			rental_prob = reduce(mul, (self.rental_dict[location_idx][current_post_movement]['prob'][current_rent_num[location_idx]] for location_idx, current_post_movement in enumerate(post_movement)))

			# Performs the rental step for the current rental possibility.
			post_rental = post_movement - current_rent_num

			# Iterates through all return possibiltiies given the current
			# rental possibility using a generator.
			for current_return_num in product(*[range(self.max_cars_list[idx] - post_rental[idx]+1) for idx in range(self.num_locations)]):
				# Computes the probability of performing current return
				# possibility.
				return_prob = reduce(mul, (self.return_dict[location_idx][current_post_rental][current_return_num[location_idx]] for location_idx, current_post_rental in enumerate(post_rental)))

				# Performs the return step for the current return possibility
				# to obtain the final state.
				final_state = post_rental + current_return_num

				# Computes the final probability by multiplying the rental
				# and return probabilties.
				final_prob = rental_prob * return_prob

				# Adds the value estimate for the current state, action and
				# new state to the final value estimate.
				final_val += final_prob * (rental_profit + self.discount * self.v[tuple(final_state.tolist())])

		return final_val

	def find_valid_moves(self, state):
		"""
		Finds all valid moves in the current state.

		Parameter(s):
		state: A valid state of the CarRental problem. The state in this case
		is the number of cars in each location.

		Return(s):
		A list containing all valid movement matrices in the current state.

		"""
		# Initialises a list to contain every possible action for the
		# current state.
		filtered_movement_matrix_list = []

		# For the current state, find every possible action and places
		# it in the filtered movement matrix list. This is done by
		# looping across each transfer pair in each movement matrix of
		# the full movement matrix list and checking if all transfer
		# pairs of the current movement matrix leads to a valid result.
		for movement_matrix in self.full_movement_matrix_list:
			# Assume that the current movement matrix is valid.
			valid_movement_matrix = True

			# For any transfer pair, if there are insufficient cars to
			# perform the transfer or if the number of received cars
			# causes a location to exceed the maximum number of cars
			# possible in a location, the movement matrix is marked as
			# invalid.
			for transfer_pair in self.transfer_index_list:
				num_movement = movement_matrix[transfer_pair]
				if num_movement > 0 and (state[transfer_pair[0]] < num_movement or state[transfer_pair[1]] + num_movement > self.max_cars_list[transfer_pair[1]]):
					valid_movement_matrix = False
					break
				elif num_movement < 0 and (state[transfer_pair[1]] < abs(num_movement) or state[transfer_pair[0]] + abs(num_movement) > self.max_cars_list[transfer_pair[0]]):
					valid_movement_matrix = False
					break

			# Adds the current movement matrix to the filtered movement
			# matrix list if the current movement matrix is valid.
			if valid_movement_matrix:
				filtered_movement_matrix_list.append(movement_matrix)

		return filtered_movement_matrix_list

	def visualise(self):
		"""
		Visualises the result of analysing the CarRental problem.

		"""
		# Obtain the current working directory.
		curr_dir = os.path.dirname(os.path.abspath(__file__))

		# Creates the required diagram and assigns a title to it, which includes
		# the number of iterations performed as part of the specified method.
		fig = plt.figure(figsize=(20, 10))
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122, projection='3d')
		graph_title = ' '.join([substr.title() for substr in self.solve_method.split('_')])
		fig.suptitle('Car Rental %s Results: %i Iterations' % (graph_title, self.current_iter), fontsize=30)

		# Converts the final policy from a dictionary to an array for
		# visualisation.
		final_policy = np.zeros_like(self.v, dtype=int)
		for key, val in self.pi.items():
			# For two locations, the only policy involved is between the
			# 0th and 1st location.
			final_policy[key] = val[0][0,1]

		# Draws the policy in the form of a contour plot in the left subplot.
		# Includes the label and tickmarks of the colorbar in the process.
		fig = heatmap(final_policy,
					  cmap=cm.coolwarm,
					  ax=ax1,
					  cbar_kws={'label': 'Cars to Transfer',
					  			'ticks': list(range(final_policy.min(), final_policy.max()+1)),
								'orientation': 'horizontal'})

		# Sets the axes labels, limits and tick marks for the left subplot.
		ax1.set_xlabel('Cars At Second Location')
		ax1.set_ylabel('Cars At First Location')
		ax1.set_ylim(0, self.max_cars_list[0] + 1)
		ax1.set_xlim(0, self.max_cars_list[1] + 1)
		ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

		# Prepares the x and y values of the right subplot by obtaining the
		# indexes of each value in the final value estimate.
		first_arr, second_arr = np.meshgrid(range(self.max_cars_list[0]+1), range(self.max_cars_list[1]+1), indexing='ij')

		# Uses the indexes and final value estimate to draw a surface plot
		# in the right subplot.
		ax2.plot_surface(first_arr, second_arr, self.v)
		ax2.text(-0.25, -0.25, self.v[0,0]+10, self.v[0,0].round().astype(int))
		ax2.text(self.max_cars_list[0]-0.25, -0.25, self.v[self.max_cars_list[0], 0]-5, self.v[self.max_cars_list[0], 0].round().astype(int))
		ax2.text(-0.25, self.max_cars_list[1], self.v[0, self.max_cars_list[1]]+5, self.v[0, self.max_cars_list[1]].round().astype(int))
		ax2.text(self.max_cars_list[0]-0.25, self.max_cars_list[1], self.v[self.max_cars_list[0], self.max_cars_list[1]]+5, self.v[self.max_cars_list[0], self.max_cars_list[1]].round().astype(int))

		# Sets the axes labels and tick marks for the right subplot.
		ax2.set_xlabel('Cars At First Location')
		ax2.set_xticks(list(range(self.max_cars_list[0] + 1)))
		ax2.set_ylabel('Cars At Second Location')
		ax2.set_yticks(list(reversed(range(self.max_cars_list[1] + 1))))
		ax2.set_zlabel('Value Estimate ($)')

		# Sets the title for both subplots.
		title_size = 15
		ax1.set_title('Optimal Policy', size=title_size)
		ax2.set_title('Optimal Value', size=title_size)

		# Saves the plotted diagram with the name of the selected method.
		if self.name:
			plt.savefig(os.path.join(curr_dir, 'carrental_%s_%s_results.png' % (self.name, self.solve_method)))
		else:
			plt.savefig(os.path.join(curr_dir, 'carrental_%s_results.png' % self.solve_method))
		plt.close()

	def policy_evaluation_state_func(self, *args):
		"""
		Performs the policy evaluation step for the given state. Only used when
		multiprocessing is enabled.

		Parameter(s):
		args: A valid state of the CarRental problem. The state in this case
		is the number of cars in each location.

		"""
		# For the current state, finds the value estimate for every
		# movement matrix in the current policy and places the value
		# estimates in a list.
		values_list = [self.step(args, movement_matrix) for movement_matrix in self.pi[args]]

		return args, sum(values_list) / len(values_list)

	def policy_evaluation_log_result(self, results):
		"""
		Stores the results of policy evaluation for every state. Only used when
		multiprocessing is enabled.

		Parameter(s):
		results: List of (state, value) tuples, the result of running policy
		evaluation.

		"""
		# Unpacks the result tuple into state and value.
		for s, val in results:
			# Updates the value estimate copy of the current state with
			# the average value estimate.
			self.new_v[s] = val

	def policy_improvement_state_func(self, state, tolerance):
		"""
		Performs the policy improvement step for the given state. Only used when
		multiprocessing is enabled.

		Parameter(s):
		args: A valid state of the CarRental problem. The state in this case
		is the number of cars in each location.

		tolerance: Used to check if the value estimate has converged and to
		decide on policies that result in the highest value estimate (the second
		use case is to handle noise in the results caused by floating point
		truncation).

		"""
		# Finds all valid movement matrices for the current state.
		filtered_movement_matrix_list = self.find_valid_moves(state)

		# For the current state, finds the value estimate for every
		# possible moveement matrix and places the value estimates in a
		# list.
		values_list = [self.step(state, movement_matrix) for movement_matrix in filtered_movement_matrix_list]

		# Finds the maximum value estimate.
		max_value = max(values_list)

		# Finds all movement matrices that correspond to the maximum
		# value estimate within a certain tolerance.
		new_movement_matrix_list = [movement_matrix for movement_matrix_idx, movement_matrix in enumerate(filtered_movement_matrix_list) if max_value - values_list[movement_matrix_idx] < tolerance]

		return state, new_movement_matrix_list

	def policy_improvement_log_result(self, results):
		"""
		Stores the results of policy improvement for every state. Only used when
		multiprocessing is enabled.

		Parameter(s):
		results: List of (state, policy) tuples, the result of running policy
		improvement.

		"""
		# Unpacks the result tuple into state and value.
		for s, pol in results:
			# Updates the current policy to the new policy for the
			# current state.
			self.new_pi[s] = pol

	def policy_iteration(self, tolerance):
		"""
		Obtains the optimum value estimate of the CarRental problem and its
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
				self.new_v = np.zeros_like(self.v)

				if self.use_multiprocessing:
					# Loops through all possible states of the CarRental problem
					# using a generator and performs policy evaluation with
					# multiprocessing.
					with Pool(processes=self.num_max_processes) as p:
						res = p.starmap_async(self.policy_evaluation_state_func,
											  list(product(*[range(x+1) for x in self.max_cars_list])),
											  callback=self.policy_evaluation_log_result)
						res.get()
				else:
					# Loops through all possible states of the CarRental problem
					# using a generator for policy evaluation.
					for state in product(*[range(x+1) for x in self.max_cars_list]):
						# For the current state, finds the value estimate for every
						# movement matrix in the current policy and places the value
						# estimates in a list.
						values_list = [self.step(state, movement_matrix) for movement_matrix in self.pi[state]]

						# Updates the value estimate copy of the current state with
						# the average value estimate.
						self.new_v[state] = sum(values_list) / len(values_list)

				# Checks if the value estimate has converged within the provided
				# tolerance.
				if np.abs(self.new_v - self.v).max() < tolerance:
					# Updates the current value estimate to the final value
					# estimate.
					self.v = self.new_v

					# Ends the infinite policy evaluation loop.
					value_stable = True

				else:
					# Updates the current value estimate to the new value
					# estimate.
					self.v = self.new_v

			# Assumes the current policy to be stable prior to policy
			# improvement.
			policy_stable = True

			if self.use_multiprocessing:
				# Makes a copy of the current policy to check for changes to
				# the policy.
				self.new_pi = deepcopy(self.pi)

				# Loops through all possible states of the CarRental problem
				# using a generator and performs policy improvement with
				# multiprocessing.
				with Pool(processes=self.num_max_processes) as p:
					res = p.starmap_async(self.policy_improvement_state_func,
										  product(product(*[range(x+1) for x in self.max_cars_list]), [tolerance]),
										  callback=self.policy_improvement_log_result)
					res.get()

				# Checks if the policy has stabilised.
				for state in self.pi:
					if not np.array_equal(self.pi[state], self.new_pi[state]):
						policy_stable = False
						break

				# Updates the current policy to the new policy for all states.
				self.pi = deepcopy(self.new_pi)

			else:
				# Loops through all possible states of the CarRental problem
				# using a generator for policy improvement.
				for state in product(*[range(x+1) for x in self.max_cars_list]):
					# Obtain the prior policy.
					old_movement_matrix_list = self.pi[state]

					# Finds all valid movement matrices for the current state.
					filtered_movement_matrix_list = self.find_valid_moves(state)

					# For the current state, finds the value estimate for every
					# possible moveement matrix and places the value estimates in a
					# list.
					values_list = [self.step(state, movement_matrix) for movement_matrix in filtered_movement_matrix_list]

					# Finds the maximum value estimate.
					max_value = max(values_list)

					# Finds all movement matrices that correspond to the maximum
					# value estimate within a certain tolerance.
					new_movement_matrix_list = [movement_matrix for movement_matrix_idx, movement_matrix in enumerate(filtered_movement_matrix_list) if max_value - values_list[movement_matrix_idx] < tolerance]

					# If the policy is still stable, checks if the policy for
					# the current state is stable.
					if policy_stable and not np.array_equal(old_movement_matrix_list, new_movement_matrix_list):
						# Set the policy_stable flag to False, indicating
						# the need to do another policy iteration loop.
						policy_stable = False

					# Updates the current policy to the new policy for the
					# current state.
					self.pi[state] = new_movement_matrix_list

			# Increments the current iteration counter by 1.
			self.current_iter += 1

		# Visualises the result of the CarRental problem for the case of two
		# locations. Higher number of locations are not as easy to visualise
		# and is thus not visualised.
		if self.num_locations == 2:
			self.visualise()

	def value_iteration_state_func(self, state, tolerance):
		"""
		Performs the value and policy update step for the given state. Only used
		when multiprocessing is enabled.

		Parameter(s):
		state: A valid state of the CarRental problem. The state in this case
		is the number of cars in each location.

		tolerance: Used to check if the value estimate has converged and to
		decide on policies that result in the highest value estimate (the second
		use case is to handle noise in the results caused by floating point
		truncation).

		"""
		# Finds all valid movement matrices for the current state.
		filtered_movement_matrix_list = self.find_valid_moves(state)

		# For the current state, finds the value estimate for every
		# movement matrix in the current policy and places the value
		# estimates in a list.
		values_list = [self.step(state, movement_matrix) for movement_matrix in filtered_movement_matrix_list]

		# Finds the maximum value estimate.
		max_value = max(values_list)

		return state, [movement_matrix for movement_matrix_idx, movement_matrix in enumerate(filtered_movement_matrix_list) if max_value - values_list[movement_matrix_idx] < tolerance], max_value

	def value_iteration_log_result(self, results):
		"""
		Stores the results of value and policy update for every state. Only used
		when multiprocessing is enabled.

		Parameter(s):
		results: List of (state, policy, value) tuples, the result of running
		value and policy updates.

		"""
		# Unpacks the result tuple into state, policy and value.
		for s, pol, val in results:
			# Finds all movement matrix indexes that correspond to the
			# maximum value estimate within a certain tolerance and uses the
			# movement matrix indexes to update the policy of the current
			# state to the new policy for that state.
			self.pi[s] = pol

			# Using the maximum value estimate, updates the value estimate
			# copy of the current state with the new value estimate for that
			# state.
			self.new_v[s] = val

	def value_iteration(self, tolerance):
		"""
		Obtains the optimum value estimate of the CarRental problem and its
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
			# Makes a copy of the value estimate initialised with 0.
			self.new_v = np.zeros_like(self.v)

			if self.use_multiprocessing:
				# Loops through all possible states of the CarRental problem
				# using a generator and performs value and policy updates with
				# multiprocessing.
				with Pool(processes=self.num_max_processes) as p:
					res = p.starmap_async(self.value_iteration_state_func,
										  product(product(*[range(x+1) for x in self.max_cars_list]), [tolerance]),
										  callback=self.value_iteration_log_result)
					res.get()

			else:
				# Loops through all possible states of the CarRental problem
				# using a generator to perform value and policy updates.
				for state in product(*[range(x+1) for x in self.max_cars_list]):
					# Finds all valid movement matrices for the current state.
					filtered_movement_matrix_list = self.find_valid_moves(state)

					# For the current state, finds the value estimate for every
					# movement matrix in the current policy and places the value
					# estimates in a list.
					values_list = [self.step(state, movement_matrix) for movement_matrix in filtered_movement_matrix_list]

					# Finds the maximum value estimate.
					max_value = max(values_list)

					# Finds all movement matrix indexes that correspond to the
					# maximum value estimate within a certain tolerance and uses the
					# movement matrix indexes to update the policy of the current
					# state to the new policy for that state.
					self.pi[state] = [movement_matrix for movement_matrix_idx, movement_matrix in enumerate(filtered_movement_matrix_list) if max_value - values_list[movement_matrix_idx] < tolerance]

					# Using the maximum value estimate, updates the value estimate
					# copy of the current state with the new value estimate for that
					# state.
					self.new_v[state] = max_value

			# Checks if the value estimate has converged within the provided
			# tolerance.
			if np.abs(self.new_v - self.v).max() < tolerance:
				# Visualises the result of the CarRental problem for the case of
				# two locations. Higher number of locations are not as easy to
				# visualise and is thus not visualised.
				if self.num_locations == 2:
					self.visualise()

				# Ends the infinite value iteration loop.
				value_stable = True

			else:
				# Updates the current value estimate to the new value estimate.
				self.v = self.new_v

			# Increments the current iteration counter by 1.
			self.current_iter += 1

	def query(self, state_list=[]):
		"""
		Provides the policy of queried state(s).  This function is used if a
		subset of the policy is desired or if the number of locations exceeds
		two, in which case a visualisation would be difficult to create.

		Parameter(s):
		state_list (optional, default []): A list of valid states that will be
		queried to obtain the relevant policies. The default is an empty list,
		which will display the entire policy.

		Return(s):
		A dictionary containing the policy of all queried states.

		"""
		# Checks if only a subset of all states is required.
		if state_list:
			# Initialises the subset dictionary.
			subset_dict = {}

			# Loops through all queried states.
			for state in state_list:
				# Ensures that the state is a valid state.
				assert state in self.pi

				# Adds the policy to the subset dictionary.
				final_dict[state] = self.pi[state]

			return final_dict
		else:
			# Returns the entire policy.
			return self.pi

	def obtain_optimum(self, solve_method='value_iteration', tolerance=1e-16):
		"""
		Obtains the optimum value estimate of the CarRental problem and its
		corresponding policy by invoking a selected solve method. The results
		are then saved into a diagram.

		Parameter(s):
		solve_method (optional, default value_iteration): Method for solving
		the CarRental problem. Currently implemented methods include
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

		# Initialises the value estimate to 0.
		self.v = np.zeros([x+1 for x in self.max_cars_list])

		# Creates a dictionary to store the policies at every possible state,
		# intialising them all to 1. A generator is used to obtain the indexes
		# of every possible state.
		self.pi = {current_location_num: [np.ones_like(self.max_transfers_arr)] \
			for current_location_num in product(*[range(x+1) for x in self.max_cars_list])}

		# Initialises the current iteration counter to 0.
		self.current_iter = 0

		# Obtains the selected solve function and performs that function to
		# solve the CarRental problem.
		solve_func = self.implemented_solve_methods[solve_method]
		solve_func(tolerance)
