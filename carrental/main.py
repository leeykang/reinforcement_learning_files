"""
Copyright Declaration (C)
From: https://github.com/leeykang/
Use and modification of information, comment(s) or code provided in this document
is granted if and only if this copyright declaration, located between lines 1 to
9 of this document, is preserved at the top of any document where such
information, comment(s) or code is/are used.

"""
from carrental import CarRental

"""
Uses the CarRental class for implementing the car rental problem.

The car rental problem refers to a task where there are multiple locations where
cars can be rented and returned every day. The objective of the task is to find
the optimal number of cars to move between locations every day to maximise the
profit of renting cars minus any costs associated with storing or moving cars
overnight.

Currently implemented methods for solving the car rental problem include
value iteration and policy iteration.

Based on Chapter 4 of Reinforcement Learning: An Introduction, second edition,
by Sutton and Barto.

"""

def example_1():
	"""
	Example 1: Obtains the solution for a given simplified car rental problem
	using value iteration and policy iteration.

	"""
	# Initialises all required inputs for the CarRental problem.
	max_cars_list = [5,6]
	transfer_dict = {(0,1): (3, 0, 2),
					 (1,0): (3, 0, 2)}
	rental_fee = 10
	add_storage_threshold_list = [20,20]
	add_storage_fee = 0
	rental_lambda_list = [3,4]
	return_lambda_list = [3,2]
	discount = 0.9
	use_multiprocessing = False

	# Initialises the CarRental problem.
	carrental = CarRental('simple_problem',
						  max_cars_list,
						  transfer_dict,
						  rental_fee,
						  add_storage_threshold_list,
						  add_storage_fee,
						  rental_lambda_list,
						  return_lambda_list,
						  discount,
						  use_multiprocessing)

	# Obtains the optimum value estimate and policy from the CarRental problem
	# using value iteration.
	carrental.obtain_optimum('value_iteration')

	# Obtains the optimum value estimate and policy from the CarRental problem
	# using policy iteration.
	carrental.obtain_optimum('policy_iteration')

def example_2():
	"""
	Example 2: Obtains the solution for a given car rental problem using value
	iteration and policy iteration with the help of multiprocessing.

	"""
	# Initialises all required inputs for the CarRental problem.
	max_cars_list = [20,20]
	transfer_dict = {(0,1): (5, 0, 2),
					 (1,0): (5, 0, 2)}
	rental_fee = 10
	add_storage_threshold_list = [20,20]
	add_storage_fee = 0
	rental_lambda_list = [3,4]
	return_lambda_list = [3,2]
	discount = 0.9
	use_multiprocessing = True
	num_max_processes = 8

	# Initialises the CarRental problem.
	carrental = CarRental('base_problem',
						  max_cars_list,
						  transfer_dict,
						  rental_fee,
						  add_storage_threshold_list,
						  add_storage_fee,
						  rental_lambda_list,
						  return_lambda_list,
						  discount,
						  use_multiprocessing,
						  num_max_processes)

	# Obtains the optimum value estimate and policy from the CarRental problem
	# using value iteration with multiprocessing.
	carrental.obtain_optimum('value_iteration')

	# Obtains the optimum value estimate and policy from the CarRental problem
	# using policy iteration with multiprocessing.
	carrental.obtain_optimum('policy_iteration')

def example_3():
	"""
	Example 3: Obtains the solution for a different given car rental problem
	using value iteration and policy iteration with the help of multiprocessing.

	"""
	# Initialises all required inputs for the CarRental problem.
	max_cars_list = [20,20]
	transfer_dict = {(0,1): (5, 1, 2),
					 (1,0): (5, 0, 2)}
	rental_fee = 10
	add_storage_threshold_list = [10,10]
	add_storage_fee = 4
	rental_lambda_list = [3,4]
	return_lambda_list = [3,2]
	discount = 0.9
	use_multiprocessing = True
	num_max_processes = 8

	# Initialises the CarRental problem.
	carrental = CarRental('additional_problem',
						  max_cars_list,
						  transfer_dict,
						  rental_fee,
						  add_storage_threshold_list,
						  add_storage_fee,
						  rental_lambda_list,
						  return_lambda_list,
						  discount,
						  use_multiprocessing,
						  num_max_processes)

	# Obtains the optimum value estimate and policy from the CarRental problem
	# using value iteration with multiprocessing.
	carrental.obtain_optimum('value_iteration')

	# Obtains the optimum value estimate and policy from the CarRental problem
	# using policy iteration with multiprocessing.
	carrental.obtain_optimum('policy_iteration')

if __name__ == '__main__':
	example_1()
	# example_2()
	# example_3()
