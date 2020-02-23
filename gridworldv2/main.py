"""
Copyright Declaration (C)
From: https://github.com/leeykang/
Use and modification of information, comment(s) or code provided in this document
is granted if and only if this copyright declaration, located between lines 1 to
9 of this document, is preserved at the top of any document where such
information, comment(s) or code is/are used.

"""
from gridworldv2 import GridworldV2

"""
Uses the GridworldV2 class for implementing the gridworld v2 problem.

The gridworld v2 problem refers to a task where a 2D grid is provided and a player
can explore the grid. The objective of the task is to obtain as many points as
possible by performing actions on the grid that provide rewards to the player.
These rewards are part of the definition of the grid.

Currently implemented methods for solving the gridworld v2 problem include
value iteration and policy iteration.

Based on Chapter 4 of Reinforcement Learning: An Introduction, second edition,
by Sutton and Barto.

"""

def example_1():
	"""
	Example 1: Obains the solution for a given infinite gridworld v2 problem
	using value iteration and policy iteration.

	"""
	# Initialises all required inputs for the GridworldV2. In this example,
	# actions list contains up, down, left and right.
	name = 'base_problem'
	size = (5,5)
	actions_list = [[-1,0],[1,0],[0,-1],[0,1]]
	terminal_coords_list = [[0,0], [4,4]]
	discount = 0.9

	# Initialises the GridworldV2.
	gridworldv2 = GridworldV2(name, size, actions_list, terminal_coords_list, discount)

	# Obtains the optimum value estimate and policy from the GridworldV2 using
	# value iteration.
	gridworldv2.obtain_optimum('value_iteration')

	# Obtains the optimum value estimate and policy from the GridworldV2 using
	# policy iteration.
	gridworldv2.obtain_optimum('policy_iteration')

if __name__ == '__main__':
	example_1()
