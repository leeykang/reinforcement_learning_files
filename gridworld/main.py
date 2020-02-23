"""
Copyright Declaration (C)
From: https://github.com/leeykang/
Use and modification of information, comment(s) or code provided in this document
is granted if and only if this copyright declaration, located between lines 1 to
9 of this document, is preserved at the top of any document where such
information, comment(s) or code is/are used.

"""
from gridworld import Gridworld

"""
Uses the Gridworld class for implementing the gridworld problem.

The gridworld problem refers to a task where a 2D grid is provided and a player
can explore the grid. The objective of the task is to obtain as many points as
possible by performing actions on the grid that provide rewards to the player.
These rewards are part of the definition of the grid.

Currently implemented methods for solving the gridworld problem include
value iteration and policy iteration.

Based on Chapter 3 of Reinforcement Learning: An Introduction, second edition,
by Sutton and Barto.

"""

def example_1():
	"""
	Example 1: Obains the solution for a given infinite gridworld problem using
	value iteration and policy iteration.

	"""
	# Initialises all required inputs for the Gridworld. In this example,
	# actions list contains up, down, left and right, and the reward dict is
	# stored in the form of key: (coord, action index), value: (reward, new coord).
	name = 'base_problem'
	size = (5,5)
	actions_list = [[-1,0],[1,0],[0,-1],[0,1]]
	reward_dict = {((0,1),1): (10, (4,1)),
				   ((0,2),0): (3, (0,2)),
				   ((0,2),1): (3, (0,2)),
				   ((0,2),2): (3, (0,2)),
				   ((0,2),3): (3, (0,2)),
				   ((1,1),3): (4, (1,4))}
	discount = 0.9

	# Initialises the Gridworld.
	gridworld = Gridworld(name, size, actions_list, reward_dict, discount)

	# Obtains the optimum value estimate and policy from the Gridworld using
	# value iteration.
	gridworld.obtain_optimum('value_iteration')

	# Obtains the optimum value estimate and policy from the Gridworld using
	# policy iteration.
	gridworld.obtain_optimum('policy_iteration')

if __name__ == '__main__':
	example_1()
