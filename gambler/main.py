"""
Copyright Declaration (C)
From: https://github.com/leeykang/
Use and modification of information, comment(s) or code provided in this document
is granted if and only if this copyright declaration, located between lines 1 to
9 of this document, is preserved at the top of any document where such
information, comment(s) or code is/are used.

"""
from gambler import Gambler

"""
Uses the Gambler class for implementing the gambler problem.

The gambler problem refers to a task where a gambler has a particular monetary
goal and decides on the amount to stake based on his current capital and the
probability of winning money.

Currently implemented methods for solving the gridworld problem include
value iteration and policy iteration.

Based on Chapter 4 of Reinforcement Learning: An Introduction, second edition,
by Sutton and Barto.

"""

def example_1():
	"""
	Example 1: Obtains the solution for a given gambler problem using value
	iteration and policy iteration.

	"""
	# Initialises all required inputs for the Gambler.
	name = 'base_problem'
	goal = 100
	win_prob = 0.4
	discount = 0.9

	# Initialises the Gambler.
	gambler = Gambler(name, goal, win_prob, discount)

	# Obtains the optimum value estimate and policy from the Gambler using
	# value iteration.
	gambler.obtain_optimum('value_iteration')

	# Obtains the optimum value estimate and policy from the Gambler using
	# policy iteration.
	gambler.obtain_optimum('policy_iteration')

if __name__ == '__main__':
	example_1()
