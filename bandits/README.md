# Bandit Algorithm
Uses Arena, Player and Bandit classes for implementing the bandit problem.
Examples of such usage is shown in main.py.

The Arena pits Players against each other for specific Bandit tasks and can be
used in parameter studies to study parameter changes in Players for specific
Bandit tasks.

The Bandit class currently implements the stationary and nonstationary generic
multi-armed bandit tasks.

	1. The stationary bandit starts with actions having means that are randomly
	initialised from a Gaussian distribution with an initial mean and standard
	deviation. The best action is defined based on these true mean values. The
	reward for taking each action is based on a different Gaussian distribution
	with the selected action's true mean as the mean and a separately defined
	standard deviation.

	2. The nonstationary bandit is the same as the stationary bandit, except
	that after every timestep of every run, the mean of each action is modified
	by adding a variable obtained from a different Gaussian distribution with a
	separately defined mean and standard deviation.

The Player class is designed to be subclassed. Currently, five player types have
been implemented by subclassing the Player class.

	 1. RandomPlayer: No computer learning involved, a random action is taken.

	 2. HumanPlayer: No computer learning involved, takes user input as the
	 	action.

	 3. QPlayer: Computer learning is involved, uses Q values to estimate the
	 true action values and decide on the current action to take.

	 4. UCBQPlayer: Computer learning is involved, uses Q values to estimate the
	 true action values, but uses both Q values and an upper confidence bound
	 term to decide on the current action to take.

	 5. GradientPlayer: Computer learning involved, uses gradient ascent to
	 obtain action preferences.

Based on Chapter 2 of Reinforcement Learning: An Introduction, second edition,
by Sutton and Barto.
