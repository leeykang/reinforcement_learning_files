import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from bandit import Bandit
from player import Player

class Arena:

	"""
	Provides the definition of an arena for Bandit(s) to be used to pit / study
	Player(s).

	"""
	def __init__(self):
		self.bandits = []
		self.players = []
		self.num_bandits = 0
		self.num_players = 0

	def add_bandits(self, bandits):
		"""
		Adds valid Bandit(s) to the Arena.

		Parameter(s):
		bandits: Either a single Bandit subclassed object or a list of Bandit
		subclassed objects.

		"""
		# If a single Bandit object is passed as input, place it into a list.
		if type(bandits) != list:
			bandits = [bandits]

		# Checks each entry in the list to ensure that they are Bandit objects.
		for bandit_idx, bandit in enumerate(bandits):
			assert isinstance(bandit, Bandit), \
				'Entry %i is of type %s, not Bandit' % (bandit_idx, type(bandit))

		# Adds all Bandit(s) to the Arena and increments Arena's num_bandits
		# by the number of added Bandit(s).
		self.bandits += bandits
		self.num_bandits += len(bandits)

	def remove_bandits(self, bandit_idxs):
		"""
		Removes Bandit(s) from the Arena.

		Parameter(s):
		bandit_idxs: Either a single integer index, corresponding to the index
		of the Bandit to be removed from the Arena, or a list of integer indexes,
		corresponding to the indexes of Bandit(s) to be removed from the Arena.
		All indexes are with reference to self.bandits.

		"""
		# If only a single index is passed as input, place it into a list.
		if type(bandit_idxs) != list:
			bandit_idxs = [bandit_idxs]

		# Sorts the bandit_idxs in descending order (preventing indexes from
		# being removed incorrectly) and removes Bandit(s) in their respective
		# indexes from self.bandits one by one.
		for bandit_idx in sorted(bandit_idxs, reverse=True):
			del self.bandits[bandit_idx]

		# Deducts Arena's num_bandits by the number of removed Bandit(s).
		self.num_bandits -= len(bandit_idxs)

	def display_bandits(self, bandit_idxs=[]):
		"""
		Displays attributes about the Bandit(s) in the Arena.

		Parameter(s):
		bandit_idxs (optional, default = []): Index(es) of Bandit(s) in the Arena
		to be displayed. The default of an empty list indicates that all Bandit(s)
		in the Arena are to be displayed.

		"""
		if len(bandit_idxs):
			# bandit_idxs is not empty, only specific Bandit index(es) are to be
			# displayed.
			for bandit_idx in bandit_idxs:
				self.bandits[bandit_idx].display()
		else:
			# bandit_idxs is empty, all Bandit(s) are to be displayed.
			for bandit in self.bandits:
				bandit.display()

	def add_players(self, players):
		"""
		Adds valid Player(s) to the Arena.

		Parameter(s):
		players: Either a single Player subclassed object or a list of Player
		subclassed objects

		"""
		# If a single Player object is passed as input, place it into a list.
		if type(players) != list:
			players = [players]

		# Checks each entry in the list to ensure that they are Player objects.
		for player_idx, player in enumerate(players):
			assert isinstance(player, Player), \
				'Entry %i is of type %s, not Player' % ((player_idx+1), type(player))

		# Adds all Player(s) to the Arena and increments Arena's num_players
		# by the number of added Player(s).
		self.players += players
		self.num_players += len(players)

	def remove_players(self, player_idxs):
		"""
		Removes Player(s) from the Arena.

		Parameter(s):
		player_idxs: Either a single integer index, corresponding to the index
		of the Player to be removed from the Arena, or a list of integer indexes,
		corresponding to the indexes of Player(s) to be removed from the Arena.
		All indexes are with reference to self.players.

		"""
		# If only a single index is passed as input, place it into a list.
		if type(player_idxs) != list:
			player_idxs = [player_idxs]

		# Sorts the player_idxs in descending order (preventing indexes from
		# being removed incorrectly) and removes Player(s) in their respective
		# indexes from self.players one by one.
		for player_idx in sorted(player_idxs, reverse=True):
			del self.players[player_idx]

		# Deducts Arena's num_players by the number of removed Player(s).
		self.num_players -= len(player_idxs)

	def display_players(self, player_idxs=[]):
		"""
		Displays attributes about the Player(s) in the Arena.

		Parameters:
		player_idxs (optional, default = []): Index(es) of Player(s) in the Arena
		to be displayed. The default of an empty list indicates that all Player(s)
		in the Arena are to be displayed.

		"""
		if len(player_idxs):
			# player_idxs is not empty, only specific Player index(es) are to be
			# displayed.
			for player_idx in player_idxs:
				self.players[player_idx].display()
		else:
			# player_idxs is empty, all Player(s) are to be displayed.
			for player in self.players:
				player.display()

	def display(self):
		"""
		Displays attributes about the Arena.

		"""
		# Displays attributes about the Bandit(s) in the Arena.
		print('NUM BANDITS: %i' % self.num_bandits)
		self.display_bandits()
		print("")

		# Displays attributes about the Player(s) in the Arena.
		print('NUM PLAYERS: %i' % self.num_players)
		self.display_players()
		print("")

	def run(self,
			run_mode='pit',
			parameter_range=None):
		"""
		Runs the Arena for all Bandit(s) and all Player(s) based on the
		specified run mode.

		Parameter(s):
		run_mode: Mode for running the Arena. Has two modes currently
		implemented:
		1. pit (default): Used to pit Player(s) against each other by evaluating
		their performance on various Bandit(s).

		2. parameter_study: Used to study the effect of changing certain
		parameters on the performance of Player(s) on various Bandit(s).

		parameter_range (optional, default None): Only used for parameter
		studies, a tuple/list with two values that corresponds to the minimum
		and maximum value of the study range of all variables in the parameter
		study.

		"""
		# Ensures that the run_mode is a valid parameter.
		assert run_mode in ['pit', 'parameter_study'], \
			'%s is not a valid run type for the Arena, try pit or parameter_study' % run_mode

		# Obtain the current working directory.
		curr_dir = os.path.dirname(os.path.abspath(__file__))

		print('RUNNING ARENA IN %s MODE' % run_mode.upper())

		for bandit_idx, bandit in enumerate(self.bandits):
			print("BANDIT NUMBER: %i" % (bandit_idx+1))

			# Flag to ensure that player parameters are updated when a new
			# Bandit problem arises.
			new_bandit = True

			# Flag to ensure that a parameter study will be started (the pit
			# run_mode is considered to be a study with only one variable).
			study_in_progress = True

			# Counter to detect the current_study_index, starts from -1 because
			# the counter is incremented at the start of every study_in_progress
			# loop.
			current_study_index = -1

			# Obtain number of actions, number of timesteps, number of runs and
			# first considered reward step for the particular Bandit.
			num_actions = bandit.num_actions
			num_timesteps = bandit.num_timesteps
			num_runs = bandit.num_runs
			first_considered_reward_step = bandit.first_considered_reward_step

			while study_in_progress:
				# The current study is assumed to have ended unless a player
				# still has available study parameters.
				study_in_progress = False

				# Incrementing the current study index by 1.
				current_study_index += 1

				# Sets up the progress bar based on num_runs.
				with tqdm(total=num_runs, desc='RUN STATUS', ascii=" .o0", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as bar:
					for run_idx in range(num_runs):
						# Analyses the Bandit at the start of each run.
						bandit.analyse()

						for t in range(bandit.num_timesteps):
							for player_idx, player in enumerate(self.players):
								if new_bandit:
									# For the first timestep of the first run of
									# a new Bandit, update the current Player's
									# parameters to those of the new Bandit.
									# For a parameter study, the current
									# Player's study variable is updated to the
									# first study value as well.
									if run_mode == 'parameter_study':
										player.update_params(num_actions=num_actions, num_runs=num_runs, \
											num_timesteps=num_timesteps, first_considered_reward_step=first_considered_reward_step, \
											**{player.study_variable: player.study_range[current_study_index]})
									else:
										player.update_params(num_actions=num_actions, num_runs=num_runs, \
											num_timesteps=num_timesteps, first_considered_reward_step=first_considered_reward_step)

									# Enable an action selection for the current
									# Player.
									player.to_run = True

									# Because a new Bandit always ensures that
									# a study / pitting can take place,
									# study_in_progress can be set to True.
									study_in_progress = True

								elif run_idx == 0 and t == 0:
									# For the first timestep of the first run of
									# an ongoing Bandit, it is definitely part
									# of a parameter study. Therefore, an
									# attempt is made to update the current
									# Player's study variable to the new study
									# value.
									try:
										player.update_params(**{player.study_variable: player.study_range[current_study_index]})

										# When the attempt is successful, the
										# current Player's action selection can
										# be enabled and study_in_progress can
										# be set to True.
										study_in_progress = True
										player.to_run = True

									except IndexError:
										# IndexError indicates that the
										# current_study_index is beyond the
										# number of values in Player's
										# study_range, meaning that the Player's
										# action selection should be disabled.
										player.to_run = False

								elif t == 0:
									# For the first timesteps of subsequent runs
									# of any Bandit, new or current, no Player
									# parameters are changed. Therefore, only
									# a simple reset of the current Player is
									# required.
									player.reset()

								if player.to_run:
									# Obtains an action from the current Player
									# based on its player_type.
									action = player.obtain_action(t)

									# Uses the Bandit to evaluate the current
									# Player's selected action.
									reward, is_optimal = bandit.evaluate(action)

									# Updates the current Player based on the
									# results obtained.
									player.update(t, action, reward, is_optimal)

							# Performs update of the current Bandit, if necessary.
							bandit.update()

							# Deactivates the new_bandit flag after the first
							# timestep of the first run.
							if new_bandit:
								new_bandit = False

						# Store each Player's reward for the current run from
						# the first considered reward step, normalised by the
						# total number of runs.
						for player in self.players:
							player.stored_optimum += (player.player_optimum[first_considered_reward_step:] / float(num_runs))
							player.stored_reward += (player.player_reward[first_considered_reward_step:] / num_runs)

						# Update the progress bar.
						bar.update(1)

				# For parameter studies, add each Player's average reward in
				# the desired range to each Player's study result.
				if run_mode == 'parameter_study':
					for player in self.players:
						if player.to_run:
							player.study_result.append(player.stored_reward.mean().item())

				else:
					# As it is not a parameter study, end the while loop.
					break

			if run_mode == 'pit':
				# For all players, visualise the average percentage of optimum
				# actions taken (the action with the highest true reward (mean))
				# for each timestep over all the runs of the current Bandit.
				for player_idx, player in enumerate(self.players):
					plt.plot(range(1+first_considered_reward_step,1+num_timesteps), player.stored_optimum, label='player%i_%s' % ((player_idx+1), player.player_type))
				plt.legend(loc="best")
				plt.savefig(os.path.join(curr_dir, 'bandit%i_optimum.png' % (bandit_idx + 1)))
				plt.close()

				# For all players, bisualise the average reward gained for each
				# timestep over all the runs of the current Bandit.
				for player_idx, player in enumerate(self.players):
					plt.plot(range(1+first_considered_reward_step,1+num_timesteps), player.stored_reward, label='player%i_%s' % ((player_idx+1), player.player_type))
				plt.legend(loc="best")
				plt.savefig(os.path.join(curr_dir, 'bandit%i_reward.png' % (bandit_idx + 1)))
				plt.close()

			else:
				# Visualise the parameter study results for all the Player(s) in
				# the current Bandit.
				for player_idx, player in enumerate(self.players):
					plt.plot(player.study_range, player.study_result, marker='x', label='player%i_%s_%s' % ((player_idx+1), player.player_type, player.study_variable))
				plt.legend(loc="best")
				plt.xscale('log', basex=2)
				plt.xlim(parameter_range)
				plt.savefig(os.path.join(curr_dir, 'bandit%i_parameter_study.png' % (bandit_idx + 1)))
				plt.close()

		print('COMPLETED ARENA IN %s MODE' % run_mode.upper())
