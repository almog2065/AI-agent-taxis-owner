import copy
import time

import numpy as np

import sample_agent
from Simulator import Simulator
import itertools

IDS = ["206571135", "207375676"]


def get_all_actions(state, player_number):
	sim = Simulator(state)
	actions_dict = {}
	for taxi, taxi_data in list(state["taxis"].items()):
		if taxi_data["player"] != player_number:
			continue
		actions_dict[taxi] = set()
		for passenger in state["passengers"].keys():
			if (state["passengers"][passenger]["destination"] == state["taxis"][taxi]["location"]
					and state["passengers"][passenger]["location"] == taxi):
				actions_dict[taxi].add(("drop off", taxi, passenger))
		if state["taxis"][taxi]["capacity"] > 0:
			for passenger in state["passengers"].keys():
				if state["passengers"][passenger]["location"] == state["taxis"][taxi]["location"]:
					actions_dict[taxi].add(("pick up", taxi, passenger))
		neighboring_tiles = sim.neighbors(state["taxis"][taxi]["location"])
		for tile in neighboring_tiles:
			actions_dict[taxi].add(("move", taxi, tile))
		actions_dict[taxi].add(("wait", taxi))
	actions = list(itertools.product(*actions_dict.values()))

	return actions


class Node:
	def __init__(self, parent, state, player_number):
		self.parent = parent
		self.wins = 0
		self.tries = 0
		actions = get_all_actions(state, player_number)
		self.childs = {action: None for action in actions}
		self.unborn_childs = len(self.childs)


# Constructing an initial tree i guess
# Each node will store mean score and the number of times it was updated
# children can be a dict with key being action and value being pointer
# parent will be a simple pointer

def check_if_legal(sim, action, player_number):
	try:
		return sim.check_if_action_legal(action, player_number)
	except:
		return False


def calculate_ucb(cur_node):
	scores = {action: 0 for action in cur_node.childs.keys()}
	for action, node in cur_node.childs.items():
		if node is None:
			continue
		base_score = node.wins / node.tries
		deviation_term = np.sqrt(2 * np.log(cur_node.tries) / node.tries)
		score = base_score + deviation_term
		scores[action] = score
	return scores


# sim should always be updated

class UCTAgent:
	def __init__(self, initial_state, player_number):
		self.ids = IDS
		self.player_number = player_number
		self.initial_state = copy.deepcopy(initial_state)
		self.current_game_state = initial_state
		self.root = Node(None, initial_state, player_number)
		self.sim = Simulator(initial_state)
		self.enemy = None
		if self.player_number == 1:
			# deleting actions that are not possible
			self.delete_impossible_actions(self.root)

		# hyper parameters
		rounds = 3000
		number_of_rollouts = 4
		turns = 40
		self.train_tree(rounds, number_of_rollouts, turns=turns)

		cur_node = self.root
		while cur_node.unborn_childs == 0 or self.should_continue(cur_node):
			action, cur_node = self.get_best_child(cur_node)
			print(action)

	def train_tree(self, rounds, number_of_rollouts, turns=200, timelimit=58):
		start = time.time()
		fails = total_count = t = 0
		while t < rounds / number_of_rollouts:
			if (total_count + 1) % (50 // number_of_rollouts) == 0 and time.time() - start > timelimit:
				break
			if (total_count + 1) % (100 // number_of_rollouts) == 0:
				print(round(self.root.wins / self.root.tries, 3))
				print(self.get_best_child(self.root)[0])
				print(t)
			parent_node = self.selection(init=True, prevent_failures=True, to_print=False)
			new_node = self.expansion(parent_node)
			total_count += 1
			# shouldn't happen now because of prevent failures. the problem it costs runtime.
			if new_node == "failed":
				fails += 1
				continue
			wins, tries = self.simulation(k=number_of_rollouts, turns=turns)
			self.backpropagation(new_node, wins, tries)
			t += 1

	# print(f"Failed runs: {fails}")

	def delete_impossible_actions(self, node):
		for action in list(node.childs.keys()):
			if not check_if_legal(self.sim, action, self.player_number):
				del node.childs[action]
				node.unborn_childs -= 1

	def check_it(self, action):
		# sim should know the current game state from act method.
		return check_if_legal(self.sim, action, self.player_number)

	def get_best_child(self, parent, check_legal=False):
		most_tries = -1
		best_action = best_node = None
		for action, node in parent.childs.items():
			# if it is used to take an action, we need to make sure it's legal.
			if node is not None and node.tries > most_tries and \
					(not check_legal or self.check_it(action)):
				best_action, best_node = (action, node)
				most_tries = node.tries
		return best_action, best_node

	def should_continue(self, node):
		for action, node in node.childs.items():
			if node is None and check_if_legal(self.sim, action, self.player_number):
				return False
		return True

	# selecting a node to expand, based on UCB1 formula
	def selection(self, init, to_print=False, prevent_failures=True):
		cur_node = self.root
		self.sim = Simulator(self.current_game_state)
		self.enemy = sample_agent.Agent(self.current_game_state, 3 - self.player_number)
		if init and self.enemy.player_number == 1:
			enemy_move = self.enemy.act(self.current_game_state)
			self.sim.act(enemy_move, self.enemy.player_number)
		action = None
		while cur_node.unborn_childs == 0 or (prevent_failures and self.should_continue(cur_node)):
			scores = calculate_ucb(cur_node)
			for action in sorted(scores, key=scores.get, reverse=True):
				if check_if_legal(self.sim, action, self.player_number):
					break
			if to_print:
				print(action)
			self.play_action_in_simulation(action)
			cur_node = cur_node.childs[action]

		return cur_node

	def play_action_in_simulation(self, action):
		self.sim.act(action, self.player_number)
		if self.player_number == 2:
			self.sim.state["turns to go"] -= 1
		new_state = self.sim.get_state()

		enemy_move = self.enemy.act(new_state)
		self.sim.act(enemy_move, self.enemy.player_number)
		if self.enemy.player_number == 2:
			self.sim.state["turns to go"] -= 1

	# adding new (empty/fresh) children to parent
	def expansion(self, parent_node):
		chosen_action = None
		for action, node in parent_node.childs.items():
			if node is None and check_if_legal(self.sim, action, self.player_number):
				chosen_action = action
				break
		if chosen_action is None:
			return "failed"
		self.play_action_in_simulation(chosen_action)
		state = self.sim.get_state()
		new_node = Node(parent_node, state, self.player_number)
		parent_node.childs[chosen_action] = new_node
		parent_node.unborn_childs -= 1
		return new_node

	def simulation(self, k=1, turns=200):
		original_state = self.sim.get_state()
		wins = 0
		for i in range(k):
			self.sim = Simulator(copy.deepcopy(original_state))
			my_dude = sample_agent.Agent(self.initial_state, self.player_number)
			agents = [my_dude, self.enemy]
			if self.player_number == 2:
				agents = list(reversed(agents))
				my_action = my_dude.act(self.sim.get_state())
				self.sim.act(my_action, my_dude.player_number)
				self.sim.state["turns to go"] -= 1
			# now it's always player 1 than player 2
			while self.sim.get_state()["turns to go"] > 0 or \
					original_state["turns to go"] - self.sim.get_state()["turns to go"] <= turns:
				action1 = agents[0].act(self.sim.get_state())
				self.sim.act(action1, agents[0].player_number)
				action2 = agents[1].act(self.sim.get_state())
				self.sim.act(action2, agents[1].player_number)
				self.sim.state["turns to go"] -= 1
			final_score = self.sim.get_score()
			if final_score[f"player {self.player_number}"] > final_score[f"player {self.enemy.player_number}"]:
				wins += 1
		return wins, k

	def backpropagation(self, node, wins, tries):
		depth = 0
		while node is not None:
			depth += 1
			node.tries += tries
			node.wins += wins
			node = node.parent

	def act(self, state):
		start = time.time()
		self.current_game_state = state
		self.sim = Simulator(state)
		# deleting actions that are not possible because of opponent's movements.
		self.delete_impossible_actions(self.root)

		rounds = 250
		number_of_rollouts = 4
		turns = 40
		self.train_tree(rounds, number_of_rollouts, timelimit=4, turns=turns)
		next_action, next_node = self.get_best_child(self.root, check_legal=False)
		self.root = next_node
		print(f"Time for action: {time.time() - start}")
		return next_action


class Agent(UCTAgent):
	def __init__(self, initial_state, player_number):
		self.ids = IDS
		super().__init__(initial_state, player_number)

	def act(self, state):
		return super().act(state)
