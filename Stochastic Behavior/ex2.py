import copy
import itertools
import json
import math
import time

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100


def product_dict(**kwargs):
	keys = kwargs.keys()
	vals = kwargs.values()
	for instance in itertools.product(*vals):
		yield dict(zip(keys, instance))


ids = ["206571135", "207375676"]


def actions(initial, state, optimal=True):
	"""Returns all the actions that can be executed in the given
		state. The result should be a tuple (or other iterable) of actions
		as defined in the problem description file"""
	all_actions_list = []
	for taxi, taxi_data in state['taxis'].items():
		actions_list = []
		# wait
		actions_list.append(("wait", taxi))
		# refuel
		if initial["map"][taxi_data["location"][0]][taxi_data["location"][1]] == 'G':
			actions_list.append(("refuel", taxi))
		# dropoff
		for passenger, pass_data in list(state["passengers"].items()):
			if tuple(taxi_data["location"]) == tuple(pass_data["destination"]) and pass_data["location"] == taxi:
				actions_list.append(("drop off", taxi, passenger))
		# pickup
		if taxi_data["capacity"] > 0:
			for passenger, pass_data in state["passengers"].items():
				if not isinstance(pass_data["location"], str) and tuple(pass_data["location"]) == tuple(
						taxi_data["location"]) \
						and tuple(pass_data["location"]) != tuple(pass_data["destination"]):
					actions_list.append(("pick up", taxi, passenger))
		# move
		if taxi_data["fuel"] > 0:
			# up
			if taxi_data["location"][0] > 0 and initial["map"][taxi_data["location"][0] - 1][
				taxi_data["location"][1]] != 'I':
				actions_list.append(("move", taxi, (taxi_data["location"][0] - 1, taxi_data["location"][1])))

			# left
			if taxi_data["location"][1] > 0 and initial["map"][taxi_data["location"][0]][
				taxi_data["location"][1] - 1] != 'I':
				actions_list.append(("move", taxi, (taxi_data["location"][0], taxi_data["location"][1] - 1)))

			# down
			if taxi_data["location"][0] < len(initial["map"]) - 1 and initial["map"][taxi_data["location"][0] + 1][
				taxi_data["location"][1]] != 'I':
				actions_list.append(("move", taxi, (taxi_data["location"][0] + 1, taxi_data["location"][1])))

			# right
			if taxi_data["location"][1] < len(initial["map"][0]) - 1 and initial["map"][taxi_data["location"][0]][
				taxi_data["location"][1] + 1] != 'I':
				actions_list.append(("move", taxi, (taxi_data["location"][0], taxi_data["location"][1] + 1)))

		all_actions_list.append(actions_list)

	legal_actions = []
	if optimal:
		legal_actions.append(("reset"))
	for action in itertools.product(*all_actions_list):
		legal = True
		for taxi_action_1 in action:
			if taxi_action_1[0] == "move":
				for taxi_action_2 in action:
					if taxi_action_1 != taxi_action_2:
						if taxi_action_2[0] == "move":
							if list(taxi_action_1[2]) == list(taxi_action_2[2]):
								legal = False
						else:
							if list(state["taxis"][taxi_action_2[1]]["location"]) == list(taxi_action_1[2]):
								legal = False
		if legal:
			legal_actions.append(action)
	return legal_actions


def reward(state, action, train=False, bfs_dict=None):
	sum_reward = 0
	if action == "reset":
		return -RESET_PENALTY
	if action == "terminate":
		return 0
	for act in action:
		if act[0] == "refuel":
			sum_reward -= REFUEL_PENALTY
		elif act[0] == "drop off":
			sum_reward += DROP_IN_DESTINATION_REWARD
	if False and train:
		for act in action:
			if act[0] == "wait":
				for pass_data in list(state["passengers"].values()):
					if pass_data["location"] == act[1] and \
							state["taxis"][act[1]]["location"] not in pass_data["possible_goals"]:
						sum_reward -= 5
			elif act[0] == "move":
				for pass_data in list(state["passengers"].values()):
					if pass_data["location"] == act[1] and bfs_dict[pass_data["destination"], act[2]] > \
							bfs_dict[pass_data["destination"], state["taxis"][act[1]]["location"]]:
						sum_reward -= 5
					elif pass_data["location"] == act[1] and bfs_dict[pass_data["destination"], act[2]] < \
							bfs_dict[pass_data["destination"], state["taxis"][act[1]]["location"]]:
						sum_reward += 5
					elif not isinstance(pass_data["location"], str) and pass_data["location"] != pass_data[
						"destination"] \
							and bfs_dict[pass_data["location"], act[2]] < bfs_dict[
						pass_data["location"], state["taxis"][act[1]]["location"]]:
						sum_reward += 2.5
					elif not isinstance(pass_data["location"], str) and pass_data["location"] != pass_data[
						"destination"] \
							and bfs_dict[pass_data["location"], act[2]] > bfs_dict[
						pass_data["location"], state["taxis"][act[1]]["location"]]:
						sum_reward -= 2.5

	return sum_reward


def apply_atomic_action(state, initial_state, atomic_action):
	"""
	apply an atomic action to the state
	"""
	taxi_name = atomic_action[1]
	if atomic_action[0] == 'move':
		state['taxis'][taxi_name]['location'] = atomic_action[2]
		state['taxis'][taxi_name]['fuel'] -= 1
		return state
	elif atomic_action[0] == 'pick up':
		passenger_name = atomic_action[2]
		state['taxis'][taxi_name]['capacity'] -= 1
		state['passengers'][passenger_name]['location'] = taxi_name
		return state
	elif atomic_action[0] == 'drop off':
		passenger_name = atomic_action[2]
		state['passengers'][passenger_name]['location'] = state['taxis'][taxi_name]['location']
		state['taxis'][taxi_name]['capacity'] += 1
		return state
	elif atomic_action[0] == 'refuel':
		state['taxis'][taxi_name]['fuel'] = initial_state['taxis'][taxi_name]['fuel']
		return state
	elif atomic_action[0] == 'wait':
		return state
	else:
		raise NotImplemented


def check_state(initial, state):
	taxi_count = {taxi: 0 for taxi in list(state["taxis"].keys())}
	for passenger, pass_data in list(state["passengers"].items()):
		if isinstance(pass_data["location"], str):
			taxi_count[pass_data["location"]] += 1
	for taxi, data_taxi in list(initial["taxis"].items()):
		if data_taxi["capacity"] - state["taxis"][taxi]["capacity"] != taxi_count[taxi]:
			return False
		if data_taxi["capacity"] < taxi_count[taxi]:
			return False
		for taxi1, data_taxi1 in list(state["taxis"].items()):
			if taxi1 == taxi:
				continue
			if tuple(state["taxis"][taxi]["location"]) == tuple(data_taxi1["location"]):
				return False
	return True


def possible_states(state, initial_state, action):
	new_state = json.loads(state)
	if action == "reset":
		new_state["taxis"] = initial_state["taxis"]
		new_state["passengers"] = initial_state["passengers"]
		return [(1, json.dumps(new_state, sort_keys=True))]
	for atomic_action in action:
		new_state = apply_atomic_action(new_state, initial_state, atomic_action)
	poss_dict = dict()
	for passenger, pass_data in list(new_state["passengers"].items()):
		pass_pos_list = []
		p = pass_data["prob_change_goal"]
		# also here we need to add the current destination seperatly somehow.
		# we need to account for 1-p, p/n, p/n,... case, that is not the 1-p+p/n, p/n, p/n, ... case
		pos_list = pass_data["possible_goals"]
		if pass_data["destination"] not in pos_list:
			pos_list.append(pass_data["destination"])
		for poss_goal in pos_list:
			add_state = copy.deepcopy(pass_data)
			add_state["destination"] = poss_goal
			if poss_goal != pass_data["destination"]:
				add_prob = p / len(pass_data["possible_goals"])
			else:
				if pass_data["destination"] in pass_data["possible_goals"]:
					add_prob = 1 - p + p / len(pass_data["possible_goals"])
				else:
					add_prob = 1 - p
			pass_pos_list.append((add_prob, add_state))
		poss_dict[passenger] = pass_pos_list
	all_pass_states = list(product_dict(**poss_dict))
	poss_list = []
	for pass_state in all_pass_states:
		total_p = 1
		alter_state = copy.deepcopy(new_state)
		for pass1, (prob, des) in list(pass_state.items()):
			total_p = total_p * prob
			alter_state["passengers"][pass1] = des
		poss_list.append((total_p, json.dumps(alter_state, sort_keys=True)))
	return poss_list


def all_states(initial):
	states_all = dict()
	for taxi, taxi_data in list(initial['taxis'].items()):
		states_taxi = []
		for i in range(len(initial["map"])):
			for j in range(len(initial["map"][0])):
				if initial["map"][i][j] == 'I':
					continue
				for fuel in range(taxi_data["fuel"] + 1):
					for capacity in range(taxi_data["capacity"] + 1):
						new_state_taxi = copy.deepcopy(initial['taxis'][taxi])
						new_state_taxi["capacity"] = capacity
						new_state_taxi["fuel"] = fuel
						new_state_taxi["location"] = (i, j)
						states_taxi.append(new_state_taxi)
		states_all[taxi] = states_taxi

	for passenger, pass_data in list(initial['passengers'].items()):
		states_pass = []
		# add here the current destination in case it's not in possible list
		pos_list = pass_data["possible_goals"]
		if pass_data["destination"] not in pos_list:
			pos_list.append(pass_data["destination"])
		for destination in pos_list:
			poss_locs = [pass_data["location"]]
			poss_locs += list(pass_data["possible_goals"])
			poss_locs += list(initial["taxis"].keys())
			for location in set(poss_locs):
				new_state_pass = copy.deepcopy(initial['passengers'][passenger])
				new_state_pass["location"] = location
				new_state_pass["destination"] = destination
				states_pass.append(new_state_pass)
		states_all[passenger] = states_pass

	states_list = list(product_dict(**states_all))
	new_states_list = []
	for state in states_list:
		new_state = {"optimal": initial["optimal"], "map": initial["map"], "taxis": dict(),
		             "passengers": dict()}
		for key, value in list(state.items()):
			if key in list(initial['taxis'].keys()):
				new_state["taxis"][key] = value
			else:
				new_state["passengers"][key] = value
		if check_state(initial, new_state):
			new_states_list.append(json.dumps(new_state, sort_keys=True))
	return new_states_list


def dynamic_programming(initial, all_states_list, action_dict, pos_states_dict, optimal=True,
                        consecutive_zero_changes_threshold=1):
	V = {(0, state): 0 for state in all_states_list}
	Q = dict()
	policy = dict()
	consecutive_zero_changes = 0
	for t in range(1, initial["turns to go"] + 1):
		policy_changes = 0
		# print(t)
		for state in all_states_list:
			max_q = -math.inf
			best_action = None
			for action in action_dict[state]:
				Q[(t, state, action)] = reward(state, action)
				for prob, pos_state in pos_states_dict[(state, action)]:
					Q[(t, state, action)] += prob * V[(t - 1, pos_state)]
				if max_q < Q[(t, state, action)]:
					max_q = Q[(t, state, action)]
					best_action = action
			policy[(t, state)] = best_action
			V[(t, state)] = max_q
			if not optimal:
				if t > 1 and policy[(t, state)] != policy[(t - 1, state)]:
					policy_changes += 1
		if t > 1 and not optimal:
			print(policy_changes)
			consecutive_zero_changes += 1 if policy_changes == 0 else -consecutive_zero_changes
		if consecutive_zero_changes == consecutive_zero_changes_threshold:
			print(f"stopped at iteration {t}")
			break

	for t1 in range(t + 1, initial["turns to go"] + 1):
		for state in all_states_list:
			policy[(t1, state)] = policy[(t1 - 1, state)]
			V[(t1, state)] = V[(t1 - 1, state)] + 0.9 * (V[(t1 - 1, state)] - V[(t1 - 2, state)])

	return policy, V, Q


class OptimalTaxiAgent:
	def __init__(self, initial):
		t0 = time.time()
		self.initial = copy.deepcopy(initial)
		# self.initial["taxis"].pop("taxis 1")
		self.all_states_list = all_states(initial)
		# print(len(self.all_states_list))
		self.action_dict = {state: actions(self.initial, json.loads(state)) for state in self.all_states_list}
		self.pos_states_dict = {(state, action): possible_states(state, initial, action) for state in
		                        self.all_states_list for action in
		                        self.action_dict[state]}
		self.policy, self.v, self.q = dynamic_programming(self.initial, self.all_states_list, self.action_dict,
		                                                  self.pos_states_dict)
		#print(f"Runtime: {time.time() - t0}")

	def act(self, state):
		state = copy.deepcopy(state)
		t = state.pop("turns to go")
		if False and t == self.initial["turns to go"]:
			print(f'The expected value of this problem is {self.v[(t, json.dumps(state, sort_keys=True))]}')
		a = self.policy[(t, json.dumps(state, sort_keys=True))]
		# print(a, self.q[(t, json.dumps(state, sort_keys=True), a)])
		return self.policy[(t, json.dumps(state, sort_keys=True))]


def guess_num_states(initial):
	pass_dict_count = {}
	taxis_num = len(initial["taxis"])
	num = get_map_size(initial) ** taxis_num
	for taxi_data in list(initial["taxis"].values()):
		num *= (taxi_data["fuel"] + 1)
	for p, pass_data in list(initial["passengers"].items()):
		poss_dests = set(pass_data["possible_goals"])
		poss_dests.add(pass_data["destination"])
		poss_locs = poss_dests.copy()
		poss_locs.add(pass_data["location"])
		pass_dict_count[p] = len(poss_dests) * (len(poss_locs) + taxis_num)
		num *= len(poss_dests) * (len(poss_locs) + taxis_num)
	return num, pass_dict_count


def get_map_size(initial):
	map_size = 0
	for i in range(len(initial["map"])):
		for j in range(len(initial["map"][0])):
			if initial["map"][i][j] != "I":
				map_size += 1
	return map_size


class TaxiAgent:
	def __init__(self, initial):
		t0 = time.time()
		self.T = initial["turns to go"]
		map_size = get_map_size(initial)
		states_count_guess, pass_dict_count = guess_num_states(initial)
		#print(states_count_guess)
		if states_count_guess < 50000:
			opt = OptimalTaxiAgent(initial)
			self.map = initial["map"]
			self.best_taxis = list(initial["taxis"].keys())
			self.best_passengers = list(initial["passengers"].keys())
			self.policy = opt.policy
			self.v = opt.v
		else:
			value_conf = []
			taxi_options = min(max(100 // map_size, 1), len(initial["taxis"].keys()))
			pass_options = 5 // taxi_options
			taxis = list(initial["taxis"].items())[:taxi_options]
			passengers = list(initial["passengers"].items())[:pass_options]
			for taxi, taxi_data in taxis:
				small_initial = copy.deepcopy(initial)
				small_initial["taxis"] = {taxi: taxi_data}
				for taxi_2, taxi_data2 in taxis:
					if taxi_2 != taxi:
						loc = taxi_data2["location"]
						small_initial["map"][loc[0]][loc[1]] = "I"
				for passenger, pass_data in passengers:
					smaller_initial = copy.deepcopy(small_initial)
					smaller_initial["passengers"] = {passenger: pass_data}
					policy, v, q = self.run(smaller_initial)
					t = smaller_initial.pop("turns to go")
					conf = (smaller_initial["map"], taxi, [passenger], policy, v)
					value_conf.append((v[(t, json.dumps(smaller_initial, sort_keys=True))], conf))
			value_conf.sort()
			if len(value_conf) >= 2 and (time.time() - t0) * pass_dict_count[value_conf[-2][1][2][0]] < 280 and \
					value_conf[-2][1][1] == value_conf[-1][1][1]:
				best_taxi = value_conf[-1][1][1]
				pass1 = value_conf[-1][1][2][0]
				pass2 = value_conf[-2][1][2][0]
				new_initial = copy.deepcopy(initial)
				for taxi_2, taxi_data2 in taxis:
					if taxi_2 != best_taxi:
						loc = taxi_data2["location"]
						new_initial["map"][loc[0]][loc[1]] = "I"
				new_initial["taxis"] = {best_taxi: new_initial["taxis"][best_taxi]}
				new_initial["passengers"] = {pass1: new_initial["passengers"][pass1],
				                             pass2: new_initial["passengers"][pass2]}
				policy, v, q = self.run(new_initial)
				t = new_initial.pop("turns to go")
				best_conf = (new_initial["map"], best_taxi, [pass1, pass2], policy, v)
			else:
				best_conf = value_conf[-1][1]
			self.map = best_conf[0]
			self.best_taxis = [best_conf[1]]
			self.best_passengers = best_conf[2]
			self.policy = best_conf[3]
			self.v = best_conf[4]
		#print(f"Final Runtime: {time.time() - t0}")

	def run(self, smaller_initial, optimal=True):
		t1 = time.time()
		all_states_list = all_states(smaller_initial)
		action_dict = {state: actions(smaller_initial, json.loads(state)) for state in all_states_list}
		pos_states_dict = {(state, action): possible_states(state, smaller_initial, action) for state in
		                   all_states_list for action in
		                   action_dict[state]}
		policy, v, q = dynamic_programming(smaller_initial, all_states_list, action_dict,
		                                   pos_states_dict, optimal)
		return policy, v, q

	def act(self, state):
		state = copy.deepcopy(state)
		all_taxis = list(state["taxis"].keys())
		t = state.pop("turns to go")
		state["map"] = self.map
		for passenger in list(state["passengers"].keys()):
			if passenger not in self.best_passengers:
				state["passengers"].pop(passenger)
		for taxi in list(state["taxis"].keys()):
			if taxi not in self.best_taxis:
				state["taxis"].pop(taxi)

		if False and t == self.T:
			print(f'The expected value of this problem is {self.v[(t, json.dumps(state, sort_keys=True))]}')
		a = self.policy[(t, json.dumps(state, sort_keys=True))]
		if a == "reset":
			return a

		adapted_a = []
		i = 0
		for taxi in all_taxis:
			if taxi not in self.best_taxis:
				adapted_a.append(("wait", taxi))
			else:
				adapted_a.append(a[i])
				i += 1
		return tuple(adapted_a)
