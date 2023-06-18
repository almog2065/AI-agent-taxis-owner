import search
import random
import math
import json
import itertools
import numpy as np

ids = ["206571135", "207375676"]


class TaxiProblem(search.Problem):
	"""This class implements a medical problem according to problem description file"""

	def __init__(self, initial):
		"""Don't forget to implement the goal test
		You should change the initial to your own representation.
		search.Problem.__init__(self, initial) creates the root node"""
		self.map = initial["map"]
		self.taxis = initial["taxis"]
		self.passengers = initial["passengers"]

		self.build_bfs_dict()

		self.gas_stations = []
		for i in range(len(self.map)):
			for j in range(len(self.map[0])):
				if self.map[i][j] == "G":
					self.gas_stations.append((i, j))

		max_fuel = max([taxi_dict["fuel"] for taxi_dict in self.taxis.values()])
		self.unsafe_pass = []
		for passenger, pass_dict in self.passengers.items():
			loc = pass_dict["location"]
			destination = pass_dict["destination"]
			dist_to_dest = self.bfs_h(loc, destination)
			min_dist_to_gs = len(self.map) + len(self.map[0])
			for gas_station in self.gas_stations:
				min_dist_to_gs = min(self.bfs_h(loc, gas_station), min_dist_to_gs)

			if min_dist_to_gs + min(dist_to_dest, min_dist_to_gs) > max_fuel:
				self.unsafe_pass.append((passenger, min(dist_to_dest, min_dist_to_gs)))

		self.total_cap = sum([taxi_data["capacity"] for _, taxi_data in self.taxis.items()])

		# create state0
		taxi_data = {taxi_name: [taxi_dict["location"], taxi_dict["fuel"], []] for taxi_name, taxi_dict in
		             self.taxis.items()}
		passenger_tuple = {pass_name: "waiting" for pass_name, pass_dict in
		                   self.passengers.items()}
		state0 = {"taxis": taxi_data, "passengers": passenger_tuple}
		state0 = json.dumps(state0, sort_keys=True)
		search.Problem.__init__(self, state0)

	def actions(self, state):
		"""Returns all the actions that can be executed in the given
		state. The result should be a tuple (or other iterable) of actions
		as defined in the problem description file"""
		state = json.loads(state)

		for passenger, pass_dict in self.passengers.items():
			if self.map[pass_dict["location"][0]][pass_dict["location"][1]] == "I" or \
					self.map[pass_dict["destination"][0]][pass_dict["destination"][1]] == "I":
				#print(f"{passenger} is the culprit")
				return []

		all_actions_list = []
		for taxi, taxi_data in state['taxis'].items():
			actions_list = []
			# wait
			actions_list.append(("wait", taxi))
			# refuel
			if self.map[taxi_data[0][0]][taxi_data[0][1]] == 'G':
				actions_list.append(("refuel", taxi))
			# dropoff
			for passenger in taxi_data[2]:
				if tuple(taxi_data[0]) == self.passengers[passenger]["destination"]:
					actions_list.append(("drop off", taxi, passenger))
			# pickup
			if len(taxi_data[2]) < self.taxis[taxi]["capacity"]:
				for passenger, pass_taxi in state["passengers"].items():
					if pass_taxi == 'waiting' and self.passengers[passenger]['location'] == tuple(taxi_data[0]):
						actions_list.append(("pick up", taxi, passenger))
			# move
			if taxi_data[1] > 0:
				# up
				if taxi_data[0][0] > 0 and self.map[taxi_data[0][0] - 1][taxi_data[0][1]] != 'I':
					actions_list.append(("move", taxi, (taxi_data[0][0] - 1, taxi_data[0][1])))

				# left
				if taxi_data[0][1] > 0 and self.map[taxi_data[0][0]][taxi_data[0][1] - 1] != 'I':
					actions_list.append(("move", taxi, (taxi_data[0][0], taxi_data[0][1] - 1)))

				# down
				if taxi_data[0][0] < len(self.map) - 1 and self.map[taxi_data[0][0] + 1][taxi_data[0][1]] != 'I':
					actions_list.append(("move", taxi, (taxi_data[0][0] + 1, taxi_data[0][1])))

				# right
				if taxi_data[0][1] < len(self.map[0]) - 1 and self.map[taxi_data[0][0]][taxi_data[0][1] + 1] != 'I':
					actions_list.append(("move", taxi, (taxi_data[0][0], taxi_data[0][1] + 1)))

			all_actions_list.append(actions_list)

		legal_actions = []
		for action in itertools.product(*all_actions_list):
			legal = True
			for taxi_action_1 in action:
				if taxi_action_1[0] == "move":
					for taxi_action_2 in action:
						if taxi_action_1 != taxi_action_2:
							if taxi_action_2[0] == "move":
								if taxi_action_1[2] == taxi_action_2[2]:
									legal = False
							else:
								if state["taxis"][taxi_action_2[1]][0] == taxi_action_1[2]:
									legal = False
			if legal:
				legal_actions.append(action)
		return legal_actions

	def result(self, state, action):
		"""Return the state that results from executing the given
		action in the given state. The action must be one of
		self.actions(state)."""
		state = json.loads(state)
		for taxi_action in action:
			if taxi_action[0] == "move":
				state["taxis"][taxi_action[1]][0] = taxi_action[2]
				state["taxis"][taxi_action[1]][1] -= 1
			elif taxi_action[0] == "pick up":
				state["taxis"][taxi_action[1]][2].append(taxi_action[2])
				state["passengers"][taxi_action[2]] = taxi_action[1]
			elif taxi_action[0] == "drop off":
				state["taxis"][taxi_action[1]][2].remove(taxi_action[2])
				state["passengers"][taxi_action[2]] = "arrived"
			elif taxi_action[0] == "refuel":
				state["taxis"][taxi_action[1]][1] = self.taxis[taxi_action[1]]["fuel"]
		return json.dumps(state, sort_keys=True)

	def goal_test(self, state):
		""" Given a state, checks if this is the goal state.
		Returns True if it is, False otherwise."""
		state = json.loads(state)
		for pass_name, pass_status in state["passengers"].items():
			if pass_status != "arrived":
				return False
		return True

	# almog - this is where you switch between the new and old huristic
	# (the new is 'all in one')
	def h(self, node):
		""" This is the heuristic. It gets a node (not a state,
		state can be accessed via node.state)
		and returns a goal distance estimate"""
		#return self.h_final(node)
		return self.h_all_in_one(node)

	def h_1(self, node):
		"""
		This is a simple heuristic
		"""
		state = json.loads(node.state)
		h_estimate = 0
		for pass_name, pass_status in state["passengers"].items():
			if pass_status == "waiting":
				h_estimate += 2
			elif pass_status != "arrived":
				h_estimate += 1
		return h_estimate / len(self.taxis.keys())

	def h_2(self, node):
		"""
		This is a slightly more sophisticated Manhattan heuristic
		"""
		state = json.loads(node.state)
		h_estimate = 0
		for pass_name, pass_status in state["passengers"].items():
			if pass_status == "waiting":
				loc = self.passengers[pass_name]["location"]
				dest = self.passengers[pass_name]["destination"]
				h_estimate += (abs(loc[0] - dest[0]) + abs(loc[1] - dest[1]))
			elif pass_status != "arrived":
				loc = state["taxis"][pass_status][0]
				dest = self.passengers[pass_name]["destination"]
				h_estimate += (abs(loc[0] - dest[0]) + abs(loc[1] - dest[1]))
		return h_estimate / len(self.taxis.keys())

	# not in use
	def h_2_upgrade(self, node):
		"""
		Accounting for pick up and drop off
		"""
		state = json.loads(node.state)
		h_estimate = 0
		for pass_name, pass_status in state["passengers"].items():
			dest = self.passengers[pass_name]["destination"]
			if pass_status == "waiting":
				loc = self.passengers[pass_name]["location"]
				h_estimate += (abs(loc[0] - dest[0]) + abs(loc[1] - dest[1])) + 2
			elif pass_status != "arrived":
				loc = state["taxis"][pass_status][0]
				h_estimate += (abs(loc[0] - dest[0]) + abs(loc[1] - dest[1])) + 1
		return h_estimate / len(self.taxis.keys())

	# not in use
	def h_almost_final(self, node):
		"""
		changing the denominator so it'll be admissable
		"""
		state = json.loads(node.state)
		h_estimate = 0
		for pass_name, pass_status in state["passengers"].items():
			dest = self.passengers[pass_name]["destination"]
			if pass_status == "waiting":
				loc = self.passengers[pass_name]["location"]
				h_estimate += (abs(loc[0] - dest[0]) + abs(loc[1] - dest[1])) + 2
			elif pass_status != "arrived":
				loc = state["taxis"][pass_status][0]
				h_estimate += (abs(loc[0] - dest[0]) + abs(loc[1] - dest[1])) + 1
		return h_estimate / min(len(self.passengers.keys()), self.total_cap)

	def h_final(self, node):
		"""
		changing the denominator so it'll be admissable
		"""
		state = json.loads(node.state)
		h_estimate = 0
		nearest_dict = {taxi: [] for taxi in self.taxis.keys()}
		for pass_name, pass_status in state["passengers"].items():
			dest = self.passengers[pass_name]["destination"]
			if pass_status == "waiting":
				loc = self.passengers[pass_name]["location"]
				h_estimate += self.bfs_h(loc, dest)
				nearest_taxi_dist = len(self.map) + len(self.map[0])
				for taxi, taxi_data in state["taxis"].items():
					dist = self.bfs_h(taxi_data[0], loc)
					nearest_taxis = []
					if dist < nearest_taxi_dist:
						nearest_taxis = [taxi]
						nearest_taxi_dist = dist
				for taxi in nearest_taxis:
					nearest_dict[taxi].append(nearest_taxi_dist)
			elif pass_status != "arrived":
				loc = state["taxis"][pass_status][0]
				h_estimate += self.bfs_h(loc, dest)
		h_basic = h_estimate / min(len(self.passengers.keys()), self.total_cap)
		# capacity because moving to pickup while moving to current destination
		h_nearest = sum([sum(distances) / (len(distances) + len(state["taxis"][taxi][2])) for taxi, distances in
		                 nearest_dict.items() if len(distances) > 0])

		return h_basic + self.h_1(node) + h_nearest

	# almog - you can experiment with the game over huristic if you want
	# return h_basic + self.h_1(node) + h_nearest + self.h_game_over(node)

	# not in use
	def h_pickup(self, node):
		state = json.loads(node.state)
		waiting_passengers = []
		for passenger, pass_status in state["passengers"].items():
			if pass_status == "waiting":
				waiting_passengers.append(self.passengers[passenger]["location"])
		if len(waiting_passengers) == 0:
			return 0
		min_dist = 9999
		for taxi_assignemnt in list(itertools.product(list(self.taxis.keys()), repeat=len(waiting_passengers))):
			taxi_pass = {taxi: [] for taxi in list(self.taxis.keys())}
			for pass_loc, taxi in zip(waiting_passengers, taxi_assignemnt):
				taxi_pass[taxi].append(pass_loc)
			total_dist = 0
			for taxi, passengers in taxi_pass.items():
				if len(passengers) == 0:
					continue
				taxi_loc = state["taxis"][taxi][0]
				taxi_cap = len(state["taxis"][taxi][2])
				taxi_fuel = state["taxis"][taxi][1]
				local_dist = 9999
				for pass_locs in [list(p) for p in itertools.permutations(passengers)]:
					dist = 0
					reg_dist = 0
					multi_tasking = 1
					prev_loc = taxi_loc
					for pass_loc in pass_locs:
						reg_dist += self.bfs_h(prev_loc, pass_loc)
						dist += self.bfs_h(prev_loc, pass_loc) / (multi_tasking + taxi_cap)
						multi_tasking += 1
						prev_loc = pass_loc
					if reg_dist > taxi_fuel:
						dist += 1
					local_dist = min(local_dist, dist)
				total_dist += local_dist
			taxis_in_use = len([taxi for taxi, passengers in taxi_pass.items() if len(passengers) > 0])
			total_dist = total_dist / taxis_in_use
			min_dist = min(min_dist, total_dist)
		return min_dist

	# almog - this was originally meant to print the hurisitic of each ingredient for anaylasis.
	# It's not updated to the new algorithm.
	def h_debugger(self, node):
		state = json.loads(node.state)
		h_estimate = 0
		pick_drop_count = 0
		nearest_dict = {taxi: [] for taxi in self.taxis.keys()}
		assign_dict = {taxi: [] for taxi in self.taxis.keys()}
		for pass_name, pass_status in state["passengers"].items():
			dest = self.passengers[pass_name]["destination"]
			if pass_status == "waiting":
				pick_drop_count += 2
				loc = self.passengers[pass_name]["location"]
				h_estimate += self.bfs_h(loc, dest)
				nearest_taxi_dist = len(self.map) + len(self.map[0])
				for taxi, taxi_data in state["taxis"].items():
					dist = self.bfs_h(taxi_data[0], loc)
					nearest_taxis = []
					if dist < nearest_taxi_dist:
						nearest_taxis = [taxi]
						nearest_taxi_dist = dist
				# elif dist == nearest_taxi_dist:
				#	nearest_taxis.append(taxi)
				for taxi in nearest_taxis:
					nearest_dict[taxi].append(nearest_taxi_dist)
					assign_dict[taxi].append(loc)
			elif pass_status != "arrived":
				pick_drop_count += 1
				loc = state["taxis"][pass_status][0]
				h_estimate += self.bfs_h(loc, dest)
		h_basic = h_estimate / min(len(self.passengers.keys()), self.total_cap)
		h_pick_drop = pick_drop_count / len(self.taxis.keys())
		# capacity because moving to pickup while moving to current destination
		h_nearest = sum([sum(distances) / (len(distances) + len(state["taxis"][taxi][2])) for taxi, distances in
		                 nearest_dict.items() if len(distances) > 0])

		# return h_basic, h_pick_drop, h_nearest, self.h_game_over(node)
		return h_basic, self.h_1(node), self.h_pickup(node), self.h_game_over(node)

	def h_all_in_one(self, node):
		state = json.loads(node.state)
		all_waiting_passengers = []
		for passenger, pass_status in state["passengers"].items():
			if pass_status == "waiting":
				all_waiting_passengers.append(passenger)

		# relaxing the problem by looking only into a maximum of 1 waiting passengers
		# keeps the computation from growing exponantially
		max_dist = self.h_all_in_one_details(state, waiting_passengers=[], calc_fuel_detour=True)
		for passenger in all_waiting_passengers:
			max_dist = max(max_dist,
			               self.h_all_in_one_details(state, waiting_passengers=[passenger], calc_fuel_detour=True))
		return max_dist

	# the main algorithm, calculates the exact distance of every possibility.
	# the calc_fuel_detour could be switched on/off to try to account for the distance we need to prolong the drive
	# in order to get to a fuel station.
	def h_all_in_one_details(self, state, waiting_passengers, calc_fuel_detour, alternative=True):
		min_dist = 9999
		# note product still has one empty valued if repeat = 0
		prod_iter = itertools.product(list(self.taxis.keys()), repeat=len(waiting_passengers))
		# each option of which taxi will take our passenger.
		for taxi_assignment in prod_iter:
			taxi_assignment = list(taxi_assignment)
			taxi_pass = {taxi: [] for taxi in list(self.taxis.keys())}
			for passenger, taxi in zip(waiting_passengers, taxi_assignment):
				taxi_pass[taxi].append((passenger, "waiting"))
				taxi_pass[taxi].append((passenger, taxi))
			total_dist = 0
			# adding the passengers that are already on the taxi
			for taxi, taxis_passengers in taxi_pass.items():
				current_pass = [(passenger, taxi) for passenger in state["taxis"][taxi][2]]
				taxis_passengers += current_pass
				if len(taxis_passengers) == 0:
					continue
				taxi_loc = state["taxis"][taxi][0]
				taxi_cap = len(state["taxis"][taxi][2])
				taxi_max_cap = self.taxis[taxi]["capacity"]
				taxi_max_fuel = self.taxis[taxi]["fuel"]
				taxi_fuel = state["taxis"][taxi][1]
				local_dist = 9999
				perm_iterator = itertools.permutations(taxis_passengers)
				for passengers in perm_iterator:
					passengers = list(passengers)
					# checking if possible
					possible = True

					distinct_pass = list(set([passenger for passenger, status in passengers]))
					index_func = lambda elem: passengers.index(elem) if elem in passengers else -1
					for passenger in distinct_pass:
						# must pickup a passenger before dropping him off
						if -1 < index_func((passenger, taxi)) < index_func((passenger, "waiting")):
							possible = False

					start_cap = taxi_cap
					# can't go over the maximal taxi's capacity
					for passenger, status in passengers:
						start_cap += 1 if status == "waiting" else -1
						if start_cap > taxi_max_cap:
							possible = False
					if not possible:
						continue

					dist = 0
					modifier = 0
					prev_loc = taxi_loc
					locs_along_way = [taxi_loc]
					for passenger, status in passengers:
						if status == "waiting":
							goal_location = self.passengers[passenger]["location"]
						else:
							goal_location = self.passengers[passenger]["destination"]
						locs_along_way.append(goal_location)
						dist += self.bfs_dict[(tuple(prev_loc), tuple(goal_location))]
						# 1 because of pickup/drop off
						modifier += 1
						prev_loc = goal_location
					# turns needed to refuel
					fuel_needed = 1 + (dist - taxi_fuel - 1) // taxi_max_fuel

					if calc_fuel_detour:
						# calculating fuel detour
						if fuel_needed > 0:
							total_detours = 0
							for i in range(len(locs_along_way) - 1):
								A = tuple(locs_along_way[i])
								B = tuple(locs_along_way[i + 1])
								if self.bfs_dict[(A, B)] > taxi_max_fuel:
									fuel_detour1 = math.inf
									fuel_detour2 = math.inf
									for gs in self.gas_stations:
										gs = tuple(gs)
										if self.bfs_dict[(A, gs)] < taxi_max_fuel:
											detour = self.bfs_dict[(A, gs)] + self.bfs_dict[(gs, B)] - self.bfs_dict[(A, B)]
											fuel_detour1 = min(fuel_detour1, detour)
										if self.bfs_dict[(gs, B)] < taxi_max_fuel:
											detour = self.bfs_dict[(A, gs)] + self.bfs_dict[(gs, B)] - self.bfs_dict[(A, B)]
											fuel_detour2 = min(fuel_detour2, detour)
									total_detours += max(fuel_detour1, fuel_detour2)
							detour_pred1 = total_detours

							dist_without_fuel = 0
							fuel_detour = 9999
							for i in range(len(locs_along_way) - 1):
								A = tuple(locs_along_way[i])
								B = tuple(locs_along_way[i + 1])
								for gs in self.gas_stations:
									gs = tuple(gs)
									detour = self.bfs_dict[(A, gs)] + self.bfs_dict[(gs, B)] - self.bfs_dict[(A, B)]
									if dist_without_fuel + self.bfs_h(A, gs) <= taxi_fuel:
										fuel_detour = min(fuel_detour, detour)
								dist_without_fuel += self.bfs_dict[(A, B)]
								if dist_without_fuel > taxi_fuel:
									break
							detour_pred2 = fuel_detour
							dist += max(detour_pred1, detour_pred2)

					dist += fuel_needed

					# accounting for pickup/dropoff
					dist += modifier
					# the min-max-min looks a bit confusing, and i admit i don't remember it's logic at the top of my head
					# sorry
					local_dist = min(local_dist, dist)
				total_dist = max(total_dist, local_dist)
			min_dist = min(min_dist, total_dist)
		return min_dist

	# not in use
	def h_fuel(self, node):
		state = json.loads(node.state)
		total_missing_fuel = []
		for taxi, taxi_data in state["taxis"].items():
			fuel_ratio = taxi_data[1] / self.taxis[taxi]["fuel"]
			total_missing_fuel.append(1 - fuel_ratio)
		return sum(total_missing_fuel)

	# not in use
	def h_cap_taxi(self, node):
		state = json.loads(node.state)
		sum = 0
		for taxi, taxi_data in state["taxis"].items():
			sum += len(taxi_data[2]) / self.taxis[taxi]["capacity"]
		return sum

	# not in use
	def h_game_over(self, node):
		state = json.loads(node.state)
		for passenger, dist_to_live in self.unsafe_pass:
			if state["passengers"][passenger] == "waiting":
				pass_loc = self.passengers[passenger]["location"]
				game_over = True
				for _, taxi_data in state["taxis"].items():
					taxi_dist = self.bfs_h(taxi_data[0], pass_loc)
					taxi_cur_fuel = taxi_data[1]
					if taxi_dist + dist_to_live <= taxi_cur_fuel:
						game_over = False
				if game_over:
					return 999999
		return 0

	def bfs_h(self, source, destination):
		if tuple(source) == tuple(destination):
			return 0
		queue = [(source, 0)]
		was_there = []
		while len(queue) > 0:
			node = queue.pop(0)
			was_there.append(node[0])
			if node[0] == destination:
				return node[1]
			else:
				loc = tuple(node[0])
				if loc[0] + 1 <= len(self.map) - 1 and self.map[loc[0] + 1][loc[1]] != 'I':
					if (loc[0] + 1, loc[1]) not in was_there:
						queue.append(((loc[0] + 1, loc[1]), node[1] + 1))
				if loc[0] - 1 >= 0 and self.map[loc[0] - 1][loc[1]] != 'I':
					if (loc[0] - 1, loc[1]) not in was_there:
						queue.append(((loc[0] - 1, loc[1]), node[1] + 1))
				if loc[1] + 1 <= len(self.map[0]) - 1 and self.map[loc[0]][loc[1] + 1] != 'I':
					if (loc[0], loc[1] + 1) not in was_there:
						queue.append(((loc[0], loc[1] + 1), node[1] + 1))
				if loc[1] - 1 >= 0 and self.map[loc[0]][loc[1] - 1] != 'I':
					if (loc[0], loc[1] - 1) not in was_there:
						queue.append(((loc[0], loc[1] - 1), node[1] + 1))
		return 0

	def build_bfs_dict(self):
		bfs_dict = {((i, j), (k, l)): math.inf for i in range(len(self.map)) for j in range(len(self.map[0]))
		            for k in range(len(self.map)) for l in range(len(self.map[0]))}
		for i in range(len(self.map)):
			for j in range(len(self.map[0])):
				if self.map[i][j] != "I":
					queue = [((i,j), 0)]
					was_there = []
					while len(queue) > 0:
						node = queue.pop(0)
						if node[0] in was_there:
							continue
						else:
							was_there.append(node[0])
							loc = tuple(node[0])
							bfs_dict[((i, j), loc)] = node[1]
							if loc[0] + 1 <= len(self.map) - 1 and self.map[loc[0] + 1][loc[1]] != 'I':
								if (loc[0] + 1, loc[1]) not in was_there:
									queue.append(((loc[0] + 1, loc[1]), node[1] + 1))
							if loc[0] - 1 >= 0 and self.map[loc[0] - 1][loc[1]] != 'I':
								if (loc[0] - 1, loc[1]) not in was_there:
									queue.append(((loc[0] - 1, loc[1]), node[1] + 1))
							if loc[1] + 1 <= len(self.map[0]) - 1 and self.map[loc[0]][loc[1] + 1] != 'I':
								if (loc[0], loc[1] + 1) not in was_there:
									queue.append(((loc[0], loc[1] + 1), node[1] + 1))
							if loc[1] - 1 >= 0 and self.map[loc[0]][loc[1] - 1] != 'I':
								if (loc[0], loc[1] - 1) not in was_there:
									queue.append(((loc[0], loc[1] - 1), node[1] + 1))
		self.bfs_dict = bfs_dict


def create_taxi_problem(game):
	return TaxiProblem(game)
