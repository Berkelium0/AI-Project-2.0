import itertools
import json
import logging
import math

import requests
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq

MAX_TIME = 2


class Node:
    def __init__(self, row, col, cell_type):
        self.row = row
        self.col = col
        self.cell_type = cell_type
        self.wearing_boots = False
        self.g = float('inf')
        self.h = 0
        self.f = 0
        self.parent = None
        self.first_step = False  # flag to track if this is the first step

    def is_first_step(self):
        self.first_step = True

    def is_not_first_step(self):
        self.first_step = False

    def __lt__(self, other):
        return self.f < other.f


def heuristic(node, goal):
    return abs(node.row - goal.row) + abs(node.col - goal.col)


def get_neighbors(node, node_map):
    neighbors = []
    rows, cols = len(node_map), len(node_map[0])
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dr, dc in movements:
        new_row, new_col = node.row + dr, node.col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbors.append(node_map[new_row][new_col])
    return neighbors


def calculate_cost(node1, node2):
    cost = 0
    if node1.cell_type != 'S':
        cost += 1
    else:
        cost += 2

    if node1.first_step:
        # Adjust cost based on wumpus starting location
        cost = cost / 2

    if node1.cell_type == 'S' and node2.cell_type != 'S' and node2.cell_type != 'W':
        cost += 1
    elif node1.cell_type != 'S' and node2.cell_type == 'S':
        cost += 1
    return cost


def reconstruct_path(start, goal):
    path = []
    current = goal

    # If the current node is the start node, return a path containing only the start and end nodes
    if current == start:
        return [(start.row, start.col), (goal.row, goal.col)]

    # Traverse back from the goal node to the start node
    while current and current != start:
        path.insert(0, (current.row, current.col))  # Insert current node to the beginning of the path
        current = current.parent

        # Check for infinite loop: if the current node's parent is already in the path, break the loop
        if current in path:
            # print("infinite loop")
            raise ValueError("Infinite loop detected in path reconstruction!")

    # Add the start node to the path
    path.insert(0, (start.row, start.col))
    return path


def astar(start, goal, node_map, max_cost):
    open_set = [(start.f, start)]
    closed_set = set()
    start.g = 0  # Update g for the start node

    while open_set:

        if len(closed_set) == 0:
            start.is_first_step()
        else:
            start.is_not_first_step()

        current_f, current_node = heapq.heappop(open_set)
        closed_set.add(current_node)

        if current_node == goal:
            return reconstruct_path(start, goal)
        if not math.isfinite(current_node.g) or current_node.g > max_cost:
            return None
        for neighbor in get_neighbors(current_node, node_map):
            if neighbor in closed_set:
                continue
            tentative_g = current_node.g + calculate_cost(current_node, neighbor)
            if neighbor.g != math.inf or tentative_g < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor, goal)
                neighbor.f = neighbor.g + neighbor.h
                if neighbor not in open_set:
                    heapq.heappush(open_set, (neighbor.f, neighbor))
    return None


def initialize_node_map(map_representation):
    node_map = []
    for row, row_data in enumerate(map_representation):
        node_row = []
        for col, cell_type in enumerate(row_data):
            node = Node(row, col, cell_type)
            node_row.append(node)
        node_map.append(node_row)
    return node_map


def calculate_next_node(current_node, direction, node_map):
    row, col = current_node.row, current_node.col
    # Update the row and column based on the direction
    if direction == "north":
        row -= 1
    elif direction == "south":
        row += 1
    elif direction == "east":
        col += 1
    elif direction == "west":
        col -= 1

    if 0 <= row < len(node_map) and 0 <= col < len(node_map[0]):
        # Return the node at the calculated position
        return node_map[row][col]
    # Get the node at the updated row and column from the node map
    else:
        # Handle the case where the next position is out of bounds
        return None


def test_paths(entrance_points, directions_list, node_map, max_cost):
    total_costs = []
    for entrance in entrance_points:
        current_node = entrance
        cost = 0
        for i, direction in enumerate(directions_list):
            if i == 0:
                current_node.is_first_step()
            else:
                current_node.is_not_first_step()
            # Calculate the next node based on the direction
            next_node = calculate_next_node(current_node, direction, node_map)

            # Check if the next node is a 'W' tile
            if next_node is None:
                cost = max_cost
            elif next_node.cell_type == 'W':
                # Add the cost until the 'W' tile and break the loop
                cost += calculate_cost(current_node, next_node)
                break
            else:
                # Update the cost and move to the next node
                cost += calculate_cost(current_node, next_node)
                current_node = next_node

        # If the loop completes without encountering a 'W' tile, set cost to max_time
        else:
            cost = max_cost

        # Store the cost for the current path
        total_costs.append(cost)

    # Calculate the average cost for each set of directions
    total_costs = capped_list = [min(value, max_cost) for value in total_costs]
    # print("total_costs", total_costs)
    average_costs = sum(total_costs) / len(total_costs)
    return average_costs


def agent_function(request_dict):
    global MAX_TIME
    MAX_TIME = request_dict["max-time"]

    temp_map = request_dict['map'].split('\n')
    map_representation = []

    # print(request_dict)

    for x in temp_map:
        map_representation.append([cell for cell in x])

    node_map = initialize_node_map(map_representation)

    entrances = []
    exits = []

    for row in node_map:
        for node in row:
            if node.cell_type == request_dict['observations']['current-cell']:
                entrances.append(node)
            elif node.cell_type == 'W':
                exits.append(node)

    entrance_exit_pairs = []
    for entrance in entrances:
        nearest_exit = min(exits, key=lambda exit_point: heuristic(entrance, exit_point))
        entrance_exit_pairs.append((entrance, nearest_exit))

    paths = []
    total_costs = {}
    for entrance, exit_point in entrance_exit_pairs:
        path = astar(entrance, exit_point, node_map, MAX_TIME)
        if path:
            directions = []
            for i in range(len(path) - 1):
                directions.append(get_direction(path[i], path[i + 1]))
            paths.append(directions)
            costs = test_paths(entrances, directions, node_map, MAX_TIME)
            total_costs[tuple(directions)] = costs

    # print("total_costs",total_costs)
    best_moveset_path = min(total_costs, key=total_costs.get, default=None)
    # print("bmp",best_moveset_path)
    if best_moveset_path:
        actions = ["GO " + best_moveset_path[i] for i in range(len(best_moveset_path))]
    else:
        actions = []
    expected_time = total_costs.get(best_moveset_path, MAX_TIME)
    if expected_time > MAX_TIME:
        expected_time = MAX_TIME

    response_dict = {
        "actions": actions,
        "expected-time": expected_time
    }
    # print(response_dict)
    return response_dict


def get_direction(current_pos, next_pos):
    current_row, current_col = current_pos
    next_row, next_col = next_pos
    if current_row < next_row:
        return "south"
    elif current_row > next_row:
        return "north"
    elif current_col < next_col:
        return "east"
    elif current_col > next_col:
        return "west"


def calculate_total_cost(path, entrance, exit_point, node_map):
    total_cost = 0
    start_location = (entrance.row, entrance.col)
    if len(path) == 2:
        total_cost = calculate_cost(entrance, exit_point)
    elif len(path) > 1:
        for i in range(len(path) - 1):
            current_node = node_map[path[i][0]][path[i][1]]
            next_node = node_map[path[i + 1][0]][path[i + 1][1]]
            total_cost += calculate_cost(current_node, next_node)
    return total_cost


def run(action_function, single_request=False):
    logger = logging.getLogger(__name__)

    with open("env-3.json", 'r') as fp:
        config = json.load(fp)

    logger.info(f'Running agent {config["agent"]} on environment {config["env"]}')
    logger.info(f'Hint: You can see how your agent performs at {config["url"]}agent/{config["env"]}/{config["agent"]}')

    actions = []
    for request_number in itertools.count():
        logger.debug(f'Iteration {request_number} (sending {len(actions)} actions)')
        response = requests.put(f'{config["url"]}/act/{config["env"]}', json={
            'agent': config['agent'],
            'pwd': config['pwd'],
            'actions': actions,
            'single_request': single_request,
        })
        if response.status_code == 200:
            response_json = response.json()
            for error in response_json['errors']:
                logger.error(f'Error message from server: {error}')
            for message in response_json['messages']:
                logger.info(f'Message from server: {message}')
            action_requests = response_json['action-requests']
            if not action_requests:
                logger.info('The server has no new action requests - waiting for 1 second.')
                time.sleep(1)
            actions = []
            for action_request in action_requests:
                actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
        elif response.status_code == 503:
            logger.warning('Server is busy - retrying in 3 seconds')
            time.sleep(3)
        else:
            logger.error(f'Status code {response.status_code}. Stopping.')
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run(agent_function, single_request=False)
