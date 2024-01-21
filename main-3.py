"""
    To use this implementation, you simply have to implement `agent_function` such that it returns a legal action.
    You can then let your agent compete on the server by calling
        python3 client_simple.py path/to/your/config.json

    The script will keep running forever.
    You can interrupt it at any time.
    The server will remember the actions you have sent.

    Note:
        By default the client bundles multiple requests for efficiency.
        This can complicate debugging.
        You can disable it by setting `single_request=True` in the last line.
"""
import itertools
import json
import logging

import requests
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import heapq

MAX_TIME = 0


class Node:
    def __init__(self, row, col, cell_type):
        self.row = row
        self.col = col
        self.cell_type = cell_type
        self.wearing_boots = False
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
        self.first_step = False  # flag to track if this is the first step

    def update_first_step(self):
        if not self.first_step:
            self.first_step = True
        else:
            self.first_step = False

    def __lt__(self, other):
        # Implement the less-than comparison for heapq
        return self.f < other.f


def heuristic(node, goal):
    # Implementing Manhattan distance heuristic
    return abs(node.row - goal.row) + abs(node.col - goal.col)


def get_neighbors(node, node_map):
    neighbors = []
    rows, cols = len(node_map), len(node_map[0])

    # Define possible movements (up, down, left, right)
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
        # Additional cost for moving within a swamp
        cost += 2

    if node1.first_step:
        # Adjust cost based on wumpus starting location
        cost = cost / 2
        node1.update_first_step()

    if node1.cell_type == 'S' and node2.cell_type != 'S':
        # Additional cost for moving from a swamp to a non-swamp
        cost += 1
    elif node1.cell_type != 'S' and node2.cell_type == 'S':
        # Additional cost for moving from a non-swamp to a swamp
        cost += 1

    return cost


def reconstruct_path(start, goal):
    path = []
    current = goal

    while current and current != start:
        path.insert(0, (current.row, current.col))
        current = current.parent

    if not path:
        # No valid path found
        return None

    return path



def astar(start, goal, node_map, max_cost):
    start.update_first_step()
    open_set = [(start.f, start)]
    closed_set = set()

    while open_set:
        current_f, current_node = heapq.heappop(open_set)
        closed_set.add(current_node)

        if current_node == goal:
            return reconstruct_path(start, goal)

        if current_node.g > max_cost:
            # Stop if the cost exceeds the maximum allowed
            return None

        for neighbor in get_neighbors(current_node, node_map):
            if neighbor in closed_set:
                continue

            tentative_g = current_node.g + calculate_cost(current_node, neighbor)
            if neighbor.g == 0 or tentative_g > neighbor.g:
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


def agent_function(request_dict):
    global MAX_TIME
    MAX_TIME = request_dict["max-time"]

    temp_map = request_dict['map'].split('\n')
    map_representation = []

    for x in temp_map:
        map_representation.append([cell for cell in x])

    node_map = initialize_node_map(map_representation)

    # Identify entrance and exit points
    entrances = []
    exits = []

    for row in node_map:
        for node in row:
            if node.cell_type == request_dict['observations']['current-cell']:
                entrances.append(node)
            elif node.cell_type == 'W':
                exits.append(node)

    # Find the nearest exit for each entrance
    entrance_exit_pairs = []
    for entrance in entrances:
        nearest_exit = min(exits, key=lambda exit_point: heuristic(entrance, exit_point))
        entrance_exit_pairs.append((entrance, nearest_exit))
    total_costs = {}
    for entrance, exit_point in entrance_exit_pairs:
        path = astar(entrance, exit_point, node_map, MAX_TIME)
        if path:
            # Update the total cost for this entrance-exit pair
            total_cost = calculate_total_cost(path, entrance, exit_point, node_map)
            total_costs[tuple(path)] = total_cost  # Convert path to tuple

    # Choose the best moveset based on total costs
    best_moveset_path = min(total_costs, key=total_costs.get, default=None)

    if best_moveset_path:
        actions = ["GO " + get_direction(best_moveset_path[i], best_moveset_path[i + 1]) for i in
                   range(len(best_moveset_path) - 1)]
        expected_time = total_costs[best_moveset_path]  # Adjust this based on your actual calculation
    else:
        # No valid moveset found
        actions = []
        expected_time = 0

    response_dict = {
        "actions": actions,
        "expected-time": expected_time
    }
    print(response_dict)
    return response_dict


def get_direction(current_pos, next_pos):
    # Helper function to determine the direction from current_pos to next_pos
    if current_pos[0] < next_pos[0]:
        return "south"
    elif current_pos[0] > next_pos[0]:
        return "north"
    elif current_pos[1] < next_pos[1]:
        return "east"
    elif current_pos[1] > next_pos[1]:
        return "west"


def calculate_total_cost(path, entrance, exit_point, node_map):
    # Calculate the total cost for the given path
    total_cost = 0
    start_location = (entrance.row, entrance.col)

    for i in range(len(path) - 1):
        current_node = node_map[path[i][0]][path[i][1]]
        next_node = node_map[path[i + 1][0]][path[i + 1][1]]
        total_cost += calculate_cost(current_node, next_node)

    return total_cost


def visualize_map(node_map):
    # Create a colormap for different cell types
    cmap = mcolors.ListedColormap(['green', 'brown', 'darkgreen', 'blue', 'red'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot each cell with its color based on cell type
    for row in node_map:
        for node in row:
            color = cmap(norm(bounds.index(['M', 'B', 'C', 'S', 'W'].index(node.cell_type))))
            rect = plt.Rectangle((node.col, node.row), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)

            # Annotate cell with cell type
            ax.text(node.col + 0.5, node.row + 0.5, node.cell_type, ha='center', va='center', color='white')

    # Set axis limits
    ax.set_xlim(0, len(node_map[0]))
    ax.set_ylim(0, len(node_map))

    # Invert y-axis to match the grid representation
    ax.invert_yaxis()

    plt.show()


def run(action_function, single_request=False):
    logger = logging.getLogger(__name__)

    with open("env-2.json", 'r') as fp:
        config = json.load(fp)

    logger.info(f'Running agent {config["agent"]} on environment {config["env"]}')
    logger.info(f'Hint: You can see how your agent performs at {config["url"]}agent/{config["env"]}/{config["agent"]}')

    actions = []
    for request_number in itertools.count():
        logger.debug(f'Iteration {request_number} (sending {len(actions)} actions)')
        # send request
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
                time.sleep(1)  # wait a moment to avoid overloading the server and then try again
            # get actions for next request
            actions = []
            for action_request in action_requests:
                actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
        elif response.status_code == 503:
            logger.warning('Server is busy - retrying in 3 seconds')
            time.sleep(3)  # server is busy - wait a moment and then try again
        else:
            # other errors (e.g. authentication problems) do not benefit from a retry
            logger.error(f'Status code {response.status_code}. Stopping.')
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    run(agent_function, single_request=False)
