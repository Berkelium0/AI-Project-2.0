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

import numpy as np
import requests
import time

from numpy import unravel_index

MAX_STEP_COUNT = 5
MAX_TIME = 0

# Create Action Point Arrays
# f = open("arrays.json", "w")
# f.write("{")
# array = [0]
# for i in range(1, 7):
#     array = array * 4
#     f.write(f'"{i}" : "{array}",')
#     array = [array]
# f.write("}")
# f.close()
a = open("arrays.json", "r")
action_point_arrays = json.load(a)


def agent_function(request_dict):
    global MAX_TIME
    print('I got the following request:')
    print(request_dict)

    MAX_TIME = request_dict['max-time']

    temp_map = request_dict['map'].split('\n')
    cave_map = []
    for x in temp_map:
        cave_map.append([cell for cell in x])

    action_list = [
        ['nn', 'ne', 'ns', 'nw'],
        ['en', 'ee', 'es', 'ew'],
        ['sn', 'se', 'ss', 'sw'],
        ['wn', 'we', 'ws', 'ww']
    ]

    action_list2 = [
        ['n', 'e', 's', 'w'],
        ['n', 'e', 's', 'w'],
        ['n', 'e', 's', 'w'],
        ['n', 'e', 's', 'w'],
        ['n', 'e', 's', 'w'],
        ['n', 'e', 's', 'w'],
    ]

    boots = False
    current_cell = request_dict['observations']["current-cell"]
    if current_cell == "B":
        current_cell = "C"
    if current_cell == "S":
        boots = True
    num_of_cells = 0

    # TODO: dynamic step count -> try from 1 action to MAX_STEP_COUNT, compare times use the shortest timed one.

    steps_and_results = {
       # "1": [],
        "2": [],
        # "3": [],
        # "4": [],
        # "5": [],
        # "6": [],
    }

    action_commands = []
    eta = 0
    for key in steps_and_results:
        key = int(key)
        action_point = action_point_arrays[f"{key}"]

        for x, line in enumerate(cave_map, start=0):
            for y, cell in enumerate(line, start=0):
                if cell == current_cell:
                    action_point = check(x, y, cave_map, action_point, key)
                    num_of_cells += 1

        actions = []
        directions = ["north", "east", "south", "west"]
        a = np.array(action_point)
        action_indexes = list(unravel_index(a.argmax(), a.shape))
        for i in range(key):
            actions.append(directions[action_indexes[i]])

        p = 1 / num_of_cells
        for x, line in enumerate(cave_map, start=0):
            for y, cell in enumerate(line, start=0):
                if cell == current_cell:
                    eta += p * expected_time(x, y, cave_map, actions, boots)
        print(eta)

        action_commands = [f"GO {action}" for action in actions]
        steps_and_results[f"{key}"] = [action_commands, eta]

    min_eta = float('inf')
    min_key = None

    for key, values in steps_and_results.items():
        act, eta = values
        if eta < min_eta:
            min_eta = eta
            min_key = key

    response_dict =  {"actions": steps_and_results[min_key][0], "expected-time": steps_and_results[min_key][1]}
    return response_dict


# TODO: Need to make expected time more modular for steps count higher than 1

def check(x, y, cm, ap, key, step=0):
    # hit_flag = False
    if step > key:
        return ap
    if x != 0:
        if (cm[x - 1][y]) == "W":
            ap[0] = [x + 1 for x in ap[0]]
        else:
            temp_ap = check(x - 1, y, cm, ap, key, step + 1)
            if temp_ap != ap:
                ap[0][0] += 0.5
    if y != 4:
        if (cm[x][y + 1]) == "W":
            ap[1] = [x + 1 for x in ap[1]]
        else:
            temp_ap = check(x, y + 1, cm, ap, key, step + 1)
            if temp_ap != ap:
                ap[1][1] += 0.5
    if x != 4:
        if (cm[x + 1][y]) == "W":
            ap[2] = [x + 1 for x in ap[2]]
        else:
            temp_ap = check(x + 1, y, cm, ap, key, step + 1)
            if temp_ap != ap:
                ap[2][2] += 0.5
    if y != 0:
        if (cm[x][y - 1]) == "W":
            ap[3] = [x + 1 for x in ap[3]]
        else:
            temp_ap = check(x, y - 1, cm, ap, key, step + 1)
            if temp_ap != ap:
                ap[3][3] += 0.5
    return ap


# TODO: Expected time -> Wear boots, check if next tile W or S take off boots depending on that
# TODO: Need to make expected time more modular for steps count higher than 1
def expected_time(x, y, cm, act, boots, step=0, total=0):
    global MAX_TIME
    if cm[x][y] == "S":
        in_swamp = True
    else:
        in_swamp = False

    cant_move = False
    if act[step] == "north":
        if x != 0:
            if (cm[x - 1][y]) == "W":
                if in_swamp:
                    if step == 0:
                        total += 1
                    else:
                        total += 2
                else:
                    if step == 0:
                        total += 0.5
                    else:
                        total += 1
            elif step == 0:
                if in_swamp:
                    total += 1
                else:
                    total += 0.5

                total = expected_time(x - 1, y, cm, act, boots, step + 1, total)
            else:
                total = MAX_TIME
        else:
            cant_move = True
    elif act[step] == "east":
        if y != 4:
            if (cm[x][y + 1]) == "W":
                if in_swamp:
                    if step == 0:
                        total += 1
                    else:
                        total += 2
                else:
                    if step == 0:
                        total += 0.5
                    else:
                        total += 1
            elif step == 0:
                if in_swamp:
                    total += 1
                else:
                    total += 0.5
                total = expected_time(x, y + 1, cm, act, boots, step + 1, total)
            else:
                total = MAX_TIME
        else:
            cant_move = True
    elif act[step] == "south":
        if x != 4:
            if (cm[x + 1][y]) == "W":
                if in_swamp:
                    if step == 0:
                        total += 1
                    else:
                        total += 2
                else:
                    if step == 0:
                        total += 0.5
                    else:
                        total += 1
            elif step == 0:
                if in_swamp:
                    total += 1
                else:
                    total += 0.5
                total = expected_time(x + 1, y, cm, act, boots, step + 1, total)
            else:
                total = MAX_TIME
        else:
            cant_move = True
    elif act[step] == "west":
        if y != 0:
            if (cm[x][y - 1]) == "W":
                if in_swamp:
                    if step == 0:
                        total += 1
                    else:
                        total += 2
                else:
                    if step == 0:
                        total += 0.5
                    else:
                        total += 1
            elif step == 0:
                if in_swamp:
                    total += 1
                else:
                    total += 0.5
                total = expected_time(x, y - 1, cm, act, boots, step + 1, total)
            else:
                total = MAX_TIME
        else:
            cant_move = True
    if cant_move:
        return MAX_TIME
    elif total > MAX_TIME:
        return MAX_TIME
    else:
        return total


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

a.close()
