# Repository for ws2324.2.0.a/ry68fesa-2

**Topic:** WS2324 Assignment 2.0.A (Warm-up): Find the Wumpus Cave

### How to Run and Dependencies

Before running this code, open the `main.py` file and change the `ENV = "env-2.json"` variable to the environment you
want to test.

This code requires the following Python libraries:

- **math:** For mathematical operations.
- **heapq:** For implementing the priority queue for A* algorithm.

### My Initial Plan

My initial plan was doing a DFS search from every starting point until they reach a 'W' tile and putting in a 'vote' for
the directions in that path, kept in a 4x4 array. This approach worked well while the maximum cost was 2.0 and the paths
were at most two tiles long. However, at env-3, the maximum cost was raised to 5 and this approach immediately showed
how costly it is. Even holding the votes required a 4x4x4x4x4 grid (!) so I decided to rewrite my code from the ground
up.

### A* Search Algorithm

After some consideration, I decided to use the A* algorithm for this problem, as it takes the cost of travel into
account and instead of searching through every single possible path, it uses heuristics to reach the goal more
efficiently.

### How The Script Works

Because that our map is made out of tiles which have their own differing attributes, I created a `Node` class to turn
every tile into a node that stores each of their position, cell type, their initial g f and h values, and parent node,
for path reconstruction.

Afterward, I created `the initialize_node_map` to initialize a 2D grid of nodes based on the provided
map representation.

Next, I put every entrance and exit point into two lists, and paired every entrance with the closest exit to
them. To calculate the nearest exit point, I used the `heuristic` function, which returns the absolute distance between
the nodes.

Now that I have my entrance and exit points, I sent each pair to the `astar` function. In this, the function takes the
entrance point and goes through every neighbor of it using the `get_neighbors` function. It calculates the true cost of
moving through that tile with the help of the `calculate_cost` function based on the types of the cells and weather if
the current tile is the first tile or not. The `astar` function runs until it reaches the exit tile or the cost goes
higher than the `MAX_COST` variable. If it reaches an exit tile, it calls the `reconstruct_path` function. This function
returns a list of the found optimal path, in coordinate form.

Next, if a path is found, it gets turned into a direction based list from a coordinate based one. This is done with the
help of `get_direction` function. It determines the direction from the current position to the next position for each
step and returns the string result. So for example,

```
[(1, 1), (1, 2), (2, 2), (2, 1)]
```

turns into,

```
["east", "south", "west" ]
``` 

and gets stored in a dictionary as the key.

The cost of this path is then calculated and is saved in the dictionary by using the `test_paths`function. This function
takes the path and every entrance point as parameters, calculates the total cost for each path, and returns the average
cost. It uses the helper function `calculate_next_node`, which calculates the next node based on the current node and
movement direction. This is necessary as we have stored the path as strings and not as coordinates.

Finally, the minimum value of the average costs is selected, and the corresponding path along with its cost is sent to
the server.