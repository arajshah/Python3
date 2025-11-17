import numpy as np
from .graph import Cell
from .utils import trace_path


def depth_first_search(graph, start, goal):
    """Depth First Search (DFS) algorithm. This algorithm is optional for P3.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    
    # Use a stack for DFS (LIFO)
    stack = [start]
    
    # Get the start node and mark it as visited
    start_node = graph.nodes[start.j][start.i]
    start_node.visited = True
    start_node.parent = None
    
    while stack:
        # Pop from the end (LIFO - stack behavior)
        current = stack.pop()
        
        # Add to visited cells for visualization
        graph.visited_cells.append(Cell(current.i, current.j))
        
        # Check if we reached the goal
        if current.i == goal.i and current.j == goal.j:
            return trace_path(goal, graph)
        
        # Get the current node
        current_node = graph.nodes[current.j][current.i]
        
        # Explore neighbors
        neighbors = graph.find_neighbors(current.i, current.j)
        for ni, nj in neighbors:
            neighbor_node = graph.nodes[nj][ni]
            
            # Skip if already visited, in collision, or occupied
            if neighbor_node.visited or graph.check_collision(ni, nj):
                continue
            
            # Mark as visited and set parent
            neighbor_node.visited = True
            neighbor_node.parent = current_node
            
            # Add to stack
            stack.append(Cell(ni, nj))
    
    # If no path was found, return an empty list.
    return []


def breadth_first_search(graph, start, goal):
    """Breadth First Search (BFS) algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    
    # Use a queue for BFS (FIFO)
    from collections import deque
    queue = deque([start])
    
    # Get the start node and mark it as visited
    start_node = graph.nodes[start.j][start.i]
    start_node.visited = True
    start_node.parent = None
    
    while queue:
        # Pop from the front (FIFO - queue behavior)
        current = queue.popleft()
        
        # Add to visited cells for visualization
        graph.visited_cells.append(Cell(current.i, current.j))
        
        # Check if we reached the goal
        if current.i == goal.i and current.j == goal.j:
            return trace_path(goal, graph)
        
        # Get the current node
        current_node = graph.nodes[current.j][current.i]
        
        # Explore neighbors
        neighbors = graph.find_neighbors(current.i, current.j)
        for ni, nj in neighbors:
            neighbor_node = graph.nodes[nj][ni]
            
            # Skip if already visited, in collision, or occupied
            if neighbor_node.visited or graph.check_collision(ni, nj):
                continue
            
            # Mark as visited and set parent
            neighbor_node.visited = True
            neighbor_node.parent = current_node
            
            # Add to queue
            queue.append(Cell(ni, nj))
    
    # If no path was found, return an empty list.
    return []


def a_star_search(graph, start, goal):
    """A* Search algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    
    import heapq
    
    # Heuristic function (Euclidean distance)
    def heuristic(i1, j1, i2, j2):
        return np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
    
    # Priority queue: stores tuples of (f_cost, counter, cell)
    # Counter ensures consistent ordering when f_costs are equal
    counter = 0
    open_set = []
    
    # Initialize start node
    start_node = graph.nodes[start.j][start.i]
    start_node.g_cost = 0
    start_node.h_cost = heuristic(start.i, start.j, goal.i, goal.j)
    start_node.f_cost = start_node.g_cost + start_node.h_cost
    start_node.parent = None
    
    heapq.heappush(open_set, (start_node.f_cost, counter, start))
    counter += 1
    
    # Keep track of cells in open set
    in_open_set = {(start.i, start.j)}
    
    while open_set:
        # Get cell with lowest f_cost
        _, _, current = heapq.heappop(open_set)
        in_open_set.discard((current.i, current.j))
        
        # Get the current node
        current_node = graph.nodes[current.j][current.i]
        
        # Skip if already visited
        if current_node.visited:
            continue
        
        # Mark as visited
        current_node.visited = True
        
        # Add to visited cells for visualization
        graph.visited_cells.append(Cell(current.i, current.j))
        
        # Check if we reached the goal
        if current.i == goal.i and current.j == goal.j:
            return trace_path(goal, graph)
        
        # Explore neighbors
        neighbors = graph.find_neighbors(current.i, current.j)
        for ni, nj in neighbors:
            neighbor_node = graph.nodes[nj][ni]
            
            # Skip if already visited or in collision
            if neighbor_node.visited or graph.check_collision(ni, nj):
                continue
            
            # Calculate cost to reach this neighbor
            # Diagonal moves cost sqrt(2), orthogonal moves cost 1
            di = abs(ni - current.i)
            dj = abs(nj - current.j)
            if di + dj == 2:  # Diagonal move
                move_cost = np.sqrt(2)
            else:  # Orthogonal move
                move_cost = 1.0
            
            tentative_g_cost = current_node.g_cost + move_cost
            
            # If this path to neighbor is better than any previous one
            if tentative_g_cost < neighbor_node.g_cost:
                neighbor_node.parent = current_node
                neighbor_node.g_cost = tentative_g_cost
                neighbor_node.h_cost = heuristic(ni, nj, goal.i, goal.j)
                neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                
                # Add to open set if not already there
                if (ni, nj) not in in_open_set:
                    heapq.heappush(open_set, (neighbor_node.f_cost, counter, Cell(ni, nj)))
                    counter += 1
                    in_open_set.add((ni, nj))
    
    # If no path was found, return an empty list.
    return []