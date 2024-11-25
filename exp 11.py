import heapq

def a_star(graph, start, goal, heuristic):
    # Priority queue to store (cost, current_node, path)
    open_list = []
    heapq.heappush(open_list, (0, start, [start]))
    
    # Dictionary to store the cost to reach each node
    g_cost = {node: float('inf') for node in graph}
    g_cost[start] = 0
    
    while open_list:
        # Pop the node with the lowest estimated total cost (f = g + h)
        current_cost, current_node, path = heapq.heappop(open_list)
        
        # If we reached the goal, return the path and cost
        if current_node == goal:
            return path, current_cost
        
        # Explore neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            tentative_g_cost = g_cost[current_node] + weight
            
            # If a better cost is found, update and add to the priority queue
            if tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor, path + [neighbor]))
    
    # Return None if no path is found
    return None, float('inf')

# Example graph represented as an adjacency list with edge weights
graph = {
    'A': {'B': 1, 'C': 3},
    'B': {'A': 1, 'D': 1, 'E': 4},
    'C': {'A': 3, 'F': 2},
    'D': {'B': 1, 'E': 1},
    'E': {'B': 4, 'D': 1, 'F': 1},
    'F': {'C': 2, 'E': 1}
}

# Example heuristic values for each node (straight-line distance to goal)
heuristic = {
    'A': 6,
    'B': 5,
    'C': 3,
    'D': 3,
    'E': 2,
    'F': 0  # Goal node has a heuristic of 0
}

# Find the shortest path from 'A' to 'F'
start = 'A'
goal = 'F'
path, cost = a_star(graph, start, goal, heuristic)
print("Shortest Path:", " -> ".join(path))
print("Total Cost:", cost)
(10)