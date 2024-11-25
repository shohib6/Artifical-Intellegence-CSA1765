import heapq
def heuristic(state, goal):
    return sum(abs(b % 3 - g % 3) + abs(b // 3 - g // 3) for b, g in zip(state, goal) if b)
def astar_search(start, goal):
    open_list, closed = [(0, start)], set()
    came_from, cost = {}, {tuple(start): 0}
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            break
        closed.add(tuple(current))
        for next_state in get_neighbors(current):
            if tuple(next_state) not in closed:
                new_cost = cost[tuple(current)] + 1
                if tuple(next_state) not in cost or new_cost < cost[tuple(next_state)]:
                    cost[tuple(next_state)] = new_cost
                    priority = new_cost + heuristic(next_state, goal)
                    heapq.heappush(open_list, (priority, next_state))
                    came_from[tuple(next_state)] = current
    return came_from, cost
def get_neighbors(state):
    neighbors, zero = [], state.index(0)
    moves = [(-3, 0), (3, 0), (-1, -1), (1, 1)]
    for move, col in moves:
        new_pos = zero + move
        if 0 <= new_pos < len(state) and (col == 0 or zero // 3 == new_pos // 3):
            neighbor = state[:]
            neighbor[zero], neighbor[new_pos] = neighbor[new_pos], neighbor[zero]
            neighbors.append(neighbor)
    return neighbors
def reconstruct_path(came_from, start, goal):
    current, path = goal, [goal]
    while current != start:
        current = came_from[tuple(current)]
        path.append(current)
    return path[::-1]
start_state = [1, 0, 3, 4, 2, 5, 6, 7, 8]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
came_from, cost = astar_search(start_state, goal_state)
path = reconstruct_path(came_from, start_state, goal_state)
print("Solution path:")
for state in path:
    print(state)
print("Cost:", cost[tuple(goal_state)])
