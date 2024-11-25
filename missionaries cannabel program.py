from collections import deque

def is_valid_state(m, c):
    return 0 <= m <= 3 and 0 <= c <= 3 and (m == 0 or m >= c) and ((3 - m) == 0 or (3 - m) >= (3 - c))

def get_next_states(state):
    m_left, c_left, b_left, m_right, c_right, b_right = state
    next_states = []
    for i in range(3):
        for j in range(3):
            if 1 <= i + j <= 2:
                if b_left:
                    new_state = (m_left - i, c_left - j, 0, m_right + i, c_right + j, 1)
                else:
                    new_state = (m_left + i, c_left + j, 1, m_right - i, c_right - j, 0)
                if is_valid_state(new_state[0], new_state[1]) and is_valid_state(new_state[3], new_state[4]):
                    next_states.append(new_state)
    return next_states

def bfs():
    start, goal = (3, 3, 1, 0, 0, 0), (0, 0, 0, 3, 3, 1)
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path + [goal]
        for next_state in get_next_states(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [state]))
    return None

def print_solution(solution):
    if solution:
        print("Solution found!")
        for i, state in enumerate(solution):
            print(f"Step {i + 1}: {state[:3]} || {state[3:]}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    print_solution(bfs())

