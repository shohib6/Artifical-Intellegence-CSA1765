from collections import deque

def solve_water_jug_problem(cap1, cap2, target):
    queue = deque([(0, 0, [])])
    visited = set([(0, 0)])
    
    while queue:
        jug1, jug2, path = queue.popleft()
        
        if jug1 == target or jug2 == target:
            path.append((jug1, jug2))
            for step in path:
                print(f"Jug1: {step[0]} liters, Jug2: {step[1]} liters")
            return
        
        for next_jug1, next_jug2 in [(cap1, jug2), (jug1, cap2), (0, jug2), (jug1, 0),
                                     (jug1 - min(jug1, cap2 - jug2), jug2 + min(jug1, cap2 - jug2)),
                                     (jug1 + min(jug2, cap1 - jug1), jug2 - min(jug2, cap1 - jug1))]:
            if (next_jug1, next_jug2) not in visited and 0 <= next_jug1 <= cap1 and 0 <= next_jug2 <= cap2:
                visited.add((next_jug1, next_jug2))
                queue.append((next_jug1, next_jug2, path + [(jug1, jug2)]))
    
    print("No solution found.")

# Example usage
solve_water_jug_problem(4, 3, 2)
