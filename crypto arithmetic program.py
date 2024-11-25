from itertools import permutations
def solve_cryptarithmetic(puzzle):
    parts = puzzle.split('+')
    left = parts[0].strip()
    right, result = parts[1].split('=')
    right = right.strip()
    result = result.strip()
    letters = set(left + right + result)
    for perm in permutations(range(10), len(letters)):
        mapping = dict(zip(letters, perm))
        if mapping[left[0]] == 0 or mapping[right[0]] == 0 or mapping[result[0]] == 0:
            continue
        left_num = int(''.join(str(mapping[char]) for char in left))
        right_num = int(''.join(str(mapping[char]) for char in right))
        result_num = int(''.join(str(mapping[char]) for char in result))
        if left_num + right_num == result_num:
            return mapping  
    return None  # No solution found
# Example usage:
puzzle = "BASE + BALL = GAMES"
solution = solve_cryptarithmetic(puzzle)
if solution:
    print("Solution found:")
    for letter, digit in solution.items():
        print(f"{letter}: {digit}")
else:
    print("No solution found.")
