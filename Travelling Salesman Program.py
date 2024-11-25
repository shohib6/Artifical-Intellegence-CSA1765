import itertools
import math
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
def total_distance(points, order):
    return sum(calculate_distance(points[order[i]], points[order[(i + 1) % len(order)]]) for i in range(len(order)))
def tsp_bruteforce(points):
    return min(
        (total_distance(points, perm), perm)
        for perm in itertools.permutations(range(len(points)))
    )
if __name__ == "__main__":
    points = [(0, 0), (1, 5), (5, 2), (6, 6)]
    min_distance, optimal_order = tsp_bruteforce(points)
    print("Minimum Distance:", min_distance)
    print("Optimal Order:", optimal_order)
