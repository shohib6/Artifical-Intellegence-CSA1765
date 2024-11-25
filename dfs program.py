def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
if __name__ == "__main__":
    graph = {
        1: [2, 3],
        2: [1, 4],
        3: [1, 5],
        4: [2],
        5: [3]
    }

    print("DFS Traversal starting from node 1:")
    dfs(graph, 1)
