from collections import deque

def bfs(graph, start):
    visited = set()  
    queue = deque([start])  
    visited.add(start)  

    while queue:
        vertex = queue.popleft()  
        print(vertex, end=" ")  
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    print("BFS Trnversal starting from node 'A':")
    bfs(graph, 'A')
