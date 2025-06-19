import heapq

# Representasi graph: adjacency list dengan biaya
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 5, 'E': 12},
    'C': {'F': 3},
    'D': {'G': 6},
    'E': {'G': 3},
    'F': {'E': 2},
    'G': {}
}

# Heuristic (estimasi jarak dari node ke goal 'G')
heuristic = {
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 7,
    'E': 3,
    'F': 6,
    'G': 0
}

def a_star_search(start, goal):
    # Priority queue untuk menyimpan node dengan f(n)
    open_list = []
    heapq.heappush(open_list, (0 + heuristic[start], 0, start, [start]))  # (f, g, current, path)

    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current == goal:
            return path, g  # path dan total cost

        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in graph[current].items():
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')  # jika tidak ditemukan

# Tes algoritma A*
start_node = 'A'
goal_node = 'G'
path, total_cost = a_star_search(start_node, goal_node)

# Output hasil
print("Rute terbaik dari", start_node, "ke", goal_node, ":", path)
print("Total biaya:", total_cost)
