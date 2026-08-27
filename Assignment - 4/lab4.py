import sys
from collections import deque

def read_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']

    idx = 0
    n, m = map(int, lines[idx].split())
    idx += 1    
    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]

    for _ in range(m):
        u, v = map(int, lines[idx].split())
        idx += 1
        children[u].append(v)
        parents[v].append(u)

    descendants = [set() for _ in range(n)]
    for i in range(n):
        visited = [False] * n
        queue = deque([i])
        visited[i] = True

        while queue:
            u = queue.popleft()
            for v in children[u]:
                if not visited[v]:
                    visited[v] = True
                    descendants[i].add(v)
                    queue.append(v)

    q = int(lines[idx])
    idx += 1
    queries = []

    for _ in range(q):
        parts = list(map(int, lines[idx].split()))
        idx += 1
        x, y = parts[0], parts[1]
        Z = set(parts[2:]) if len(parts) > 2 else set()
        queries.append((x, y, Z))

    return n, children, parents, descendants, queries


def has_active_trail(x, y, Z, children, parents, descendants):
    stack = []
    visited = set()

    for w in children[x]:
        state = (w, x, 1)
        stack.append(state)
        visited.add(state)

    for w in parents[x]:
        state = (w, x, -1)
        stack.append(state)
        visited.add(state)

    while stack:
        v, u, arrived_dir = stack.pop()
        if v == y:
            return True

        for w in children[v]:
            if w == u:
                continue

            leave_dir = 1
            blocked = False

            if arrived_dir == 1 and leave_dir == -1:
                if v not in Z and not (descendants[v] & Z):
                    blocked = True
            else:
                if v in Z:
                    blocked = True

            if not blocked:
                state = (w, v, leave_dir)
                if state not in visited:
                    visited.add(state)
                    stack.append(state)

        for w in parents[v]:
            if w == u:
                continue

            leave_dir = -1
            blocked = False

            if arrived_dir == 1 and leave_dir == -1:
                if v not in Z and not (descendants[v] & Z):
                    blocked = True
            else:
                if v in Z:
                    blocked = True

            if not blocked:
                state = (w, v, leave_dir)
                if state not in visited:
                    visited.add(state)
                    stack.append(state)

    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python lab4.py <input_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    n, children, parents, descendants, queries = read_input(filename)
    
    with open('lab4_output.txt', 'w') as out_file:
        for x, y, Z in queries:
            if has_active_trail(x, y, Z, children, parents, descendants):
                out_file.write("NO\n")
            else:
                out_file.write("YES\n")

if __name__ == "__main__":
    main()