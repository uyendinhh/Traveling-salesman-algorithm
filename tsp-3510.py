import sys
import numpy as np
import random
import copy 
import time


class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class TSP:
    def __init__(self):
        self.nodes = []
        self.node_distances = None
        self.timeout = 0
    
    def generate_node_distances(self):
        # get the distance using the normal vector
        z = np.array([complex(node.x, node.y) for node in self.nodes])
        self.node_distances = np.around(abs(z[..., np.newaxis] - z)) # round to nearest integer number
        self.node_distances = self.node_distances.astype(int) # convert from float -> int
    
    def generate_initial_path(self): 
        n = len(self.node_distances)
        start_node = random.randint(0, n - 1)
        all_neighbors = random.sample(range(0, n), n)
        unvisited_neighbors = range(0, n)
        visited_neighbors = [start_node]
        
        # sort the distance matrix and keep their index
        sorted_cost_matrix = []
        for i in range(n):
            sorted_distance = np.argsort(self.node_distances[i]) 
            sorted_cost_matrix.append(sorted_distance)
        
        while len(visited_neighbors) < n:
            unvisited_neighbors = list(set(unvisited_neighbors).difference(visited_neighbors))
            number_of_unvisited_neighbors = len(unvisited_neighbors) 
            last_visited_node = visited_neighbors[-1]

            # get the first, second, or third closest neighbor of the last visited node randomly
            probability = (number_of_unvisited_neighbors - 1) if number_of_unvisited_neighbors < 5 else random.randint(1, 4)
            next_neighbor_to_visit = sorted_cost_matrix[last_visited_node][probability]

            # if the next node to visit has already been visited, find another node
            if next_neighbor_to_visit in visited_neighbors and number_of_unvisited_neighbors:
                random_neighbor_idx = random.randint(0, number_of_unvisited_neighbors - 1) if number_of_unvisited_neighbors > 1 else 0
                next_neighbor_to_visit = unvisited_neighbors[random_neighbor_idx]

            visited_neighbors.append(next_neighbor_to_visit)

        return visited_neighbors
            
    
    def run(self):
        if (len(sys.argv) < 3):
            print('Arguments should contain: <input-file-nam> <output-file-name> <time>')
            return

        # read data from the given file 
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]
        time_limit = sys.argv[3]
        
        self.timeout = time.time() + int(time_limit) # 3 mins
        
        with open(input_file_name) as f:
            lines = f.readlines()
            for line in lines:
                if line != ('\n'):
                    vertex, x, y = line.split()
                    node = Node(int(vertex), float(x), float(y))
                    self.nodes.append(node)

        n = len(self.nodes)
        self.generate_node_distances()

        initial_path = self.generate_initial_path()
        best_path, best_cost = self.two_opt(initial_path)
 
        while time.time() < self.timeout:
            better_path, better_cost = self.path_after_swapping_edges(best_path)
            if best_cost == better_cost:
                break
            if better_cost < best_cost:
                best_path = better_path
                best_cost = better_cost

        # since the first node starts at 1, increase 1 to every node to convert the nodes back their original form
        for i in range(n):
            best_path[i] += 1

        best_path.append(best_path[0])

        # save the path and its cost to a text file
        with open(output_file_name, 'a') as f:
            f.write('Path: ' + str(best_path))       
            f.write(', Cost: ' + str(best_cost) + '\n')
        print('final cost', best_cost)

    def path_after_swapping_edges(self, cur_path):
        a, b, c, d = random.sample(range(0, len(self.nodes)), 4)
        new_path = copy.deepcopy(cur_path)
        # swap 2 edges randomly
        new_path[a], new_path[b], new_path[c], new_path[d] = new_path[d], new_path[c], new_path[b], new_path[a]
        return self.two_opt(new_path)
            
    def calculate_path_cost(self, path):
        n = len(self.nodes)
        cost = self.node_distances
        total_cost = 0
        for i in range(0, n - 1):
            total_cost += cost[path[i]][path[i+1]]
        total_cost += cost[path[n-1]][path[0]]
        return total_cost

    def should_swap_edges(self, v1, v2, k1, k2):
        old_edge1 = self.node_distances[v1][v2] 
        old_edge2 = self.node_distances[k1][k2]

        new_edge1 = self.node_distances[v1][k1]
        new_edge2 = self.node_distances[v2][k2]

        return (new_edge1 + new_edge2) < (old_edge1 + old_edge2)


    def two_opt(self, cur_path):
        best_path_found = False
        while not best_path_found:
            best_path_found = False
            for i in range(1, len(cur_path) - 2):
                for j in range(i + 1, len(cur_path)):
                    if j - i == 1: continue
                    if time.time() > self.timeout:
                        break
                    if self.should_swap_edges(cur_path[i - 1], cur_path[i], cur_path[j - 1], cur_path[j]):
                        best_path_found = True
                        cur_path[i], cur_path[j - 1]= cur_path[j-1], cur_path[i]
        cost = self.calculate_path_cost(cur_path)
        return (cur_path, cost)
    
if __name__ == "__main__":
    TSP().run()
