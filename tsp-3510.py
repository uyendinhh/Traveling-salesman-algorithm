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
            probability = (number_of_unvisited_neighbors - 1) if number_of_unvisited_neighbors < 5 else random.randint(1, 1)
            next_neighbor_to_visit = sorted_cost_matrix[last_visited_node][probability]
            if next_neighbor_to_visit in visited_neighbors and number_of_unvisited_neighbors:
                random_neighbor_idx = random.randint(0, number_of_unvisited_neighbors - 1) if number_of_unvisited_neighbors > 1 else 0
                next_neighbor_to_visit = unvisited_neighbors[random_neighbor_idx] 
            visited_neighbors.append(next_neighbor_to_visit)
        return visited_neighbors
            
    
    def run(self):
        if (len(sys.argv) < 3):
            return
        # read data in the given file 
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

        self.generate_node_distances()
        n = len(self.nodes)
        initial_path = self.generate_initial_path()
        best_path, best_cost = self.two_opt(initial_path, self.node_distances) 
        while time.time() < self.timeout:
            better_path, better_cost = self.swap_edges(best_path)
            if best_cost == better_cost:
                break
            if better_cost < best_cost:
                best_path = better_path
                best_cost = better_cost
        for i in range(n):
            best_path[i] -= 1
        with open(output_file_name, 'a') as f:
            f.write('Path: ' + str(best_path))       
            f.write(', Cost: ' + str(best_cost) + '\n')
        print('final_cost', best_cost)

    def swap_edges(self, cur_path):
        a, b, c, d = random.sample(range(0, len(self.nodes)), 4)
        new_path = copy.deepcopy(cur_path)
        # swap 2 edges randomly
        new_path[a], new_path[b], new_path[c], new_path[d] = new_path[d], new_path[c], new_path[b], new_path[a]  
        return self.two_opt(new_path, self.node_distances)
            
    def calculate_path_cost(self, path):
        n = len(self.nodes)
        cost = self.node_distances
        total_cost = 0
        for i in range(0, n - 1):
            total_cost += cost[path[i]][path[i+1]]
        total_cost += cost[path[n-1]][path[0]]
        return total_cost

    def cost_change(self, cost_mat, n1, n2, n3, n4):
        return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]


    def two_opt(self, route, cost_mat):
        best = route
        improved = True
        old_cost = 0
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if time.time() > self.timeout:
                        break
                    if j - i == 1: continue
                    if self.cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            route = best
        cost = self.calculate_path_cost(best)
        return (best, cost)
    
if __name__ == "__main__":
    TSP().run()
