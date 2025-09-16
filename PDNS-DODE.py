import networkx as nx
import random
import numpy as np
import math
import time


def read_edge_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines_2 = lines[1:]
        edges = [tuple(map(int, line.strip().split())) for line in lines_2]
        G = nx.Graph()
        G.add_edges_from(edges)
    return G


def pagerank_descent_initialization(G, pop_size, k, div):
    iter_count = 0
    population_matrix = None
    pagerank = nx.pagerank(G)
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    sorted_node_list = [node for node, _ in sorted_nodes]
    init_seed = sorted_node_list[:k]
    all_nodes = list(G.nodes())

    while iter_count < 2 * pop_size:
        current_solution = []
        up_bound = min(k * (iter_count // (pop_size + 20)), len(sorted_node_list))
        candidate_pool = sorted_node_list[:up_bound]
        available_candidates = [node for node in candidate_pool if node not in init_seed]

        for node in init_seed:
            if random.random() > div:
                if available_candidates:
                    replacement = random.choice(available_candidates)
                    current_solution.append(replacement)
                    available_candidates.remove(replacement)
                else:
                    replacement = random.choice([n for n in all_nodes if n not in init_seed])
                    current_solution.append(replacement)
            else:
                current_solution.append(node)

        iter_count += 1
        solution_vector = np.array(current_solution).reshape(1, -1)
        if population_matrix is None:
            population_matrix = solution_vector
        else:
            population_matrix = np.vstack([population_matrix, solution_vector])

    return population_matrix


def pdns_get_three_hop_neighbors(seed_set, G):
    all_neighbors = set(seed_set)
    current_hop = set(seed_set)

    for _ in range(3):
        next_hop = set()
        for node in current_hop:
            for neighbor in G.neighbors(node):
                if neighbor not in all_neighbors:
                    next_hop.add(neighbor)
        current_hop = next_hop
        all_neighbors.update(current_hop)

    all_neighbors.difference_update(seed_set)
    return all_neighbors


def pdns_pr_threshold_filter(node_set, pr_threshold, pagerank):
    for node in node_set.copy():
        pr_value = pagerank.get(node)
        if pr_value is not None and pr_value < pr_threshold:
            node_set.remove(node)
    return node_set


def pdns_generate_candidates(G, seed_set, pagerank):
    three_hop_neighbors = pdns_get_three_hop_neighbors(seed_set, G)
    pr_values = {node: pagerank[node] for node in three_hop_neighbors}
    pr_threshold = max(pr_values.values()) / 5
    filtered_candidates = pdns_pr_threshold_filter(three_hop_neighbors, pr_threshold, pagerank)
    return filtered_candidates


def edv_fitness(G, seed_set, k):
    one_hop_neighbors = set()
    for node in seed_set:
        one_hop_neighbors.update(G.neighbors(node))
    one_hop_neighbors.difference_update(seed_set)

    edv = k
    activation_prob = 0.01
    for neighbor in one_hop_neighbors:
        common_nodes = set(G.neighbors(neighbor)) & set(seed_set)
        ti = len(common_nodes)
        edv += 1 - (1 - activation_prob) ** ti
    return edv


def batch_edv_fitness(G, solution_matrix, k):
    fitness_list = []
    for solution in solution_matrix:
        fitness = edv_fitness(G, solution, k)
        fitness_list.append(fitness)
    return fitness_list


def population_stratification(population, fitness_values, pop_size):
    sorted_indices = sorted(range(len(fitness_values)), key=lambda x: fitness_values[x], reverse=True)
    elite_solution = population[sorted_indices[0]]
    superior_solutions = np.array(population[sorted_indices[:pop_size]])
    ordinary_solutions = np.array(population[sorted_indices[pop_size:]])
    return elite_solution, superior_solutions, ordinary_solutions


def ooa_mutate(solution, pdns_candidates, cr, search_radius, pagerank):
    mutated_solution = solution.copy()
    sorted_candidates = sorted(pdns_candidates, key=lambda x: pagerank[x], reverse=True)
    keep_count = math.ceil(len(sorted_candidates) * search_radius)
    selected_candidates = sorted_candidates[:keep_count]

    for i in range(len(mutated_solution)):
        if random.random() < cr:
            candidate_copy = selected_candidates.copy()
            if not candidate_copy:
                break
            new_node = random.choice(candidate_copy)
            while new_node in mutated_solution and candidate_copy:
                candidate_copy.remove(new_node)
                if candidate_copy:
                    new_node = random.choice(candidate_copy)
            if new_node not in mutated_solution:
                mutated_solution[i] = new_node
    return mutated_solution


def update_ordinary_solutions(mutated_superior, elite_solution, prob):
    common_nodes = set(mutated_superior) & set(elite_solution)
    unique_superior = [n for n in mutated_superior if n not in common_nodes]
    unique_elite = [n for n in elite_solution if n not in common_nodes]
    new_solution = list(common_nodes)
    while len(new_solution) < len(mutated_superior):
        if random.random() > prob and unique_superior:
            selected = random.choice(unique_superior)
            new_solution.append(selected)
            unique_superior.remove(selected)
        elif unique_elite:
            selected = random.choice(unique_elite)
            new_solution.append(selected)
            unique_elite.remove(selected)
        else:
            selected = random.choice(unique_superior)
            new_solution.append(selected)
            unique_superior.remove(selected)
    return np.array(new_solution)


def update_population(elite, ordinary_solutions, superior_solutions, G, k, pop_size):
    merged_population = np.vstack((superior_solutions, ordinary_solutions))
    merged_fitness = batch_edv_fitness(G, merged_population, k)
    sorted_indices = sorted(range(len(merged_fitness)), key=lambda x: merged_fitness[x], reverse=True)
    new_superior = np.array(merged_population[sorted_indices[:pop_size]])
    max_fitness = max(merged_fitness)
    current_elite_fitness = edv_fitness(G, elite, k)
    if max_fitness > current_elite_fitness:
        new_elite = new_superior[0]
        new_elite_fitness = max_fitness
    else:
        new_elite = elite
        new_elite_fitness = current_elite_fitness
    return new_elite, new_superior, new_elite_fitness


def de_adaptive_mutation(G, population, pop_size, k, max_pagerank, pagerank):
    mutated_population = []
    for i in range(pop_size):
        individual = population[i].copy()
        pdns_candidates = pdns_generate_candidates(G, individual, pagerank)
        for j in range(k):
            current_node = individual[j]
            adaptive_prob = pagerank[current_node] / (2 * max_pagerank)
            if random.random() < adaptive_prob:
                new_node = DDS(individual, pdns_candidates)
                if new_node is not None:
                    individual[j] = new_node
        mutated_population.append(individual)
    return np.array(mutated_population)


def DDS(current_solution, candidate_set):
    candidates = list(candidate_set)
    if not candidates:
        return None
    selected = random.choice(candidates)
    while selected in current_solution and candidates:
        candidates.remove(selected)
        if candidates:
            selected = random.choice(candidates)
    return selected if selected not in current_solution else None


def de_adaptive_crossover(G, population, mutated_pop, pop_size, k, cr_min, cr_max, elite, pagerank):
    elite_edv = edv_fitness(G, elite, k)
    population_edv = [edv_fitness(G, ind, k) for ind in population]
    sorted_nodes = sorted(G.nodes, key=lambda x: pagerank[x], reverse=True)

    crossover_pop = []
    for i in range(pop_size):
        cr = cr_min + (cr_max - cr_min) * (population_edv[i] / elite_edv)
        cr = max(min(cr, cr_max), cr_min)

        current_ind = []
        up_bound = k * (i + 5)
        if up_bound > len(sorted_nodes):
            up_bound = len(sorted_nodes)
        candidate_pool = sorted_nodes[:up_bound]

        for j in range(k):
            orig_node = population[i][j]
            mutate_node = mutated_pop[i][j]
            if random.random() < cr or pagerank[orig_node] <= pagerank[mutate_node]:
                selected = mutate_node
            else:
                selected = orig_node
            while selected in current_ind:
                available = [n for n in candidate_pool if n not in current_ind]
                if not available:
                    available = [n for n in G.nodes if n not in current_ind]
                selected = random.choice(available)
            current_ind.append(selected)
        crossover_pop.append(current_ind)
    return np.array(crossover_pop)


def de_selection(G, orig_pop, crossover_pop, k, pop_size):
    selected_pop = []
    for i in range(pop_size):
        orig_fitness = edv_fitness(G, orig_pop[i], k)
        cross_fitness = edv_fitness(G, crossover_pop[i], k)
        if cross_fitness > orig_fitness:
            selected_pop.append(crossover_pop[i])
        else:
            selected_pop.append(orig_pop[i])
    return np.array(selected_pop)


def get_best_solution(population, G, k):
    fitness_list = [edv_fitness(G, ind, k) for ind in population]
    max_idx = np.argmax(fitness_list)
    return population[max_idx], fitness_list[max_idx]


def local_search(G, seed_set, pr_threshold, k, pagerank):
    current_solution = sorted(seed_set, key=lambda x: pagerank[x])
    current_fitness = edv_fitness(G, current_solution, k)
    fitness_cache = {tuple(current_solution): current_fitness}

    for idx in range(len(current_solution)):
        node = current_solution[idx]
        three_hop = pdns_get_three_hop_neighbors([node], G)
        valid_candidates = [n for n in three_hop if pagerank[n] >= pr_threshold]

        for candidate in valid_candidates:
            new_solution = current_solution.copy()
            new_solution[idx] = candidate
            new_sol_tuple = tuple(new_solution)
            if new_sol_tuple in fitness_cache:
                new_fitness = fitness_cache[new_sol_tuple]
            else:
                new_fitness = edv_fitness(G, new_solution, k)
                fitness_cache[new_sol_tuple] = new_fitness
            if new_fitness > current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness
    return current_solution, current_fitness


def PDNS_DODE(G, k, pop_size, T_max, cr, pr_threshold, cr_min, cr_max, div):
    pagerank = nx.pagerank(G)
    max_pagerank = max(pagerank.values())
    init_population = pagerank_descent_initialization(G, pop_size, k, div)
    init_fitness = batch_edv_fitness(G, init_population, k)
    elite, superior_solutions, ordinary_solutions = population_stratification(init_population, init_fitness, pop_size)

    iter_count = 0
    start_total = time.time()
    stagnation_count = 0
    prev_elite_fitness = edv_fitness(G, elite, k)

    while iter_count < T_max:
        start_iter = time.time()
        pdns_radius = 0.2 + ((T_max - iter_count) * 0.8) / T_max

        for i in range(pop_size):
            current_superior = superior_solutions[i].copy()
            pdns_candidates = pdns_generate_candidates(G, current_superior, pagerank)
            mutated_superior = ooa_mutate(current_superior, pdns_candidates, cr, pdns_radius, pagerank)
            ordinary_solutions[i] = update_ordinary_solutions(mutated_superior, elite, 0.5)

        elite, superior_solutions, elite_fitness = update_population(elite, ordinary_solutions, superior_solutions, G,
                                                                     k, pop_size)

        mutated_pop = de_adaptive_mutation(G, superior_solutions, pop_size, k, max_pagerank, pagerank)
        crossover_pop = de_adaptive_crossover(G, superior_solutions, mutated_pop, pop_size, k, cr_min, cr_max, elite,
                                              pagerank)
        superior_solutions = de_selection(G, superior_solutions, crossover_pop, k, pop_size)
        elite, elite_fitness = get_best_solution(superior_solutions, G, k)

        end_iter = time.time()
        iter_count += 1
        print(
            f'迭代第{iter_count}次  用时:{end_iter - start_iter:.4f}s  适应度值:{elite_fitness:.4f}  种子集:{set(elite)}')

        if abs(elite_fitness - prev_elite_fitness) < 1e-6:
            stagnation_count += 1
            if stagnation_count == 10:
                print("早停策略触发，提前进入局部搜索阶段")
                break
        else:
            stagnation_count = 0
        prev_elite_fitness = elite_fitness

    start_local = time.time()
    final_seed, final_fitness = local_search(G, elite, pr_threshold, k, pagerank)
    end_local = time.time()
    end_total = time.time()

    print(f'局部搜索后  用时:{end_local - start_local:.4f}s  适应度值:{final_fitness:.4f}  种子集:{set(final_seed)}')
    print(
        f'PDNS-DODE迭代共{iter_count}次；k={k}  总用时:{end_total - start_total:.4f}s  最优适应度:{final_fitness:.4f}  最优种子集:{set(final_seed)}')
    return final_seed


if __name__ == '__main__':
    data_path = "your_edge_list.txt"
    G = read_edge_list(data_path)
    seed_size = 50
    pop_size = 80
    max_iter = 80
    ooa_cr = 0.1
    div_prob = 0.4
    de_cr_min = 0.2
    de_cr_max = 0.7

    pagerank = nx.pagerank(G)
    pr_max = max(pagerank.values())
    local_pr_threshold = pr_max / 5

    optimal_seeds = PDNS_DODE(
        G=G,
        k=seed_size,
        pop_size=pop_size,
        T_max=max_iter,
        cr=ooa_cr,
        pr_threshold=local_pr_threshold,
        cr_min=de_cr_min,
        cr_max=de_cr_max,
        div=div_prob
    )
