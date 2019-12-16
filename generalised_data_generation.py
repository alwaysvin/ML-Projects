import networkx as nx
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import deque
import copy
# Global Variables used across functions



def return_required_results(no_of_nodes, data_length, mu, sigma, coeff_lo, coeff_hi, cat_num):
    x_1_hot, x_final_dict, G, bin_limits = main(no_of_nodes, data_length, mu, sigma, coeff_lo, coeff_hi, cat_num)
    return x_1_hot, x_final_dict, G, bin_limits

def copy_x_dict_tograph(x_dict, G):
    for i in x_dict.keys():
        G.node[i]['data'] = x_dict[i]    
    return G
   
##############Parameters##############
# node_idx- Index of node
# G - Graph object
# q_list - Queue object
# x_dict - x_dict is the dictionry storing value for each node.
def update_all_subsequent_nodes(node_idx, G, q_list, x_dict):
    # push all out edges on stack  
    if list(G.out_edges(node_idx)) != []:
        for i in G.out_edges(node_idx):
            q_list.append(i)
        #q_list.append(list(G.out_edges(node_idx)))
    # Pop 1 element and update its values
    if len(q_list) != 0:
        top = q_list.pop()
        
        # x_dict stores initialized values so only update in parent node is added to child node
        G.node[top[1]]['data'] = copy.deepcopy(x_dict[top[1]])
        for i in G.in_edges(top[1]):
            if i[1] != top[1]:
                input('what did you do')
            
            G.node[top[1]]['data'] += G.node[i[0]]['data'] * G.get_edge_data(i[0], top[1])['weight'][0]

        

        # and update as per edge
        G, q_list = update_all_subsequent_nodes(top[1], G, q_list, x_dict)
    
    return G, q_list
    
    
    
def populate_graph_as_per_weights(x_dict, G):
    
    # First find all nodes with no incoming edges
    root_nodes = []
    for i in G.nodes():
        x = list(G.in_edges(i))
        if  x == []:
            root_nodes.append(i)
    
    # Now for each such nodes do DFS and update all the 
    q_list = deque()        
    for i in root_nodes:
        G, q_list = update_all_subsequent_nodes(i, G, q_list, x_dict)
    print('Hopefully this should be empty', q_list)
    #nx.dfs_edges()

def copy_graph_to_x_dict(x_dict, G):
    for i in x_dict.keys():
        x_dict[i] = copy.deepcopy(G.node[i]['data'])
    return x_dict

def construct_1_hot_vector(x_dict, bin_limits, G, data_length, cat_num):
        x_final_dict = {}
        for i in x_dict.keys():
            x_final_dict[i] = np.searchsorted(bin_limits, x_dict[i], side="left")
    
        # Now convert number to 1 hot vector
        x_1_hot = {}
        for i in x_final_dict.keys():
            #all_zeros = np.zeros((data_length, cat_num, 1))
            all_zeros = np.zeros((data_length, cat_num))
        
            for j in range(data_length):        
                all_zeros[j, x_final_dict[i][j]] = 1            
            x_1_hot[i] = all_zeros
        
        return x_1_hot, x_final_dict
        
        
        
def main(no_of_nodes = 10, data_length = 1, mu = 0, sigma = 1, coeff_lo = -10, coeff_hi =10, cat_num = 10):
    
    x = list(range(no_of_nodes))
    
    # creat dictionary to store all data points
    x_dict = {key: np.random.normal(mu, sigma, data_length) for key in x}
    #x_dict = {key: np.random.randint(0, 10, data_length) for key in x}
    
    G = nx.Graph()    
    G = nx.Graph()
    G = G.to_directed()
    G.add_nodes_from(x)
    nx.set_node_attributes(G, G.nodes, 'data')
    G = copy_x_dict_tograph(x_dict, G)
    
    iterator = permutations(x, 2)
    possible_combinations = sum(1 for _ in iterator)
    iterator = permutations(x, 2) # This is too wierd
    for [i, j] in iterator:
        
        edge_prob = 1/possible_combinations
        edge_prob= 0.1
        if random.random() > edge_prob:
            continue
        coeff = np.random.randint(coeff_lo, coeff_hi, size = 1)
        G.add_edge(i, j, weight = coeff)
        if nx.dag.has_cycle(G):
            G.remove_edge(i, j)
            continue
        # Update random variable as per newly added edge
        #x_dict[j] += x_dict[i] * coeff
    # before updating
    populate_graph_as_per_weights(x_dict, G)
    
    # Copy back as x_dict is being used further
    x_dict = copy_graph_to_x_dict(x_dict, G)

    # Convert data to 1 hot vector
    
    ## First get category based on region it lies in
    mean_whole  = np.mean([np.mean(x_dict[i]) for i in x_dict.keys()])
    squared_sum = 0
    for i in x_dict.keys():
        squared_sum = squared_sum + np.sum(np.square(x_dict[i] - mean_whole))
    
    std_whole = (np.sqrt(squared_sum))/ (no_of_nodes * data_length)
    upper_l    = mean_whole + 2 * std_whole
    lower_l    = mean_whole - 2 * std_whole
    
    bin_limits = np.linspace(upper_l, lower_l, cat_num - 1) # fix this cat_num - 1
    bin_limits = np.sort(bin_limits)
    
    x_1_hot, x_final_dict = construct_1_hot_vector(x_dict, bin_limits, G, data_length, cat_num)
    nx.draw(G,pos=nx.spring_layout(G), with_labels = True)
    print('Printing usefull data', G.number_of_edges())
    return x_1_hot, x_final_dict, G, bin_limits
    

if __name__ == "__main__": 
    main()