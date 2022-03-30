
from pyclbr import Class
from networkx.algorithms.centrality import group

from descriptive import Person_links

from itertools import product
import math
import random
import pandas as pd
import networkx as nx
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path
import time
from static_network.node_initialization import initialize_nodes, spatial
from static_network.hash_groups import hash_groups
# from static_network.household_initialization import initialize_households
# from static_network.edge_initialization_homophily import initialize_edges_links
from static_network.edge_initilization import initialize_edges_links

if __name__ == '__main__':
    # TODO
    # Spatial
    # Interlayer correlation

    '''
    Hash function
    '''

    hash_dict, rehash_dict = hash_groups()
    
    '''
    Initialize nodes 
    '''
    

    df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')
    all_nodes, group_nodes, etn_dict = initialize_nodes(df_nodes, hash_dict)


    area_dict, node_area = spatial(etn_dict, group_nodes)

    # area_dict, node_area = [0,1]



    # exit()
    '''
    Initialize edges for multiple layers
    '''

    layers = 'buren', 'f'

    barabasi = 0


    reciprocities = 0,1
    
    for layer_number, layer in enumerate(layers):

    #     for percentage in percentages:    

            df_edges = pd.read_csv(f'Data/tab_{layer}.csv')
            source, destination, source_id, destination_id = initialize_edges_links(df_edges, all_nodes, layer ,group_nodes, hash_dict,area_dict,node_area, id_source = None, id_destination = None, source = None, destination = None, barabasi = barabasi, percentage = 0, reciprocity = 0.8, spatial = True)

            d = {'source_id': list(source_id), 'destination_id':list(destination_id), 'source_group': list(source), 'destination_group': list(destination)}

            df_ = pd.DataFrame(d)
            
            df_.to_csv(f'Data/NW_data2/{layer}_test.csv')

            # house_holds = initialize_edges_links(df_edges, all_nodes, layer ,group_nodes, hash_dict,area_dict,node_area, id_source = None, id_destination = None, source = None, destination = None, barabasi = barabasi, percentage = 0, reciprocity = 1, spatial = True)

            # count = 0
            # for household in house_holds:
            #     if len(household.agents) > 0:
            #         count += len(household.agents)
            #         # print(len(household.agents))

            # print(count)
            




    


    