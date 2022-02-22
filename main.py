
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
from static_network.node_initialization import hash_groups, initialize_nodes, spatial
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

    x = 0
    for etn in etn_dict.values():
        x += len(etn)
    print(x)

    area_dict, node_area = spatial(etn_dict, group_nodes)

    print(len(node_area))


    # exit()
    '''
    Initialize edges for multiple layers
    '''

    layers ='familie', 'huishouden'

    barabasi = 0


    reciprocities = 0,1
    
    for layer_number, layer in enumerate(layers):

        for reciprocity in reciprocities: 

        #     for percentage in percentages:    

                    
                
                df_edges = pd.read_csv(f'Data/tab_{layer}.csv')
                source, destination, source_id, destination_id = initialize_edges_links(df_edges, all_nodes, layer ,group_nodes, hash_dict,area_dict,node_area, id_source = None, id_destination = None, source = None, destination = None, barabasi = barabasi, percentage = 0, reciprocity = 0, spatial = False)


                d = {'source_id': list(source_id), 'destination_id':list(destination_id), 'source_group': list(source), 'destination_group': list(destination)}

                df_ = pd.DataFrame(d)
                
                df_.to_csv(f'Data/NW_data2/{layer}_nw_barabasi={barabasi}_reciprocity_{reciprocity}_.csv')




    


    