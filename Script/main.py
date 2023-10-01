
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
from static_network.node_initialization import initialize_nodes, spatial, workplace_dictionary, workplace_distribution
from static_network.hash_groups import hash_groups

from static_network.household_initialization import make_households
# from static_network.household_initialization import initialize_households
# from static_network.edge_initialization_homophily import initialize_edges_links
from static_network.edge_initilization import initialize_edges_links



from SALib.test_functions import Ishigami
from analysis import analysis

import matplotlib as mpl
import matplotlib.font_manager as fm#  Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2



def main(layer, param_values, area_name):
    x = 0
    for i in param_values:
    
        # print('hallo')
        def roundQuarter(x):
            return round(round(x * 3) / 3.0,2)
    

        # x,y,z equal sized 1D arrays
        '''
        Hash function
        '''

        hash_dict, rehash_dict = hash_groups()
        
        '''
        Initialize nodes 
        '''


        df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')
        df_edges = pd.read_csv(f'Data/tab_{layer}.csv')


        area_dict, node_area = [0,1]

        print(i)
        fraction,reciprocity,transitivity, workplaces = i
        
        
        if layer == 'buren':
          

            print(fraction,reciprocity,specify)
            s = roundQuarter(specify)

          


            if s != 0:
                fraction =1
            print(s)
            area_n = area_name[s]    
            specify = s
            workplaces = False

            print(fraction,reciprocity,s, transitivity)

            # exit()
            all_nodes, group_nodes, etn_dict, nodes_group, work_group_dict, person_work_dict, person_degree_count = initialize_nodes(hash_dict, area_n)

            area_dict, node_area = spatial(etn_dict, group_nodes, area_n)
            source, destination, source_id, destination_id = initialize_edges_links(df_edges, all_nodes, layer ,group_nodes,nodes_group, hash_dict,area_dict,node_area, work_group_dict, person_work_dict,area_name[1], fraction = float(fraction), reciprocity = float(reciprocity),transitivity = float(transitivity), spatial = int(s), workplaces = int(workplaces), i = i, person_degree_count = person_degree_count) 

        elif layer == 'werkschool':
            
   

            print(transitivity)
            s = round(workplaces)
            workplaces = round(workplaces)
            all_nodes, group_nodes, etn_dict, nodes_group, work_group_dict, person_work_dict, person_degree_count = initialize_nodes(hash_dict, area_name[1])

            source, destination, source_id, destination_id = initialize_edges_links(df_edges, all_nodes, layer ,group_nodes,nodes_group, hash_dict,area_dict,node_area, work_group_dict, person_work_dict,area_name[1], fraction = float(fraction), reciprocity = float(reciprocity),transitivity = float(transitivity), spatial = 0, workplaces = int(workplaces), i = i, person_degree_count = person_degree_count) 

        elif layer == 'familie':

            workplaces = False

            s = round(specify)
            transitivity = 1

            print(transitivity)
            all_nodes, group_nodes, etn_dict, nodes_group, work_group_dict, person_work_dict, person_degree_count = initialize_nodes(hash_dict, area_name[1])

            source, destination, source_id, destination_id = initialize_edges_links(df_edges, all_nodes, layer ,group_nodes,nodes_group, hash_dict,area_dict,node_area, work_group_dict, person_work_dict,area_name[1], fraction = float(fraction), reciprocity = float(reciprocity),transitivity = float(transitivity), spatial = int(s), workplaces = int(workplaces), i = i, person_degree_count = person_degree_count) 

        elif layer == 'huishouden':

            s = 0
            all_nodes, group_nodes, etn_dict, nodes_group, work_group_dict, person_work_dict, person_degree_count = initialize_nodes(hash_dict, 'wijken')

            area_dict, node_area = spatial(etn_dict, group_nodes, 'wijken')

            source, destination, source_id, destination_id = initialize_edges_links(df_edges, all_nodes, layer ,group_nodes,nodes_group, hash_dict,area_dict,node_area, work_group_dict, person_work_dict,area_name[1], fraction = float(fraction), reciprocity = float(reciprocity),transitivity = float(transitivity), spatial = int(s), workplaces = False, i = i, person_degree_count = person_degree_count) 
            if s == 1:
                source_nodes, destination_nodes, source_group, destination_group, household_ids, household_sizes = make_households(hash_dict, all_nodes, group_nodes,area_dict, node_area)

        print('\n\n\n')
 
        print('\n\n\n')

        d = {'source_id': list(source_id), 'destination_id':list(destination_id)}

        df_ = pd.DataFrame(d)


        
        df_['source_group'] = source
        df_['destination_group'] =destination

        # df_.to_csv(f'Experiment_{layer}/experiment_{x}_Transitivity.csv')

        # exit()
        x += 1
        analysis(layer, df_, x, fraction ,reciprocity,transitivity,workplaces)
        

        if x%10 == 0:
            df_.to_csv(f'Experiment_{layer}/experiment_{x}.csv')





        


        