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
import pandas as pd
from static_network.node_initialization import initialize_nodes, spatial
from static_network.hash_groups import hash_groups


class Initialize_edges:

    def __init__(self, layer, s):
        
        self.area_names = {0: '22gebieden', 0.33:  '22gebieden', 0.67: 'wijken', 1: 'buurten'}
        self.area_name_dict = {'22gebieden': 21, 'wijken': 98, 'buurten': 480}

        self.area_name = self.area_names[s]
        self.n_areas = self.area_name_dict[self.area_name]
        self.area_n = self.area_name_dict[self.area_name]  
        self.hash_dict, self.rehash_dict = hash_groups()

        # Initalization functions
        self.all_nodes, self.group_nodes, self.etn_dict, self.nodes_group, self.work_group_dict, self.person_work_dict, self.person_degree_count = initialize_nodes(self.hash_dict, self.area_name)
        self.area_dict, self.node_area = spatial(self.etn_dict, self.group_nodes, self.area_name)

        # Edge data
        self.df_edges = pd.read_csv(f'Data/tab_{layer}.csv')
        self.df_edges = pd.concat([self.df_edges]*10, ignore_index=True)
        self.df_edges.n = self.df_edges.n/10
        self.total_edges = self.df_edges.n.sum()

        # Parameters
        self.layer = layer


    def make_kf_for_spec(self,area_workplace_dict):

            for workplace in area_workplace_dict.keys():
                self.barabasi_dict[workplace]  = {}
                for group in range(240):
                    self.barabasi_dict[workplace][group] = []
            


            for workplace in area_workplace_dict.keys():
                for group in range(240):
                    if area_workplace_dict[workplace][group]:
                        chosen_ones = random.sample(area_workplace_dict[workplace][group], k=(math.ceil(len(area_workplace_dict[workplace][group])*(self.fraction))))
                        self.barabasi_dict[workplace][group].extend(chosen_ones)

    def initialize_rows(self):
        initial_list = []
        for row in self.df_edges.iterrows():


            row = row[1]

            dst = self.hash_dict[f"{row['lft_dst']}, {row['etngrp_dst']}, {row['geslacht_dst']}, {row['oplniv_dst']}"]
            src = self.hash_dict[f"{row['lft_src']}, {row['etngrp_src']}, {row['geslacht_src']}, {row['oplniv_src']}"]

            initial_list.append(
                (
                src,
                dst,
                int(row['n'])
                )
                )


            self.max_dict[(src,dst)] = int(row['n']) * 10
        
            self.symetry_dict[(src,dst)] = 0
        return np.array(initial_list)

    def initialize_arrays(self):
        
        self.source = []
        self.destination = []
        self.id_source = []
        self.id_destination = []
        self.overig = [0, 1, 2, 18, 19, 20, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 66, 67, 68, 99, 100, 101, 105, 106, 107, 108,
        109, 110, 132, 133, 134, 144, 145, 146, 180, 181, 182]

        self.group_connection = {}
        self.barabasi_dict = {}

        self.symetry_dict = {}
        self.max_dict = {}
        
        
        # Initialize all nodes with own link dictionary
        self.link_dictionary = {}

        self.done_connections = set()

        self.edges_layd = 0
        self.tot_connections = 0
        
        
    

        for node in self.all_nodes:
            self.link_dictionary[node] = set()

  
    
        # Make from all strings integers so we do not have toe loop over pandas dataframe but over numpy array
        self.initial_list = self.initialize_rows()


    def household_family_correlation(self):

        houshold_df = pd.read_csv(f"./Data/Experiments/Experiments7/huishouden_experiment_1.csv")
        houshold_df =  houshold_df[(~houshold_df['source_group'].isin(self.overig) ) & (~houshold_df['destination_group'].isin(self.overig))]

        print(houshold_df.shape)
      
        x = random.sample(list(houshold_df.household_id.unique()), k =100000)

        houshold_df = houshold_df[houshold_df['household_id'].isin(x)]

        c = 0
        for row in houshold_df.itertuples():
        
            _,_,source_id, destination_id, source_group, destination_group, _ = row


            if (source_group,destination_group)  in self.symetry_dict and (destination_group,source_group) in self.symetry_dict and\
                self.symetry_dict[(destination_group, source_group)] < self.max_dict[(destination_group, source_group)] and \
                self.symetry_dict[(source_group,destination_group)] < self.max_dict[(source_group,destination_group)] :

                self.add_nodes_to_network(source_group, destination_group, source_id, destination_id)
        
                c += 1

    def area_check(self, src_node, dst_group, area_r):
        
        complete =True

        # Get area of source group
        area = self.node_area[src_node]
        
        if np.random.uniform() < 0.01:

            if np.random.uniform()<0.5:
                area += 1
            else :
                area -= 1
                
            area %= self.n_areas

        area += area_r
        area %= self.n_areas

        dst_area_nodes = self.area_dict[area][dst_group]
        if len(dst_area_nodes)> 0:

            
            dst_area_nodes = self.area_dict[area][dst_group]
            area_r += 1
            dst_node = random.choices(dst_area_nodes)[0]
            
        else:
            area_r +=1
            complete = False
            dst_node = -1

        

        return dst_node, area_r, complete    

    def work_place_check(self,src_node, dst_group, dst_nodes, dst_nodes_bin):

        workplace = self.person_work_dict[src_node]
        # print(len(dst_area_nodes))
        dst_workplace_nodes = self.work_group_dict[workplace][dst_group]

        


        if len(dst_workplace_nodes) > 10:
            dst_node = random.choices(dst_workplace_nodes)[0]
        else:
            dst_node = random.choices(dst_nodes)[0]


        if self.fraction:
                
            if self.barabasi_dict[self.person_work_dict[src_node]][dst_group] and len(dst_workplace_nodes) > 10:
                dst_node = random.choices(self.barabasi_dict[self.person_work_dict[src_node]][dst_group])[0]
            elif len(dst_workplace_nodes) > 10:
                dst_node = dst_node = random.choices(dst_nodes)[0]
            else: 
                dst_node = random.choices(dst_nodes)[0]
            
        return dst_node

    def add_nodes_to_network(self,src_group, dst_group, src_node, dst_node):
        
        self.source.append(src_group)
        self.destination.append(dst_group)
    
        self.id_source.append(src_node)
        self.id_destination.append(dst_node)

        self.link_dictionary[src_node].add(dst_node)
        self.person_degree_count[dst_node] += 1

        self.symetry_dict[(src_group,dst_group)] += 1
                        
        self.edges_layd +=1
        
    
    def reciprocity_check(self,src_node, dst_node, src_group, dst_group, T = False):
        if self.reciprocity > np.random.uniform() and \
            src_node not in self.link_dictionary[dst_node] and \
                (dst_group, src_group) in self.symetry_dict and \
                    self.symetry_dict[(dst_group, src_group)] < self.max_dict[(dst_group, src_group)]:


                    self.add_nodes_to_network(dst_group, src_group, dst_node, src_node)

                    if src_group == dst_group and not T:
                        self.i += 1
                    return True
                    

    def transitivity_check(self,src_node,dst_node,dst_group):

    
                            
        # Look at the friends of the source node
        friends_dst = self.link_dictionary[dst_node]

        # Source node is the initial source node
        src_node_t = src_node

        for dst_node_t in friends_dst:

            if self.transitivity  < np.random.uniform():
                continue
            # New destination node is a random friend of the original destination node
            # dst_node_t =  random.choice(list(friends_dst))
            
            # Look at the groups of the nodes
            src_group_t = self.nodes_group[src_node_t]
            dst_group_t = self.nodes_group[dst_node_t]
                    

            # Check if new destination node isn't already a friend
            # Check if the node pair is inside the symetry dict, which means that bot groups have connections
            # Check if the amount of connections between the two is not already maximal

            if dst_node_t not in self.link_dictionary[src_node_t] and \
                (src_group_t, dst_group_t) in self.symetry_dict and \
                    self.symetry_dict[(src_group_t, dst_group_t)] < self.max_dict[(src_group_t, dst_group_t)]:

        
                if dst_group_t == dst_group:
                    self.i+=1
                
                self.add_nodes_to_network(src_group_t, dst_group_t, src_node_t, dst_node_t)
                
        

                if self.reciprocity_check(src_node_t, dst_node_t, src_group_t, dst_group_t, True):
                    if (dst_group_t == dst_group) and (dst_group_t == src_group_t):
                            self.i += 1
                        
                    
            
    def create_network(self):
   
        df = pd.DataFrame()

        df['source_id'] = self.id_source
        df['destination_id'] = self.id_destination
        df['source_group'] = self.source
        df['destination_group'] =self.destination


        return df

    def model_run(self,fraction,transitivity,reciprocity,specification, make_network = True):
        print(reciprocity, transitivity, fraction)
        self.fraction = fraction
        self.reciprocity = reciprocity
        self.transitivity = transitivity
        self.spatial = specification
        self.workplaces = round(specification)
        if specification != 0 and self.layer == 'buren': self.fraction= 1
        

        self.make_kf_for_spec(self.work_group_dict)
        
        if self.layer == 'familie' and self.spatial:

            self.household_family_correlation()

        for row in self.initial_list:

            # Identifies source and destination group
            src_group = row[0]
            dst_group = row[1]
            connections = row[2]
            
            self.group_connection[(src_group,dst_group)] = self.tot_connections

            self.tot_connections += connections

            if src_group in self.overig or dst_group in self.overig:
                continue
        
            
            
            self.i = 0    
            
            
               
            if (self.reciprocity or self.transitivity) and (src_group, dst_group) in self.symetry_dict:
        
                self.i = self.symetry_dict[(src_group, dst_group)] % (self.max_dict[(src_group, dst_group)]/10)
                # c += self.symetry_dict[(src_group, dst_group)]
                
                if self.symetry_dict[(src_group, dst_group)] >= self.max_dict[(src_group, dst_group)]:
                    self.i = connections
                    
                 

            # Initialize dictionary with the node id as key and initial edges, 1, as value
            dst_nodes = self.group_nodes[dst_group]
            src_nodes = self.group_nodes[src_group]

                
            # If Barabasi parameter is on take a sample to put in the initial bin (1 percent is standard)
            # ! Denk aan Barabasi in een school of op werk
            
            if self.fraction: 
                
                dst_nodes_bin = random.sample(dst_nodes, k=(math.ceil(len(dst_nodes)*self.fraction)))

        

            self.done_connections.add((src_group, dst_group))
            # print(connections)

            while self.i < connections:
            
                if self.edges_layd % 100000 == 0:
                    # print(symetry_dict)
                    # pass
                    print(self.edges_layd/self.total_edges)
                    print(self.edges_layd, len(self.source))

                area_r = 0
                c = 0
                while True:
                    
                    c+=1
                    # Take random source node and destination node based on the groups
                
                    
                    src_node = random.choices(src_nodes)[0]
                    dst_node = random.choices(dst_nodes)[0]
                    
            
                    # CHECK IN WHICH AREA HE IS FROM ==> MAKE DICTIONARY WHERE NODE IS KEY AND AREA IS VALUE
                    # CHOOSE A RANDOM OTHE NODE FROM THAT AREA WITH GROUP SPECIFICS ==> MAKE DICTIONARY WHERE AREA AND GROUP IS KEY AND LIST OF NODES IS VALUE
                    if self.layer == 'buren' and self.spatial:
                        
                            
                        # Get area of source group
                        area = self.node_area[src_node]
                        
                        if np.random.uniform() < 0.01:

                            if np.random.uniform()<0.5:
                                area += 1
                            else :
                                area -= 1
                                
                            area %= self.n_areas


            
                        # if area_r < 5:
                        #     area += area_r
                        area += area_r
                        area %= self.n_areas

                        dst_area_nodes = self.area_dict[area][dst_group]
                        if len(dst_area_nodes)> 0:
            
                            
                            dst_area_nodes = self.area_dict[area][dst_group]
                            area_r += 1
                            dst_node = random.choices(dst_area_nodes)[0]

                
                            # area_r += 1
                            # if np.random.uniform()<0.5:
                            #     area += np.random.randint(1,6)
                            # else :
                            #     area -= np.random.randint(1,6)
                            
                        else:
                            # dst_node = random.choices(dst_nodes)[0]

                            # if np.random.uniform()<0.5:
                            #     area += np.random.randint(1,6)
                            # else :
                            #     area -= np.random.randint(1,6)
                            area_r +=1
                            continue

                    #! We can also do this based on area code, for instance use for this 22 gebieden, iig voor school kids kan dit interresant zijn

                    elif self.layer == 'werkschool' and self.workplaces:
                        
                        dst_node = self.work_place_check(src_node, dst_group, dst_nodes, dst_nodes_bin)

                    

                    # Checks if the source and destination node are not the same and checks if they aren't already linked
                    if dst_node != src_node and dst_node not in self.link_dictionary[src_node]:
                        
                        # # If Barabasi append a random node to the bin
                        if self.fraction and not self.workplaces:
                            dst_nodes_bin.append(random.choices(dst_nodes)[0])
                
                            # Add the chosen node to the bin so the chosen node gets a higher weight 
                            if np.random.uniform() > self.fraction : dst_nodes_bin.append(dst_node)

                        elif self.fraction and self.workplaces and self.layer == 'werkschool':     

                            if self.work_group_dict[self.person_work_dict[src_node]][dst_group]:

                                r_node = random.choices(self.work_group_dict[self.person_work_dict[src_node]][dst_group])[0]
                                self.barabasi_dict[self.person_work_dict[src_node]][dst_group].append(r_node)
                                if np.random.uniform() > self.fraction: self.barabasi_dict[self.person_work_dict[src_node]][dst_group].append(dst_node)

                        self.add_nodes_to_network(src_group, dst_group, src_node, dst_node)

                        self.i += 1
                        # Appends both nodes to lists
        
                        
                        self.reciprocity_check(src_node, dst_node, src_group, dst_group)
                                
                        self.transitivity_check(src_node,dst_node,dst_group)
                    break
        print(self.edges_layd, len(self.source))

        if make_network:
            return self.create_network()