

from pyclbr import Class
from networkx.algorithms.centrality import group
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


def hash_groups():
    '''
    Making a hash dictionary based on the groups
    Making the hash and rehash dictionaries
    '''

    # Initializing hash dictionaries
    hash_dict = {}
    rehash_dict = {}

    # Read oplniv dataframe
    df = pd.read_csv('Data/tab_n_(with oplniv).csv')

    # Hash every group
    for i in range(df.shape[0]):
        group = df.iloc[i]
                
        age = group['lft']
        etnc = group['etngrp']
        gender = group['geslacht']
        education = group['oplniv']


        hash_dict[f'{age}, {etnc}, {gender}, {education}'] = i
        rehash_dict[i] = f'{age}, {etnc}, {gender}, {education}' 
    
    return hash_dict, rehash_dict

def spatial(group_dict, group_nodes):
    '''
    Makes 2 dictionaries: areas_groups, node_area

    areas_group: is the dictionary which takes as key area and group and returns the nodes in that area of the group
    node_area: returns a dictionary whith as key the node and value the area in which the node is present

    '''
    a = 0
    b = 0
    
    print(len(group_nodes))
    # Import spatial data (see spatial data.ipynb to see how csv is constructed)
    df_sp = pd.read_csv('.\Data\Spatial_data\spatial_data_wijken.csv')

    # get unique areas and etnicities and groups
    group_list = df_sp.groupby(['geslacht','etngrp', 'oplniv']).size().reset_index().to_numpy()[:, [0, 1, 2]]
    area_list = df_sp['code'].unique()
    

    
    
    groups = list(group_nodes.keys())

    # Initialize Dictionaries
    areas_groups = {}
    node_area = {}
    areas_df = {} 
    areas = {}

    # Loop through area's and initialize areas with a second dictionary
    for area in area_list:
        areas[area] = {}
        areas_df[area] = {}
        areas_groups[area] = {}

        # Give areas df a value (max amount of people represented in that area)
        # Initialize areas and group_areas with a list for each area
        for group in group_list:
            gender, etn, edu = group
            
            amount = df_sp[(df_sp['geslacht'] == gender) & (df_sp['oplniv'] == edu) & (df_sp['etngrp'] == etn)]
            areas_df[area][tuple(group)] = float(amount[amount['code'] == area]['percentage']/100)
            areas[area][tuple(group)] = []
            
        for group in groups:
            areas_groups[area][group] = []
    
    # Loop through the groups 
    for group in group_dict:
        
        group = tuple(group)
        # Shuffle areas and nodes to put the nodes randomly in the areas
        random.shuffle(group_dict[group])
        random.shuffle(area_list)

        # Initialize area and iterator
        area = 0
        i = 0
 
        x = True
        

        # Loop through all nodes with ethnicity etn
        while i < len(group_dict[group]):
            
            
            if area == 1 and x == True:
                x = False
                a += 1
                
       
                
#             print(len(group_dict[group]), i, group)
            # node is i_th node of the list
            
            node = group_dict[group][i]
            
            # Go to next area
            
#             print(len(group_dict[group])* areas_df[area][group])
            
    
            # Check if area is full and area is not greater than 22
            if len(areas[area][group]) < math.ceil(areas_df[area][group] * sum(areas_df[area].values())) and area < 99:

                # Append node to area
                areas[area][group].append(node)
                areas_groups[area][node[1]].append(node[0])
                node_area[node[0]] = area
                i+=1
                
            else:
                area+=1
                
                if area > 98:
                    
                    b += 1
                    print(b)
                    
                    rest = len(group_dict[group]) - i
                    
                    print(i/len(group_dict[group]))
                    tot = i
                    
                    for j in area_list:
                        percentage = areas_df[j][group]/tot
                        
                        areas_df[j][group] += math.ceil(rest * percentage)
                    
                    area = 0
                

    return areas_groups, node_area
    
def initialize_nodes(df, hash_dict):
    '''
    Initilizes all the nodes for the network
    '''
    

    # Initialize list with nodes and nodes per group (example: Man,"[0,20)",1,Autochtoon)
    all_nodes = []
    group_nodes = {}
    nodes_group = {}
    df_sp = pd.read_csv('Data\Spatial_data\spatial_data_wijken.csv')
    
    print(df_sp)

    group_list = df_sp.groupby(['geslacht','etngrp', 'oplniv']).size().reset_index().to_numpy()[:, [0, 1, 2]]

    group_dict = {tuple(group) : [] for group in group_list}
    

    id = 0
 

    # Loops through all lines in the tab_n file and get group properties of each line
    for i in range(df.shape[0]):
        group = df.iloc[i]

        
        age = group['lft']
        etnc = group['etngrp']
        gender = group['geslacht']
        education = group['oplniv']
        nodes = []
        
        hd = hash_dict[f'{age}, {etnc}, {gender}, {education}']


        # Makes n (the size of the group) nodes of the group
        for _ in range(int(group['n'])):
            

            node = (id, hd)
            nodes.append(node[0])
            all_nodes.append(id)
            group_dict[tuple([gender, etnc, education])].append(node)
            nodes_group[id] = hd

            id += 1

            
   
                
                
        # make a dictionary with as key the group properties and value a list of nodes of the group
        group_nodes[hd] = nodes


    return all_nodes, group_nodes, group_dict, nodes_group

