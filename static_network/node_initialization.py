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
    df = pd.read_csv('./Data/tab_n_(with oplniv).csv')

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

def spatial(etn_dict, group_nodes):
    '''
    Makes 2 dictionaries: areas_groups, node_area

    areas_group: is the dictionary which takes as key area and group and returns the nodes in that area of the group
    node_area: returns a dictionary whith as key the node and value the area in which the node is present

    '''

    # Import spatial data (see spatial data.ipynb to see how csv is constructed)
    df_sp = pd.read_csv('Data\Spatial_data\spatial_data.csv')

    # get unique areas and etnicities and groups
    area_list = df_sp['area_code'].unique()
    etn_list = df_sp['variabele'].unique()
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
        for etn in etn_list:
            areas_df[area][etn] = int(df_sp[(df_sp['variabele'] == etn) & (df_sp['area_code'] == area)]['amount_per_area'])
            areas[area][etn] = []
        for group in groups:
            areas_groups[area][group] = []

    # Loop through the 5 ethnicities (A, M, T, S, O)
    for etn in etn_dict:

        # Shuffle areas and nodes to put the nodes randomly in the areas
        random.shuffle(etn_dict[etn])
        random.shuffle(area_list)

        # Initialize area and iterator
        area = 0
        i = 0

        # Loop through all nodes with ethnicity etn
        while i < len(etn_dict[etn]):
            
            # node is i_th node of the list
            node = etn_dict[etn][i]

            # Check if area is full and area is not greater than 22
            if len(areas[area][etn]) < areas_df[area][etn] and area < 22:

                # Append node to area
                areas[area][etn].append(node)
                areas_groups[area][node[1]].append(node[0])
                node_area[node[0]] = area
                i+=1

            # Go to next area
            else:
                area+=1
                if area > 21:
                    area = 0

    return areas_groups, node_area
def initialize_nodes(df, hash_dict):
    '''
    Initilizes all the nodes for the network
    '''

    # Initialize list with nodes and nodes per group (example: Man,"[0,20)",1,Autochtoon)
    all_nodes = []
    group_nodes = {}
    df_sp = pd.read_csv('Data\Spatial_data\spatial_data.csv')
    etn_list = df_sp['variabele'].unique()
    etn_dict = {etn : [] for etn in etn_list}
    
    
    areas = {}
    areas_df = {}
    

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
        for _ in range(math.ceil((group['n']))):
             
            node = (id, hd)
            nodes.append(node[0])
            all_nodes.append(id)
            etn_dict[etnc].append(node)

            id += 1
            
   
                
                
        # make a dictionary with as key the group properties and value a list of nodes of the group
        group_nodes[hd] = nodes




    return all_nodes, group_nodes, etn_dict
