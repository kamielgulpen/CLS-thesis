

from asyncore import compact_traceback
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
from scipy import rand
import scipy.stats as stats


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

def spatial(group_dict, group_nodes, area_name):
    '''
    Makes 2 dictionaries: areas_groups, node_area

    areas_group: is the dictionary which takes as key area and group and returns the nodes in that area of the group
    node_area: returns a dictionary whith as key the node and value the area in which the node is present

    '''
    a = 0
    b = 0
    
    print(len(group_nodes))
    # Import spatial data (see spatial data.ipynb to see how csv is constructed)
    df_sp = pd.read_csv(f'.\Data\Spatial_data\spatial_data_{area_name}.csv')

    area_dict = {'22gebieden': 21, 'wijken': 98, 'buurten': 480}

    n_areas = area_dict[area_name]

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
            if len(areas[area][group]) < math.ceil(areas_df[area][group] * sum(areas_df[area].values())) and area < n_areas + 1:

                # Append node to area
                areas[area][group].append(node)
                areas_groups[area][node[1]].append(node[0])
                node_area[node[0]] = area
                i+=1
                
            else:
                area+=1
                
                if area > n_areas:
                    
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

def workplace_distribution(distribution, n_companies):
    n_companies = n_companies + 500
    if distribution == 'normal':
        lower, upper = 500, n_companies
        mu, sigma = n_companies/2, n_companies/4

        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        
    
    elif distribution == 'exponential':
        lower, upper, scale = 0, n_companies, n_companies/2
        X = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)

  
    distribution = X.rvs(861000).astype(int)
    
   

   
    return np.array(distribution)

def school_distribution(n_schools):
    # Je maakt een dictionary met een kans dat je op een school komt op basis van je ethnischiteit
    # Je maakt een 
    df_sp = pd.read_csv('./Data/Spatial_data/Ethnic_data.csv')
    school = {}
    for ethniciteit in df_sp['variabele'].unique():
        school[ethniciteit] = []
        for i in df_sp[df_sp['variabele'] == ethniciteit].itertuples():
   
            school[ethniciteit].extend([i[-1]] * int(i[-2]/2))

        random.shuffle(school[ethniciteit])

        print(len(school[ethniciteit]))
      
    return school

def workplace_dictionary(n_companies):
    # Wat hebben we nodig?
    # We geven een persoon een werkplek en een werkplek een persoon
    # Werkplek dictionary ziet er dan als volgt uit: dict[werkplek][group] = alle mensen die werken in die groep
    # En Je krijgt ook een dictionay van alle dict[persoon] = werkplek

    work_group_dict = {}
    for i in range(n_companies + 500):
        work_group_dict[i] = {}
        for j in range(240):
            work_group_dict[i][j] = []
    
    return work_group_dict
    


def initialize_nodes(hash_dict, area_name):
    '''
    Initilizes all the nodes for the network
    '''

    df = pd.read_csv('Data/tab_n_(with oplniv).csv')

    n_companies = 500

    # Basisscholen hebben alleen maar een niveau van 1
    # Kinderen op basisschool

    basisschool_n = 61918

    # Kinderen op middelbareschool kan niveau 1 en 2

    middelbareschool_n = 41561

    # Kinderen op MBO kan niveau 1 en 2

    MBO_n = 17320

    Total_n = basisschool_n + middelbareschool_n + MBO_n

    percentage_scholier = Total_n/df[df['lft'] == '[0,20)'].sum()['n'] 

    print(percentage_scholier)
    # Initialize list with nodes and nodes per group (example: Man,"[0,20)",1,Autochtoon)
    all_nodes = []
    group_nodes = {}
    nodes_group = {}

    work_sample = workplace_distribution('normal', n_companies)
    school_sample = school_distribution(100)

    df_sp = pd.read_csv(f'Data\Spatial_data\spatial_data_{area_name}.csv')
    
    work_group_dict = workplace_dictionary(n_companies)
    
    group_list = df_sp.groupby(['geslacht','etngrp', 'oplniv']).size().reset_index().to_numpy()[:, [0, 1, 2]]

    group_dict = {tuple(group) : [] for group in group_list}
    
    person_work_dict = {}

    person_degree_count = {}
    
    id = 0
 
    c = 0
    # Loops through all lines in the tab_n file and get group properties of each line
    for i in range(df.shape[0]):

        student_count = 0

        group = df.iloc[i]

        age = group['lft']
        etnc = group['etngrp']
        gender = group['geslacht']
        education = group['oplniv']
        n = int(group['n'])
        nodes = []
        
        hd = hash_dict[f'{age}, {etnc}, {gender}, {education}']


        # Makes n (the size of the group) nodes of the group
        for _ in range(n):
            
            x =  np.random.uniform() < 0.01
            if (age == '[0,20)' and education < 3 and student_count < (n * percentage_scholier)) or x:

               
                workplace = school_sample[etnc].pop()
                person_work_dict[id] = workplace
                work_group_dict[workplace][hd].append(id)
                
                if not x:
                    c+=1
                    student_count +=1
            else:           
                
                workplace = work_sample[id]
                person_work_dict[id] = workplace
                work_group_dict[workplace][hd].append(id) 

            node = (id, hd)
            nodes.append(node[0])
            all_nodes.append(id)
            group_dict[tuple([gender, etnc, education])].append(node)
            nodes_group[id] = hd
            person_degree_count[id] = 0
            id += 1

            

            
   
                
                
        # make a dictionary with as key the group properties and value a list of nodes of the group
        group_nodes[hd] = nodes


    return all_nodes, group_nodes, group_dict, nodes_group, work_group_dict, person_work_dict, person_degree_count

