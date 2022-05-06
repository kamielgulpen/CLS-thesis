
import matplotlib.pyplot as plt
import importlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import sys
import seaborn as sns


from model import Amsterdam_social_network
sys.path.append('./')
# C:\Users\KGulp\Documents\Computational Science\CLS_thesis\static_network\node_initialization.py
# from CLS_thesis.static_network.node_initialization import hash_groups
from static_network import hash_groups

from static_network import node_initialization

import networkx as nx

#TODO before thursday 04/28/2022
    # Make decision event such as moving out and getting a partner (and getting a child)

    # Make sample out of the population accordingly to how they are distributed

    # Make new network based on probability function

    # Make probability function based on how you meet people ==> example Neighbours will only meet people in their neighbourhood etc.

    # Add weird connection based on boltman sim annealing
    # Determine the Gamma of the dristributions
    # Do a sensitivity analysis

hash_dict, rehash_dict = hash_groups.hash_groups()

'''
Initialize nodes 
'''


df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')
all_nodes, group_nodes, etn_dict, nodes_group = node_initialization.initialize_nodes(df_nodes, hash_dict)




# '''

# '''

# df1 = pd.read_csv('./Data/NW_data/overlap.csv')
# df2 = pd.read_csv('./Data/NW_data/hh_test.csv')
# df3 = pd.read_csv('./Data/NW_data/buren_test.csv')
# df4 = pd.read_csv('./Data/NW_data/werkschool_test.csv')

# print('''
# -------------------------------------------------------------- 
# Data is imported
# -------------------------------------------------------------- 
#  ''')
# family_network  = nx.from_pandas_edgelist(df1, 'source_id', 'destination_id' ,create_using=nx.DiGraph())
# household_network  = nx.from_pandas_edgelist(df1, 'source_id', 'destination_id' ,create_using=nx.DiGraph())
# neighbour_network =  nx.from_pandas_edgelist(df1, 'source_id', 'destination_id' ,create_using=nx.DiGraph())
# workschool_network = nx.from_pandas_edgelist(df1, 'source_id', 'destination_id' ,create_using=nx.DiGraph())
# dummy = False
# print('''
# -------------------------------------------------------------- 
# Networks are made
# --------------------------------------------------------------
# ''')

family_network = nx.erdos_renyi_graph(1000, .005, seed=None)
household_network = nx.erdos_renyi_graph(1000, .002)
neighbour_network = nx.erdos_renyi_graph(1000, .01, seed=None, directed=True)
workschool_network = nx.erdos_renyi_graph(1000, .01, seed=None, directed=True)
dummy = True
all_nodes = range(1000)

model = Amsterdam_social_network(
    agents= all_nodes,
    household_network=household_network,
    family_network=family_network,
    neighbour_network = neighbour_network,
    workschool_network= workschool_network,
    nodes_group = nodes_group,
    rehash_dict = rehash_dict,
    hash_dict = hash_dict,
    max_iters=3000,
    dummy=dummy
)


for t in range(10):
    model.step()

model_df = model.dc.get_model_vars_dataframe()
agent_df = model.dc.get_agent_vars_dataframe()

agent_df.to_csv('s.csv')

model_df.to_csv('m.csv')


    # print(agent_df.loc[i])
f_df = model_df.iloc[9]['work/school']
 

    
a_df = agent_df.loc[9]
    # print(agent_df.loc[i]['age'].max())
    
    # print(agent_df.loc[i]['education'].value_counts())
    # print('\n')


    # print(agent_df[agent_df['Step'] == i]['age'].mean())
# df = pd.read_csv('../overlap.csv')
# Data\NW_data2\huishouden_nw_barabasi=0.csv

group_s = []
group_d = []

# print(agent_df.loc[3]['group'].index)

print(f_df)
print(a_df)

# def categorise(row, a_df):  
#     print(type(row['target']))
#     return a_df['age_group'].iloc[int(row['target'])]


# f_df['t_age'] = f_df.apply(lambda row: categorise(row, a_df), axis=1)

# # print(list(f_df['source']))
# # for i in list(f_df['source']):
# #     group_s.append(nodes_group[i])

# # for j in list(f_df['target']):
# #     print(j)
# #     group_d.append(nodes_group[j])

# f_df.to_csv('age_diff.csv')