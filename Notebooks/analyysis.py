import glob

import networkx as nx
import pandas as pd
import math
import numpy as np

import pandas as pd
import igraph as ig
import pylab 

import powerlaw

import matplotlib.pyplot as plt
# from py3plex.core import multinet

from itertools import chain
import random
import seaborn as sn

def get_type_node(G, id):
    
    print(list(G[id].values()))
    return list(G[id].values())[0]['source_group']

def hash_groups():
    hash_dict = {}
    rehash_dict = {}

    df = pd.read_csv('Data/tab_n_(with oplniv).csv')

    for i in range(df.shape[0]):
        group = df.iloc[i]
                
        age = group['lft']
        etnc = group['etngrp']
        gender = group['geslacht']
        education = group['oplniv']


        hash_dict[f'{age}, {etnc}, {gender}, {education}'] = i
        rehash_dict[i] = f'{age}, {etnc}, {gender}, {education}' 
    
    return hash_dict, rehash_dict

hash_dict, rehash_dict = hash_groups()

rehash_dict[12]

df = pd.DataFrame()
for filepath in glob.iglob('Data/NW_data2/*.csv'):
    values = []

    print(filepath)
    p = pd.read_csv(filepath)

    g = ig.Graph.TupleList(p[['source_id','destination_id']].itertuples(index=False), directed=True, weights=False)
    # g.transitivity_undirected()
    print('loaded')
    edge_list = g.get_edgelist()

    edge_list = np.array(edge_list)

    max_e = np.max(edge_list.flatten())
    values.append(max_e)

    rn = np.random.randint(max_e, size=(100,2))

    x = rn[:,0]
    y = rn[:,1]
    
    sp = g.shortest_paths(x, y, mode='in')

    print('shortest path')
    sp = np.array(sp)

    sp1 = sp.copy()
    sp1[sp == np.inf] = 100
    # sp = sp.flatten()

    # np.mean(sp1)
    # # list(sp).count(np.inf)
    # # len(sp)

    values.append(np.mean(sp.flatten()))

    values.append(np.mean(sp1.flatten()))

    values.append(g.assortativity_degree())

    degrees = g.degree(mode = 'in')
    sn.histplot(data=degrees, bins = 100)

    plt.savefig(f'{filepath}_histogram.png')

    plt.close()
    reciprocity = g.reciprocity()
    values.append(reciprocity)


    x = sorted((list(p['destination_id'].value_counts())))

    from collections import Counter

    y = list(Counter(x).values())
    x = list(Counter(x).keys())
    # # df_s['destination_id'].value_counts()

    plt.plot(x,y)

    plt.xlabel('in_degree')
    plt.ylabel('frequency')
    plt.yscale('log')
    plt.xscale('log')

    plt.savefig(f'{filepath}_loglog.pdf')

    plt.close()

    df[filepath] = values

df.to_csv('all_values.csv')