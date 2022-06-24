import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import pandas as pd
import leidenalg as la
import seaborn as sn
import numpy as np
optimiser = la.Optimiser()
import os
import matplotlib as mpl
import matplotlib.font_manager as fm#  Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

overig = [0, 1, 2, 18, 19, 20, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45,
 46,
 47,
 51,
 52,
 53,
 57,
 58,
 59,
 63,
 64,
 65,
 66,
 67,
 68,
 99,
 100,
 101,
 105,
 106,
 107,
 108,
 109,
 110,
 132,
 133,
 134,
 144,
 145,
 146,
 180,
 181,
 182]


for i in [6,7,10]:
    print(i)
    rootdir = f'..\Data\Experiments\Experiments{i}'
    print(rootdir)
    nodes = []
    assortivities = []
    modularities = []
    reciprocities = []
    small_world = []
    transitivity = []
    # werkschool_experiment_0.csv
    # df = pd.read_csv(f'../Data/Experiments/Experiments{i}/werkschool_experiment_0.csv')
    # print(list(os.walk(rootdir)))
    for subdir, dirs, files in os.walk(rootdir):
        
        for file in files:
            print(file)
            if file[0] == 's' or file[0] == "d" or file[0] == "i":
                continue

            path =  os.path.join(subdir,file)
            
        
            df = pd.read_csv(path)
            print('df is imported')
            tuples = [tuple([a, b]) for a, b in zip(df['source_id'], df['destination_id'])]
            Gm = ig.Graph.TupleList(tuples, directed = True)

            print('Graph is generated')
            nodes.append(Gm.vcount())

            print('nodes are added')
            partition = la.find_partition(Gm, la.ModularityVertexPartition)
            modularities.append(Gm.modularity(partition))

            print('Modularity is added')
            assortivities.append(Gm.assortativity_degree())

            print('assortivity is added')
            rn = np.random.randint(Gm.vcount(), size=(100,2))



            x = rn[:,0]

            y = rn[:,1]

            sp = Gm.shortest_paths(x, y, mode='in')

            sp = np.array(sp).flatten()

            # # sp = list(sp[sp == np.inf] = 0.0)

            # sp = list(sp).remove(np.inf)


            reciprocities.append(Gm.reciprocity())
            print('reciprocity is added')


            small_world.append(np.mean(sp))
    #         print('small-world is added')
            transitivity.append(Gm.transitivity_undirected())
            print('transitivity is added')
    
            df2 = pd.DataFrame()
            df2['nodes'] = nodes
            df2['assortivity'] = assortivities
            df2['modularity'] = modularities
            df2['reciprocity'] = reciprocities
            df2['transitivity'] = transitivity
            df2['small-world'] = small_world

            df2.to_csv(f'{subdir}/statistics_{file[-5]}.csv')
            
        #     break
        # break

# experiment = ['1','6','8', '9']

# experiment = ['2','6','8', '9']

# experiment = ['4','5','7', '10']


# layer = ['werkschool', 'buren', 'huishouden', 'familie'] 

# n_df = pd.DataFrame()
# n_df_zo = pd.DataFrame()
# for i, j in zip(experiment,layer):    
    
#     experiment = i
#     layer = j


#     group_dict = {
#                     'werkschool':{'same':'6', 'other':['116','11']}, 
#                     'familie':{'same':'39', 'other':['39','75']},
#                     'huishouden':{'same':'6', 'other':['81','156']},
#                     'buren':{'same':'6', 'other':['11','6']}, 

#                  }

#     df = pd.read_csv(f'../Data/Experiments/Experiments{experiment}/{layer}_experiment_0.csv')

#     df_zo = df[(~df['source_group'].isin(overig) ) & (~df['destination_group'].isin(overig))]
    
#     n_df_zo = pd.concat([n_df_zo,df_zo])
#     n_df = pd.concat([n_df,df])


# print('made it!')

# tuples = [tuple([a, b]) for a, b in zip(n_df_zo['source_id'], n_df_zo['destination_id'])]
# Gm = ig.Graph.TupleList(tuples, directed = True)


# print('nodes are added')
# print(Gm.vcount())

# print('Modularity is added')
# partition = la.find_partition(Gm, la.ModularityVertexPartition)
# print(Gm.modularity(partition))

# print('assortivity is added')
# print(Gm.assortativity_degree())


# rn = np.random.randint(Gm.vcount(), size=(100,2))



# x = rn[:,0]

# y = rn[:,1]

# sp = Gm.shortest_paths(x, y, mode='in')

# sp = np.array(sp).flatten()

# # # sp = list(sp[sp == np.inf] = 0.0)

# # sp = list(sp).remove(np.inf)


# print('reciprocity is added')
# print(Gm.reciprocity())

# print('small-world is added')
# print(np.mean(sp))


# print('transitivity is added')
# print(Gm.transitivity_undirected())

# print('Max indegree')
# print(max(Gm.indegree()))



# import glob

# import networkx as nx
# import pandas as pd
# import math
# import numpy as np

# import pandas as pd
# import igraph as ig
# import pylab 

# import powerlaw

# import matplotlib.pyplot as plt
# # from py3plex.core import multinet

# from itertools import chain
# import random
# import seaborn as sn

# def get_type_node(G, id):
    
#     print(list(G[id].values()))
#     return list(G[id].values())[0]['source_group']

# def hash_groups():
#     hash_dict = {}
#     rehash_dict = {}

#     df = pd.read_csv('Data/tab_n_(with oplniv).csv')

#     for i in range(df.shape[0]):
#         group = df.iloc[i]
                
#         age = group['lft']
#         etnc = group['etngrp']
#         gender = group['geslacht']
#         education = group['oplniv']


#         hash_dict[f'{age}, {etnc}, {gender}, {education}'] = i
#         rehash_dict[i] = f'{age}, {etnc}, {gender}, {education}' 
    
#     return hash_dict, rehash_dict

# hash_dict, rehash_dict = hash_groups()

# rehash_dict[12]

# df = pd.DataFrame()
# for filepath in glob.iglob('Data/NW_data2/*.csv'):
#     values = []

#     print(filepath)
#     p = pd.read_csv(filepath)

#     g = ig.Graph.TupleList(p[['source_id','destination_id']].itertuples(index=False), directed=True, weights=False)
#     # g.transitivity_undirected()
#     print('loaded')
#     edge_list = g.get_edgelist()

#     edge_list = np.array(edge_list)

#     max_e = np.max(edge_list.flatten())
#     values.append(max_e)

#     rn = np.random.randint(max_e, size=(100,2))

#     x = rn[:,0]
#     y = rn[:,1]
    
#     sp = g.shortest_paths(x, y, mode='in')

#     print('shortest path')
#     sp = np.array(sp)

#     sp1 = sp.copy()
#     sp1[sp == np.inf] = 100
#     # sp = sp.flatten()

#     # np.mean(sp1)
#     # # list(sp).count(np.inf)
#     # # len(sp)

#     values.append(np.mean(sp.flatten()))

#     values.append(np.mean(sp1.flatten()))

#     values.append(g.assortativity_degree())

#     degrees = g.degree(mode = 'in')
#     sn.histplot(data=degrees, bins = 100)

#     plt.savefig(f'{filepath}_histogram.png')

#     plt.close()
#     reciprocity = g.reciprocity()
#     values.append(reciprocity)


#     x = sorted((list(p['destination_id'].value_counts())))

#     from collections import Counter

#     y = list(Counter(x).values())
#     x = list(Counter(x).keys())
#     # # df_s['destination_id'].value_counts()

#     plt.plot(x,y)

#     plt.xlabel('in_degree')
#     plt.ylabel('frequency')
#     plt.yscale('log')
#     plt.xscale('log')

#     plt.savefig(f'{filepath}_loglog.pdf')

#     plt.close()

#     df[filepath] = values

# df.to_csv('all_values.csv')