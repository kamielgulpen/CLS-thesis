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
from scipy.stats import skew
import random
font_names = [f.name for f in fm.fontManager.ttflist]
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2


def analysis(layer, df, experiment, fraction,reciprocity, transitivity, specification):
    
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



    # df = df[(~df['source_group'].isin(overig) ) & (~df['destination_group'].isin(overig))]


    group_dict = {
                    'werkschool':{'same':'6', 'other':['116','11']}, 
                    'familie':{'same':'39', 'other':['39','75']},
                    'huishouden':{'same':'6', 'other':['81','156']},
                    'buren':{'same':'6', 'other':['11','6']}, 

                }

    same = group_dict[layer]['same']
    first = group_dict[layer]['other'][0]
    second = group_dict[layer]['other'][1]

    df2 = df[(df['source_group'] == int(same)) & (df['destination_group'] == int(same))]
            
    tuples = [tuple([a, b]) for a, b in zip(df2['source_id'], df2['destination_id'])]
    Gm2 = ig.Graph.TupleList(tuples, directed = True)

    import os
    path = f"Experiment_{layer}"
    if experiment == 0:
    # define the name of the directory to be created
        
        
        # define the access rights

        try:
            os.mkdir(path)

        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)

        
        try:
            os.mkdir(f'{path}/Figures')

        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)

    
        f= open(f"{path}/output.txt","w+")
        f.write("N,R,T,M,A,S,S_am,max,skew_a,skew_g\n")
        x= open(f"{path}/input.txt","w+")

        x.write("fraction,reciprocity,transitivity,specification\n")
       
    tuples = [tuple([a, b]) for a, b in zip(df['source_id'], df['destination_id'])]
    Gm = ig.Graph.TupleList(tuples, directed = True)
    
    print('nodes are added')


    N = Gm.vcount()

    print('Modularity is added')
    partition = la.find_partition(Gm, la.ModularityVertexPartition)


    M = Gm.modularity(partition)

    print('assortivity is added')
    A = Gm.assortativity_degree()


    # rn = np.random.randint(N, size=(100,2))
    print(len(Gm.vs.indices))
    rn = random.sample(list(Gm.vs.indices), 200)


    x = rn[:100]

    y = rn[100:]

    sp = Gm.shortest_paths(x, y, mode="in")

    sp = np.array(sp).flatten()

    

    sp = np.delete(sp, np.where(sp == np.inf))

    
    if len(sp) < 30:
        sp = [0]

    amount_S = len(sp)
    print('reciprocity is added')
    R = Gm.reciprocity()

    print('small-world is added')
    S = np.mean(sp)


    print('transitivity is added')
    T = Gm.transitivity_undirected()
    f = open(f"{path}/output.txt","a")

    m = max(Gm.indegree())

    sk = skew(Gm.indegree())

    sk_g = skew(Gm2.indegree())
    f.write(f"{N},{R},{T},{M},{A},{S},{amount_S},{m},{sk},{sk_g}\n")

    x= open(f"{path}/input.txt","a")

    x.write(f"{fraction},{reciprocity},{transitivity},{specification}\n")

    import matplotlib.pyplot as plt
    sn.histplot(data=Gm.indegree(), bins = 50)

    plt.xlabel('Node indegree')
    plt.ylabel('Number of nodes')
    plt.tight_layout()


    plt.savefig(f'{path}/Figures/in_degree_whole_layer_zo_{experiment}.pdf')

    plt.clf()
    import matplotlib.pyplot as plt

    sn.histplot(data=Gm2.indegree(), bins = 50)

    plt.xlabel('Node degrees')
    plt.ylabel('Number of nodes')
    plt.tight_layout()

    plt.savefig(f'{path}/Figures/in_degree_within_group_{same}_zo_{experiment}.pdf')
    # plt.show()

    