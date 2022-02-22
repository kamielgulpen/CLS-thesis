from itertools import groupby
import pandas as pd
import numpy as np
import scipy
from scipy.spatial import distance, distance_matrix
from scipy import optimize



def prob_function(d_ij, b, alpha):
    p_ij =1/( 1 + ((1/b * d_ij) ** alpha))
    
    return p_ij

def function(alpha, dij, pij):
    
    def f(b):
        pij - prob_function(dij, b, alpha)

    sol = optimize.newton_krylov(f,  0.1, 10)

    return sol

def get_pij(df_edges, df_nodes):

    # probability group= total_connections of group/connections with other group
    # probability node = (total_connections of group / total amount of persons in that group)/ 
    # (connections with other group/total amount of persons in that group)

    # E|k| = average degree between groups or network? I think between groups
    # We do no know alpha and b

    # f(b) = E|k|  - 1/N * ssummation of all probabilities with p(alpha, b)
    # because all probabilities are the same between the groups we can just say
    # f(b) = E|k| - 1/(N+M)/2 * N*M * prob(alpha, b)
    # This is between a group but for full network it will be different

    # Simulate small proportion for the whole network
    # Calculate it for each node instead of group

            #group1     group2      group3
    
    #group1  d=y        d=y         d=y
    #group2  ..         ..          ...
    #group3  ...        ..          ..


        #group1     group2      group3
    
    #group1  p=y        p=y         p=y
    #group2  ..         ..          ...
    #group3  ...        ..          ..


    # geslacht_src,lft_src,oplniv_src,etngrp_src,geslacht_dst,lft_dst,oplniv_dst,etngrp_dst
    df_edges['source_group'] = df_edges['geslacht_src'] + "_" +  df_edges['lft_src'] + '_' + df_edges['etngrp_src']  + '_' +df_edges['oplniv_src'].astype(str)
    df_edges['destination_group'] = df_edges['geslacht_dst'] + "_" +  df_edges['lft_dst'] + '_' + df_edges['etngrp_dst']  + '_' +df_edges['oplniv_dst'].astype(str)
    
    df_edges = df_edges.sort_values(by= ['source_group', 'destination_group'])
    new_df2 = pd.DataFrame()

    new_df = df_edges[['source_group', 'destination_group', 'n']].sort_values(by=['source_group'])

    new_df2[['geslacht_src','lft_src','oplniv_src','etngrp_src','geslacht_dst','lft_dst','oplniv_dst','etngrp_dst']] = df_edges[['geslacht_src','lft_src','oplniv_src','etngrp_src','geslacht_dst','lft_dst','oplniv_dst','etngrp_dst']].apply(lambda x: pd.factorize(x)[0])


    print(new_df2)

    source_group_df = new_df.groupby('source_group').sum().reset_index().sort_values(by=['source_group'])
    destination_group_df = new_df.groupby('destination_group').sum().reset_index().sort_values(by=['destination_group'])


    print(source_group_df)

    print(source_group_df)

    probabilities = []
    distances = []
    coordinates = []
    for i, j in zip(new_df.iterrows(), new_df2.iterrows()):

        row = i[1]
        sg = row['source_group']
        dg = row['destination_group']
        n = row['n']

        tot_n_s = source_group_df[source_group_df['source_group'] == sg]['n']
        
        prob = n/tot_n_s

        probabilities.append(prob)

        row2 = j[1]

        coordinates_s = row2['geslacht_src'], row2['lft_src'], row2['oplniv_src'], row2['etngrp_src']
        coordinates_d =  row2['geslacht_dst'], row2['lft_dst'], row2['oplniv_dst'], row2['etngrp_dst']

        
        dist = np.linalg.norm(np.array(coordinates_s)-np.array(coordinates_d))        
     
        distances.append(dist)

    return np.array(distances), np.array(probabilities)
     



    # pd.factorize(s)

df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')

#

'''
Initialize edges for multiple layers
'''

layer ='werkschool'
        
    
df_edges = pd.read_csv(f'Data/tab_{layer}.csv')

print('jall')
dij, pij = get_pij(df_edges, df_nodes)

alpha = 2

function(alpha, dij, pij)