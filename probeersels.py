import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import pandas as pd

df = pd.read_csv('Data/NW_data/buren_nw.csv')
G = nx.from_pandas_edgelist(df, 'source_id', 'destination_id',['source_group', 'destination_group'] ,create_using=nx.DiGraph())

import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
# G = nx.karate_club_graph()
# compute the best partition
partion = community_louvain.best_partition(G)
