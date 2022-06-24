import pandas as pd
from static_network.Edge_Initialization import Initialize_edges
from analysis import analysis
import sys
import multiprocessing as mp
from SALib.sample import saltelli
import numpy as np

def roundQuarter(x):
        return round(round(x * 3) / 3.0,2)

def main(layer, param_values):
    x = 0

    # if layer == 'buren':
    #     edge_innit1 = Initialize_edges(layer, 0)

    #     edge_innit2 = Initialize_edges(layer, 0.33)
    
    #     edge_innit3 = Initialize_edges(layer, 0.67)
    
    #     edge_innit4 = Initialize_edges(layer, 1)
    # else:
    #     edge_innit = Initialize_edges(layer, 0)
    edge_innit = Initialize_edges(layer, 0)
    for count, i in enumerate(param_values):

        f,r,t = i

        s = 0


        
        edge_innit.initialize_arrays()
        df = edge_innit.model_run(fraction=f, reciprocity=r, transitivity = t ,specification = s)
        analysis(layer= layer, df = df, experiment = count, fraction=f ,reciprocity=r,specification = s,transitivity = t)


if __name__ == "__main__":

    df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')
    df_edges = pd.read_csv(f'Data/tab_buren.csv')

    layer = sys.argv[1]
    num_workers = mp.cpu_count() - 2 

    num_workers = 1
    layer = sys.argv[1]

    problem = {
            'num_vars': 3,
            'names': ['x1', 'x2', 'x3'],
            'bounds': [[0,1], [0,1],[0,1]]
            }

    # Generate samples
    param_values = saltelli.sample(problem, 100)
    # s = np.where(param_values[:,2] > 0.5, 1, 0)
    


    param_values

    processes = []


    arrays = np.array_split(param_values, num_workers)

    arrays = [array for array in arrays]



    

    for i in arrays:
        print(i)
        p = mp.Process(target=main, args=(layer, i))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()




# area_dict, node_area = [0,1]

# fraction,reciprocity,s = [1,1,0]

# specify = round(s)
# workplaces = specify

# s = 1
# t = 1
# f = 1
# r = 1



# edge_innit = Initialize_edges(layer, 1)

# df = edge_innit.model_run(fraction=f, reciprocity=r, transitivity = t ,specification = s)

# df.to_csv(f'Experiment_{layer}/experiment_{1}_Transitivity.csv')

# # exit()
