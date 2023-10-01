import pandas as pd
from static_network.Edge_Initialization import Initialize_edges
from analysis import analysis
import sys
import multiprocessing as mp
from SALib.sample import saltelli
import numpy as np
import time



def roundQuarter(x):
        return round(round(x * 3) / 3.0,2)

def main(layer, param_values):
    
    for count, i in enumerate(param_values):

        df_total = pd.DataFrame()
        for layer in ['werkschool', 'buren', 'huishouden', 'familie']:
            layer = 'familie'
            # get the start time
            st = time.process_time()
            f,r,t = i
            f,r,t = 1,1,1
            edge_innit = Initialize_edges(layer, 0)
            edge_innit.initialize_arrays()

            df = edge_innit.model_run(fraction=f, reciprocity=r, transitivity = t ,specification = 0)

            df.to_csv('x.csv')
            # exit()

            df_total = pd.concat([df_total,df])
            # get the end time
            et = time.process_time()

            # get execution time
            res = et - st
            print('CPU Execution time:', res, 'seconds')
            analysis(layer= layer, df = df, experiment = count, fraction=f ,reciprocity=r,specification = 0,transitivity = t)
            exit()
        layer = 'total'
        analysis(layer= layer, df = df_total, experiment = count, fraction=f ,reciprocity=r,specification = 0,transitivity = t)


if __name__ == "__main__":

    df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')
    df_edges = pd.read_csv(f'Data/tab_buren.csv')

    layer = sys.argv[1]
    num_workers = mp.cpu_count() - 6
    num_workers = 1


    problem = {
            'num_vars': 3,
            'names': ['x1', 'x2', 'x3'],
            'bounds': [[0,1], [0,1],[0,1]]
            }

    # Generate samples
    param_values = saltelli.sample(problem, 68)
    # s = np.where(param_values[:,2] > 0.5, 1, 0)
    
    np.savetxt('param_values',param_values)
    
    param_values = np.loadtxt('param_values')

    divided_param_values = int(param_values.shape[0]/4)
    # Take a certain amount of the param values
    param_values = param_values[0:divided_param_values]

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