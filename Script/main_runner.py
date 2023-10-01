from SALib.sample import saltelli
from main import main
import sys
import multiprocessing as mp
import numpy as np

if __name__ == '__main__':
    
    num_workers = mp.cpu_count()  -2

 
    num_workers =  1

    layer = sys.argv[1]

    problem = {
            'num_vars': 4,
            'names': ['x1', 'x2', 'x3', 'x4'],
            'bounds': [[0,1], [0,1],[0,1],[0,1]]
            }

    # Generate samples
    param_values = saltelli.sample(problem, 100)
    # s = np.where(param_values[:,2] > 0.5, 1, 0)

    x = 0 
    area_name = {0: '22gebieden', 0.33:  '22gebieden', 0.67: 'wijken', 1: 'buurten'}
    


    # param_values = [[0.21972656, 0.09667969, 0.],[0.67675781, 0.09667969, 0.51855469]]

    print(param_values)

    processes = []


    arrays = np.array_split(param_values, num_workers)

    arrays = [array[5:] for array in arrays]

    
    print(arrays)
    for i in arrays:
        print(i)
        p = mp.Process(target=main, args=(layer, i, area_name,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()