import multiprocessing as mp
import time

def sleepy_man(sec):
    print('Starting to sleep')
    time.sleep(1)
    print(sec)


if __name__ == '__main__':
    
    num_workers = mp.cpu_count()  

    pool = mp.Pool(num_workers)

    tasks = [2,3,4,5,3,4,5]
    for task in tasks:
        pool.apply_async(sleepy_man, args = (task,))

    pool.close()
    pool.join()