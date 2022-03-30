import pandas as pd

def hash_groups():
    '''
    Making a hash dictionary based on the groups
    Making the hash and rehash dictionaries
    '''

    # Initializing hash dictionaries
    hash_dict = {}
    rehash_dict = {}

    # Read oplniv dataframe
    df = pd.read_csv('./Data/tab_n_(with oplniv).csv')
    
    # Hash every group
    for i in range(df.shape[0]):
        group = df.iloc[i]
                
        age = group['lft']
        etnc = group['etngrp']
        gender = group['geslacht']
        education = group['oplniv']


        hash_dict[f'{age}, {etnc}, {gender}, {education}'] = i
        rehash_dict[i] = f'{age}, {etnc}, {gender}, {education}' 
    
    return hash_dict, rehash_dict