import pandas as pd
from  hash_groups import hash_groups
import random
import numpy as np
import itertools

# * First way is by using the distribution provided by 
# Get the people in to 240 different bins (bins that are based on their characters)
# Get household distribution ==> Take person out of the data
# Make that household complete by taking a person out of the population which resembles
# Do this till all persons have a household
    
def initialize_nodes(df, hash_dict):
    '''
    Initilizes all the nodes for the network
    '''
    
    # Initialize list with nodes and nodes per group (example: Man,"[0,20)",1,Autochtoon)
    all_nodes = []
    group_nodes = {}

    all_nodes_t = set()
    df_sp = pd.read_csv('Data\Spatial_data\spatial_data_22gebieden.csv')
    
    group_list = df_sp.groupby(['geslacht','etngrp', 'oplniv']).size().reset_index().to_numpy()[:, [0, 1, 2]]
    print(group_list)
    group_dict = {tuple(group) : [] for group in group_list}
    

    id = 0
 

    # Loops through all lines in the tab_n file and get group properties of each line
    for i in range(df.shape[0]):
        group = df.iloc[i]

        
        age = group['lft']
        etnc = group['etngrp']
        gender = group['geslacht']
        education = group['oplniv']
        nodes = []
        
        hd = (f'{age}', f'{etnc}', f'{gender}', f'{education}')

        
        # Makes n (the size of the group) nodes of the group
        for _ in range(int(group['n'])):

            node = (id, hd)
            nodes.append(node[0])
            all_nodes.append(id)
            group_dict[tuple([gender, etnc, education])].append(node)

            id += 1

            all_nodes_t.add(node)

                
        
        # make a dictionary with as key the group properties and value a list of nodes of the group
        group_nodes[hd] = nodes


        
    return all_nodes, group_nodes, group_dict, all_nodes_t


def make_households(df, hh_df, group_nodes, prob, hh_prob, total_amount_agents, all_nodes_t):

    # Take a person
    # Check to which household it consists
    # Make the household complete
    # Add household to dictionary
    # * Do this till all agents are used

    source_nodes = []
    destination_nodes = []

    source_group = []
    destination_group = []

    households = {}
    
    added_nodes = set()
    i = 0

    household_ids = []
    household_sizes = []

    

            
    print(total_amount_agents)
    while len(added_nodes) != total_amount_agents:
        hh = hh_df.sample(n=1)

        aantal = float(hh['aantalhh2'])

    
        if i < 20000 and aantal < 4:
            
            continue

        elif i < 50000 and aantal < 3:
            continue
       
        
        full_hh = hh_df[hh_df['nohhold'] == float(hh['nohhold'])]

        # print(full_hh)
        # First choose person based on age, education and gender
        # Than base its ethnicity on the distribution etnicity 
        etnc = random.choices(list(prob['etngrp']), weights = list(prob['probability']))[0]

        hh_list = []
        group_list = []
        
        for count, person in enumerate(full_hh.itertuples()):


            Index, _, nohhold, geslacht, aantalhh, aantalki, positie, aantalhh2, age, education, _ = person

            education = int(education)
            if positie == 'child living at home' or count == 0:
               etnc = etnc
            else:
                etnc =  etnc
                probabilities = hh_prob[hh_prob['Unnamed: 0'] == etnc]
                etnc = random.choices(list(probabilities)[1:], weights = list(probabilities.iloc[0])[1:])[0]
                
            
            # First person adult
            group = group_nodes[(f'{age}', f'{etnc}', f'{geslacht}', f'{education}')]

            if len(group) == 0:
                # hh_not_completed = True
                continue

            agent = random.choices(group, k=1)[0]

            hh_list.append(agent)
          
            group_list.append(hash_dict[f'{age}, {etnc}, {geslacht}, {education}'])

            
           
            group_nodes[(f'{age}', f'{etnc}', f'{geslacht}', f'{education}')].remove(agent)
           

      


        if i%1000 == 1:
            print('households = ', i, len(households))
            print('nodes = ', len(added_nodes))
        
            print('avg_household = ', len(added_nodes)/ i)

            print(len(household_ids), len(source_nodes))
            # print(np.mean(sizes))
            # print(np.mean(hh_sizes))

        if hh_list:
            
           
            # print(hh_list)
            households[i] = hh_list
            i+=1

        

       
        household_sizes.append(len(hh_list))
        added_nodes.update(hh_list)

        if len(hh_list) < 2:
            continue

     
        nodes = (np.array(list(itertools.combinations(hh_list, 2))))
        
        source_nodes.extend(nodes[:,0])
        destination_nodes.extend(nodes[:,1])

        source_nodes.extend(nodes[:,1])
        destination_nodes.extend(nodes[:,0])

        group_nodes2 = (np.array(list(itertools.combinations(group_list, 2))))


        source_group.extend(group_nodes2[:,0])
        destination_group.extend(group_nodes2[:,1])

        source_group.extend(group_nodes2[:,1])
        destination_group.extend(group_nodes2[:,0])

        household_ids.extend([i] * len(group_nodes2[:,0]) * 2)



    return source_nodes, destination_nodes, source_group, destination_group, household_ids, household_sizes


if __name__ == '__main__':
    
    hash_dict, rehash_dict = hash_groups()

    df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')

    hh_df = pd.read_csv('Data/Household_data/DHS_household_.csv')

    hh_etn_prob = pd.read_csv('Data/Household_data/ethnicity_probabilities.csv')
    
    all_nodes, group_nodes, group_dict, all_nodes_t = initialize_nodes(df_nodes, hash_dict)

    # print(list(group_nodes.keys()))

    prob = df_nodes.groupby(by='etngrp').sum().reset_index()[['etngrp', 'n']]

    # print(prob)
    normalized_df=(prob['n'])/prob['n'].sum()

    # print(normalized_df)
    prob['probability'] = normalized_df

    
    total_amount_agents = int(df_nodes.n.sum())

    source_nodes, destination_nodes, source_group, destination_group , household_ids, household_sizes = make_households(df_nodes,hh_df, group_nodes, prob, hh_etn_prob, total_amount_agents, all_nodes_t)

    d = {'source_id': list(source_nodes), 'destination_id':list(destination_nodes), 'source_group': list(source_group), 'destination_group': list(destination_group), 'household_id': list(household_ids)}

    df_ = pd.DataFrame(d)


    df2_ = pd.DataFrame({'size': household_sizes})
    df_.to_csv(f'Data/NW_data2/hh_test2.csv')

    df2_.to_csv('Data/NW_data2/hh_test2_sizes.csv')