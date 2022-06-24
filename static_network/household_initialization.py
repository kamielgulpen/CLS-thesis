import pandas as pd

# from node_initialization import spatial, initialize_nodes
import random
import numpy as np
import itertools

# * First way is by using the distribution provided by 
# Get the people in to 240 different bins (bins that are based on their characters)
# Get household distribution ==> Take person out of the data
# Make that household complete by taking a person out of the population which resembles
# Do this till all persons have a household
    

def make_households(hash_dict, all_nodes, group_nodes,area_dict, node_area):

    # Take a person
    # Check to which household it consists
    # Make the household complete
    # Add household to dictionary
    # * Do this till all agents are used



    df_nodes = pd.read_csv('Data/tab_n(with oplniv).csv')

    hh_df = pd.read_csv('Data/Household_data/DHS_household_.csv')
    
    hh_prob = pd.read_csv('Data/Household_data/ethnicity_probabilities.csv')


    prob = df_nodes.groupby(by='etngrp').sum().reset_index()[['etngrp', 'n']]

    # print(prob)
    normalized_df=(prob['n'])/prob['n'].sum()

    # print(normalized_df)
    prob['probability'] = normalized_df

    
    total_amount_agents = int(df_nodes.n.sum())

    source_nodes = []
    destination_nodes = []

    source_group = []
    destination_group = []

    households = {}
    
    added_nodes = set()
    i = 0

    household_ids = []
    household_sizes = []
            
    
    while len(added_nodes) != total_amount_agents:
        x = 1

        if i < 17401 :
            x = 5
        elif i < 17401 + 32883 :
            x = 4
        elif i < 17401 + 32883 + 43208 :
            x = 3
        elif i <= 17401 + 32883 + 43208 + 124787:
            x = 2
        else:
            break
        
        
        hh_df2  =hh_df[hh_df['aantalhh2'] == x]

        if x == 5:
            hh_df2  =hh_df[hh_df['aantalhh2'] > x - 1]  
 
        hh = hh_df2.sample(n=1)

        full_hh = hh_df2[hh_df2['nohhold'] == float(hh['nohhold'])]

        # print(full_hh)
        # First choose person based on age, education and gender
        # Than base its ethnicity on the distribution etnicity 
        etnc = random.choices(list(prob['etngrp']), weights = list(prob['probability']))[0]

        hh_list = []
        group_list = []
           
        for count, person in enumerate(full_hh.itertuples()):

            # print(person)


            Index, _, nohhold, geslacht, aantalhh, aantalki, positie, aantalhh2, age, education, _ = person

        
            education = int(education)
            if positie == 'child living at home' or count == 0:
               etnc = etnc
            else:
                etnc =  etnc
                probabilities = hh_prob[hh_prob['Unnamed: 0'] == etnc]
                etnc = random.choices(list(probabilities)[1:], weights = list(probabilities.iloc[0])[1:])[0]
                
            
            # First person adult
            group = group_nodes[hash_dict[f'{age}, {etnc}, {geslacht}, {education}']]

            # Second person
            if count > 0:
                group  = area_dict[area][hash_dict[f'{age}, {etnc}, {geslacht}, {education}']]

            if len(group) == 0:
                # hh_not_completed = True
                continue

            agent = random.choices(group, k=1)[0]

            if agent in hh_list:
                break

            hh_list.append(agent)
          
            group_list.append(hash_dict[f'{age}, {etnc}, {geslacht}, {education}'])

            if count == 0:
                area = node_area[agent]
            
        if len(hh_list) != x:
            continue
        
        for x, y in zip(hh_list,group_list):

            # print(hh_list,group_list)
            # print(x,y)
            group_nodes[y].remove(x)
            area_dict[area][y].remove(x)

        if i%1000 == 1:
            print('households = ', i, len(households))
            print('nodes = ', len(added_nodes))
        
            print('avg_household = ', len(added_nodes)/ i)

            print(len(household_ids), len(source_nodes))
            # print(np.mean(sizes))
            # print(np.mean(hh_sizes))

        if hh_list:
            
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
    



        source_nodes, destination_nodes, source_group, destination_group , household_ids, household_sizes = make_households(hh_df, group_nodes, prob, hh_etn_prob, total_amount_agents, hash_dict,area_dict, node_area)

        d = {'source_id': list(source_nodes), 'destination_id':list(destination_nodes), 'source_group': list(source_group), 'destination_group': list(destination_group), 'household_id': list(household_ids)}

        df_ = pd.DataFrame(d)


        df2_ = pd.DataFrame({'size': household_sizes})
        # df_.to_csv(f'Data/NW_data2/hh_test3.csv')

        df_.to_csv(f'../Data/Experiments7/huishouden_experiment_{i}.csv')