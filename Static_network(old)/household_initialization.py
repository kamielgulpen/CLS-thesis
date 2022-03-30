import itertools
import pandas as pd
import numpy as np
from wandb import agent
from hash_groups import hash_groups
import random
import math

import time
class Household:

    def __init__(self, id, size):
        self.id = id
        self.size = size

        self.agents = {}

    # def check_constraints(self, group, group_dictionary, symetric_dictionary):

    #     # print('1')

    #     # return True
    #     # print('3')

    #     for roomie_group in self.groups:
    #         if (group, roomie_group) not in symetric_dictionary:
    #             return False

    #         # print('4',  group_dictionary[(group, roomie_group)], symetric_dictionary[(group, roomie_group)])
    #         # if group_dictionary[(group, roomie_group)] <=  symetric_dictionary[(group, roomie_group)]:
    #             return False

    #         # print('5')
    #     return True

    def add_roomie(self, agent, group, group_dictionary, i):
        
        if i == 'swap':
            self.agents[agent[0]] = agent[1]

            return True

        if i > 0:
            self.agents[agent] = group
            return True

        for roomie_group in self.agents.values():
            if (roomie_group, group) not in group_dictionary:
                return False

        self.agents[agent] = group

        return True

    def remove_roomie(self, agent, r_agents):
        if agent not in self.agents:
            print(self.agents, agent, r_agents)
        self.agents.pop(agent)


def initialize_households():
    households = []
    hh_1 = (3697117 / (4946326 + 3097117)) * 467208
    hh_2 = (2640717 / (4946326 + 3097117)) * 467208
    hh_3 = (740588 / (4946326 + 3097117)) * 467208
    hh_4 = (758249 / (4946326 + 3097117)) * 467208
    hh_5 = (106772 / (4946326 + 3097117)) * 467208

    hh_values = [
        hh_1,
        hh_1 + hh_2,
        hh_1 + hh_2 + hh_3,
        hh_1 + hh_2 + hh_3 + hh_4,
        hh_1 + hh_2 + hh_3 + hh_4 + hh_5
    ]

    index = 0
    for i in range(1, 457208):

        if i < hh_values[index]:
            size = index + 1
        else:

            index += 1
            size = index + 1
        if size == 5:
            size = np.inf

        households.append(Household(i, size))

    return households


def initialize_nodes(hash_dict, households, stats_dict):
    '''
    Initilizes all the nodes for the network
    '''
    df = pd.read_csv('./Data/tab_n(with oplniv).csv')

    full_households = []

    # Initialize list with nodes and nodes per group (example: Man,"[0,20)",1,Autochtoon)
    all_nodes = []
    group_nodes = {}
    df_sp = pd.read_csv('./Data/Spatial_data/spatial_data_22gebieden.csv')

    group_list = df_sp.groupby(['geslacht', 'etngrp', 'oplniv']).size(
    ).reset_index().to_numpy()[:, [0, 1, 2]]
    print(group_list)
    group_dict = {tuple(group): [] for group in group_list}

    id = 0

    # Loops through all lines in the tab_n file and get group properties of each line
    for i in range(df.shape[0]):
        group = df.iloc[i]

        age = group['lft']
        etnc = group['etngrp']
        gender = group['geslacht']
        education = group['oplniv']
        nodes = []

        hd = hash_dict[f'{age}, {etnc}, {gender}, {education}']

        # Makes n (the size of the group) nodes of the group
        # for _ in range(int(group['n']/10)):
        for _ in range(int(group['n'])):
            i = 0
            while True:
                household_i = random.randint(0, len(households)-1)
                household = households[household_i]

                if not household.add_roomie(id, hd, stats_dict, i):
                    i += 1
                    continue

                if household.size == len(household.agents):
                    full_households.append(household)
                    households.pop(household_i)                   
                break

            node = (id, hd)
            nodes.append(node[0])
            all_nodes.append(id)
            group_dict[tuple([gender, etnc, education])].append(node)

            id += 1

            if id % 10000 == 0:
                print(id)

        # make a dictionary with as key the group properties and value a list of nodes of the group
        group_nodes[hd] = nodes
        print(len(households), len(full_households))
    full_households.extend(households)

    return full_households


def get_stats_household(hash_dict, households):
    df = pd.read_csv('./Data/tab_huishouden.csv')

    true_stats = {}
    manu_stats = {}
    edges_tru = 0
   

    for row in df.iterrows():

        row = row[1]

        dst = hash_dict[f"{row['lft_dst']}, {row['etngrp_dst']}, {row['geslacht_dst']}, {row['oplniv_dst']}"]
        src = hash_dict[f"{row['lft_src']}, {row['etngrp_src']}, {row['geslacht_src']}, {row['oplniv_src']}"]

        true_stats[(src, dst)] = int(row['n'])
        manu_stats[(src, dst)] = 0

        edges_tru += int(row['n'])



    return true_stats, manu_stats


def get_arti_stats(manu_stats, stats_dict, households):

    for i in range(240):
        for j in range(240):
            manu_stats[(i,j)] = 0
            if (i,j) not in stats_dict:
                stats_dict[(i,j)] = 0


    for household in households:

        combinations = list(itertools.combinations(household.agents.values(), 2))

    
        for combi in combinations:
            manu_stats[combi] += 1
    
 
    return manu_stats, stats_dict

        
def get_score(arti_stats, stats_dict, old_stats = 0, new = False, changed_values = False):

    score = 0
    old_score = 0

    if new:
        for key in changed_values:
            score += abs(stats_dict[key] - arti_stats[key])

            # print(arti_stats[key], old_stats[key])
            old_score += abs(stats_dict[key] - old_stats[key])
        return score, old_score

    for key in stats_dict:
        score += abs(stats_dict[key] - arti_stats[key])

    
    return score

def acceptance_probability(cost, temperature):
    """
    Calculate probability of accepting new cost
    """
    if cost < 0:
        return 1
    else:
        p = np.exp(- (cost) / temperature)
        return p

def swap_swap(arti_stats, stats_dict, households):


    # Get old score 
    old_score = get_score(arti_stats, stats_dict)

    old_dict = arti_stats.copy()
    old_households = households.copy()

    c = 0
    while True:
        c+=1
        # Take two households
        hhs = [random.choice(households) for _ in range(2)]

        # Take two random agents from household

        if not hhs[1].agents.items() or not hhs[0].agents.items():
            continue
        
    
        r_agents = [random.choice(list(household.agents.items())) for household in hhs]

        if r_agents[0] == r_agents[1]:
            continue
        # Swap households
        hhs[1].add_roomie(r_agents[0],0, 0, 'swap')
        hhs[0].add_roomie(r_agents[1],0, 0 , 'swap')

        hhs[1].remove_roomie(r_agents[1][0], r_agents)
        hhs[0].remove_roomie(r_agents[0][0], r_agents)

        # Get new score

        # Check whitch groups the change concerns
        changed_groups = []
        for i, group in enumerate(hhs[0].agents.values()):
            if i == len(hhs[0].agents.values()):
                continue

            arti_stats[(r_agents[1][1], group)] += 1
            arti_stats[(group, r_agents[1][1])] += 1   

            
            arti_stats[(r_agents[0][1], group)] -= 1
            arti_stats[(group, r_agents[0][1])] -= 1

            changed_groups.extend([(group, r_agents[1][1]), (group, r_agents[1][1]), (r_agents[0][1], group), (group, r_agents[0][1])])

        for i, group in enumerate(hhs[1].agents.values()):
            if i == len(hhs[1].agents.values()):
                continue

            arti_stats[(r_agents[0][1], group)] += 1
            arti_stats[(group, r_agents[0][1])] += 1   

            
            arti_stats[(r_agents[1][1], group)] -= 1
            arti_stats[(group, r_agents[1][1])] -= 1

            changed_groups.extend([(r_agents[0][1], group), (group, r_agents[0][1]), (r_agents[1][1], group), (group, r_agents[1][1])])
        # Add or substract the new values to the groups


        # Check if old score is better than new score
        cost, old_cost = get_score(arti_stats, stats_dict, old_dict ,True, changed_groups) 

       

            # check if next state to be accepted
        ap = acceptance_probability((cost-old_cost), 1.0e-200)

        if np.random.uniform() < ap:
            old_score = (old_score - old_cost) + cost
            old_households = households
            old_dict = arti_stats.copy()
           
           
        else:
            households = old_households
            arti_stats = old_dict.copy()
         

        # elif random.uniform(0, 1) < math.exp(- (old_score-new_score) / temp):
                       
        # exit()

        if c % 1000 == 0:

           
            print(old_score)
        
    # Is old score is better change back

    # If new score is better keep changes




if __name__ == '__main__':

    hash_dict, rehash_dict = hash_groups()
    households = initialize_households()

    stats_dict, manu_stats = get_stats_household(hash_dict, households)
    households = initialize_nodes(hash_dict, households, stats_dict)

    arti_stats, stats_dict = get_arti_stats(manu_stats,stats_dict, households)

    print(sum(arti_stats.values()))
    print(sum(stats_dict.values()))
    swap_swap(arti_stats, stats_dict, households)

