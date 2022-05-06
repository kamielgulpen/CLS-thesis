import math
from turtle import update

import numpy as np
import networkx as nx
from mesa import Agent
import random
import time
from scipy.spatial import distance

class Citizen(Agent):

    def __init__(
        self,
        unique_id,
        model,
        group,
        born =False
    ):

        super().__init__(unique_id, model)

        self.group = group
        self.group_id = -1
        self.character_coordinates = [-1,-1,-1,-1,-1,-1,-1,-1]
        self.deceased = False
        self.born = born
        self.unwrap_group(self.group)

        self.get_age(self.age_group, self.gender)
        self.get_group_id()
        self.get_character_coordinates()

        self.child_probability = self.model.child_probabilities[self.age]
        
  
    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """

        # self.check_mortality()
        # self.update_child_probability()
        # self.make_child()


        # if self.age == 16 or self.age == 18:
        #     self.update_education()
        
        self.update_age()

        self.get_age_group()

        self.get_character_coordinates()
        
        # if self.model.iteration >0:
        #     self.meet_people()

        # print(self.group_id, self.child_probability)
    
    def get_age_group(self):
        
        if self.age > -1 and self.age < 20:
            self.age_group = '[0,20)' 
        elif self.age > 19 and self.age < 30:
            self.age_group = '[20,30)'
        elif self.age > 29 and self.age < 40:
            self.age_group = '[30,40)'
        elif self.age > 39 and self.age < 50:
            self.age_group = '[40,50)'
        elif self.age > 49 and self.age < 60:
            self.age_group = '[50,60)'
        elif self.age > 59 and self.age < 70:
            self.age_group = '[60,70)'
        elif self.age > 69 and self.age < 80:
            self.age_group = '[70,80)'
        else:
            self.age_group = '[80,120]'
        
    def update_age(self):
        
        self.age += 1

    def check_mortality(self):
        
        
        if np.random.uniform() < self.model.mortality_rates[self.age][self.gender] or self.age > 120:
            self.deceased = True
            # print(self.age)
            # print(self.unique_id)
          
            self.model.family_network.remove_node(self.unique_id)
            self.model.household_network.remove_node(self.unique_id)
            self.model.workschool_network.remove_node(self.unique_id)
            self.model.neighbour_network.remove_node(self.unique_id)

            self.model.schedule.remove(self)

            self.model.deaths += 1

    def unwrap_group(self, group):
        
        # print(group)
        self.age_group, self.ethnicity, self.gender, self.education = group.split(', ')
        
        self.education = int(self.education)
    def get_age(self, age_group, gender):

        if self.born:
            self.age = 0
            return 0

        probabilities = self.model.age_distribution[age_group][gender]
        ages = self.model.age_distribution[age_group]['Age']
    
        self.age = random.choices(ages, probabilities)[0]



    def get_group_id(self):
        self.group_id = self.model.hash_dict[f'{self.age_group}, {self.ethnicity}, {self.gender}, {self.education}']
          
    def get_character_coordinates(self):
        self.character_coordinates = self.model.coordinates_dict[self.group_id]

    def update_child_probability(self):
        self.child_probability = self.model.child_probabilities[self.age][self.gender]

    def make_child(self):
   
        if self.child_probability > np.random.uniform():

            self.model.make_new_agent(self)

            self.model.births +=1 

    def update_education(self):
        #* bron https://digitaal.scp.nl/ssn2020/onderwijs/
        #* https://www.cbs.nl/nl-nl/nieuws/2019/33/verschil-levensverwachting-hoog-en-laagopgeleid-groeit/opleidingsniveau

        if self.age == 16 and 0.8 > np.random.uniform() and self.education == 1:
            self.education += 1

        elif self.age == 18 and self.education == 2 and 0.5125 > np.random.uniform():

            self.education +=1
    
    def meet_people(self):

        #! The probability function does not work as not all groups are represented equally therfore there must be a slight correction for people with a different ethnicity
        #! Check the connections of each ethnicity
        #! Je kan 2 dingen doen of probability corrigeren of per ethniciteit evenveel mensen naar voren brengen om een connectie te maken
        
        # k = int(self.model.dc.get_model_vars_dataframe()['agent_count'].iloc[self.model.iteration]/100)

        d = 0
        for layer in  ['buren', 'werkschool', 'huishouden', 'familie']:
                
            # agents =random.choices(self.model.all_agents, k=k)

            for i in range(240):
                p = self.model.layer_probability_dict[layer][(self.group_id,i)]
                uniform =  np.random.rand(len(self.model.groups[i]))
                for count, j in enumerate(self.model.groups[i]):
                    
                    if p > uniform[count]:
                
                        self.model.add_to_network(self, j.unique_id, layer)

                        self.model.remove_from_network(self, '', layer)
                
           
             
                


               

    #TODO before thursday 04/28/2022
        # Make decision event such as moving out and getting a partner (and getting a child)
        # Make sample out of the population accordingly to how they are distributed
        # Make new network based on probability function
        # Make probability function based on how you meet people ==> example Neighbours will only meet people in their neighbourhood etc.
        # Add weird connection based on boltman sim annealing
        # Determine the Gamma of the dristributions
        # Do a sensitivity analysis
        # Normalize homophily parameter
        
