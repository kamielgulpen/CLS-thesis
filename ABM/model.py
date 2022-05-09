from tokenize import group
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import Grid
from mesa.datacollection import DataCollector
import networkx as nx
import random
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from torch import int64
from agent import Citizen
import pandas as pd
import itertools
from mesa.datacollection import DataCollector
import math
import numpy as np
from scipy.spatial import distance

class Amsterdam_social_network(Model):


    def __init__(
        self,
        agents,
        household_network,
        family_network,
        neighbour_network,
        workschool_network,
        nodes_group,
        rehash_dict,
        hash_dict,
        max_iters,
        dummy = False
        
    ):
        super().__init__()
        
        self.agents = agents
        self.household_network = household_network
        self.family_network = family_network
        self.neighbour_network =  neighbour_network
        self.workschool_network = workschool_network
        self.nodes_group = nodes_group
        self.rehash_dict = rehash_dict
        self.hash_dict = hash_dict
        self.max_iters = max_iters
        self.dummy = dummy
        self.coordinates_character = {}
  
        self.iteration = 0
        self.max_id = 0
        self.deaths = 0
        self.births = 0
        
        self.schedule = RandomActivation(self)
        model_reporters={
            "agent_count": lambda m: self.get_agent_count(m), 
            "deaths": lambda m : self.get_deaths_count(m), 
            "births": lambda m : self.get_births_count(m),
            "family": lambda m : self.get_network(m.family_network),
            "household": lambda m : self.get_network(m.household_network),
            "work/school": lambda m : self.get_network(m.workschool_network),
            "neighbour": lambda m : self.get_network(m.neighbour_network),
            }

        agent_reporters={
            "age": lambda a: a.age,
            "age_group": lambda a: a.age_group,
            "education": lambda a: a.education,
            "ethnicity": lambda a: a.ethnicity,
            "gender": lambda a: a.gender,
            "group": lambda a:  a.group_id
        }

        self.dc = DataCollector(model_reporters=model_reporters,agent_reporters=agent_reporters )

        self.all_agents = []
        self.groups = self.make_group_dict()

        self.age_distribution = self.age_distibution_per_age()

        self.mortality_rates = self.mortality_rate()

        self.coordinates_dict = self.make_characteristics_coordinates()
        
        self.child_probabilities = self.child_probability()
        self.layer_probability_dict = self.get_probability_dictionary()
      
        for agent in agents:
            
            # For full model
            group_number = nodes_group[agent]
            group = rehash_dict[group_number]

            # For small model
            if dummy:
                group_number = random.randint(0,239)
                group = rehash_dict[group_number]

            
            citizen = Citizen(
                agent,
                self,
                group,
                )
            
            
            self.schedule.add(citizen)

            self.max_id += 1
            self.all_agents.append(citizen)
            # add citizen id to the list
           
    def step(self):
        """
        Advance the model by one step and collect data.
        """
        print('step')
 
        self.dc.collect(self)
        self.deaths = 0
        self.births = 0
        
        # self.get_group_distribution()
        self.schedule.step()
        
        self.iteration += 1
    

    def make_group_dict(self):
        '''
        Initializes a dictionary with all the groups 

        :returns: dict -- keys: int - group number, values: set - empty set
        '''

        group_dictionary = {}
        for i in range(2400):
            group_dictionary[i] = set()
        
    
        return group_dictionary

    def age_distibution_per_age(self):
        '''
        This method makes a distribution for each age group

        :returns: dict -- keys: age group, values: probabilities
        '''
        #* Bron: https://www.cbs.nl/nl-nl/visualisaties/dashboard-bevolking/bevolkingspiramide
        ages_groups = [
        '[0,20)',
        '[20,30)',
        '[30,40)',
        '[40,50)',
        '[50,60)',
        '[60,70)',
        '[70,80)',
        '[80,120]'
        ]

        age_frames = {}
        for i in range(8):
            
            df = pd.read_csv(f'Data/Other/Leeftijdsopbouw_group_{i}.csv')
            
            fractions_men = df['fractie_Mannen'].tolist()
            fractions_women = df['fractie_Vrouwen'].tolist()

            ages = df['Leeftijd'].tolist()
            age_frames[ages_groups[i]] = ({'Age' :ages , 'Man': fractions_men, 'Vrouw': fractions_women})

        
            
        return age_frames
        

    def mortality_rate(self):
        '''
        Calculates the mortality probability of a person

        :returns: nested dict -- keys: age and gender, value: probability
        '''
        #* Bron: https://www.cbs.nl/nl-nl/visualisaties/dashboard-bevolking/bevolkingsgroei/overlijden

        # Import mortality data from CBS
        df = pd.read_csv(f'Data/Other/Leeftijd overledenen.csv')

        # Initialize dictionary
        mortality_rate = {}

        # Loops over the dataframe and create a nested dictionary with gender and age as keys
        # And probability as value
        for i in df.iterrows():
            
            i = i[1]
            d = {'Man': float(i['percentage_m']), 'Vrouw': float(i['percentage_v'])}

            for j in range(int(i['Leeftijd_min']), int(i['Leeftijd_max'])):
               
                
                mortality_rate[j] = d
        
        return mortality_rate
            
    def child_probability(self):
        '''
        Calculates the child probability of a person

        :returns: nested dict -- keys: age and gender, value: probability
        '''

        #* Bron: https://www.cbs.nl/nl-nl/nieuws/2019/19/leeftijd-moeder-bij-eerste-kind-stijgt-naar-29-9-jaar
        #* Bron: https://www.cbs.nl/nl-nl/visualisaties/dashboard-bevolking/levensloop/kinderen-krijgen

        # Import mortality data from CBS
        df = pd.read_csv(f'Data/Other/child_probability.csv')

        # Initialize dictionary
        child_probability = {}

        # Loops over the dataframe and create a nested dictionary with gender and age as keys
        # And probability as value
        for i in df.iterrows():
            
            i = i[1]
            d = {'Man': float(i['Mannen']) * (10187/(1 - 0.4256772401018754))/861000, 'Vrouw': float(i['Vrouwen']) * (10187/(1 - 0.4256772401018754))/861000}

            for j in range(int(i['Leeftijd_min']), int(i['Leeftijd_max'])):
               
                
                child_probability[j] = d 
        
        return child_probability

           
    def make_characteristics_coordinates(self):
        '''
        This method makes from the list of characteristics: gender, age, education, ethnicity a coordinate.
        Where each characterisic is resembled by a dimension. Ethnicity is a categocical variable with more than 2
        categories and therefore it is represented in 5 dimensions where each dimension is represented by a ethnicity.

        :returns: dict -- where the keys are the group id's and the values are the coordinates
        '''
        
        distance = {}

        distances = []

        ethn_dist = []
           
        count = 0
        for i in range(2):
            for j in range(80):
                for k in range(3):
                    for l in np.identity(5):
                        distance[count] = [i,j,k]
                        ethn_dist.append(l)
                        distance[count].extend(l)
                        distance[count] = np.array(distance[count])

                        distances.append(tuple(distance[count]))
                        self.coordinates_character[tuple(distance[count])] = count
                        count += 1

        combinations = np.array(list(itertools.product(distances, distances)))
        combinations_ethn = np.array(list(itertools.product(ethn_dist, ethn_dist)))
        
        distances = np.absolute(combinations[:,1] - combinations[:,0])
        ethn_dist =  np.array(np.absolute((combinations_ethn[:,1] - combinations_ethn[:,0])).any(axis=1), dtype=int).reshape((5760000,1))
        

        distances = np.hstack((distances, ethn_dist))

        return distances

    def get_distance(self, coordinates_agent_1, coordinates_agent_2, weights = False, distance_type='Euclidian'):
        '''
        This method calculates the distance in homophily between two agents

        :param coordinates of first agent: coordinates_agent_1
        :type list
        :param coordinates of second agent: coordinates_agent_2
        :type list
        :param type of distance: distance_type
        :type str

        :returns: float or list -- distance between two nodes
        '''
        
        # Create two numpy arrays from the coordinates
        # a and b reprecent a list of the coordiantes of two agents

        if distance_type == 'Manhattan':
            distance = [0]*4

            
            distance[0] = np.linalg.norm(self.coordinates_dict[coordinates_agent_1][0]-self.coordinates_dict[coordinates_agent_2][0])
            distance[1] = np.linalg.norm(self.coordinates_dict[coordinates_agent_1][1]-self.coordinates_dict[coordinates_agent_2][1])
            distance[2] = np.linalg.norm(self.coordinates_dict[coordinates_agent_1][2]-self.coordinates_dict[coordinates_agent_2][2])
            distance[3] = round(np.linalg.norm(self.coordinates_dict[coordinates_agent_1][3:]-self.coordinates_dict[coordinates_agent_2][3:]),0)
            

        elif distance_type == 'Euclidian':
            # Get euclidean distance  
            # return distance.cdist([coordinates_agent_1], [coordinates_agent_2], 'euclidean')
            weights = True
            if not weights:
                distance = np.linalg.norm(self.coordinates_dict[coordinates_agent_1]-self.coordinates_dict[coordinates_agent_2])
            else:
                distance = [0]*4

                
                distance[0] = np.linalg.norm(self.coordinates_dict[coordinates_agent_1][0]-self.coordinates_dict[coordinates_agent_2][0])
                distance[1] = np.linalg.norm(self.coordinates_dict[coordinates_agent_1][1]-self.coordinates_dict[coordinates_agent_2][1])
                distance[2] = np.linalg.norm(self.coordinates_dict[coordinates_agent_1][2]-self.coordinates_dict[coordinates_agent_2][2])
                distance[3] = np.linalg.norm(self.coordinates_dict[coordinates_agent_1][3:]-self.coordinates_dict[coordinates_agent_2][3:])
                

        else:
            ValueError("distance_type has to be Manhattan or Euclidean")
            

        return distance

    def get_connection_probability(self,  d_ij, layer, weights = False):
        '''
        This method calculates the probability of a connection 
        between two agents based on their disance

        :param --  distance between agents: d_ij
        :type list or float
        :param layer of the network: layer
        :type string
        :param type of weights of the characteristics: weights
        :type bool

        :returns: float -- distance between two nodes
        '''

        weights = True
        if layer == 'buren':
            alpha, a, beta = 0.8476302729791598, 0.00010058247905386748, 1.000000002209136e-05
            X1,X2,X3,X4 = np.array([-6.54545759e-04, -9.37835720e-05, -2.86801454e-04, -3.44638837e-04]) 

        elif layer == 'werk/school':
            alpha, a, beta = 2.0229665059979687, 0.0003592290518517478, 0.0014684449493073915
            X1,X2,X3,X4 = np.array([-4.21151980e-04, -6.08069159e-05, -1.73513776e-04, -2.18553946e-04])

        elif layer == 'huishouden':
            alpha, a, beta = 4.837506803289496, 1.8657612786520468e-05, 0.03531088386379604
            X1,X2,X3,X4 = np.array([-6.55219306e-04, -9.55356137e-05,  4.49769922e-04, -2.62261278e-04])

        else:
            alpha, a, beta = 5.1629729202556405, 8.549049832600861e-05, 0.04806617965081799
            X1,X2,X3,X4 = np.array([-1.07855419e-03, -3.27486360e-05,  7.02328689e-06, -2.82151353e-04])

        if weights == False:
            p_ij = ( a+ (beta *d_ij)) * np.exp(- alpha * d_ij)
        else : 
            
            d_ijg = d_ij[:,0]
            d_ija = d_ij[:,1]/(80/7)
            d_ijo = d_ij[:,2] 
            d_ije = d_ij[:,3] 

            p_ij = ( a+ (beta * (X1 * d_ije  + X3 * d_ijg + X2 * d_ija + X4* d_ijo))) * np.exp(- alpha * (X1 * d_ije  + X3 * d_ijg + X2 * d_ija + X4 * d_ijo))


        return p_ij

    def make_new_agent(self, parent):
        '''
        This method makes a new agent and adds the agent to all necessary networks

        :param --  distance between agents: parent
        :type agent object

        '''

        self.max_id +=1

        gender = random.choice(['Man', 'Vrouw'])

        group = f'[0,20), {parent.ethnicity}, {gender}, 1'
        
  
        citizen = Citizen(
                self.max_id, # new agent id
                self, 
                group, # Check group of parents
                born = True
                )
        

        self.schedule.add(citizen)

        self.family_network.add_node(self.max_id)
        self.household_network.add_node(self.max_id)
        self.workschool_network.add_node(self.max_id)
        self.neighbour_network.add_node(self.max_id)

        
        self.family_network.add_edges_from([tuple([self.max_id,i] ) for i in list(self.family_network[parent.unique_id])])  # using a list of edge tuples
        self.household_network.add_edges_from([tuple([self.max_id,i] ) for i in list(self.household_network[parent.unique_id])])
        self.neighbour_network.add_edges_from([tuple([self.max_id,i] ) for i in list(self.neighbour_network[parent.unique_id])]) 

    # def get_group_distribution(self):

    #     distribution = []
    #     for i in range(240):
            
    #         distribution.append(len(self.groups[i]))

    #     # Normalised [0,1]

    #     distribution = np.array(distribution)
    #     pd_groups = (distribution/ np.sum(distribution))




    #     self.pd_groups = pd_groups
    #     return distribution
    
    def get_probability_dictionary(self):
        '''
        This method calculates the probability of a connection 
        between two agents based on their disance

        :param --  distance between agents: d_ij
        :type list or float
        :param layer of the network: layer
        :type string
        :param type of weights of the characteristics: weights
        :type bool

        :returns: float -- distance between two nodes
        '''
        layer_probability_dict = {}
        

        for layer in ['huishouden', 'familie', 'buren', 'werkschool']:
         
            p_ij = self.get_connection_probability(self.coordinates_dict, layer)

            layer_probability_dict[layer] = p_ij.reshape((2400,2400))
            #         probability_dict[(i,j)] = self.get_distance(i,j)
            # layer_probability_dict[layer] = probability_dict

        return layer_probability_dict
       
    def add_to_network(self, agent1, agent2, layer):
        '''
        This method makes connection within the network between two agents

        :param --  agent: agent1
        :type agent object
        :param --  agent: agent2
        :type agent object
        :param layer of the network: layer
        :type string

        '''

        if layer == 'familie':
            self.family_network.add_edge(agent1.unique_id, agent2)  # using a list of edge tuples
     
        elif layer == 'huishouden':
            self.household_network.add_edge(agent1.unique_id, agent2)
  
        elif layer == 'buren':
            self.neighbour_network.add_edge(agent1.unique_id, agent2)
           

        else: 
            self.workschool_network.add_edge(agent1.unique_id, agent2)
    

    def remove_from_network(self, agent1, agent2, layer):
       
        '''
        This method removes connection within the network between two agents

        :param --  agent: agent1
        :type agent object
        :param --  agent: agent2
        :type agent object
        :param layer of the network: layer
        :type string

        '''
        if layer == 'familie':

            if agent2 == '':
                agent2 = list(self.family_network[agent1.unique_id])[0]

            self.family_network.remove_edge(agent1.unique_id, agent2)  # using a list of edge tuples
          
        elif layer == 'huishouden':
            if agent2 == '':
                agent2 =  list(self.household_network[agent1.unique_id])[0]

            self.household_network.remove_edge(agent1.unique_id, agent2)  # using a list of edge tuples
 
        elif layer == 'buren':
            if agent2 == '':
                agent2 = list(self.neighbour_network[agent1.unique_id])[0]

            self.neighbour_network.remove_edge(agent1.unique_id, agent2)  # using a list of edge tuples

        else: 
            if agent2 == '':
                agent2 = list(self.workschool_network[agent1.unique_id])[0]

            self.workschool_network.remove_edge(agent1.unique_id, agent2)  # using a list of edge tuples


    @staticmethod
    def get_agent_count(model):
        """
        Helper method to count agents.
        """
        
        return len( model.schedule.agents)

    @staticmethod
    def get_deaths_count(model):
        """
        Helper method to count deaths.
        """
        
        return model.deaths

    @staticmethod
    def get_births_count(model):
        """
        Helper method to count births.
        """
        return model.births

    @staticmethod
    def get_network(G):
        """
        Helper method to count births.
        """
        return nx.to_pandas_edgelist(G)
