import pandas as pd


class Network:

    def __init__(self, total_nodes, nodes, total_edges, edges, source_nodes, destination_nodes, group_nodes):

        self.nodes = nodes
        self.edges = edges

        self.total_nodes = total_nodes
        self.total_edges = total_edges
        
        self.group_probabilities = {}
        # self.destination_nodes = destination_nodes
        # self.source_nodes = source_nodes
        # self.node_probabilities = {}

        def initialize_goup_probabilities(self):
            for group in group_nodes.keys():
                node_probability = {}

                for node in group_nodes[group]:
                    node_probability[node] = 1/len(group_nodes[group])
                    

            self.group_probabilities[group] = [] * len(group_nodes[group])


        def initialize_node_probabilities(self):
            for node in self.nodes:
                node.probability = 1/self.total_edges
                self.node_probabilities[node] = 1/self.total_edges

        def update_node_probability(self, node):
            self.node_probabilities[node] = node.links/self.total_edges
            node.probability = node.links/self.total_edges

        
        

    

if __name__ == 'main':
    pass