class Group:
    def __init__(self, id, nodes, initial_probabilities):
        self.id = id
        self.nodes = nodes

        self.zip_iterator = zip(nodes, initial_probabilities)
        self.link_probabilities = dict(self.zip_iterator)
