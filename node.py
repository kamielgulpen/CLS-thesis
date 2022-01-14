class Node:

    def __init__(self, id, age, etnicity, education, gender, amount):
        self.id = id
        self.age = age
        self.etnicity = etnicity
        self.education = education
        self.gender = gender
        self.amount = amount

        
        self.links = []
        self.probability = 0

        # def initialize_probability(self, total_links):
        #     self.probability = 1/total_links

        # def update_probabilty(self, total_links):
        #     self.probability = (self.links/total_links)
        

    def __str__(self):
        return f'{self.id}, {self.age}, {self.etnicity}, {self.education}'