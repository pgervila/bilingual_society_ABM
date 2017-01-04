# IMPORT LIBS
import networkx as nx

class Agent:
    def __init__(self, unique_id, language, S):
        self.unique_id = unique_id
        self.language = language
        self.S = S
    def move_random(self):
        pass