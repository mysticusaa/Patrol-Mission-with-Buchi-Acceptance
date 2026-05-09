import numpy as np
import random
import matplotlib.pyplot as plt

class BuchiPatrolSystem:
    """
    Defines the physical grid world - 10×10 grid
    """
    def __init__(self):
        self.grid_size = 10
        # need to visit all checkpoints to complete one cycle
        # format - (row, col)
        # row - 0 (top) to 9 (bottom)
        # col - 0 (left) to 9 (right)
        self.targets = {'A': (1, 2), 'B': (8, 2), 'C': (5, 8)}
        # Base station - agent must visit base after crossing a risk zone
        self.base = (2, 7)
        # Restricted zones - episode ends if agent reaches a restricted area
        self.restricted = [(0, 1),(0, 0),(1, 0),(0, 9),(0, 8),(1, 9),(9, 0),(8, 0),(9, 1),(9, 9),(8, 9),(9, 8)]
        # Risk zone - must go to base before continuing with the cycle if the agent visits a risky zone
        self.risk = [(3, 4), (4, 4), (5, 4), (6, 4), (7, 4)]

    def get_label(self, position):
        """
        Returns the 'semantic label of a grid cell'
        We need this because the automaton does not know about the grid geometry, it only understands labels like 'risk' or 'A'.
        The position can only have one label, so we check the most severe conditions first.
        """
        if position in self.restricted: 
            return "restricted"
        if position in self.risk:
            return "risk"
        if position == self.base: 
            return "base"
        for label, coord in self.targets.items():
            if position == coord: 
                return label
        return "safe"