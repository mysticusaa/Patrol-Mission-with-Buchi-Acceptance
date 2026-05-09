import numpy as np
import random
import matplotlib.pyplot as plt
from environment import BuchiPatrolSystem
from automaton import BuchiAutomaton
from ql_algo import QLearningPatrol

# Instantiate the three components
env = BuchiPatrolSystem()
aut = BuchiAutomaton()
agent = QLearningPatrol(env, aut)

"""
Part 1 - Training
Runs 5000 episodes, each upto 500 steps
After every step, Q-table is updated
After every episode, epsilon is decayed
"""
print("Training agent with episode resets on Restricted hits...")
agent.train(episodes=5000)

"""
Part 2 - Plot Convergence
Shows reward rising from 0 and plateauing (proof that the agent learned)"""
agent.plot_convergence()

# Reset global cycle counter so it only counts evaluation cycles (not training)
aut.cycles = 0

"""
Part 3 - Evaluation
Runs greedy policy for 800 steps with verbose output
Returns the path taken for visualization.
"""
path = agent.run_policy(800)

"""
Part 4 - Visualization
Draws the grid with the agent's evaluation path overlaid
"""
agent.visualize(path)

# Print how many complete Büchi cycles the agent achieved in evaluation
print(f"Num of cycles is {aut.cycles}")

