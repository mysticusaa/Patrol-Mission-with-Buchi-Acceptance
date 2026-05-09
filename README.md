# Patrol-Mission-with-Buchi-Acceptance
A Logic-Constrained Reinforcement Learning (LCRL) framework where a Q-Learning agent navigates a 10x10 grid. It uses a Büchi Automaton to enforce a multi-stage patrol mission (A-B-C cycle) while respecting hard safety constraints (Restricted Zones) and soft recovery protocols (Risk Zones).

## Environment - environment.py
This file defines the gridworld. It consists a function that returns the semantic label of a grid cell.

## Automaton - automaton.py
This file implements the Buchi acceptance condition and safety constraints. The transition function in the file handles the logic of Buchi acceptance and logic constraints. It also consists of an helper function for the frontier reset at the start of every episode.

## Q Learning - ql_algo.py
The RL agent learns the policy via Q-Learning. It maintains a 'q_table' which stores the expected reward of each possible action at each state. The Q-table is the Python dict mapping product states with the array of 4 Q-values. The agent uses this to pick the action with the highest Q-value.

## main.py
This file instantiaes the three components - environment, automaton, and agent. 
1. Training - The agent runs 5000 episodes, each upto 500 steps. After every step, the Q-table is updated and after every episode, epsilon value is decayed.
2. Plotting Convergence - It shows the reward accumulated by the agent in every episode. It rises from 0 and plateaus at some point, showing that the agent has learned.
3. Evaluation - It runs the greedy policy for 800 steps with verbose output and returns the path taken for visualization.
4. Visualization - It draws the grid with the agent's evaluation path overlaid.

## Instructions to run the code
```bash
python main.py
```

## Output
A convergence graph and path visualization graph, with verbose output of checkpoints visited and frontiers at every point and the number of cycles in total at the end of the 800 episodes are shown.
