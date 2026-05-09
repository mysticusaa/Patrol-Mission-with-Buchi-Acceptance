import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningPatrol:
    """
    The RL agent learns a patrol policy via Q-Learning
    Q-Learning
        Q-learning maintains a table Q[state][action] = expected future reward
        After each step, it updates this table using the Bellman equation:
            Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
        where,
            α (alpha)  = learning rate — how fast to update (0.2 here)
            γ (gamma)  = discount factor — how much future rewards matter (0.95)
            r          = reward received this step
            s'         = next state
            max Q(s',·)= best possible Q-value from the next state
        Over thousands of episodes, Q[s][a] converges to the true value of
        taking action a from state s.
    Q-Table
        A Python dict mapping product states - array of 4 Q-values (one per action)
        Product state = (grid position, automaton state)
                      = ((row, col), (needs_base_reset, frontier_tuple))
        
        Example entry:
            {((5,5), (False, ('A','B','C'))): [2.3, 1.1, -0.5, 3.7]}
             ↑ position  ↑ automaton state    ↑UP  ↑DOWN ↑LEFT ↑RIGHT
        
        The agent picks the action with the highest Q-value (argmax)
        """
    def __init__(self, env, automaton):
        self.env = env
        self.aut = automaton
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.action_map = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        # Q-table: starts empty, entries are created on the first visit to each state
        self.q_table = {}
        # learning rate - how much each new experience updates the Q-value
        self.alpha = 0.2
        # discount factor - how much the agent values future rewards vs immediate
        self.gamma = 0.95
        # epsilon greedy exploration parameters
        self.epsilon = 1.0          # start with exploring fully
        self.epsilon_min = 0.05     # this is the lowest epsilon can b
        self.epsilon_decay = 0.998  # decay per episode
        # changed epsilon from a fixed 0.2 to this

    def get_q_state(self, pos):
        """
        Constructs the 'product state' - Q-table key
        This helps because two visits to the same position with different frontiers should have different q-values
        By combining position + automaton state into one key, the Q-table learns a different policy for each combination automatically
        """
        return (pos, self.aut.get_current_state())
    
    def choose_Action(self, state, train=True):
        """
        epsilon-greedy action selection
        During training, uses epsilon (starts high, decays over time)
        During evaluation (train=False), always picks argmax - no randomness
        Also initialises the Q-table entry for unseen states to zero
        """
        if state not in self.q_table:
            # first time visiting this state - initialises Q-values to 0
            self.q_table[state] = np.zeros(len(self.actions))
        if train and random.random() < self.epsilon:
            return random.randint(0, 3)             # explore - random action
        return np.argmax(self.q_table[state])       # exploit - best known action
    
    def train(self, episodes=5000, max_steps_per_episode=500):
        """
        Training loops - runs N episodes and fills the Q-table
        Structure:
            outer loop: episodes (each is one run of the agent) 
                - reset automaton + position at the start of each episode
                inner loop: steps (each is one move within the episode)
                    - pick action (epsilon-greedy)
                    - move on grid
                    - get label + reward from automaton
                    - update Q-table (Bellman equation)
                    - if terminal, end epsiode early
                after inner loop:
                    - record episode reward for convergence plot
                    - decay epsilon
        """
        # stores total reward per episode for convergence plot
        self.history = []

        for episode in range(episodes):
            pos = (5, 5)            # fixed start position 
            self.aut.reset()        # fresh automaton state: full frontier, not compromised
            episode_reward = 0

            for step in range(max_steps_per_episode):
                # 1. Get current product state
                current_q_state = self.get_q_state(pos)
                # 2. Choose epsilon-greedy action
                action_index = self.choose_Action(current_q_state, train=True)

                # 3. Move on the grid
                dr, dc = self.action_map[self.actions[action_index]]
                # Clamp to [0,9] so agent can't walk off the grid
                new_pos = (max(0, min(9, pos[0] + dr)), max(0, min(9, pos[1] + dc)))

                # 4. Get label and reward after the move
                label = self.env.get_label(new_pos)
                reward, done = self.aut.transition(label)

                # 5. Bellman Q-table update
                next_q_state = self.get_q_state(new_pos)
                # if not in the table, initialise with 0s
                if next_q_state not in self.q_table:
                    self.q_table[next_q_state] = np.zeros(len(self.actions))

                if done:
                    # no future if 'done'
                    best_next_q = 0
                else:
                    # otherwise the max q-value
                    best_next_q = np.max(self.q_table[next_q_state])
                # reward + discounted best future Q-value
                td_target = reward + self.gamma * best_next_q
                # TD error - how wrong the current update was 
                # Update Q(s, a) by alpha * td_error
                self.q_table[current_q_state][action_index] += self.alpha * (td_target - self.q_table[current_q_state][action_index])
                
                episode_reward += reward
                pos = new_pos
                if done: 
                    break
            
            # After each episode
            # decay epsilon - agent explores less as it learns more
            self.history.append(episode_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def run_policy(self, max_steps=500):
        """
        Runs the greedy policy and returns the path taken
        During training, we intentionally add randomness so the agent discovers the whole state space.
        For evaluation, we want to see the best policy the agent has learned (just argmax).
        Returns
            path:   list of (row, col) positions - used by visualize()
        """
        pos = (5, 5)
        path = [pos]
        self.aut.reset()
        
        print(f"\n--- Evaluation (Max {max_steps} steps) ---")
        for i in range(max_steps):
            state = self.get_q_state(pos)
            # Pick the best known action
            action_index = np.argmax(self.q_table.get(state, np.zeros(4)))
            dr, dc = self.action_map[self.actions[action_index]]
            pos = (max(0, min(9, pos[0] + dr)), max(0, min(9, pos[1] + dc)))
            
            label = self.env.get_label(pos)
            reward, done = self.aut.transition(label, verbose=True)
            path.append(pos)

            if done:
                break
        return path

    def visualize(self, path):
        """
        Draws the grid and agent's path on top of it
        COLOR KEY:
            black      = restricted zone (instant death)
            blue       = risk zone (compromises system)
            cyan       = base station (resets compromised flag)
            lime green = patrol targets A, B, C
            white      = safe (no special meaning)
            red line   = agent's path during evaluation
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks(np.arange(-0.5, 10, 1)); ax.set_yticks(np.arange(-0.5, 10, 1))
        ax.grid(which='both', color='grey', alpha=0.3)
        
        for r in range(10):
            for c in range(10):
                l = self.env.get_label((r, c))
                color = 'white'
                if l == 'restricted': 
                    color = 'black'
                elif l == 'base': 
                    color = 'cyan'
                elif l == 'risk': 
                    color = 'blue'
                elif l in ['A', 'B', 'C']: 
                    color = 'lime'
                ax.add_patch(plt.Rectangle((c-0.5, 9-r-0.5), 1, 1, color=color, alpha=0.3))
                if l in ['A', 'B', 'C', 'base']: 
                    ax.text(c, 9-r, l, ha='center', va='center')

        # Convert path from (row,col) to matplotlib (x,y)
        rows = [9-p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, color='red', alpha=0.6, linewidth=1, marker='o', markersize=3)
        plt.title("Büchi Patrol: Red Line = Agent Path | Blue = Risk | Black = Restricted")
        plt.show()

    def plot_convergence(self):
        """
        Plots total reward per episode over training
        The rolling average (smoothed line) makes the trend visible despite high variance of individual episodes
        The variance in the raw signal comes from:
              - Random actions (epsilon exploration)
              - Episodes where the agent hits a restricted zone early
        """
        window = 100        # average over 100 episodes at a time
        smoothed = np.convolve(self.history, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(10,4))
        plt.plot(self.history, alpha=0.3, color='steelblue', linewidth=0.6, label='raw')
        plt.plot(np.arange(window-1, len(self.history)), smoothed, color='steelblue', linewidth=1.5, label=f'rolling avg (w={window})')
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title("Training Convergence")
        plt.legend()
        plt.tight_layout()
        plt.show()