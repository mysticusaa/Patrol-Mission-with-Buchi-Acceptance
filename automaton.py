import numpy as np
import random
import matplotlib.pyplot as plt

class BuchiAutomaton:
    """
    Implements the Buchi acceptance condition and safety constraints.
    BUCHI AUTOMATON
        A Buchi Automaton defines an infinite-horizon task - the agent must visit a set of accepting states (A, B, C) infinitely often.
        One full visit to all accepting states - One Buchi cycle
    
    FRONTIER
        The frontier is the set of targets not yet visited in the current cycle.
        It starts as {A, B, C} (set of accepting states). As the agent visits targets, they are removed.
        When the frontier becomes empty, the cycle is complete and resets to {A, B, C}.

    PRODUCT MDP CONNECTION
        The automaton's state (needs_base_reset, frontier) combined with the grid position forms the 'product state' used by the Q-learning agent.
    """
    def __init__(self, accepting_labels=['A', 'B', 'C']):
        # set of accepting states defined
        self.accepting_labels = set(accepting_labels)
        # initialise frontier to set of accepting states
        self.frontier = set(accepting_labels)
        # safety flag - becomes true when agent crosses the risk zone and needs to go to base
        self.needs_base_reset = False  
        self.cycles = 0  

    def transition(self, label, verbose=False):
        """
        Handles the logic of the Büchi condition and safety constraints.

        Called once per step with the label of the cell the agent just moved into
        After every move, we need to know 
            a. What reward does the agent get?
            b. Does the episode end here?
        This function also updates the automaton's internal state (frontier, needs_base_reset)

        Returns: 
            (reward, is_terminal)
            reward      : float — reward signal for the Q-learning update
            is_terminal : bool  — True means episode ends immediately
        """
        # global cycles
        # 1. RESTRICTED ZONE: Immediate death
        if label == "restricted":
            if verbose:
                print(" [!!!] CRITICAL FAILURE: Entered restricted zone. Episode terminated.")
            return -100, True
        
        # 2. RISK ZONE: Compromised state
        if label == "risk":
            if not self.needs_base_reset:
                if verbose:
                    print(" [!] RISK DETECTED: System compromised. Go to Base to reset.")
                self.needs_base_reset = True
            return -5, False

        # 3. BASE STATION: Clearing the risk
        if label == "base":
            if self.needs_base_reset:
                if verbose: 
                    print(" [+] Back at Base. Penalty cleared. Resuming the cycle...")
                self.needs_base_reset = False
                return 20, False
            return 0, False # No special reward if not compromised
            
        # 4. PATROL: Checkpoints A, B, C
        if label in self.accepting_labels:
            # PROTOCOL VIOLATION: If agent visits a target while compromised, it dies.
            if self.needs_base_reset: 
                if verbose:
                    print(f" [!!!] PROTOCOL VIOLATION: Visited {label} while compromised. Terminating.")
                return -50, True 

            # Checkpoint progress
            if label in self.frontier:
                if verbose: 
                    print(f" [*] Visited {label}. (Remaining: {self.frontier - {label}})")
                
                self.frontier.remove(label)
                
                # Check if entire patrol cycle is complete
                if not self.frontier:
                    if verbose: 
                        print(" [BÜCHI] CYCLE COMPLETE! Resetting frontier.")
                    self.frontier = set(self.accepting_labels)
                    self.cycles += 1
                    return 150, False # Return big reward immediately

                return 50, False # Return progress reward immediately
            else:
                # Already visited this target in the current cycle
                return 0, False
        
        # 5. SAFE/DEFAULT STEP
        # small negative reward at every safe cell; otherwise the agent has no incentive to move efficiently
        return -0.1, False

    def reset(self):
        """
        Resets automaton state at the start of each training episode
        Called by train() at the beginning of every episode so each episode starts with a full frontier and no compromised flag
        """
        self.frontier = set(self.accepting_labels)
        self.needs_base_reset = False

    def get_current_state(self):
        """
        Returns the automaton's current state as a hashable tuple
            a. needs_base_reset : bool
            b. frontier         : set (not hashable, so we sort and convert to tuple)
        This is combined with grid position in get_q_state() to form the full product state used as the Q-table key.
        """
        return (self.needs_base_reset, tuple(sorted(list(self.frontier))))
