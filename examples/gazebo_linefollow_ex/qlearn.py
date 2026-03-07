import random
import pickle
import csv
import os


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''

        # Check if the file exists
        path = filename + ".pickle"
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q = pickle.load(f)
            print("Loaded file: {}".format(path))
        else:
            print("No existing pickle found at {}. Starting fresh.".format(path))
        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''

        with open(filename + ".pickle", "wb") as f:
            pickle.dump(self.q, f)

        with open(filename + ".csv", "w", newline='') as f:
            writer = csv.writer(f)
            for (state, action), value in self.q.items():
                writer.writerow([state, action, value])

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''

        # Choose a random action with probability epsilon
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)
            # Handle ties by choosing a random action among those with max Q
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            action = random.choice(best_actions)

        if return_q:
            return action, self.getQ(state, action)
        return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''

        # Find Q for current (state1, action1)
        old_q = self.getQ(state1, action1)

        # Find max Q for state2
        max_q = max([self.getQ(state2, a) for a in self.actions], default=0.0)

        # Update Q for (state1, action1)
        self.q[(state1, action1)] = old_q + self.alpha * (reward + self.gamma * max_q - old_q)
