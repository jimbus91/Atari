import pickle
import numpy as np
import gymnasium as gym
from ale_py import ALEInterface

# Create the ALE interface
ale = ALEInterface()

# Enable the display screen to show the game screen while the AI is playing.
ale.setBool('display_screen', True)

# Enable sound
ale.setBool('sound', True)

# Load the Ms. PacMan ROM
#ale.loadROM("C:\\Users\\jimbu\\Atari\\Ms. PacMan.a26") #Windows
ale.loadROM("/Users/beusse/Atari/Ms. PacMan.a26") #Mac

# Get the size of the action space
num_actions = len(ale.getMinimalActionSet())
valid_actions = set(ale.getMinimalActionSet())

# Define the Q-table with the size of the state space and action space
screen_height, screen_width = ale.getScreenDims()
q_table = np.random.rand(screen_height * screen_width, num_actions)

# load the q_table
try:
    with open('q_table_Ms.PacMan.pkl', 'rb') as f:
        q_table = pickle.load(f)
        print("Loading previously trained Q-table")
except:
    q_table = np.random.rand(screen_height * screen_width, num_actions)
    print("Q-table not found, starting with new Q-table")

# Define the learning rate, discount factor, and exploration rate (epsilon)
alpha = 0.5
gamma = 0.9
epsilon = 9.0

#A good starting point for the learning rate (alpha) is typically between 0.1 and 0.5. 
#A low learning rate means that the agent will update its Q-values slowly, which can
# make the learning process more stable but also slower. A high learning rate means that
# the agent will update its Q-values quickly, which can make the learning process faster
# but also more unstable.

#A good starting point for the discount factor (gamma) is typically between 0.5 and 0.9.
# A low discount factor means that the agent will prioritize short-term rewards over
# long-term rewards, while a high discount factor means that the agent will prioritize
# long-term rewards over short-term rewards.

#A good starting point for the exploration rate (epsilon) is typically between 0.1 and 0.5.
#A low exploration rate means that the agent will mostly follow the Q-table, while a high
#exploration rate means that the agent will explore the state space more.

# Define the number of episodes to train the AI
num_episodes = 10000

# Train the AI
for episode in range(num_episodes):
    # Reset the environment
    ale.reset_game()

    # Set the initial reward to zero
    total_reward = 0

    # Run the episode
    while not ale.game_over():
        
        # Get the current state
        state = ale.getScreenRGB()
        state = state.flatten() 

        # Choose an action according to the Q-table and an exploration strategy
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])
            
        # Take the action and observe the next state and reward
        reward = ale.act(action)

        # Get the next state
        next_state = ale.getScreenRGB()
        next_state = next_state.flatten()

        # Update the Q-value for the current state and action
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))

        # Update the total reward
        total_reward += reward

        # Update the current state
        state = next_state

        # Check if the episode is over
        if ale.game_over():
           break

        # Save the Q-table every 500 episodes
        if episode % 500 == 0:
            with open('q_table_Ms.PacMan', 'wb') as f:
                pickle.dump(q_table, f)

    # Print the total reward for the episode
    print("Episode: {}, Total reward: {}".format(episode, total_reward))