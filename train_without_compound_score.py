import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv("aaplCombined.csv")
# Calculate the 5-day moving average
data['5-day MA'] = data['Close'].rolling(window=5).mean()
data = data.dropna()
train_data = data[:1513]
test_data = data[1513:]



# Define hyperparameters
batch_size = 50
gamma = 0.95
memory_size = 10000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
update_target_network = 100
# Define the number of episodes
num_episodes = 50

# Define the deep neural network model
model = keras.models.Sequential([
    keras.layers.Dense(64, input_shape=(5,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='linear')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse')

# Define the target network model
target_model = keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

# Define the experience replay function
def experience_replay(agent_memory):
    # Sample a batch from agent memory
    batch_indices = np.random.choice(len(agent_memory), batch_size, replace=False)
    batch = [agent_memory[i] for i in batch_indices]

    # Unpack the batch into states, actions, rewards, and next states
    states = np.array([b[0] for b in batch])
    actions = np.array([b[1] for b in batch])
    rewards = np.array([b[2] for b in batch])
    next_states = np.array([b[3] for b in batch])
    done_flags = np.array([b[4] for b in batch])

    # Predict the Q values for the current states and the next states
    current_Q = model.predict(states)
    next_Q = target_model.predict(next_states)

    # Calculate the target Q values
    target_Q = np.zeros(batch_size)
    for i in range(batch_size):
        if done_flags[i]:
            target_Q[i] = rewards[i]
        else:
            target_Q[i] = rewards[i] + gamma * np.max(next_Q[i])

    # Update the Q values for the current states and actions
    current_Q[np.arange(batch_size), actions.astype(int)] = target_Q

    # Train the deep neural network with the updated Q values
    model.fit(states, current_Q, epochs=1, verbose=0)

# Load the stock market data
data = train_data

# Initialize the agent memory
agent_memory = []

# Initialize the cash and stock holdings
cash_held = 10000
stocks_held = 25
initial_portfolio_value = 10000
remaining_cash = 10000 - data['Open'][0] * stocks_held

# Initialize exploration rate
exploration_rate = epsilon

# Iterate through each episode
for episode in range(num_episodes):
    # Initialize portfolio value, cash, and stocks held
    portfolio_value = initial_portfolio_value
    cash_held = remaining_cash
    stocks_held = 25
    print('number of episode', episode)
    # Iterate through each day in the dataset
    for i in range(len(data)):
        # Define the state for the current day
        state = np.array([data['Open'][i], data['5-day MA'][i], portfolio_value, cash_held, stocks_held])

        # Take an action using an epsilon-greedy policy
        if np.random.rand() < exploration_rate:
            action = random.randrange(3)
        else:
            Q_values = model.predict(np.array([state]))
            action = np.argmax(Q_values)

        # Calculate the reward for the action taken
        price_change_percentage = ((data['Open'][i] - data['5-day MA'][i]) / data['5-day MA'][i]) * 100
        reward = price_change_percentage

        # Calculate the reward for the action taken
        if action == 0:
            stocks_bought = cash_held // data['Open'][i]
            cash_held -= stocks_bought * data['Open'][i]
            stocks_held += stocks_bought
        elif action == 1:
            cash_gained = stocks_held * data['Open'][i]
            cash_held += cash_gained
            stocks_held = 0
        elif action == 2:
            pass

        # Update the portfolio value with the current cash and stock holdings
        portfolio_value = cash_held + stocks_held * data['Open'][i]

        # Update the agent memory with the current state, action, reward, next state, and done flag
        next_state = np.array([data['Open'][i], data['5-day MA'][i], portfolio_value, cash_held, stocks_held])
        done_flag = (i == len(data) - 1)
        agent_memory.append((state, action, reward, next_state, done_flag))

        # Update the exploration rate
        exploration_rate = max(epsilon_min, exploration_rate * epsilon_decay)

        # Update the target network every 'update_target_network' steps
        if len(agent_memory) % update_target_network == 0:
            target_model.set_weights(model.get_weights())

        # Perform experience replay every step after memory_size is reached
        if len(agent_memory) >= memory_size:
            experience_replay(agent_memory)
#Save the model            
model.save('/mnt/home/mahaseth/CMSEProj/my_model.h5')
