import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class UIEnvironment:
    def __init__(self):
        self.layout = [0, 1]  # Two buttons: positions 0 and 1
        self.user_preferences = [0.8, 0.2]  # User prefers button 0 more
        self.current_step = 0
        self.max_steps = 10  # Number of steps per episode

    def reset(self):
        self.layout = [0, 1]  # Reset button positions
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # Actions: 0 = Swap buttons, 1 = Keep layout
        if action == 0:
            self.layout = self.layout[::-1]  # Swap button positions
        
        # Simulate user interaction
        clicks = np.dot(self.layout, self.user_preferences)
        reward = clicks  # Reward is based on simulated clicks
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done

    def _get_state(self):
        return np.array(self.layout)

env = UIEnvironment()

# Define Q-network
model = Sequential([
    Dense(16, input_dim=2, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='linear')  # Two actions: Swap or Keep
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Training parameters
episodes = 100
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        state_input = np.reshape(state, [1, 2])
        
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = np.random.choice([0, 1])  # Random action
        else:
            q_values = model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])  # Best action
        
        # Take action and observe result
        next_state, reward, done = env.step(action)
        next_state_input = np.reshape(next_state, [1, 2])
        
        # Update Q-values
        q_values = model.predict(state_input, verbose=0)
        q_next = model.predict(next_state_input, verbose=0)
        # Bellman Equation for Q-learning
        q_values[0][action] = reward + gamma * np.max(q_next[0]) * (1 - done)
        
        # Train the model
        model.fit(state_input, q_values, verbose=0)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

print("Training Complete!")

state = env.reset()
total_reward = 0

print("\nTesting Trained Agent:")
while True:
    state_input = np.reshape(state, [1, 2])
    q_values = model.predict(state_input, verbose=0)
    action = np.argmax(q_values[0])  # Choose the best action

    next_state, reward, done = env.step(action)
    total_reward += reward
    state = next_state
    
    print(f"State: {state}, Action: {action}, Reward: {reward}")
    
    if done:
        break

print(f"Total Reward: {total_reward}")
