import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simulated stock price environment
class StockEnvironment:
    def __init__(self, prices):
        self.prices = prices
        self.n_steps = len(prices)
        self.current_step = 0
        self.cash = 1000  # Starting cash
        self.stock_owned = 0  # Starting stock
        self.net_worth = 1000  # Net worth = cash + (stock_owned * current_price)
    
    def reset(self):
        self.current_step = 0
        self.cash = 1000
        self.stock_owned = 0
        self.net_worth = 1000
        return self._get_state()
    
    def step(self, action):
        current_price = self.prices[self.current_step]
        
        # Actions: 0 = hold, 1 = buy, 2 = sell
        if action == 1:  # Buy one unit of stock
            if self.cash >= current_price:
                self.cash -= current_price
                self.stock_owned += 1
        elif action == 2:  # Sell one unit of stock
            if self.stock_owned > 0:
                self.cash += current_price
                self.stock_owned -= 1
        
        # Update step and calculate new net worth
        self.current_step += 1
        done = self.current_step == self.n_steps - 1
        self.net_worth = self.cash + self.stock_owned * current_price
        reward = self.net_worth  # Reward is the net worth
        
        return self._get_state(), reward, done

    def _get_state(self):
        current_price = self.prices[self.current_step]
        return np.array([self.cash, self.stock_owned, current_price])

# Create the Q-network
model = Sequential([
    Dense(24, input_dim=3, activation='relu'),
    Dense(24, activation='relu'),
    Dense(3, activation='linear')  # Actions: hold, buy, sell
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Training parameters
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 100
stock_prices = np.random.randint(50, 150, size=200)  # Simulated stock prices
env = StockEnvironment(stock_prices)

# Training loop
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        state_input = np.reshape(state, [1, 3])
        
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = np.random.choice([0, 1, 2])  # Random action
        else:
            q_values = model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])
        
        # Take action and observe the environment
        next_state, reward, done = env.step(action)
        next_state_input = np.reshape(next_state, [1, 3])
        
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
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

print("Training Complete!")
