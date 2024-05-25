# jax_dqn_reinforcement_learning.py

import jax
import jax.numpy as jnp
import numpy as np
import gym
from collections import deque
import random

# Define the Q-network
def q_network(inputs, num_actions):
    x = jax.nn.relu(jax.nn.dense(inputs, 64))
    x = jax.nn.relu(jax.nn.dense(x, 64))
    q_values = jax.nn.dense(x, num_actions)
    return q_values

# Define the loss function
def loss_fn(params, inputs, actions, targets):
    q_values = q_network(inputs, num_actions)
    q_values_selected = jnp.take_along_axis(q_values, actions.reshape(-1, 1), axis=1).squeeze()
    loss = jnp.mean(jax.lax.square(q_values_selected - targets))
    return loss

# Define the update function
@jax.jit
def update(params, inputs, actions, targets, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, inputs, actions, targets)
    updates, optimizer = optimizer.update(grads, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer, loss

# Define the epsilon-greedy policy
def epsilon_greedy_policy(params, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        q_values = q_network(state, num_actions)
        return jnp.argmax(q_values)

# Create the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return jnp.array(states), jnp.array(actions), jnp.array(rewards), jnp.array(next_states), jnp.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# Set hyperparameters
num_episodes = 1000
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
target_update_frequency = 10

# Create the environment
env = gym.make('CartPole-v1')
num_actions = env.action_space.n

# Initialize the Q-network and target network
params = jax.random.uniform(jax.random.PRNGKey(0), (env.observation_space.shape[0], num_actions))
target_params = params
optimizer = optax.adam(learning_rate)

# Create the replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Select an action using epsilon-greedy policy
        epsilon = max(epsilon_end, epsilon_start * epsilon_decay ** episode)
        action = epsilon_greedy_policy(params, state, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # Store the transition in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        
        # Update the Q-network
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            next_q_values = q_network(next_states, num_actions)
            max_next_q_values = jnp.max(next_q_values, axis=1)
            targets = rewards + (1 - dones) * gamma * max_next_q_values
            
            params, optimizer, loss = update(params, states, actions, targets, optimizer)
        
        state = next_state
    
    # Update the target network
    if episode % target_update_frequency == 0:
        target_params = params
    
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

# Close the environment
env.close()
