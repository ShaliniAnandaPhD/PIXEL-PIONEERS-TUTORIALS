# jax_dqn_cartpole.py

import jax
import jax.numpy as jnp
import numpy as np
import gym
from collections import deque
import random

# Define the Q-network
def jax_q_network(states, num_actions, hidden_size):
    x = jax.nn.dense(states, hidden_size)
    x = jax.nn.relu(x)
    q_values = jax.nn.dense(x, num_actions)
    return q_values

# Define the loss function
def jax_loss_fn(params, states, actions, rewards, next_states, dones, gamma):
    q_values = jax_q_network(states, num_actions=2, hidden_size=128)
    next_q_values = jax_q_network(next_states, num_actions=2, hidden_size=128)
    
    q_values = jnp.take_along_axis(q_values, actions.reshape(-1, 1), axis=1).squeeze()
    next_q_values = jnp.max(next_q_values, axis=1)
    
    targets = rewards + gamma * next_q_values * (1 - dones)
    loss = jnp.mean(jax.lax.square(targets - q_values))
    return loss

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# Train the agent
def jax_train(env, params, optimizer, replay_buffer, num_episodes, batch_size, gamma, epsilon):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = jax_q_network(state[np.newaxis], num_actions=2, hidden_size=128)
                action = jnp.argmax(q_values)
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                loss_value, grads = jax.value_and_grad(jax_loss_fn)(params, states, actions, rewards, next_states, dones, gamma)
                params = optimizer.update(grads, params)
        
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
    
    return params

# Main function
def main():
    env = gym.make('CartPole-v1')
    
    # Initialize Q-network parameters
    rng = jax.random.PRNGKey(0)
    params = jax.random.normal(rng, (4, 128))
    
    # Initialize optimizer
    optimizer = jax.optim.Adam(learning_rate=0.001)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Train the agent
    params = jax_train(env, params, optimizer, replay_buffer, num_episodes=200, batch_size=64, gamma=0.99, epsilon=0.1)
    
    # Evaluate the trained agent
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        q_values = jax_q_network(state[np.newaxis], num_actions=2, hidden_size=128)
        action = jnp.argmax(q_values)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        env.render()
    
    print(f"Evaluation Reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    main()
