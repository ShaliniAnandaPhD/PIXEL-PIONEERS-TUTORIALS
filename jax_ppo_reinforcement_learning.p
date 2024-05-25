# jax_ppo_reinforcement_learning.py

import jax
import jax.numpy as jnp
import numpy as np
import gym
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial

# Define the policy network
class PolicyNetwork(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

# Define the value network
class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Define the PPO algorithm
def ppo_update(state, batch, clip_ratio, value_coef, entropy_coef):
    def loss_fn(params, old_log_probs, old_values, advantages, returns):
        obs, actions = batch
        
        # Compute the policy and value predictions
        log_probs = PolicyNetwork().apply(params["policy"], obs)
        log_probs = log_probs[jnp.arange(log_probs.shape[0]), actions]
        values = ValueNetwork().apply(params["value"], obs)
        values = values.squeeze()
        
        # Compute the policy loss
        ratio = jnp.exp(log_probs - old_log_probs)
        policy_loss1 = ratio * advantages
        policy_loss2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(policy_loss1, policy_loss2))
        
        # Compute the value loss
        value_loss = jnp.mean(jnp.square(returns - values))
        
        # Compute the entropy bonus
        entropy_bonus = -jnp.mean(jnp.exp(log_probs) * log_probs)
        
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
        return total_loss, (policy_loss, value_loss, entropy_bonus)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    
    obs, actions = batch
    old_log_probs = PolicyNetwork().apply(state.params["policy"], obs)
    old_log_probs = old_log_probs[jnp.arange(old_log_probs.shape[0]), actions]
    old_values = ValueNetwork().apply(state.params["value"], obs).squeeze()
    advantages = returns - old_values
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    (_, (policy_loss, value_loss, entropy_bonus)), grads = grad_fn(
        state.params, old_log_probs, old_values, advantages, returns)
    state = state.apply_gradients(grads=grads)
    
    return state, (policy_loss, value_loss, entropy_bonus)

# Create the custom game environment
class CustomEnvironment(gym.Env):
    def __init__(self):
        super(CustomEnvironment, self).__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.action_space = gym.spaces.Discrete(2)
    
    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(3,))
        return self.state
    
    def step(self, action):
        self.state += np.random.uniform(low=-0.01, high=0.01, size=(3,))
        reward = 1.0 if action == (self.state[0] > 0) else -1.0
        done = np.abs(self.state).max() > 1.0
        return self.state, reward, done, {}

# Set hyperparameters
num_envs = 16
num_steps = 128
num_epochs = 10
batch_size = 64
gamma = 0.99
clip_ratio = 0.2
value_coef = 0.5
entropy_coef = 0.01
learning_rate = 3e-4

# Create the environment and the networks
env = CustomEnvironment()
policy_model = PolicyNetwork(action_dim=env.action_space.n)
value_model = ValueNetwork()
policy_params = policy_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
value_params = value_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
params = {"policy": policy_params["params"], "value": value_params["params"]}

# Create the optimizer and the training state
tx = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=None, params=params, tx=tx)

# Training loop
@partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, num_steps):
    returns = []
    for i in range(num_steps):
        obs, actions, rewards, dones = batch[i]
        values = ValueNetwork().apply(state.params["value"], obs)
        returns.append(rewards + gamma * (1 - dones) * values)
    returns = jnp.stack(returns[::-1]).squeeze()
    
    def scan_fn(carry, inputs):
        state, batch = carry
        state, _ = ppo_update(state, batch, clip_ratio, value_coef, entropy_coef)
        return state, None
    
    state, _ = jax.lax.scan(scan_fn, state, (batch, returns))
    return state

for epoch in range(num_epochs):
    batch = []
    obs = env.reset()
    for _ in range(num_steps):
        probs = jax.nn.softmax(PolicyNetwork().apply(state.params["policy"], obs[jnp.newaxis, ...]))
        actions = jax.random.categorical(jax.random.PRNGKey(0), probs)
        next_obs, rewards, dones, _ = env.step(actions)
        batch.append((obs, actions, rewards, dones))
        obs = next_obs
    
    state = train_step(state, batch, num_steps)
    
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Evaluation
@jax.jit
def evaluate(params, obs):
    probs = jax.nn.softmax(PolicyNetwork().apply(params["policy"], obs[jnp.newaxis, ...]))
    action = jnp.argmax(probs)
    return action

rewards = []
obs = env.reset()
done = False
while not done:
    action = evaluate(state.params, obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)
    env.render()
print(f"Evaluation Reward: {np.sum(rewards)}")

env.close()
