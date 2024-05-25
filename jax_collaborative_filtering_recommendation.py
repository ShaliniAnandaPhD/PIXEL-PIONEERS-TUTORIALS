# jax_collaborative_filtering_recommendation.py

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import mean_squared_error

# Define the collaborative filtering model
def jax_collaborative_filtering(user_item_matrix, num_features, num_iterations, learning_rate):
    num_users, num_items = user_item_matrix.shape
    
    # Initialize user and item feature matrices randomly
    user_features = jnp.random.normal(size=(num_users, num_features))
    item_features = jnp.random.normal(size=(num_items, num_features))
    
    # Optimize the user and item feature matrices
    for _ in range(num_iterations):
        # Compute the predicted ratings
        predicted_ratings = jnp.dot(user_features, item_features.T)
        
        # Compute the error between predicted and actual ratings
        error = user_item_matrix - predicted_ratings
        
        # Update user and item feature matrices using gradient descent
        user_features -= learning_rate * jnp.dot(error, item_features)
        item_features -= learning_rate * jnp.dot(error.T, user_features)
    
    return user_features, item_features

# Prepare the user-item rating matrix
def prepare_data(ratings):
    num_users = len(np.unique(ratings[:, 0]))
    num_items = len(np.unique(ratings[:, 1]))
    
    user_item_matrix = np.zeros((num_users, num_items))
    for user_id, item_id, rating in ratings:
        user_item_matrix[int(user_id) - 1, int(item_id) - 1] = rating
    
    return user_item_matrix

# Evaluate the recommendation system using RMSE
def evaluate_recommendation(user_item_matrix, user_features, item_features):
    predicted_ratings = jnp.dot(user_features, item_features.T)
    actual_ratings = user_item_matrix[user_item_matrix.nonzero()]
    predicted_ratings = predicted_ratings[user_item_matrix.nonzero()]
    
    mse = mean_squared_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mse)
    return rmse

# Generate movie recommendations for a user
def recommend_movies(user_id, user_features, item_features, movie_titles, top_n=5):
    user_ratings = jnp.dot(user_features[user_id - 1], item_features.T)
    movie_indices = jnp.argsort(user_ratings)[::-1][:top_n]
    recommended_movies = [movie_titles[i] for i in movie_indices]
    return recommended_movies

# Example usage
# Simulated movie rating data
ratings = np.array([
    [1, 1, 4],
    [1, 2, 3],
    [1, 3, 5],
    [2, 1, 5],
    [2, 2, 4],
    [3, 1, 3],
    [3, 3, 4],
    [4, 2, 3],
    [4, 3, 5],
])

movie_titles = ["Movie A", "Movie B", "Movie C"]

# Prepare the user-item rating matrix
user_item_matrix = prepare_data(ratings)

# Set the hyperparameters
num_features = 2
num_iterations = 100
learning_rate = 0.01

# Perform collaborative filtering
user_features, item_features = jax_collaborative_filtering(user_item_matrix, num_features, num_iterations, learning_rate)

# Evaluate the recommendation system
rmse = evaluate_recommendation(user_item_matrix, user_features, item_features)
print(f"RMSE: {rmse:.3f}")

# Generate movie recommendations for a user
user_id = 1
recommended_movies = recommend_movies(user_id, user_features, item_features, movie_titles)
print(f"Recommended movies for User {user_id}: {recommended_movies}")
