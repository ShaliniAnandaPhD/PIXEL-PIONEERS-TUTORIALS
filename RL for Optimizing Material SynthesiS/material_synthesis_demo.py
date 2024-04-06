import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enhanced mock function to simulate model prediction and provide a history of adjustments
def simulate_optimization(target_properties, steps=10):
    history = []
    conditions = np.array([20.0, 20.0])  # Initial conditions

    for _ in range(steps):
        # Simulate adjustment towards target properties
        adjustment = np.random.normal(0, 2, size=2)
        conditions += adjustment * (target_properties - conditions) / 10
        history.append(conditions.copy())
        
    return conditions, np.array(history)

# Plotting function
def plot_optimization_history(history, target_properties):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    time = np.arange(history.shape[0])
    
    for i, label in enumerate(["Hardness", "Conductivity"]):
        axs[i].plot(time, history[:, i], label="Optimized Value")
        axs[i].hline(target_properties[i], color='r', linestyle='--', label="Target Value")
        axs[i].set_ylabel(label)
        axs[i].legend()
        axs[i].grid(True)
    
    plt.xlabel("Optimization Step")
    st.pyplot(fig)

# Streamlit UI Enhancement
st.title('Advanced Material Synthesis Optimization Demo')

st.markdown("""
This interactive demo simulates the optimization of synthesis conditions for materials using reinforcement learning. 
Adjust the sliders to set your target material properties, and watch the algorithm optimize towards these goals.
""")

# Input sliders for target properties
target_hardness = st.slider('Target Hardness', min_value=0.0, max_value=100.0, value=50.0, step=0.5)
target_conductivity = st.slider('Target Conductivity', min_value=0.0, max_value=100.0, value=50.0, step=0.5)

target_properties = np.array([target_hardness, target_conductivity])

# Button to perform optimization
if st.button('Optimize Synthesis Conditions'):
    optimal_conditions, history = simulate_optimization(target_properties)
    
    st.write("### Optimal Conditions:")
    st.write(f"- Hardness: {optimal_conditions[0]:.2f}")
    st.write(f"- Conductivity: {optimal_conditions[1]:.2f}")
    
    st.write("### Optimization Process Visualization")
    plot_optimization_history(history, target_properties)

# Additional Information
st.markdown("""
## How It Works
- The algorithm simulates an RL agent's decision-making process to adjust synthesis conditions towards the target properties.
- The optimization process is visualized to show the dynamic adjustments and convergence towards the target conditions.
""")
