import streamlit as st
import numpy as np
import plotly.graph_objects as go

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

# Plotting function using Plotly
def plot_optimization_history(history, target_properties):
    steps = list(range(history.shape[0]))
    fig = go.Figure()

    # Adding Hardness trace
    fig.add_trace(go.Scatter(x=steps, y=history[:, 0], mode='lines+markers', name='Hardness'))
    fig.add_hline(y=target_properties[0], line=dict(color="red", width=2, dash="dash"), name="Target Hardness")
    
    # Adding Conductivity trace
    fig.add_trace(go.Scatter(x=steps, y=history[:, 1], mode='lines+markers', name='Conductivity'))
    fig.add_hline(y=target_properties[1], line=dict(color="green", width=2, dash="dash"), name="Target Conductivity")

    # Enhancing the layout
    fig.update_layout(title="Optimization Process Visualization",
                      xaxis_title="Optimization Step",
                      yaxis_title="Property Value",
                      legend_title="Properties")
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI Enhancement
st.title('Advanced Material Synthesis Optimization Demo with Interactive Visuals')

st.markdown("""
This interactive demo simulates the optimization of synthesis conditions for materials using reinforcement learning, visualized with Plotly for an engaging user experience. Adjust the sliders to set your target material properties, and observe how the algorithm optimizes towards these goals.
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

st.markdown("""
## How It Works
- The algorithm simulates an RL agent's decision-making process to adjust synthesis conditions towards the target properties.
- The optimization process is visualized with Plotly, showing the dynamic adjustments and convergence towards the target conditions.
""")
