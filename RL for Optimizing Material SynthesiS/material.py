import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Initialize session state if it's not already initialized
if 'history' not in st.session_state:
    st.session_state.history = []
if 'target_properties' not in st.session_state:
    st.session_state.target_properties = np.array([50.0, 50.0])  # Default target properties

# Simulate an optimization step
def optimization_step():
    if len(st.session_state.history) == 0:
        conditions = np.array([20.0, 20.0])  # Initial conditions
    else:
        conditions = st.session_state.history[-1] + np.random.normal(0, 2, size=2)  # Random step
    
    st.session_state.history.append(conditions)
    plot_optimization_history()

# Plotting function using Plotly, modified to use session state for history
def plot_optimization_history():
    history = np.array(st.session_state.history)
    steps = list(range(len(history)))
    fig = go.Figure()

    # Adding traces for Hardness and Conductivity
    fig.add_trace(go.Scatter(x=steps, y=history[:, 0], mode='lines+markers', name='Hardness'))
    fig.add_trace(go.Scatter(x=steps, y=history[:, 1], mode='lines+markers', name='Conductivity'))

    # Target lines
    fig.add_hline(y=st.session_state.target_properties[0], line=dict(color="red", width=2, dash="dash"), name="Target Hardness")
    fig.add_hline(y=st.session_state.target_properties[1], line=dict(color="green", width=2, dash="dash"), name="Target Conductivity")

    # Layout adjustments
    fig.update_layout(title="Optimization Process Visualization in 'Real-Time'",
                      xaxis_title="Step",
                      yaxis_title="Property Value",
                      legend_title="Properties")
    st.plotly_chart(fig, use_container_width=True)

st.title("Real-Time RL Optimization Simulation")

# Target property inputs
st.session_state.target_properties[0] = st.slider('Target Hardness', min_value=0.0, max_value=100.0, value=50.0, step=0.5)
st.session_state.target_properties[1] = st.slider('Target Conductivity', min_value=0.0, max_value=100.0, value=50.0, step=0.5)

# Button to perform optimization step
if st.button('Perform Optimization Step'):
    optimization_step()

# Reset button to clear the history and start over
if st.button('Reset Optimization'):
    st.session_state.history = []
