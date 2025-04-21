import streamlit as st
import torch
import numpy as np
from network import build_network
from visualizer import visualize_network
import random

st.set_page_config(layout="wide")

st.title("Neural Network Forward Pass Visualizer")

# Sidebar Inputs
with st.sidebar:
    seed = st.number_input("Random Seed", value=42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    depth = st.slider("Depth (number of hidden layers)", 1, 5, 2)
    hidden_size = st.slider("Hidden Layer Size", 1, 10, 5)
    output_size = st.slider("Output Layer Size", 1, 5, 1)

    activation = st.selectbox("Activation Function", ["ReLU", "LeakyReLU", "Tanh", "Sigmoid"])

    x_input = st.text_input("Input values (comma-separated)", "-1.0, -0.5, 0.0, 0.5, 1.0")
    x_vals = np.array([float(x.strip()) for x in x_input.split(",")]).reshape(-1, 1)
    x_tensor = torch.tensor(x_vals, dtype=torch.float32)

# Build and visualize the model
model = build_network(input_size=1, hidden_size=hidden_size, depth=depth, 
                      output_size=output_size, activation=activation)

visualize_network(model, x_tensor, activation_name=activation)