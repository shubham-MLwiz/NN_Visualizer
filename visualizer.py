# # Handles plotting of weights, biases, activations, intermediate calculations.

import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import plot_tensor, plot_weight_heatmap, plot_weight_histogram


def visualize_network(model, x_tensor, activation_name):
    st.subheader("Layer-by-Layer Visualization")

    x_vals = x_tensor.detach().numpy().flatten()
    a = x_tensor
    layer_outputs = []

    # Forward pass with recording
    for idx, layer in enumerate(model):
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            wx = a @ layer.weight.T
            wx_plus_b = wx + layer.bias.unsqueeze(0)

            layer_outputs.append({
                "type": "Linear",
                "input": a.detach().numpy(),
                "w": w,
                "b": b,
                "wx": wx.detach().numpy(),
                "wx+b": wx_plus_b.detach().numpy(),
            })

            a = wx_plus_b

        elif isinstance(layer, torch.nn.Module):  # Activation
            act_output = layer(a)
            layer_outputs[-1]["activation"] = act_output.detach().numpy()
            layer_outputs[-1]["act_fn"] = activation_name
            a = act_output

    # Render Layer-by-layer details
    for i, out in enumerate(layer_outputs):
        st.markdown(f"### Layer {i+1} ({out['type']})")
        cols = st.columns(6)

        plot_tensor(out["input"], x_vals, f"x (Input)", cols[0])
        # plot_tensor(out["w"], None, f"Weights (w)", cols[1])  # No x_vals here
        plot_weight_heatmap(out["w"], f"Weights (w)", cols[1])
        plot_weight_histogram(out["w"], f"Weights (w)", cols[5])

        plot_tensor(out["wx"], x_vals, f"wx", cols[2])
        plot_tensor(out["wx+b"], x_vals, f"wx + b", cols[3])

        if "activation" in out:
            plot_tensor(out["activation"], x_vals, f"Activation Output ({out['act_fn']})", cols[4])

    # =============================
    # Option B: Mini Network Graph
    # =============================
    st.markdown("---")
    st.markdown("### Network Diagram with Neuron Outputs")

    fig, ax = plt.subplots(figsize=(10, 4 + len(layer_outputs)))
    layer_width = 1.0
    h_spacing = 2.0
    v_spacing = 1.0

    max_neurons = max(layer["activation"].shape[1] if "activation" in layer else layer["wx+b"].shape[1]
                      for layer in layer_outputs)

    for layer_idx, layer in enumerate(layer_outputs):
        outputs = layer["activation"] if "activation" in layer else layer["wx+b"]
        num_neurons = outputs.shape[1]

        x = layer_idx * h_spacing
        for n in range(num_neurons):
            y = (max_neurons - num_neurons) / 2 + n

            # Draw neuron box
            rect = plt.Rectangle((x, y), layer_width, 0.8, edgecolor='black', facecolor='#e6f2ff')
            ax.add_patch(rect)
            ax.text(x + 0.5 * layer_width, y + 0.4, f"L{layer_idx+1}\nN{n+1}",
                    ha='center', va='center', fontsize=8)

            # Draw tiny plot below
            ax_inset = fig.add_axes([0.1 + 0.1*layer_idx, 0.05 + 0.1*n, 0.08, 0.05])
            ax_inset.plot(x_vals, outputs[:, n], color='blue')
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.set_title("", fontsize=6)
            ax_inset.grid(True)

    # Draw connections
    for layer_idx in range(len(layer_outputs) - 1):
        curr_outputs = layer_outputs[layer_idx]["activation"] if "activation" in layer_outputs[layer_idx] else layer_outputs[layer_idx]["wx+b"]
        next_outputs = layer_outputs[layer_idx + 1]["wx+b"]
        for i in range(curr_outputs.shape[1]):
            for j in range(next_outputs.shape[1]):
                x1 = layer_idx * h_spacing + layer_width
                y1 = (max_neurons - curr_outputs.shape[1]) / 2 + i + 0.4
                x2 = (layer_idx + 1) * h_spacing
                y2 = (max_neurons - next_outputs.shape[1]) / 2 + j + 0.4
                ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.6)

    ax.set_xlim(-1, len(layer_outputs)*h_spacing)
    ax.set_ylim(-1, max_neurons + 1)
    ax.axis('off')
    st.pyplot(fig)
