import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def plot_tensor(y_vals, x_vals, label, container):
    fig, ax = plt.subplots()
    y_vals = np.array(y_vals)

    if y_vals.ndim == 2:
        if x_vals is not None and x_vals.shape[0] == y_vals.shape[0]:
            for i in range(y_vals.shape[1]):
                ax.plot(x_vals, y_vals[:, i], label=f"Neuron {i+1}")
        else:
            for i in range(y_vals.shape[0]):
                ax.plot(range(y_vals.shape[1]), y_vals[i], label=f"Input {i+1}")
    else:
        if x_vals is not None and len(x_vals) == len(y_vals):
            ax.plot(x_vals, y_vals, label=label)
        else:
            ax.plot(range(len(y_vals)), y_vals, label=label)

    ax.set_title(label)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True)
    if x_vals is not None:
        ax.set_xlim(min(x_vals), max(x_vals))
    container.pyplot(fig)


def plot_weight_heatmap(w, label, container):
    fig, ax = plt.subplots()
    cax = ax.imshow(w, aspect="auto", cmap="coolwarm")
    ax.set_title(f"{label} Heatmap")
    ax.set_xlabel("Input Neurons")
    ax.set_ylabel("Output Neurons")
    fig.colorbar(cax, ax=ax)
    container.pyplot(fig)

def plot_weight_histogram(w, label, container):
    fig, ax = plt.subplots()
    ax.hist(w.flatten(), bins=10, color="#1f77b4", edgecolor="black")
    ax.set_title(f"{label} Histogram")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    container.pyplot(fig)
