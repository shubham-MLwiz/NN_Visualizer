# Neural Network Forward Visualizer (Streamlit)

Interactive app to visualize each transformation through a neural network:
- Inputs, weights, biases, linear outputs (wx + b), and activations
- Support for different activation functions and network shapes
- Layer-by-layer plots for batch input
- Reproducible results using seed
- Modular design for future extensions (backprop, gradients, etc.)

# How to run the app

`streamlit run app.py`

### RuntimeError: Tried to instantiate class '__path__._path'
That error is a known non-breaking issue in certain versions of Streamlit and PyTorch.
It happens during hot-reload in development mode. Streamlit tries to introspect all modules to detect file changes. When it reaches into PyTorch internals (torch.classes), it hits something that isn't meant to be introspected — especially with custom class registration (__path__._path).

This is not your fault and doesn't impact your app's actual functionality.
It's a known harmless **PyTorch x Streamlit** logging issue
