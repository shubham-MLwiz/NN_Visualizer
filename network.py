import torch.nn as nn

def build_network(input_size, hidden_size, depth, output_size, activation):
    act_fn_map = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(),
    }

    act_fn = act_fn_map[activation]

    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(act_fn)
        input_size = hidden_size

    layers.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*layers)