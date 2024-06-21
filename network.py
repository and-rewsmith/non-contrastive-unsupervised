import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LEARNING_RATE = 0.05


class LayerLocalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, batch_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.optimizers = []

        # Create the layers with weights for bottom-up, top-down, and recurrent connections
        for _ in range(num_layers):
            layer = {
                'bottom_up': nn.Parameter(torch.randn(input_dim, output_dim)),
                'top_down': nn.Parameter(torch.randn(output_dim, output_dim)),
                'recurrent': nn.Parameter(torch.randn(output_dim, output_dim))
            }
            self.layers.append(nn.ParameterDict(layer))

            # Create separate optimizers for each set of parameters in the layer
            self.optimizers.append({
                'bottom_up': optim.Adam([layer['bottom_up']], lr=LEARNING_RATE),
                'top_down': optim.Adam([layer['top_down']], lr=LEARNING_RATE),
                'recurrent': optim.Adam([layer['recurrent']], lr=LEARNING_RATE),
            })

        self.activations = [torch.zeros(batch_size, output_dim) for _ in range(num_layers)]

    def forward(self, bottom_input, top_input):
        for i, layer in enumerate(self.layers):
            bottom_up_act = torch.mm(bottom_input.detach(), layer['bottom_up']) if i == 0 else torch.mm(
                self.activations[i-1].detach(), layer['bottom_up'])
            top_down_act = torch.mm(top_input.detach(), layer['top_down']) if i == self.num_layers - \
                1 else torch.mm(self.activations[i+1].detach(), layer['top_down'])
            recurrent_act = torch.mm(self.activations[i].detach(), layer['recurrent'])

            # Sum contributions and apply non-linearity
            total_input = bottom_up_act + top_down_act + recurrent_act
            self.activations[i] = F.leaky_relu(total_input)

        # Compute loss
        running_sum = 0
        for act in self.activations:
            running_sum += torch.mean(act.pow(2))
        running_sum /= len(self.activations)

        loss = running_sum

        # Update weights using their respective optimizers
        for i, layer in enumerate(self.layers):
            self.optimizers[i]['bottom_up'].zero_grad()
            self.optimizers[i]['top_down'].zero_grad()
            self.optimizers[i]['recurrent'].zero_grad()

            loss.backward()  # Compute gradients for this layer's weights

            self.optimizers[i]['bottom_up'].step()
            self.optimizers[i]['top_down'].step()
            self.optimizers[i]['recurrent'].step()

        return loss


# Example usage:
model = LayerLocalNetwork(input_dim=10, output_dim=10, num_layers=1, batch_size=1)
bottom_input = torch.eye(10)[0].reshape(1, -1)  # One-hot vector for bottom input
top_input = torch.eye(10)[9].reshape(1, -1)    # One-hot vector for top input

for i in range(0, 100):
    loss = model(bottom_input, top_input)
    print("Loss:", f"{loss.item(): .2f}")
