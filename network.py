import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BOTTOM_DIM = 10
TOP_DIM = 10
NUM_LAYERS = 1
BATCH_SIZE = 5
NUM_EPOCHS = 1
LEARNING_RATE = 0.05
ITERATIONS = 50


class InputPairsDataset(Dataset):
    def __init__(self, num_samples, input_dim):
        # Generate pairs of indices, ensuring they match in your desired way
        # For simplicity, using identity matrix pairs here as placeholders
        self.bottom_inputs = [torch.eye(input_dim)[i].reshape(1, -1).squeeze(0) for i in range(input_dim)]
        self.top_inputs = [torch.eye(input_dim)[i].reshape(1, -1).squeeze(0) for i in range(input_dim)]

        self.input_dim = input_dim
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.bottom_inputs[idx % self.input_dim], self.top_inputs[idx % self.input_dim]


class LayerLocalNetwork(nn.Module):
    def __init__(self, bottom_dim, top_dim, num_layers=1, batch_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.optimizers = []

        for _ in range(num_layers):
            layer = {
                'bottom_up': nn.Parameter(torch.randn(bottom_dim, top_dim)),
                'top_down': nn.Parameter(torch.randn(top_dim, top_dim)),
                'recurrent': nn.Parameter(torch.randn(top_dim, top_dim))
            }
            self.layers.append(nn.ParameterDict(layer))
            self.optimizers.append({
                'bottom_up': optim.Adam([layer['bottom_up']], lr=LEARNING_RATE),
                'top_down': optim.Adam([layer['top_down']], lr=LEARNING_RATE),
                'recurrent': optim.Adam([layer['recurrent']], lr=LEARNING_RATE),
            })

        self.activations = [torch.zeros(batch_size, top_dim) for _ in range(num_layers)]

    def forward(self, bottom_input, top_input):
        for i, layer in enumerate(self.layers):
            bottom_up_act = torch.mm(bottom_input.detach(), layer['bottom_up']) if i == 0 else torch.mm(
                self.activations[i-1].detach(), layer['bottom_up'])
            top_down_act = torch.mm(top_input.detach(), layer['top_down']) if i == self.num_layers - 1 else torch.mm(
                self.activations[i+1].detach(), layer['top_down'])
            recurrent_act = torch.mm(self.activations[i].detach(), layer['recurrent'])

            total_input = bottom_up_act + top_down_act + recurrent_act
            self.activations[i] = F.leaky_relu(total_input)

        loss = self.compute_loss()

        for i, layer in enumerate(self.layers):
            self.optimizers[i]['bottom_up'].zero_grad()
            self.optimizers[i]['top_down'].zero_grad()
            self.optimizers[i]['recurrent'].zero_grad()

            loss.backward()

            self.optimizers[i]['bottom_up'].step()
            self.optimizers[i]['top_down'].step()
            self.optimizers[i]['recurrent'].step()

        return loss

    def compute_loss(self):
        # Standard loss computation
        running_sum = 0
        for act in self.activations:
            running_sum += torch.mean(act.pow(2))
        standard_loss = running_sum / len(self.activations)

        # Hebbian loss computation
        hebbian_loss = 0
        for act in self.activations:
            hebbian_loss += self.generate_lpl_loss_hebbian(act)

        # Combine losses
        total_loss = standard_loss + hebbian_loss  # Consider weighting factors if necessary
        return total_loss

    def generate_lpl_loss_hebbian(self, activations):
        mean_act = torch.mean(activations, dim=0)
        mean_subtracted = activations - mean_act
        sigma_squared = torch.sum(mean_subtracted ** 2, dim=0) / (activations.shape[0] - 1)
        loss = -torch.log(sigma_squared + 1e-10).sum() / sigma_squared.shape[0]
        return loss


dataset = InputPairsDataset(num_samples=100, input_dim=10)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Example usage:
model = LayerLocalNetwork(bottom_dim=BOTTOM_DIM, top_dim=TOP_DIM, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE)


for epoch in range(NUM_EPOCHS):
    print("Epoch:", epoch)
    for bottom_input, top_input in dataloader:
        for i in range(ITERATIONS):
            loss = model(bottom_input, top_input)
            print("Loss:", f"{loss.item(): .2f}")
    print()

print("======TRYING NEGATIVE SAMPLES======")


bottom_input = torch.eye(10)[0].reshape(1, -1)  # One-hot vector for bottom input
top_input = torch.eye(10)[1].reshape(1, -1)    # One-hot vector for top input

for i in range(75):
    loss = model(bottom_input, top_input)
    print("Loss:", f"{loss.item(): .2f}")


print("======TRYING POSITIVE SAMPLE AGAIN======")

for epoch in range(NUM_EPOCHS):
    print("Epoch:", epoch)
    for bottom_input, top_input in dataloader:
        for i in range(ITERATIONS):
            loss = model(bottom_input, top_input)
            print("Loss:", f"{loss.item(): .2f}")
    print()
