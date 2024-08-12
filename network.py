from collections import deque
import math

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

ITERATIONS = 50
NUM_EPOCHS = 100

INPUT_DIM = 10
NUM_LAYERS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.01

HIDDEN_DIM = 10

# STANDARD_LOSS_SCALE = 30
# HEBBIAN_LOSS_SCALE = 1
# PREDICTIVE_LOSS_SCALE = 10

STANDARD_LOSS_SCALE = 30
HEBBIAN_LOSS_SCALE = 15
PREDICTIVE_LOSS_SCALE = 1


class InputPairsDataset(Dataset):
    def __init__(self, num_samples, input_dim):
        # Generate pairs of indices, ensuring they match in your desired way
        # For simplicity, using identity matrix pairs here as placeholders
        self.inputs = [torch.eye(input_dim)[i].reshape(1, -1).squeeze(0) for i in range(input_dim)]

        self.input_dim = input_dim
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx % self.input_dim], self.inputs[idx % self.input_dim], self.inputs[(idx + 1) % self.input_dim]


def amplified_initialization_(in_features: int, param: torch.Tensor, amplification_factor: float = 3.0) -> None:
    """Amplified initialization for Linear layers."""
    # Compute the standard deviation for He initialization
    std = (2.0 / in_features) ** 0.5
    # Amplify the standard deviation
    amplified_std = std * amplification_factor
    # Initialize weights with amplified standard deviation
    nn.init.normal_(param, mean=0, std=amplified_std)


class LayerLocalNetwork(nn.Module):
    def __init__(self, bottom_dim, top_dim, num_layers=1, batch_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.optimizers = []

        for n in range(num_layers):
            layer = {
                'bottom_up': nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM)),
                'top_down': nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM)),
                'recurrent': nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM))
            }

            if n == num_layers - 1:
                amplified_initialization_(bottom_dim, layer['top_down'])
            else:
                nn.init.uniform_(layer['top_down'], -0.05, 0.05)

            nn.init.kaiming_uniform_(layer['bottom_up'])
            nn.init.orthogonal_(layer['recurrent'], gain=math.sqrt(2))

            self.layers.append(nn.ParameterDict(layer))
            self.optimizers.append({
                'bottom_up': optim.Adam([layer['bottom_up']], lr=LEARNING_RATE),
                'top_down': optim.Adam([layer['top_down']], lr=LEARNING_RATE),
                'recurrent': optim.Adam([layer['recurrent']], lr=LEARNING_RATE),
            })

        self.activations = [torch.zeros(batch_size, top_dim) for _ in range(num_layers)]

    def resize_activations(self, new_batch_size):
        # don't zero activations, just trim them
        for i in range(self.num_layers):
            self.activations[i] = self.activations[i][:new_batch_size]

    def forward(self, bottom_input, top_input):
        for i, layer in enumerate(self.layers):
            self.optimizers[i]['bottom_up'].zero_grad()
            self.optimizers[i]['top_down'].zero_grad()
            self.optimizers[i]['recurrent'].zero_grad()

        old_activations = [act.detach().clone() for act in self.activations]
        for i, layer in enumerate(self.layers):
            bottom_up_act = torch.mm(bottom_input.detach(), layer['bottom_up']) if i == 0 else torch.mm(
                old_activations[i-1], layer['bottom_up'])
            top_down_act = torch.mm(top_input.detach(), layer['top_down']) if i == self.num_layers - 1 else torch.mm(
                old_activations[i+1], layer['top_down'])
            recurrent_act = torch.mm(old_activations[i], layer['recurrent'])

            total_input = bottom_up_act + top_down_act + recurrent_act
            total_input = F.leaky_relu(total_input)
            # print("total_input: ", total_input.mean())
            # self.activations[i] = torch.clamp(total_input, min=-1, max=1)
            self.activations[i] = total_input

        loss, energy = self.compute_energy(self.activations)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

        for i, layer in enumerate(self.layers):
            self.optimizers[i]['bottom_up'].step()
            self.optimizers[i]['top_down'].step()
            self.optimizers[i]['recurrent'].step()

        for i in range(0, len(self.activations)):
            self.activations[i] = self.activations[i].detach()

        return loss, energy

    # TODO: maybe this should be FF similar softmax
    # TODO: decorrelative loss
    def compute_energy(self, old_activations: Tensor):
        # Push energy down proportional to activations
        running_sum = 0
        for act in self.activations:
            # pow the activations, average across neurons in layer, average across batches
            running_sum += torch.mean(torch.mean(act.pow(2), dim=1), dim=0)  # type: ignore
        standard_loss = running_sum / len(self.activations)

        # Hebbian loss computation: encourage variance at neuron level
        hebbian_loss = 0
        for act in self.activations:
            hebbian_loss += self.generate_lpl_loss_hebbian(act)

        # predictive_loss = 0
        # for i, act in enumerate(self.activations):
        #     individual_predictive_loss = (act - old_activations[i]) ** 2
        #     individual_predictive_loss = torch.sum(individual_predictive_loss, dim=1)
        #     individual_predictive_loss = torch.sum(individual_predictive_loss, dim=0)
        #     individual_predictive_loss = individual_predictive_loss / (2 * act.shape[0] * act.shape[1])
        #     predictive_loss += individual_predictive_loss

        # Combine losses
        standard_loss = STANDARD_LOSS_SCALE * standard_loss
        hebbian_loss = HEBBIAN_LOSS_SCALE * hebbian_loss
        # predictive_loss = PREDICTIVE_LOSS_SCALE * predictive_loss

        # total_loss = standard_loss + hebbian_loss + predictive_loss
        # wandb.log({"standard_loss": standard_loss, "hebbian_loss": hebbian_loss,
        #           "predictive_loss": predictive_loss, "total_loss": total_loss})

        # total_loss = standard_loss
        # wandb.log({"standard_loss": standard_loss, "total_loss": total_loss})

        total_loss = standard_loss + hebbian_loss
        wandb.log({"standard_loss": standard_loss, "hebbian_loss": hebbian_loss, "total_loss": total_loss})

        return total_loss, standard_loss

    def generate_lpl_loss_hebbian(self, activations):
        mean_act = torch.mean(activations, dim=0)
        mean_subtracted = activations - mean_act
        sigma_squared = torch.sum(mean_subtracted ** 2, dim=0) / (activations.shape[0] - 1)
        loss = -torch.log(sigma_squared + 1e-10).sum() / sigma_squared.shape[0]
        return loss


if __name__ == "__main__":
    # Set print options for tensors
    torch.set_printoptions(threshold=5000, edgeitems=2000)
    torch.manual_seed(1234)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(
        project="non-contrastive-unsupervised",
        config={
            "architecture": "multilayer-regularized",
        }
    )

    dataset = InputPairsDataset(num_samples=100, input_dim=INPUT_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Example usage:
    model = LayerLocalNetwork(bottom_dim=INPUT_DIM, top_dim=INPUT_DIM, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE)

    print("======TRYING POSITIVE SAMPLE FULL BATCH SIZE======")
    wandb.log({"scenario": 0})

    for epoch in range(NUM_EPOCHS):
        print("Epoch:", epoch)
        for bottom_input, top_input, _ in dataloader:
            running_sum = 0
            for i in range(ITERATIONS):
                loss, energy = model(bottom_input, top_input)
                running_sum += energy.item()
                wandb.log({"energy": energy})

            wandb.log({"average_energy": running_sum / ITERATIONS})

    print("======TRYING NEGATIVE SAMPLES FULL BATCH SIZE======")
    wandb.log({"scenario": 1})

    for epoch in range(NUM_EPOCHS):
        print("Epoch:", epoch)
        for bottom_input, top_input, _ in dataloader:
            # shuffle the top input along first dimension
            top_input = top_input[torch.randperm(top_input.size()[0])]
            bottom_input = bottom_input[torch.randperm(bottom_input.size()[0])]

            running_sum = 0
            for i in range(ITERATIONS):
                loss, energy = model(bottom_input, top_input)
                running_sum += energy.item()
                wandb.log({"energy": energy})
            wandb.log({"average_energy": running_sum / ITERATIONS})

    # print("======TRYING POSITIVE SAMPLE LOW BATCH SIZE======")
    # wandb.log({"scenario": 2})
    # new_batch_size = 5
    # model.resize_activations(new_batch_size)

    # for epoch in range(NUM_EPOCHS):
    #     print("Epoch:", epoch)
    #     for bottom_input, top_input, _ in dataloader:
    #         running_sum = 0
    #         for i in range(ITERATIONS):
    #             bottom_input = bottom_input[0:new_batch_size]
    #             top_input = bottom_input[0:new_batch_size]
    #             assert bottom_input.shape == (new_batch_size, INPUT_DIM)
    #             assert top_input.shape == (new_batch_size, INPUT_DIM)

    #             loss, energy = model(bottom_input, top_input)
    #             running_sum += energy.item()
    #             wandb.log({"energy": energy})
    #         wandb.log({"average_energy": running_sum / ITERATIONS})

    # print("======TRYING NEGATIVE SAMPLES LOW BATCH SIZE======")
    # wandb.log({"scenario": 3})
    # new_batch_size = 5

    # for epoch in range(NUM_EPOCHS):
    #     print("Epoch:", epoch)
    #     for bottom_input, top_input, _ in dataloader:
    #         running_sum = 0
    #         for i in range(ITERATIONS):
    #             bottom_input = bottom_input[0:new_batch_size]
    #             top_input = bottom_input[0:new_batch_size]
    #             assert bottom_input.shape == (new_batch_size, INPUT_DIM)
    #             assert top_input.shape == (new_batch_size, INPUT_DIM)

    #             # for every top vec in batch, make it a one hot vector with first index as 1
    #             top_input = torch.eye(INPUT_DIM)[0].reshape(1, -1).repeat(new_batch_size, 1)

    #             loss, energy = model(bottom_input, top_input)
    #             running_sum += energy.item()
    #             wandb.log({"energy": energy})
    #         wandb.log({"average_energy": running_sum / ITERATIONS})
