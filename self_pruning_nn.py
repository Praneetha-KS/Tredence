import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Part 1: Prunable Linear Layer
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# Neural Network using prunable layers
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_all_gates(self):
        return [
            self.fc1.get_gates(),
            self.fc2.get_gates(),
            self.fc3.get_gates()
        ]

# Sparsity Loss (L1 on gates)
def sparsity_loss(model):
    loss = 0
    for gates in model.get_all_gates():
        loss += torch.sum(gates)
    return loss

# Data Loaders (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Training Function
def train(model, device, train_loader, optimizer, lambda_sparse):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ce_loss = F.cross_entropy(output, target)
        sp_loss = sparsity_loss(model)
        loss = ce_loss + lambda_sparse * sp_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Sparsity Calculation
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0
    for gates in model.get_all_gates():
        total += gates.numel()
        pruned += torch.sum(gates < threshold).item()
    return 100 * pruned / total

# Main Experiment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambdas = [1e-5, 1e-4, 1e-3]  # low, medium, high
results = []
for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")
    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):  # you can increase
        loss = train(model, device, train_loader, optimizer, lam)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    acc = test(model, device, test_loader)
    sparsity = calculate_sparsity(model)
    results.append((lam, acc, sparsity))
    print(f"Lambda: {lam}, Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")

print("\nFinal Results:")
print("Lambda\tAccuracy\tSparsity")
for r in results:
    print(f"{r[0]}\t{r[1]:.2f}\t\t{r[2]:.2f}")