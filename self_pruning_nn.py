import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Prunable Linear Layer
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

# 2. Neural Network
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

# 3. Sparsity Loss
def sparsity_loss(model):
    loss = 0
    for gates in model.get_all_gates():
        loss += torch.sum(gates)
    return loss

# 4. Plot Functions (BONUS)
def plot_gate_distribution(model, save_path="gate_distribution.png"):
    all_gates = []
    for g in model.get_all_gates():
        all_gates.extend(g.detach().cpu().numpy().flatten())
    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value (0 = pruned, 1 = active)")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")
    plt.show()


def plot_lambda_vs_metrics(results):
    lambdas = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    sparsities = [r[2] for r in results]
    plt.figure()
    plt.plot(lambdas, accuracies, marker='o', label="Accuracy")
    plt.plot(lambdas, sparsities, marker='s', label="Sparsity")
    plt.xscale("log")
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Value")
    plt.title("Lambda vs Accuracy & Sparsity")
    plt.legend()
    plt.savefig("lambda_vs_metrics.png")
    print("Lambda vs metrics plot saved!")
    plt.show()

# 5. Data Loaders (with normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# 6. Training Function
def train(model, device, loader, optimizer, lambda_sparse):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ce_loss = F.cross_entropy(output, target)
        sp_loss = sparsity_loss(model)
        loss = ce_loss + lambda_sparse * sp_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 7. Evaluation
def test(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100 * correct / len(loader.dataset)

# 8. Sparsity Calculation
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0
    for gates in model.get_all_gates():
        total += gates.numel()
        pruned += torch.sum(gates < threshold).item()
    return 100 * pruned / total

# 9. Main Experiment
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    lambdas = [1e-5, 1e-4, 1e-3]
    results = []
    best_model = None
    best_acc = 0
    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")
        model = PrunableNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            loss = train(model, device, train_loader, optimizer, lam)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        acc = test(model, device, test_loader)
        sparsity = calculate_sparsity(model)
        results.append((lam, acc, sparsity))
        print(f"Lambda: {lam}, Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            torch.save(model, "best_model.pth")
    # Final Results
    print("\nFINAL RESULTS")
    print("Lambda\tAccuracy\tSparsity")
    for r in results:
        print(f"{r[0]}\t{r[1]:.2f}%\t\t{r[2]:.2f}%")
    plot_gate_distribution(best_model)
    plot_lambda_vs_metrics(results)
if __name__ == "__main__":
    main()
