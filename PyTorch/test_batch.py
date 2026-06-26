import torch
import torch.optim as optim
from model import LYT
from losses import CombinedLoss

print("Initializing...")
device = torch.device('mps')
model = LYT().to(device)
criterion = CombinedLoss(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

inputs = torch.randn(1, 3, 256, 256).to(device)
targets = torch.randn(1, 3, 256, 256).to(device)

print("Forward model...")
outputs = model(inputs)
print("Forward loss...")
loss = criterion(outputs, targets)
print("Backward loss...")
loss.backward()
print("Optimizer step...")
optimizer.step()
print("Done!")
