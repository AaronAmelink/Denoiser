import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 15
DEVICE = torch.device('cpu')
LOG_INTERVAL=500
LEARNING_RATE = 0.001

def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.ToTensor()),
  batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.ToTensor()),
  batch_size=BATCH_SIZE, shuffle=True)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # [B, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                # [B, 16, 14, 14]
            nn.Conv2d(16, 32, 3, padding=1),# [B, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                # [B, 32, 7, 7]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        noisy_data = add_noise(data)
        output = model(noisy_data)
        loss = F.binary_cross_entropy(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:  # Ignore target
            data = data.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, data, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average BCE loss: {:.4f}\n'.format(test_loss))

def main():
    model = DenoisingAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(1, EPOCHS):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
        scheduler.step()

        # Show the last image of the last epoch (reconstructed and original)

    model.eval()
    with torch.no_grad():
        data_iter = iter(test_loader)
        images, _ = next(data_iter)
        images = images.to(DEVICE)
        noised = add_noise(images)
        outputs = model(noised)
        # Take the last image in the batch
        noise_img = noised[-1].cpu().view(28, 28).numpy()
        orig_img = images[-1].cpu().view(28, 28).numpy()
        recon_img = outputs[-1].cpu().view(28, 28).numpy()

        fig, axs = plt.subplots(3, 5, figsize=(12, 5))
        for i in range(5):
            orig_img = images[i].cpu().view(28, 28).numpy()
            noise_img = noised[i].cpu().view(28, 28).numpy()
            recon_img = outputs[i].cpu().view(28, 28).numpy()
            axs[0, i].imshow(orig_img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            axs[0, i].set_title('Original')
            axs[0, i].axis('off')
            axs[1, i].imshow(noise_img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            axs[1, i].set_title('Noised')
            axs[1, i].axis('off')
            axs[2, i].imshow(recon_img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            axs[2, i].set_title('Reconstructed')
            axs[2, i].axis('off')
        plt.show()

if __name__ == '__main__':
    main()