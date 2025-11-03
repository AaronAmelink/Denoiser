import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from piq import ssim
from scipy.stats import multivariate_normal
from Metrics import overlay_error_heatmap, patchwise_errors, patch_error_vs_content, patch_content_variance, patch_content_edge, extract_latent_vectors, plot_tsne_umap

BATCH_SIZE = 64
EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_INTERVAL = 10
LEARNING_RATE = 1e-3
PATCH_SIZE = 16
STRIDE = 8
NOISE_FACTOR = 0.15
TITLE = f"PATCH {PATCH_SIZE}, STRIDE {STRIDE}"
SUBTITLE = f"NOISE FACTOR {NOISE_FACTOR}, EPOCHS {EPOCHS}"


def add_noise(inputs, noise_factor=NOISE_FACTOR):
    noisy = inputs + noise_factor * torch.randn_like(inputs).to(inputs.device)
    return torch.clamp(noisy, 0., 1.)

def run_all_diagnostics(model, device, test_dataset, test_loader, patch_size=PATCH_SIZE):
    print("Running patchwise error diagnostics...")
    mse_list, ssim_list = patchwise_errors(model, device, test_dataset)

    # Pick the first image id in test_dataset for visualization
    target_image_id = test_dataset.locations[0][0]

    print("Generating overlay error heatmap...")
    model.eval()
    with torch.no_grad():
        # reconstruct image from patches for heatmap overlay
        patches = []
        for i in range(len(test_dataset)):
            patch, noised, img_id = test_dataset[i]
            if img_id != target_image_id:
                continue
            noised = noised.unsqueeze(0).to(device)
            output = model(noised)
            patches.append(output.cpu().squeeze(0))
        recon_full = test_dataset.reconstruct_image(patches, target_image_id)

    overlay_error_heatmap(model, test_dataset, recon_full, target_image_id,
                          metric='mse', patch_size=patch_size, device=device)

    print("Analyzing error vs patch content variance...")
    #patch_error_vs_content(model, device, test_dataset, patch_content_variance)

    print("Analyzing error vs patch content edge magnitude...")
    #patch_error_vs_content(model, device, test_dataset, patch_content_edge)

    print("Extracting latent vectors for dimensionality reduction visualization...")
    latents = extract_latent_vectors(model, device, test_dataset, max_samples=1000)

    print("Plotting t-SNE of latent space...")
    plot_tsne_umap(latents, method='tsne')

    print("Plotting UMAP of latent space...")
    plot_tsne_umap(latents, method='umap')


class PatchDataset(Dataset):
    def __init__(self, root_dir, patch_size=32, stride=None, image_ids=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size  # default no overlap
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((256,256))
        self.transform = add_noise
        self.patches = []
        self.locations = []  # (image_id_in_dataset, top, left)
        self.image_info = []
        self.weight_mask = self._gaussian_weight_mask(self.patch_size).unsqueeze(0)  # shape [1, H, W]

        self._prepare_patches(image_ids)

    def _prepare_patches(self, image_ids):
        all_images = sorted([
            f for f in os.listdir(self.root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])

        if image_ids is None:
            image_ids = list(range(len(all_images)))

        for idx_in_dataset, image_idx in enumerate(image_ids):
            filename = all_images[image_idx]
            path = os.path.join(self.root_dir, filename)
            img = Image.open(path).convert('RGB')
            w, h = img.size
            if self.patch_size is not None and (w < self.patch_size or h < self.patch_size):
                continue  # Skip small images

            self.image_info.append((filename, w, h))

            # Sliding window with stride smaller than patch size for overlap
            if self.patch_size:
                for top in range(0, h - self.patch_size + 1, self.stride):
                    for left in range(0, w - self.patch_size + 1, self.stride):
                        patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                        self.patches.append(patch)
                        self.locations.append((idx_in_dataset, top, left))
            else:
                self.patches.append(img)
                self.locations.append(((idx_in_dataset, 0, 0)))
                
    def _gaussian_weight_mask(self, size):

        x, y = np.mgrid[0:size, 0:size]
        pos = np.dstack((x, y))
        mu = [size // 2, size // 2]
        sigma = [size / 4, size / 4]  # Adjust to control how soft the edges are
        rv = multivariate_normal(mean=mu, cov=[[sigma[0]**2, 0], [0, sigma[1]**2]])
        mask = rv.pdf(pos)
        mask = mask / mask.max()  # Normalize to 1.0
        return torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if (self.patch_size == None):
            patch = self.resize(patch)  # Ensure consistent size before stacking
        patch = self.to_tensor(patch)
        noised = self.transform(patch)
        image_id, top, left = self.locations[idx]
        return patch, noised, image_id

    def reconstruct_image(self, recon_patches, image_id):
        if self.patch_size is None:
            return recon_patches[0]  # No patching used

        filename, w, h = self.image_info[image_id]
        num_channels = recon_patches[0].shape[0]
        full_img = torch.zeros((num_channels, h, w), dtype=torch.float32)
        count_img = torch.zeros((num_channels, h, w), dtype=torch.float32)

        # Move weight mask to device only once
        weight = self.weight_mask.to(recon_patches[0].device)  # [1, H, W]

        for i, (img_id, top, left) in enumerate(self.locations):
            if img_id != image_id:
                continue

            patch = recon_patches[i] * weight  # Apply Gaussian weight mask
            full_img[:, top:top + self.patch_size, left:left + self.patch_size] += patch
            count_img[:, top:top + self.patch_size, left:left + self.patch_size] += weight

        count_img[count_img == 0] = 1  # Avoid division by zero
        return full_img / count_img


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(target)  # target is noisy input, data is clean
        ssim_value = ssim(output, data, data_range=1.0)  # images are in [0, 1]
        loss = 0.8 * F.mse_loss(output, data) + 0.2 * (1 - ssim_value)
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
        for (batch_idx, (data, target, _)) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(target)
            ssim_value = ssim(output, data, data_range=1.0)  # images are in [0, 1]
            test_loss += 0.8 * F.mse_loss(output, data) + 0.2 * (1 - ssim_value)
            if batch_idx % LOG_INTERVAL == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    1, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss.item()))
    test_loss /= len(test_loader)
    return test_loss


def main():
    # First, load all images to get total number of images
    root_dir = "./files/flickr30k_images"
    all_images = sorted([
        f for f in os.listdir(root_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])
    num_images = len(all_images)

    # Shuffle and split at image level
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    train_img_ids = indices[:int(0.8 * num_images)]
    test_img_ids = indices[int(0.8 * num_images):int(0.805 * num_images)]

    print(f"Total images: {num_images}, train: {len(train_img_ids)}, test: {len(test_img_ids)}")

    train_dataset = PatchDataset(root_dir, patch_size=PATCH_SIZE, stride=STRIDE, image_ids=train_img_ids)
    test_dataset = PatchDataset(root_dir, patch_size=PATCH_SIZE, stride=STRIDE, image_ids=test_img_ids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DenoisingAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    DATASET_LEN = len(train_loader.dataset)
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        print(f"Epoch {epoch} complete. Testing model...")
        AVERAGE_LOSS = test(model, DEVICE, test_loader)
        scheduler.step()
        
    print("Training complete.")
        
    # Visualization on first test image
    model.eval()
    with torch.no_grad():
        target_image_id = test_dataset.locations[0][0]  # Gets the first valid image_id
        patches = []
        noised_patches = []
        reconstructed_patches = []

        # Iterate all patches in test dataset to get patches from image_id_to_recon
        for i in range(len(test_dataset)):
            patch, noised, img_id = test_dataset[i]
            if img_id != target_image_id:
                continue
            patch = patch.unsqueeze(0).to(DEVICE)
            noised = noised.unsqueeze(0).to(DEVICE)

            output = model(noised)

            patches.append(patch.cpu().squeeze(0))
            noised_patches.append(noised.cpu().squeeze(0))
            reconstructed_patches.append(output.cpu().squeeze(0))
            
        # Reconstruct full images from patches
        original_full = test_dataset.reconstruct_image(patches, image_id=target_image_id)
        noised_full = test_dataset.reconstruct_image(noised_patches, image_id=target_image_id)
        recon_full = test_dataset.reconstruct_image(reconstructed_patches, image_id=target_image_id)

        # Convert to [H, W, 3] for plotting
        orig_np = original_full.permute(1, 2, 0).numpy()
        noise_np = noised_full.permute(1, 2, 0).numpy()
        recon_np = recon_full.permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(top=0.65)  # Add extra space at the top for titles
        fig.suptitle(TITLE, fontsize=16, y=0.98)
        fig.text(0.5, 0.92, SUBTITLE, ha='center', fontsize=10)
        fig.text(0.5, 0.89, f"Average Loss: {AVERAGE_LOSS}, Dataset Length: {DATASET_LEN}", ha='center', fontsize=10)
        print("Output shape:", recon_np.shape)
        axs[0].imshow(orig_np.clip(0, 1))
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(noise_np.clip(0, 1))
        axs[1].set_title("Noised Image")
        axs[1].axis('off')

        axs[2].imshow(recon_np.clip(0, 1))
        axs[2].set_title("Reconstructed Image")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()
        
        run_all_diagnostics(model, DEVICE, test_dataset, test_loader, patch_size=PATCH_SIZE)
    
    model_save_path = f"denoising_autoencoder_{PATCH_SIZE}_{STRIDE}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == '__main__':
    main()
