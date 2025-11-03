import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from piq import ssim
import seaborn as sns
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import pearsonr
import cv2
from sklearn.manifold import TSNE
import umap as umap
from torch.utils.data import DataLoader


def overlay_error_heatmap(model, dataset, recon_full, target_image_id, metric='mse', patch_size=32, device='cuda'):

    error_map = torch.zeros((recon_full.shape[1], recon_full.shape[2]))
    count_map = torch.zeros_like(error_map)

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            patch, noised, img_id = dataset[i]
            if img_id != target_image_id:
                continue

            patch = patch.unsqueeze(0).to(device)
            noised = noised.unsqueeze(0).to(device)
            output = model(noised)

            if metric == 'mse':
                error = F.mse_loss(output, patch).item()
            elif metric == 'ssim':
                error = 1 - ssim(output, patch, data_range=1.0).item()
            else:
                raise ValueError("Unsupported metric. Use 'mse' or 'ssim'.")

            _, top, left = dataset.locations[i]
            error_map[top:top + patch_size, left:left + patch_size] += error
            count_map[top:top + patch_size, left:left + patch_size] += 1

    count_map[count_map == 0] = 1
    avg_error_map = error_map / count_map
    avg_error_np = avg_error_map.numpy()

    recon_np = recon_full.permute(1, 2, 0).numpy().clip(0, 1)
    two_color_cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

    plt.figure(figsize=(8, 6))
    plt.imshow(recon_np)
    plt.imshow(avg_error_np, cmap=two_color_cmap, alpha=0.5, norm=Normalize(vmin=0, vmax=np.percentile(avg_error_np, 99)))
    plt.colorbar(label=f"Patch Error ({metric.upper()})")
    plt.title("Heatmap of Reconstruction Error Overlayed on Reconstructed Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def patchwise_errors(model, device, dataset):
    model.eval()
    mse_list = []
    ssim_list = []

    with torch.no_grad():
        for i in range(max(len(dataset), 500)):
            clean, noisy, _ = dataset[i]
            clean = clean.unsqueeze(0).to(device)
            noisy = noisy.unsqueeze(0).to(device)

            output = model(noisy)

            mse = F.mse_loss(output, clean).item()
            ssim_val = ssim(output, clean, data_range=1.0).item()

            mse_list.append(mse)
            ssim_list.append(ssim_val)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(mse_list, bins=50, ax=axs[0], color='blue')
    axs[0].set_title('Patch-wise MSE Distribution')
    axs[0].set_xlabel('MSE')
    axs[0].set_ylabel('Count')

    sns.histplot(ssim_list, bins=50, ax=axs[1], color='green')
    axs[1].set_title('Patch-wise SSIM Distribution')
    axs[1].set_xlabel('SSIM')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    return mse_list, ssim_list



def patch_content_variance(patch_tensor):
    # patch_tensor shape: [C, H, W], values in [0, 1]
    patch_np = patch_tensor.numpy()
    return patch_np.var()

def patch_content_edge(patch_tensor):
    # Convert patch to grayscale and compute edge magnitude using Sobel
    patch_np = patch_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    gray = cv2.cvtColor((patch_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return edge_mag.mean()

def patch_error_vs_content(model, device, dataset, content_func):
    errors = []
    contents = []

    model.eval()
    with torch.no_grad():
        for clean, noisy, _ in dataset:
            clean = clean.to(device)
            noisy = noisy.to(device).unsqueeze(0)  # add batch dim

            denoised = model(noisy)
            denoised = denoised.squeeze(0)  # remove batch dim

            error = F.mse_loss(denoised, clean, reduction='mean').cpu().item()
            content = content_func(clean.cpu())  # ensure CPU tensor or numpy

            errors.append(error)
            contents.append(content)

    errors = np.array(errors)
    contents = np.array(contents)

    # Compute Pearson correlation
    corr, p_val = pearsonr(errors, contents)
    print(f"Pearson correlation: {corr:.4f} (p={p_val:.4g})")

    # Plot
    plt.scatter(contents, errors, alpha=0.5)
    plt.xlabel("Patch Content")
    plt.ylabel("Denoising Error")
    plt.title(f"Error vs. {content_func.__name__}\nPearson r = {corr:.4f}")
    plt.grid(True)
    plt.show()
    

def extract_latent_vectors(model, device, dataset, max_samples=1000, batch_size=1):
    model.eval()
    latents = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        total_samples = 0
        for batch in loader:
            patch, noisy, _ = batch
            noisy = noisy.to(device)
            latent = model.encoder(noisy)
            latent_flat = latent.reshape(latent.size(0), -1).cpu().numpy()
            latents.append(latent_flat)

            total_samples += latent_flat.shape[0]
            if total_samples >= max_samples:
                break

    latents = np.concatenate(latents, axis=0)
    return latents[:max_samples]  # trim in case of overflow

def plot_tsne_umap(latents, method='tsne', labels=None):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedding = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, palette='tab10', s=10, legend='full')
    else:
        sns.scatterplot(x=embedding[:,0], y=embedding[:,1], s=10)

    plt.title(f'{method.upper()} Visualization of Latent Space')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.show()