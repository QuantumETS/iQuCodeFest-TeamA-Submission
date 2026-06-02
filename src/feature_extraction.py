import numpy as np
from dataset_loader import load_brain_tumor_dataset
import matplotlib.pyplot as plt
from pathlib import Path
from plotting import save_fig


N_SAMPLES = 10

X, y = load_brain_tumor_dataset("data/archive/Data")


def get_random_subset(X, y, n_samples):
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    return X[indices], y[indices]


X_random_subset, y_random_subset = get_random_subset(X, y, N_SAMPLES)

print("Subset of images shape:", X_random_subset.shape)
print("Subset of labels shape:", y_random_subset.shape)


def show_sample_images(images, labels, num_images=10):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(num_images):
        img, lbl = images[i], labels[i]
        ax = axes[i // 5, i % 5]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Label: {lbl}")
        ax.axis("off")
    plt.show()


# show_sample_images(X_random_subset, y_random_subset, num_images=N_SAMPLES)


def extract_with_PCA(x_train, x_test, n_components=50):
    # Retire la dimension du canal et aplatit chaque image.
    # Forme initiale : [N, 1, 28, 28]
    # Forme finale : [N, 784]
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    print(
        f"Values before scaling: mean={x_train_flat.mean():.4f}, std={x_train_flat.std():.4f}"
    )

    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)

    print(
        f"Values after scaling: mean={x_train_scaled.mean():.4f}, std={x_train_scaled.std():.4f}"
    )

    from sklearn.decomposition import PCA

    n_features = 4

    pca = PCA(n_components=n_features)

    x_train_reduced = pca.fit_transform(x_train_scaled)
    x_test_reduced = pca.transform(x_test_scaled)

    print(f"Shape after PCA: {x_train_reduced.shape}, {x_test_reduced.shape}")

    def visualize_preprocessing_and_pca(n_features=8, idx=0):
        """
        Visualize preprocessing steps and PCA features for one example image.

        Parameters:
        - idx: index of the image to visualize (default: 0)
        - n_features: number of PCA components to display (default: 8)
        """
        # Original tensor image
        orig = x_train[idx].squeeze()
        # Flattened view reshaped back
        # flat = x_train_flat[idx].reshape(256, 256)
        # Inverse transform a scaled sample for visualization
        # scaled_unscaled = scaler.inverse_transform(
        #     x_train_scaled[idx].reshape(1, -1)
        # ).reshape(256, 256)
        # PCA reconstruction (inverse transform -> then inverse scale -> reshape)
        pca_recon_scaled = pca.inverse_transform(x_train_reduced[idx : idx + 1])
        pca_recon = scaler.inverse_transform(pca_recon_scaled).reshape(256, 256)

        # Principal components as images
        components = pca.components_.reshape(pca.n_components_, 256, 256)

        # Create figure with 4 + n_features subplots
        num_cols = min(4, n_features + 4)
        num_rows = (n_features + 4 + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
        axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

        # Plot preprocessing steps
        axes[0].imshow(orig, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[3].imshow(pca_recon, cmap="gray")
        axes[3].set_title("PCA Reconstruction")
        axes[3].axis("off")

        # Plot all n_features principal components
        for i in range(n_features):
            ax = axes[4 + i]
            im = ax.imshow(components[i], cmap="seismic")
            ax.set_title(f"PC {i + 1}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # Hide unused subplots
        for i in range(4 + n_features, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    # Call the function
    visualize_preprocessing_and_pca(n_features, idx=0)


# extract_with_PCA(X_random_subset, X_random_subset, n_components=8)


def extract_with_convolution(x_train, x_test, n_features=4):
    from torch import nn
    import torch

    def _to_channel_first_tensor(images):
        if not torch.is_tensor(images):
            images = torch.from_numpy(images)

        images = images.float()

        # Convert grayscale images from [N, H, W, 1] to [N, 1, H, W].
        if images.ndim == 4 and images.shape[-1] == 1:
            images = images.permute(0, 3, 1, 2)

        return images

    class SimpleConvPreprocessor(nn.Module):
        def __init__(self, n_features=4):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=n_features, kernel_size=2, stride=2
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.AdaptiveAvgPool2d((13, 13)),
                nn.Flatten(),
                nn.Linear(n_features * 13 * 13, n_features),
            )

        def forward(self, x):
            return self.features(x)

    import matplotlib.pyplot as plt
    import visualtorch
    from torch import nn

    conv_preprocessor = SimpleConvPreprocessor(n_features)

    img = visualtorch.layered_view(
        conv_preprocessor,
        input_shape=(1, 1, 256, 256),
        legend=True,
        scale_xy=3,
        scale_z=0.3,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.imshow(img)
    fig = plt.gcf()

    save_fig(
        Path("figures") / "feature_extraction",
        filename="conv_preprocessor.png",
        fig=fig,
    )

    x_train_torch = _to_channel_first_tensor(x_train)
    x_test_torch = _to_channel_first_tensor(x_test)

    with torch.no_grad():
        x_train_features = conv_preprocessor(x_train_torch)
        x_test_features = conv_preprocessor(x_test_torch)

    x_train_reduced = x_train_features.numpy()
    x_test_reduced = x_test_features.numpy()

    print(
        f"Shape after convolutional feature extraction: {x_train_reduced.shape}, {x_test_reduced.shape}"
    )

    def visualize_preprocessing_and_convolution(n_features=8, idx=0):
        orig = x_train_torch[idx].squeeze().detach().cpu().numpy()

        with torch.no_grad():
            # Keep the convolutional feature maps before Flatten/Linear
            conv_output = conv_preprocessor.features[:3](x_train_torch[idx : idx + 1])

        feature_maps = conv_output.squeeze(0).cpu().numpy()  # [4, 13, 13]
        num_maps = min(feature_maps.shape[0], 8)

        fig, axes = plt.subplots(1, 1 + num_maps, figsize=(12, 3))
        axes = np.asarray(axes).reshape(-1)

        axes[0].imshow(orig, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        for i in range(num_maps):
            im = axes[1 + i].imshow(feature_maps[i], cmap="viridis")
            axes[1 + i].set_title(f"Feature Map {i + 1}")
            axes[1 + i].axis("off")
            fig.colorbar(im, ax=axes[1 + i], fraction=0.046, pad=0.02)

        plt.tight_layout()
        fig = plt.gcf()

        save_fig(
            Path("figures") / "feature_extraction",
            filename=f"conv_features_{idx}.png",
            fig=fig,
        )

    for i in range(5):
        visualize_preprocessing_and_convolution(n_features, idx=i)


extract_with_convolution(X_random_subset, X_random_subset, n_features=8)
