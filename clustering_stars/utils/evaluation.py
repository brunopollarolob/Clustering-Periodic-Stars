"""
Evaluation utilities for clustering models.
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def train_and_evaluate(model_class, train_loader, val_loader, test_loader, num_repeats=5, epochs=50, lr=0.001, in_dim=50, z_dim=10, num_classes=5):
    """
    Train and evaluate a clustering model multiple times, reporting average performance metrics.
    
    Args:
        model_class: The model class to instantiate (e.g., AutoencoderMCDSVDD)
        train_loader: PyTorch DataLoader for training data
        val_loader: PyTorch DataLoader for validation data
        test_loader: PyTorch DataLoader for test data
        num_repeats: Number of times to train and evaluate the model
        epochs: Number of training epochs
        lr: Learning rate for optimization
        in_dim: Input dimension for the model
        z_dim: Latent space dimension
        num_classes: Number of classes in the dataset
        
    Returns:
        dict: Dictionary containing lists of metrics and their summaries
    """
    silhouette_scores = []
    davies_bouldin_scores = []

    for i in range(num_repeats):
        print(f"Run {i + 1}/{num_repeats}")

        # Initialize a new model
        model = model_class(in_dim=in_dim, z_dim=z_dim, num_classes=num_classes)

        # Train Autoencoder
        print("Training Autoencoder...")
        model.train_autoencoder(train_loader, val_loader, epochs=epochs, lr=lr)

        # Set centers for MCDSVDD
        model.set_centers(train_loader)

        # Train MCDSVDD
        print("Training MCDSVDD...")
        model.train_mcdsvdd(train_loader, val_loader, epochs=epochs, lr=lr)

        # Evaluate the latent space
        print("Evaluating...")
        model.set_centers(test_loader)
        z = model.get_latent_space(test_loader)
        z_feats = z[0].cpu().numpy()
        z_labels = z[1].cpu().numpy()

        # Compute metrics
        silhouette = silhouette_score(z_feats, z_labels)
        davies_bouldin = davies_bouldin_score(z_feats, z_labels)

        print(f"Run {i + 1} - Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {davies_bouldin:.4f}")

        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)

    # Compute mean and standard deviation
    silhouette_mean = np.mean(silhouette_scores)
    silhouette_std = np.std(silhouette_scores)
    davies_bouldin_mean = np.mean(davies_bouldin_scores)
    davies_bouldin_std = np.std(davies_bouldin_scores)

    print("\nFinal Results:")
    print(f"Silhouette Score: {silhouette_mean:.4f} ± {silhouette_std:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_mean:.4f} ± {davies_bouldin_std:.4f}")

    return {
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "silhouette_mean": silhouette_mean,
        "silhouette_std": silhouette_std,
        "davies_bouldin_mean": davies_bouldin_mean,
        "davies_bouldin_std": davies_bouldin_std
    }