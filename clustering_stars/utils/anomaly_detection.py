"""
Anomaly detection utilities for clustering stars.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.metrics import silhouette_score, davies_bouldin_score

def plot_umap_projection(data_loader, title, model, label_encoder=None):
    """
    Plot 2D projection of the latent space with UMAP and identify anomalies.
    
    Args:
        data_loader: PyTorch DataLoader containing the data
        title: String title for the plot
        model: Trained model with forward and set_centers methods
        label_encoder: Optional LabelEncoder used to convert numeric labels to class names
        
    Returns:
        predictions: Tensor indicating whether each point is normal (0) or anomalous (1)
        davies_bouldin: Davies-Bouldin score for the projection
        silhouette: Silhouette score for the projection
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Lists to store anomaly scores and labels
    anomaly_scores = []
    labels = []

    # Iterate through the data loader to get anomaly scores
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.float(), y.long()
            z, _ = model.forward(x)

            # Calculate anomaly scores (distance to the centers)
            scores = torch.norm(z - model.c[y], p=2, dim=1)  # Use the center corresponding to each class
            anomaly_scores.append(scores)
            labels.append(y)

    # Concatenate results into a single tensor
    anomaly_scores = torch.cat(anomaly_scores)
    labels = torch.cat(labels)

    # Calculate the mean radius for each class
    num_classes = len(torch.unique(labels))  # Assuming classes are numbered from 0 to n-1
    radios = {}

    for i in range(num_classes):
        # Filter the scores for class i
        class_scores = anomaly_scores[labels == i]

        # Calculate the mean radius for class i
        mean_radius = class_scores.mean().item() + class_scores.std().item()
        radios[i] = mean_radius

    # Classify examples as anomalies if their anomaly score > mean radius of their class
    predictions = []

    for i in range(len(anomaly_scores)):
        # Get the class of the current example
        class_id = labels[i].item()

        # Compare the anomaly score with the mean radius of the class
        if anomaly_scores[i].item() > radios[class_id]:
            predictions.append(1)  # 1 indicates anomaly
        else:
            predictions.append(0)  # 0 indicates normality

    # Convert predictions to a tensor
    predictions = torch.tensor(predictions)

    # Count anomalies
    num_anomalies = (predictions == 1).sum().item()
    num_normals = (predictions == 0).sum().item()

    # Count anomalies per class
    class_counts = {}
    for i in range(num_classes):
        class_counts[i] = torch.sum(predictions[labels == i]).item()

    # Convert numeric class IDs to class names if label_encoder is provided
    if label_encoder is not None:
        class_names = [label_encoder.classes_[i] for i in range(num_classes)]
        class_counts_named = {class_names[i]: count for i, count in class_counts.items()}
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]
        class_counts_named = class_counts

    print(f"Anomalies in {title}: {num_anomalies} / {len(predictions)}")
    print(f"Normal examples in {title}: {num_normals} / {len(predictions)}")
    print(f"Anomalies per class in {title}: {class_counts_named}")

    # Get latent space
    model.set_centers(data_loader)
    z = model.get_latent_space(data_loader)

    # Prepare data for UMAP
    z_feats = z[0].cpu().numpy()
    z_labels = z[1].cpu().numpy()
    umap_reducer = umap.UMAP(n_components=2)
    embedding = umap_reducer.fit_transform(z_feats)

    # Calculate clustering metrics
    silhouette = silhouette_score(z_feats, z_labels)
    davies_bouldin = davies_bouldin_score(z_feats, z_labels)
    print(f'Silhouette Score: {silhouette}')
    print(f'Davies-Bouldin Score: {davies_bouldin}')

    # Filter anomalies from the embedding
    anomalous_indices = np.where(predictions == 1)[0]
    embedding_anomalous = embedding[anomalous_indices]
    z_labels_anomalous = z_labels[anomalous_indices]

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=z_labels, cmap='Spectral', s=20)
    scatter_anomalous = plt.scatter(embedding_anomalous[:, 0], embedding_anomalous[:, 1], c=z_labels_anomalous, cmap='Spectral', marker='x', s=30)

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f"UMAP Projection of Latent Space - {title}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Create a legend with the class names
    handles, _ = scatter.legend_elements()
    if label_encoder is not None:
        labels = [label_encoder.classes_[label] for label in range(num_classes)]
    else:
        labels = [f"Class {i}" for i in range(num_classes)]
    normal_legend = plt.legend(handles, labels, title="Class")

    # Legend for anomalies
    handles_anomalous, _ = scatter_anomalous.legend_elements()
    plt.legend(handles_anomalous, labels, title="Anomalous Class")

    plt.show()

    return predictions, davies_bouldin, silhouette