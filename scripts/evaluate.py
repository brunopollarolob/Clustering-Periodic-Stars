#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for evaluating clustering and anomaly detection models.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Add the root directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AutoencoderMCDSVDD, AutoencoderGMM
from clustering_stars.utils.anomaly_detection import plot_umap_projection
from clustering_stars.utils.visualization import visualize_latent_space, plot_anomaly_scores

def load_model(model_path, device='cpu'):
    """
    Load a trained model from a file.
    
    Args:
        model_path (str): Path to the model file.
        device (str): Device to run the model on ('cpu' or 'cuda').
        
    Returns:
        tuple: (model, model_info) Loaded model and model information.
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint['model_type']
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    num_classes = checkpoint['num_classes']
    
    print(f"Model type: {model_type}")
    print(f"Input dimension: {input_dim}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Number of classes: {num_classes}")
    
    if 'class_names' in checkpoint:
        print(f"Class names: {checkpoint['class_names']}")
    
    if model_type.lower() == 'mcdsvdd':
        model = AutoencoderMCDSVDD(in_dim=input_dim, z_dim=latent_dim, num_classes=num_classes)
    elif model_type.lower() == 'gmm':
        model = AutoencoderGMM(input_dim=input_dim, latent_dim=latent_dim, n_gmm=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def prepare_data(data_file, labels_file, batch_size=64):
    """
    Prepare data for evaluation.
    
    Args:
        data_file (str): Path to the features file.
        labels_file (str): Path to the labels file.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (dataloader, label_encoder, df) DataLoader, label encoder, and concatenated dataframe.
    """
    print(f"Loading data from {data_file} and {labels_file}...")
    features = pd.read_parquet(data_file)
    labels = pd.read_parquet(labels_file)
    
    # Prepare features
    if 'oid' in features.columns:
        features_index = features['oid']
        features = features.set_index('oid')
    
    if 'oid' in labels.columns:
        labels = labels.set_index('oid')
    
    # Select only the class column
    class_column = 'alerceclass' if 'alerceclass' in labels.columns else labels.columns[0]
    labels = labels[[class_column]]
    
    # Fill missing values
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    from preprocessing import fill_missing_values
    features_filled = fill_missing_values(features, numeric_columns)
    
    # Concatenate labels and features
    df_concatenated = pd.concat([labels, features_filled], axis=1)
    df_concatenated.dropna(inplace=True)
    
    # Split into features and labels
    X = df_concatenated.drop(columns=[class_column]).values
    y = df_concatenated[class_column].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, label_encoder, df_concatenated

def evaluate_model(model, dataloader, label_encoder, model_type, output_dir=None):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model.
        dataloader: DataLoader with data to evaluate.
        label_encoder: Label encoder.
        model_type (str): Model type ('mcdsvdd' or 'gmm').
        output_dir (str): Directory to save results.
        
    Returns:
        dict: Evaluation results.
    """
    print(f"Evaluating model of type {model_type}...")
    
    results = {}
    
    if model_type.lower() == 'mcdsvdd':
        # Initialize centers for MCDSVDD
        model.set_centers(dataloader)
        
        # Detect anomalies and visualize the latent space
        print("Detecting anomalies and visualizing latent space...")
        predictions, davies_bouldin, silhouette = plot_umap_projection(dataloader, 'Evaluation', model)
        
        # Visualize anomaly score distribution
        print("Visualizing anomaly scores...")
        df_scores = plot_anomaly_scores(model, dataloader)
        
        results = {
            'predictions': predictions.numpy(),
            'davies_bouldin': davies_bouldin,
            'silhouette': silhouette,
            'anomaly_scores': df_scores
        }
        
    elif model_type.lower() == 'gmm':
        # Visualize the latent space
        print("Visualizing latent space...")
        z = model.get_latent_space(dataloader)
        visualize_latent_space(z, label_encoder)
        
        # Calculate metrics
        z_feats = z[0].cpu().numpy()
        z_labels = z[1].cpu().numpy()
        silhouette = silhouette_score(z_feats, z_labels)
        davies_bouldin = davies_bouldin_score(z_feats, z_labels)
        
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        results = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
    
    # Save results if a directory was specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            for key, value in results.items():
                if isinstance(value, (float, int)):
                    f.write(f"{key}: {value}\n")
            
        print(f"Metrics saved to {metrics_file}")
    
    return results

def plot_learning_curves(history, model_type, output_dir=None):
    """
    Visualize the learning curves of the model.
    
    Args:
        history: Training history.
        model_type (str): Model type ('mcdsvdd' or 'gmm').
        output_dir (str): Directory to save plots.
    """
    if model_type.lower() == 'mcdsvdd':
        # Autoencoder loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['autoencoder']['train_loss'], label='Training')
        plt.plot(history['autoencoder']['val_loss'], label='Validation')
        plt.title('Autoencoder Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # MCDSVDD loss
        plt.subplot(1, 2, 2)
        plt.plot(history['mcdsvdd']['train_loss'], label='Training')
        plt.plot(history['mcdsvdd']['val_loss'], label='Validation')
        plt.title('MCDSVDD Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
    elif model_type.lower() == 'gmm':
        # Total loss
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history['gmm']['loss'])
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Reconstruction error
        plt.subplot(1, 3, 2)
        plt.plot(history['gmm']['reconstruction'])
        plt.title('Reconstruction Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Energy
        plt.subplot(1, 3, 3)
        plt.plot(history['gmm']['energy'])
        plt.title('Energy')
        plt.xlabel('Epoch')
        plt.ylabel('Energy')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot if a directory was specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, f'learning_curves_{model_type}.png')
        plt.savefig(plot_file)
        print(f"Learning curves saved to {plot_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Evaluation of clustering and anomaly detection models')
    
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the features file (.parquet)')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to the labels file (.parquet)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--plot_history', action='store_true',
                        help='Visualize learning curves')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the model on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Load model
    model, checkpoint = load_model(args.model_file, args.device)
    
    # Prepare data
    dataloader, label_encoder, df = prepare_data(args.data_file, args.labels_file, args.batch_size)
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        label_encoder=label_encoder,
        model_type=checkpoint['model_type'],
        output_dir=args.output_dir
    )
    
    # Visualize learning curves if requested
    if args.plot_history and 'history' in checkpoint:
        plot_learning_curves(
            history=checkpoint['history'],
            model_type=checkpoint['model_type'],
            output_dir=args.output_dir
        )
    
if __name__ == "__main__":
    main()