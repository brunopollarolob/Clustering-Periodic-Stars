#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for training clustering and anomaly detection models.
Supports training both AutoencoderMCDSVDD and AutoencoderGMM.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Add the root directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AutoencoderMCDSVDD, AutoencoderGMM
from preprocessing import prepare_data, fill_missing_values, balance_data

def train_model(train_loader, val_loader, model_type='mcdsvdd', input_dim=50, 
                latent_dim=10, num_classes=5, epochs=50, lr=0.001):
    """
    Train a model of the specified type.
    
    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        model_type (str): Model type ('mcdsvdd' or 'gmm').
        input_dim (int): Input dimension.
        latent_dim (int): Latent space dimension.
        num_classes (int): Number of classes.
        epochs (int): Training epochs.
        lr (float): Learning rate.
        
    Returns:
        tuple: (model, history) Trained model and loss history.
    """
    if model_type.lower() == 'mcdsvdd':
        print(f"Training AutoencoderMCDSVDD (in_dim={input_dim}, z_dim={latent_dim}, num_classes={num_classes})...")
        model = AutoencoderMCDSVDD(in_dim=input_dim, z_dim=latent_dim, num_classes=num_classes)
        
        # Phase 1: Train autoencoder
        print(f"Phase 1: Training autoencoder for {epochs} epochs...")
        train_loss, val_loss = model.train_autoencoder(train_loader, val_loader, epochs=epochs, lr=lr)
        
        # Set centers for MCDSVDD
        model.set_centers(train_loader)
        
        # Phase 2: Train MCDSVDD
        print(f"Phase 2: Training MCDSVDD for {epochs} epochs...")
        train_loss_mcdsvdd, val_loss_mcdsvdd = model.train_mcdsvdd(train_loader, val_loader, epochs=epochs, lr=lr)
        
        history = {
            'autoencoder': {
                'train_loss': train_loss,
                'val_loss': val_loss
            },
            'mcdsvdd': {
                'train_loss': train_loss_mcdsvdd,
                'val_loss': val_loss_mcdsvdd
            }
        }
        
    elif model_type.lower() == 'gmm':
        print(f"Training AutoencoderGMM (input_dim={input_dim}, latent_dim={latent_dim}, n_gmm={num_classes})...")
        model = AutoencoderGMM(input_dim=input_dim, latent_dim=latent_dim, n_gmm=num_classes, num_epochs=epochs)
        
        # Train GMM model
        print(f"Training AutoencoderGMM for {epochs} epochs...")
        model.train_model(train_loader)
        
        history = {
            'gmm': {
                'loss': model.loss_history,
                'reconstruction': model.reconstruction_history,
                'energy': model.energy_history
            }
        }
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(
        description='Training of clustering and anomaly detection models')
    
    parser.add_argument('--model_type', type=str, default='mcdsvdd', choices=['mcdsvdd', 'gmm'],
                        help='Model type to train (mcdsvdd or gmm)')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the features file (.parquet)')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to the labels file (.parquet)')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent space dimension')
    parser.add_argument('--use_smote', action='store_true',
                        help='Use SMOTE for balancing the data')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_file} and {args.labels_file}...")
    features = pd.read_parquet(args.data_file)
    labels = pd.read_parquet(args.labels_file)
    
    # Prepare features
    print("Preparing features...")
    if 'oid' in features.columns:
        features = features.set_index('oid')
    
    if 'oid' in labels.columns:
        labels = labels.set_index('oid')
    
    # Select only the class column
    class_column = 'alerceclass' if 'alerceclass' in labels.columns else labels.columns[0]
    labels = labels[[class_column]]
    
    # Fill missing values
    numeric_columns = features.select_dtypes(include=[np.number]).columns
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
    
    # Split into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    if args.use_smote:
        # Balance data with SMOTE
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.long)
    else:
        # Convert to tensors without applying SMOTE
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Input dimension (number of features)
    input_dim = X_scaled.shape[1]
    
    # Number of classes
    num_classes = len(np.unique(y_encoded))
    
    # Train model
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model_type=args.model_type,
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        epochs=args.epochs
    )
    
    # Save trained model
    if args.save_model:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
        
        print(f"Saving model to {args.save_model}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': args.model_type,
            'input_dim': input_dim,
            'latent_dim': args.latent_dim,
            'num_classes': num_classes,
            'class_names': list(label_encoder.classes_),
            'history': history
        }, args.save_model)
        
        print(f"Model successfully saved to {args.save_model}")
    
if __name__ == "__main__":
    main()