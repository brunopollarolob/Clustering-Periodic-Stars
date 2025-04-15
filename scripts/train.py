#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar modelos de clustering y detección de anomalías.
Permite entrenar tanto AutoencoderMCDSVDD como AutoencoderGMM.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Añadir el directorio raíz al path para importar módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AutoencoderMCDSVDD, AutoencoderGMM
from preprocessing import prepare_data, fill_missing_values, balance_data

def train_model(train_loader, val_loader, model_type='mcdsvdd', input_dim=50, 
                latent_dim=10, num_classes=5, epochs=50, lr=0.001):
    """
    Entrena un modelo del tipo especificado.
    
    Args:
        train_loader: DataLoader para entrenamiento.
        val_loader: DataLoader para validación.
        model_type (str): Tipo de modelo ('mcdsvdd' o 'gmm').
        input_dim (int): Dimensión de entrada.
        latent_dim (int): Dimensión del espacio latente.
        num_classes (int): Número de clases.
        epochs (int): Épocas de entrenamiento.
        lr (float): Learning rate.
        
    Returns:
        tuple: (model, history) Modelo entrenado e historial de pérdidas.
    """
    if model_type.lower() == 'mcdsvdd':
        print(f"Entrenando AutoencoderMCDSVDD (in_dim={input_dim}, z_dim={latent_dim}, num_classes={num_classes})...")
        model = AutoencoderMCDSVDD(in_dim=input_dim, z_dim=latent_dim, num_classes=num_classes)
        
        # Fase 1: Entrenar autoencoder
        print(f"Fase 1: Entrenando autoencoder por {epochs} épocas...")
        train_loss, val_loss = model.train_autoencoder(train_loader, val_loader, epochs=epochs, lr=lr)
        
        # Establecer centros para MCDSVDD
        model.set_centers(train_loader)
        
        # Fase 2: Entrenar MCDSVDD
        print(f"Fase 2: Entrenando MCDSVDD por {epochs} épocas...")
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
        print(f"Entrenando AutoencoderGMM (input_dim={input_dim}, latent_dim={latent_dim}, n_gmm={num_classes})...")
        model = AutoencoderGMM(input_dim=input_dim, latent_dim=latent_dim, n_gmm=num_classes, num_epochs=epochs)
        
        # Entrenar modelo GMM
        print(f"Entrenando AutoencoderGMM por {epochs} épocas...")
        model.train_model(train_loader)
        
        history = {
            'gmm': {
                'loss': model.loss_history,
                'reconstruction': model.reconstruction_history,
                'energy': model.energy_history
            }
        }
        
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(
        description='Entrenamiento de modelos de clustering y detección de anomalías')
    
    parser.add_argument('--model_type', type=str, default='mcdsvdd', choices=['mcdsvdd', 'gmm'],
                        help='Tipo de modelo a entrenar (mcdsvdd o gmm)')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Ruta al archivo con características (.parquet)')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Ruta al archivo con etiquetas (.parquet)')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Ruta para guardar el modelo entrenado')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Tamaño de batch')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Dimensión del espacio latente')
    parser.add_argument('--use_smote', action='store_true',
                        help='Usar SMOTE para balancear los datos')
    
    args = parser.parse_args()
    
    # Cargar datos
    print(f"Cargando datos desde {args.data_file} y {args.labels_file}...")
    features = pd.read_parquet(args.data_file)
    labels = pd.read_parquet(args.labels_file)
    
    # Preparar características
    print("Preparando características...")
    if 'oid' in features.columns:
        features = features.set_index('oid')
    
    if 'oid' in labels.columns:
        labels = labels.set_index('oid')
    
    # Seleccionar solo la columna de clase
    class_column = 'alerceclass' if 'alerceclass' in labels.columns else labels.columns[0]
    labels = labels[[class_column]]
    
    # Llenar valores faltantes
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    features_filled = fill_missing_values(features, numeric_columns)
    
    # Concatenar etiquetas y características
    df_concatenated = pd.concat([labels, features_filled], axis=1)
    df_concatenated.dropna(inplace=True)
    
    # Dividir en características y etiquetas
    X = df_concatenated.drop(columns=[class_column]).values
    y = df_concatenated[class_column].values
    
    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # División en conjuntos de entrenamiento, validación y prueba
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    if args.use_smote:
        # Balancear datos con SMOTE
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Convertir a tensores
        X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.long)
    else:
        # Convertir a tensores sin aplicar SMOTE
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Crear datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Crear dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Dimensión de entrada (número de características)
    input_dim = X_scaled.shape[1]
    
    # Número de clases
    num_classes = len(np.unique(y_encoded))
    
    # Entrenar modelo
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model_type=args.model_type,
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        epochs=args.epochs
    )
    
    # Guardar modelo entrenado
    if args.save_model:
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
        
        print(f"Guardando modelo en {args.save_model}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': args.model_type,
            'input_dim': input_dim,
            'latent_dim': args.latent_dim,
            'num_classes': num_classes,
            'class_names': list(label_encoder.classes_),
            'history': history
        }, args.save_model)
        
        print(f"Modelo guardado exitosamente en {args.save_model}")
    
if __name__ == "__main__":
    main()