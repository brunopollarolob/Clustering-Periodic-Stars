#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para evaluar modelos de clustering y detección de anomalías.
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

# Añadir el directorio raíz al path para importar módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AutoencoderMCDSVDD, AutoencoderGMM
from clustering_stars.utils.anomaly_detection import plot_umap_projection
from clustering_stars.utils.visualization import visualize_latent_space, plot_anomaly_scores

def load_model(model_path, device='cpu'):
    """
    Carga un modelo entrenado desde un archivo.
    
    Args:
        model_path (str): Ruta al archivo de modelo.
        device (str): Dispositivo para ejecutar el modelo ('cpu' o 'cuda').
        
    Returns:
        tuple: (model, model_info) Modelo cargado e información del modelo.
    """
    print(f"Cargando modelo desde {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint['model_type']
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    num_classes = checkpoint['num_classes']
    
    print(f"Tipo de modelo: {model_type}")
    print(f"Dimensión de entrada: {input_dim}")
    print(f"Dimensión latente: {latent_dim}")
    print(f"Número de clases: {num_classes}")
    
    if 'class_names' in checkpoint:
        print(f"Nombres de clases: {checkpoint['class_names']}")
    
    if model_type.lower() == 'mcdsvdd':
        model = AutoencoderMCDSVDD(in_dim=input_dim, z_dim=latent_dim, num_classes=num_classes)
    elif model_type.lower() == 'gmm':
        model = AutoencoderGMM(input_dim=input_dim, latent_dim=latent_dim, n_gmm=num_classes)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def prepare_data(data_file, labels_file, batch_size=64):
    """
    Prepara los datos para evaluación.
    
    Args:
        data_file (str): Ruta al archivo de características.
        labels_file (str): Ruta al archivo de etiquetas.
        batch_size (int): Tamaño de batch.
        
    Returns:
        tuple: (dataloader, label_encoder) DataLoader y codificador de etiquetas.
    """
    print(f"Cargando datos desde {data_file} y {labels_file}...")
    features = pd.read_parquet(data_file)
    labels = pd.read_parquet(labels_file)
    
    # Preparar características
    if 'oid' in features.columns:
        features_index = features['oid']
        features = features.set_index('oid')
    
    if 'oid' in labels.columns:
        labels = labels.set_index('oid')
    
    # Seleccionar solo la columna de clase
    class_column = 'alerceclass' if 'alerceclass' in labels.columns else labels.columns[0]
    labels = labels[[class_column]]
    
    # Llenar valores faltantes
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    from preprocessing import fill_missing_values
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
    
    # Convertir a tensores
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    # Crear dataset y dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, label_encoder, df_concatenated

def evaluate_model(model, dataloader, label_encoder, model_type, output_dir=None):
    """
    Evalúa un modelo entrenado.
    
    Args:
        model: Modelo entrenado.
        dataloader: DataLoader con datos a evaluar.
        label_encoder: Codificador de etiquetas.
        model_type (str): Tipo de modelo ('mcdsvdd' o 'gmm').
        output_dir (str): Directorio para guardar resultados.
        
    Returns:
        dict: Resultados de la evaluación.
    """
    print(f"Evaluando modelo de tipo {model_type}...")
    
    results = {}
    
    if model_type.lower() == 'mcdsvdd':
        # Inicializar centros para MCDSVDD
        model.set_centers(dataloader)
        
        # Detectar anomalías y visualizar el espacio latente
        print("Detectando anomalías y visualizando espacio latente...")
        predictions, davies_bouldin, silhouette = plot_umap_projection(dataloader, 'Evaluación', model)
        
        # Visualizar distribución de puntuaciones de anomalía
        print("Visualizando puntuaciones de anomalía...")
        df_scores = plot_anomaly_scores(model, dataloader)
        
        results = {
            'predictions': predictions.numpy(),
            'davies_bouldin': davies_bouldin,
            'silhouette': silhouette,
            'anomaly_scores': df_scores
        }
        
    elif model_type.lower() == 'gmm':
        # Visualizar el espacio latente
        print("Visualizando espacio latente...")
        z = model.get_latent_space(dataloader)
        visualize_latent_space(z, label_encoder)
        
        # Calcular métricas
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
    
    # Guardar resultados si se especificó un directorio
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar métricas
        metrics_file = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            for key, value in results.items():
                if isinstance(value, (float, int)):
                    f.write(f"{key}: {value}\n")
            
        print(f"Métricas guardadas en {metrics_file}")
    
    return results

def plot_learning_curves(history, model_type, output_dir=None):
    """
    Visualiza las curvas de aprendizaje del modelo.
    
    Args:
        history: Historial de entrenamiento.
        model_type (str): Tipo de modelo ('mcdsvdd' o 'gmm').
        output_dir (str): Directorio para guardar gráficos.
    """
    if model_type.lower() == 'mcdsvdd':
        # Pérdida del autoencoder
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['autoencoder']['train_loss'], label='Entrenamiento')
        plt.plot(history['autoencoder']['val_loss'], label='Validación')
        plt.title('Pérdida del Autoencoder')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Pérdida MCDSVDD
        plt.subplot(1, 2, 2)
        plt.plot(history['mcdsvdd']['train_loss'], label='Entrenamiento')
        plt.plot(history['mcdsvdd']['val_loss'], label='Validación')
        plt.title('Pérdida MCDSVDD')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
    elif model_type.lower() == 'gmm':
        # Pérdida total
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history['gmm']['loss'])
        plt.title('Pérdida Total')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Error de reconstrucción
        plt.subplot(1, 3, 2)
        plt.plot(history['gmm']['reconstruction'])
        plt.title('Error de Reconstrucción')
        plt.xlabel('Época')
        plt.ylabel('Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Energía
        plt.subplot(1, 3, 3)
        plt.plot(history['gmm']['energy'])
        plt.title('Energía')
        plt.xlabel('Época')
        plt.ylabel('Energía')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar gráfico si se especificó un directorio
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, f'learning_curves_{model_type}.png')
        plt.savefig(plot_file)
        print(f"Curvas de aprendizaje guardadas en {plot_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Evaluación de modelos de clustering y detección de anomalías')
    
    parser.add_argument('--model_file', type=str, required=True,
                        help='Ruta al archivo de modelo entrenado')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Ruta al archivo con características (.parquet)')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Ruta al archivo con etiquetas (.parquet)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directorio para guardar resultados')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Tamaño de batch para evaluación')
    parser.add_argument('--plot_history', action='store_true',
                        help='Visualizar curvas de aprendizaje')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Dispositivo para ejecutar el modelo (cpu o cuda)')
    
    args = parser.parse_args()
    
    # Cargar modelo
    model, checkpoint = load_model(args.model_file, args.device)
    
    # Preparar datos
    dataloader, label_encoder, df = prepare_data(args.data_file, args.labels_file, args.batch_size)
    
    # Evaluar modelo
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        label_encoder=label_encoder,
        model_type=checkpoint['model_type'],
        output_dir=args.output_dir
    )
    
    # Visualizar curvas de aprendizaje si se solicita
    if args.plot_history and 'history' in checkpoint:
        plot_learning_curves(
            history=checkpoint['history'],
            model_type=checkpoint['model_type'],
            output_dir=args.output_dir
        )
    
if __name__ == "__main__":
    main()