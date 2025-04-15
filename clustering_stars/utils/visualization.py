"""
Visualization utilities for clustering and anomaly detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import seaborn as sns
import pandas as pd
import lightkurve as lk

def visualize_latent_space(model, dataloader, label_encoder=None, n_components=2, random_state=42):
    """
    Visualize the latent space of a model using UMAP dimensionality reduction.
    
    Args:
        model: Trained model with get_latent_space method
        dataloader: PyTorch DataLoader containing the data
        label_encoder: Optional LabelEncoder used to convert numeric labels to class names
        n_components: Number of components for UMAP projection (default: 2)
        random_state: Random state for reproducibility
        
    Returns:
        z_feats: The latent features before UMAP projection
        z_labels: The class labels for each point
        embedding: The UMAP projection of the latent features
    """
    # Extract latent representations from the data
    z = model.get_latent_space(dataloader)
    z_feats = z[0].cpu().numpy()
    z_labels = z[1].cpu().numpy()
    
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(z_feats)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=z_labels, cmap='Spectral', s=30, alpha=0.8)
    
    # Add equal aspect ratio and labels
    plt.gca().set_aspect('equal', 'datalim')
    plt.title("UMAP Projection of Latent Space", fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Create legend
    handles, _ = scatter.legend_elements()
    if label_encoder is not None:
        num_classes = len(np.unique(z_labels))
        legend_labels = [label_encoder.classes_[i] for i in range(num_classes)]
    else:
        legend_labels = [f"Class {i}" for i in np.unique(z_labels)]
    
    plt.legend(handles, legend_labels, title="Class", title_fontsize=12)
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.show()
    
    return z_feats, z_labels, embedding

def plot_feature_distributions(df, class_column, feature_columns, class_names=None):
    """
    Create boxplots showing the distributions of features by class.
    
    Args:
        df: DataFrame containing the data
        class_column: Name of the column containing class labels
        feature_columns: List of column names for features to plot
        class_names: Optional dictionary mapping class values to display names
    """
    # Plot distributions by class for selected features
    for feature in feature_columns:
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(x=class_column, y=feature, data=df)
        
        # Apply class name mapping if provided
        if class_names is not None:
            labels = [class_names.get(item.get_text(), item.get_text()) 
                      for item in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
        
        plt.title(f'Distribution of {feature} by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_periodic_light_curves(lc_data, oids, periods=None, class_names=None):
    """
    Plot folded light curves for the given object IDs.
    
    Args:
        lc_data: DataFrame containing light curve data with 'oid', 'mjd', 'magpsf', 'sigmapsf' columns
        oids: List of object IDs to plot
        periods: Dictionary mapping object IDs to their periods
        class_names: Dictionary mapping object IDs to their class names
    """
    if not isinstance(oids, list):
        oids = [oids]
        
    # Calculate number of rows and columns for subplot grid
    n_objects = len(oids)
    n_cols = min(3, n_objects)
    n_rows = (n_objects + n_cols - 1) // n_cols
    
    # Create figure and subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows * n_cols == 1:  # If only one subplot
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Color mapping for visualization consistency
    class_colors = {}
    if class_names:
        unique_classes = set(class_names.values())
        cmap = plt.cm.get_cmap('tab10', len(unique_classes))
        class_colors = {cls: cmap(i) for i, cls in enumerate(unique_classes)}
    
    # Plot each light curve
    for i, oid in enumerate(oids):
        if i >= len(axes):
            break
            
        # Filter data for current OID
        subset = lc_data[lc_data['oid'] == oid]
        
        if subset.empty:
            axes[i].text(0.5, 0.5, f"No data for OID: {oid}", 
                         ha='center', va='center', transform=axes[i].transAxes)
            continue
            
        # Create a LightCurve using MJD and MagPSF values
        time = subset['mjd'].values
        flux = subset['magpsf'].values
        flux_err = subset['sigmapsf'].values if 'sigmapsf' in subset.columns else None
        
        lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
        
        # Get period for folding
        period = periods.get(oid) if periods else None
        
        # Determine marker color based on class
        color = 'blue'
        if class_names and oid in class_names:
            color = class_colors.get(class_names[oid], 'blue')
        
        # Plot normal or folded light curve
        if period:
            lc_folded = lc.fold(period=period)
            lc_folded.scatter(ax=axes[i], color=color, s=30, label=f"Period: {period:.2f}")
            axes[i].set_title(f'OID: {oid} - Class: {class_names.get(oid, "Unknown")}')
            axes[i].set_xlabel('Phase')
            axes[i].set_ylabel('Folded Magnitude')
            axes[i].legend()
        else:
            lc.scatter(ax=axes[i], color=color, s=30)
            axes[i].set_title(f'OID: {oid} - Class: {class_names.get(oid, "Unknown")}')
            axes[i].set_xlabel('Time (MJD)')
            axes[i].set_ylabel('Magnitude')
            
        # Invert y-axis for astronomical magnitude (smaller values are brighter)
        axes[i].invert_yaxis()
    
    # Hide unused subplots
    for i in range(n_objects, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes