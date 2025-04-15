"""
Clustering-Periodic-Stars package - Tools for clustering and anomaly detection in light curves of periodic stars.
"""

# Import main classes from existing files
from models import AutoencoderMCDSVDD, AutoencoderGMM, AverageMeter
from preprocessing import fill_missing_values, balance_data, prepare_data

# Import utilities
from .utils.anomaly_detection import plot_umap_projection
from .utils.evaluation import train_and_evaluate
from .utils.visualization import visualize_latent_space

__all__ = [
    'AutoencoderMCDSVDD',
    'AutoencoderGMM', 
    'AverageMeter',
    'fill_missing_values',
    'balance_data',
    'prepare_data',
    'plot_umap_projection',
    'train_and_evaluate',
    'visualize_latent_space'
]