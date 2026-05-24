"""
Auxiliary Tasks Package

Contains:
1. Binary Classification (Footwear vs Non-Footwear)
2. Dimensionality Reduction Visualizations (t-SNE, PCA)
"""

from .binary_classification import (
    convert_to_binary_labels,
    get_binary_data_loaders,
    get_binary_data_numpy,
    CLASS_NAMES_BINARY,
    FOOTWEAR_CLASSES,
    NON_FOOTWEAR_CLASSES
)

from .dimensionality_reduction import (
    load_sample_data,
    plot_tsne,
    plot_pca,
    plot_pca_variance,
    generate_all_visualizations
)

__all__ = [
    'convert_to_binary_labels',
    'get_binary_data_loaders',
    'get_binary_data_numpy',
    'CLASS_NAMES_BINARY',
    'FOOTWEAR_CLASSES',
    'NON_FOOTWEAR_CLASSES',
    'load_sample_data',
    'plot_tsne',
    'plot_pca',
    'plot_pca_variance',
    'generate_all_visualizations'
]
