"""
Auxiliary Task 2: Dimensionality Reduction Visualization (t-SNE and PCA)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import os


CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_sample_data(n_samples=5000, data_root='data', seed=42):
    """
    Load a subset of Fashion-MNIST for visualization
    
    Args:
        n_samples: Number of samples to use
        data_root: Data directory
        seed: Random seed
    
    Returns:
        features, labels
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Load test set for visualization
    dataset = FashionMNIST(root=data_root, train=False, transform=transform, download=True)
    
    # Convert to numpy
    features = []
    labels = []
    for img, label in dataset:
        features.append(img.numpy().flatten())
        labels.append(label)
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Subsample
    if n_samples < len(features):
        np.random.seed(seed)
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    return features, labels


def plot_tsne(features, labels, save_path=None, perplexity=30, n_iter=1000):
    """
    Create t-SNE visualization
    
    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Class labels
        save_path: Path to save plot
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
    """
    print(f"Computing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, 
                random_state=42, verbose=1)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Create color palette
    colors = sns.color_palette('tab10', n_colors=10)
    
    for i in range(10):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=CLASS_NAMES[i], 
                   alpha=0.6, s=20, edgecolors='none')
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f't-SNE Visualization of Fashion-MNIST\n(perplexity={perplexity}, {len(features)} samples)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE plot saved to {save_path}")
    
    plt.close()
    
    return features_2d


def plot_pca(features, labels, save_path=None, n_components=2):
    """
    Create PCA visualization
    
    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Class labels
        save_path: Path to save plot
        n_components: Number of PCA components
    """
    print(f"Computing PCA (n_components={n_components})...")
    
    pca = PCA(n_components=n_components)
    features_2d = pca.fit_transform(features)
    
    explained_var = pca.explained_variance_ratio_
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Create color palette
    colors = sns.color_palette('tab10', n_colors=10)
    
    for i in range(10):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=CLASS_NAMES[i], 
                   alpha=0.6, s=20, edgecolors='none')
    
    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
    plt.title(f'PCA Visualization of Fashion-MNIST\n({len(features)} samples)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PCA plot saved to {save_path}")
    
    plt.close()
    
    return features_2d, pca


def plot_pca_variance(features, save_path=None, n_components=50):
    """
    Plot explained variance by PCA components
    
    Args:
        features: Feature vectors
        save_path: Path to save plot
        n_components: Number of components to analyze
    """
    print(f"Computing PCA variance analysis...")
    
    pca = PCA(n_components=min(n_components, features.shape[1]))
    pca.fit(features)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Individual variance
    ax1.bar(range(1, len(explained_var)+1), explained_var, alpha=0.8, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each PC', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 
             marker='o', linewidth=2, markersize=4, color='darkred')
    ax2.axhline(y=0.95, color='green', linestyle='--', label='95% variance')
    ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PCA variance plot saved to {save_path}")
    
    plt.close()
    
    # Print summary
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")


def generate_all_visualizations(n_samples=5000, output_dir='images/visualizations', 
                                data_root='data'):
    """
    Generate all dimensionality reduction visualizations
    
    Args:
        n_samples: Number of samples to visualize
        output_dir: Directory to save plots
        data_root: Data directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("="*70)
    print("GENERATING DIMENSIONALITY REDUCTION VISUALIZATIONS")
    print("="*70)
    features, labels = load_sample_data(n_samples=n_samples, data_root=data_root)
    
    # t-SNE
    print("\n" + "-"*70)
    tsne_path = os.path.join(output_dir, 'tsne_visualization.png')
    plot_tsne(features, labels, save_path=tsne_path)
    
    # PCA 2D
    print("\n" + "-"*70)
    pca_path = os.path.join(output_dir, 'pca_visualization.png')
    plot_pca(features, labels, save_path=pca_path)
    
    # PCA Variance
    print("\n" + "-"*70)
    pca_var_path = os.path.join(output_dir, 'pca_variance_analysis.png')
    plot_pca_variance(features, save_path=pca_var_path)
    
    print("\n" + "="*70)
    print(f"✅ All visualizations saved to {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    generate_all_visualizations()
