"""
Auxiliary Task 2 Extended: Dimensionality Reduction Impact on Model Performance

This script compares model performance on:
1. Original 784-dimensional features
2. PCA-reduced features with varying components (50, 100, 200, 400)

Models tested: SVM-RBF (best SVM variant), Neural Network
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import time
import pandas as pd

from models.svm.model import SVMClassifier
from models.traditional.model import SimpleNN
from models.cnn.model import CNN


CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data_numpy(data_root='data'):
    """Load Fashion-MNIST as numpy arrays"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = FashionMNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root=data_root, train=False, transform=transform, download=True)
    
    # Convert to numpy
    X_train, y_train = [], []
    for img, label in train_dataset:
        X_train.append(img.numpy().flatten())
        y_train.append(label)
    
    X_test, y_test = [], []
    for img, label in test_dataset:
        X_test.append(img.numpy().flatten())
        y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Split train into train + val
    X_val = X_train[50000:]
    y_val = y_train[50000:]
    X_train = X_train[:50000]
    y_train = y_train[:50000]
    
    print(f"Data loaded - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def apply_pca_reduction(X_train, X_val, X_test, n_components):
    """
    Apply PCA dimensionality reduction with scaling
    
    Returns:
        X_train_pca, X_val_pca, X_test_pca, pca_model, variance_explained
    """
    print(f"\nApplying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    # Apply standard scaling after PCA for better SVM convergence
    scaler = StandardScaler()
    X_train_pca = scaler.fit_transform(X_train_pca)
    X_val_pca = scaler.transform(X_val_pca)
    X_test_pca = scaler.transform(X_test_pca)
    
    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"✓ Variance explained: {variance_explained:.2f}% (scaled)")
    
    return X_train_pca, X_val_pca, X_test_pca, pca, variance_explained


def train_svm_with_pca(X_train, y_train, X_val, y_val, X_test, y_test, 
                       n_components, kernel='rbf', C=1.0):
    """Train SVM on PCA-reduced features"""
    print(f"\n{'='*70}")
    print(f"SVM ({kernel.upper()}) - {n_components} PCA Components")
    print(f"{'='*70}")
    
    if n_components < X_train.shape[1]:
        X_train_pca, X_val_pca, X_test_pca, pca, var_exp = apply_pca_reduction(
            X_train, X_val, X_test, n_components
        )
    else:
        # No PCA, but still scale for SVM
        print(f"\nUsing full {n_components} features (no PCA, with scaling)")
        scaler = StandardScaler()
        X_train_pca = scaler.fit_transform(X_train)
        X_val_pca = scaler.transform(X_val)
        X_test_pca = scaler.transform(X_test)
        var_exp = 100.0
    
    # Train SVM
    start_time = time.time()
    svm = SVMClassifier(kernel=kernel, C=C, max_iter=5000, random_state=42)
    svm.fit(X_train_pca, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    train_pred = svm.predict(X_train_pca)
    val_pred = svm.predict(X_val_pca)
    test_pred = svm.predict(X_test_pca)
    
    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    return {
        'n_components': n_components,
        'variance_explained': var_exp,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_time': train_time,
        'test_predictions': test_pred
    }


class SimpleNN_PCA(nn.Module):
    """Simple NN adapted for variable input dimensions"""
    def __init__(self, input_dim, num_classes=10, hidden_size=256, dropout_rate=0.3):
        super(SimpleNN_PCA, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def train_nn_with_pca(X_train, y_train, X_val, y_val, X_test, y_test,
                      n_components, epochs=20, lr=0.001, batch_size=128):
    """Train Neural Network on PCA-reduced features"""
    print(f"\n{'='*70}")
    print(f"Neural Network - {n_components} PCA Components")
    print(f"{'='*70}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if n_components < X_train.shape[1]:
        X_train_pca, X_val_pca, X_test_pca, pca, var_exp = apply_pca_reduction(
            X_train, X_val, X_test, n_components
        )
    else:
        X_train_pca, X_val_pca, X_test_pca = X_train, X_val, X_test
        var_exp = 100.0
        print(f"Using full {n_components} features (no PCA)")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_pca)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_pca)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test_pca)
    y_test_t = torch.LongTensor(y_test)
    
    # Data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = SimpleNN_PCA(input_dim=X_train_pca.shape[1], hidden_size=256, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    start_time = time.time()
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = 100 * val_correct / len(y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Val Acc: {val_acc:.2f}%")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    def evaluate(loader):
        correct = 0
        all_preds = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                correct += (predicted == y_batch.to(device)).sum().item()
        return 100 * correct / len(loader.dataset), np.array(all_preds)
    
    train_acc, _ = evaluate(train_loader)
    val_acc, _ = evaluate(val_loader)
    test_acc, test_pred = evaluate(test_loader)
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    return {
        'n_components': n_components,
        'variance_explained': var_exp,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_time': train_time,
        'test_predictions': test_pred
    }


def plot_dimensionality_comparison(results_dict, save_dir='images/visualizations'):
    """
    Plot comparison of model performance across different PCA dimensions
    
    Args:
        results_dict: {'SVM': [results], 'NN': [results]}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Test Accuracy vs Components
    ax = axes[0, 0]
    for model_name, results in results_dict.items():
        components = [r['n_components'] for r in results]
        test_accs = [r['test_acc'] for r in results]
        ax.plot(components, test_accs, marker='o', linewidth=2, 
                markersize=8, label=model_name)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy vs Dimensionality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Training Time vs Components
    ax = axes[0, 1]
    for model_name, results in results_dict.items():
        components = [r['n_components'] for r in results]
        times = [r['train_time'] for r in results]
        ax.plot(components, times, marker='s', linewidth=2, 
                markersize=8, label=model_name)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time vs Dimensionality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 3: Variance Explained vs Components
    ax = axes[1, 0]
    for model_name, results in results_dict.items():
        components = [r['n_components'] for r in results]
        variance = [r['variance_explained'] for r in results]
        ax.plot(components, variance, marker='^', linewidth=2, 
                markersize=8, label=model_name)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('Variance Captured vs Dimensionality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95%')
    
    # Plot 4: Accuracy Drop vs Speedup
    ax = axes[1, 1]
    for model_name, results in results_dict.items():
        # Calculate relative to full dimensionality
        full_result = results[-1]  # Assuming last is full
        acc_drops = [full_result['test_acc'] - r['test_acc'] for r in results]
        speedups = [full_result['train_time'] / r['train_time'] for r in results]
        
        ax.scatter(speedups, acc_drops, s=150, alpha=0.7, label=model_name)
        
        # Annotate with component counts
        for r, x, y in zip(results, speedups, acc_drops):
            if r['n_components'] != 784:
                ax.annotate(f"{r['n_components']}", (x, y), 
                           fontsize=9, ha='center', va='bottom')
    
    ax.set_xlabel('Training Speedup (×)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Accuracy vs Speed Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pca_dimensionality_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {save_path}")
    plt.close()


def create_summary_table(results_dict, save_dir='images/visualizations'):
    """Create summary table comparing all experiments"""
    rows = []
    
    for model_name, results in results_dict.items():
        for r in results:
            rows.append({
                'Model': model_name,
                'Components': r['n_components'],
                'Variance (%)': f"{r['variance_explained']:.2f}",
                'Train Acc (%)': f"{r['train_acc']:.2f}",
                'Val Acc (%)': f"{r['val_acc']:.2f}",
                'Test Acc (%)': f"{r['test_acc']:.2f}",
                'Train Time (s)': f"{r['train_time']:.2f}"
            })
    
    df = pd.DataFrame(rows)
    
    # Print to console
    print("\n" + "="*90)
    print("DIMENSIONALITY REDUCTION EXPERIMENT RESULTS")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'pca_dimensionality_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    return df


def main():
    """Run comprehensive dimensionality reduction experiments"""
    print("="*70)
    print("DIMENSIONALITY REDUCTION IMPACT ON MODEL PERFORMANCE")
    print("="*70)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_numpy()
    
    # PCA component configurations to test
    pca_configs = [50, 100, 200, 400, 784]  # 784 = full dimensionality
    
    results = {}
    
    # Test SVM
    print("\n" + "#"*70)
    print("# TESTING SVM WITH DIFFERENT DIMENSIONALITIES")
    print("#"*70)
    
    svm_results = []
    for n_comp in pca_configs:
        result = train_svm_with_pca(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_components=n_comp, kernel='rbf', C=1.0
        )
        svm_results.append(result)
    
    results['SVM'] = svm_results
    
    # Test Neural Network
    print("\n" + "#"*70)
    print("# TESTING NEURAL NETWORK WITH DIFFERENT DIMENSIONALITIES")
    print("#"*70)
    
    nn_results = []
    for n_comp in pca_configs:
        result = train_nn_with_pca(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_components=n_comp, epochs=20, lr=0.001
        )
        nn_results.append(result)
    
    results['NN'] = nn_results
    
    # Create visualizations and summary
    plot_dimensionality_comparison(results)
    create_summary_table(results)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    for model_name, model_results in results.items():
        full_result = model_results[-1]
        best_result = max(model_results, key=lambda x: x['test_acc'])
        
        print(f"\n{model_name}:")
        print(f"  Full (784-D): {full_result['test_acc']:.2f}% acc, {full_result['train_time']:.2f}s")
        print(f"  Best ({best_result['n_components']}-D): {best_result['test_acc']:.2f}% acc, "
              f"{best_result['train_time']:.2f}s")
        
        if best_result['n_components'] < 784:
            acc_drop = full_result['test_acc'] - best_result['test_acc']
            speedup = full_result['train_time'] / best_result['train_time']
            print(f"  → {speedup:.2f}× speedup with only {acc_drop:.2f}% accuracy drop!")
        else:
            print(f"  → Full dimensionality is best for accuracy")
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
