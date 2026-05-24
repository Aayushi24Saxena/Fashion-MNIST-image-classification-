"""
Auxiliary Task 1: Binary Classification - Footwear vs Non-Footwear

Fashion-MNIST classes:
Footwear: Sandal (5), Sneaker (7), Ankle boot (9)
Non-Footwear: T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Coat (4), Shirt (6), Bag (8)
"""
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, TensorDataset


# Binary class mapping
FOOTWEAR_CLASSES = [5, 7, 9]  # Sandal, Sneaker, Ankle boot
NON_FOOTWEAR_CLASSES = [0, 1, 2, 3, 4, 6, 8]  # Everything else

CLASS_NAMES_BINARY = ['Non-Footwear', 'Footwear']


def convert_to_binary_labels(labels):
    """
    Convert 10-class labels to binary labels
    
    Args:
        labels: Original labels (0-9)
    
    Returns:
        Binary labels (0: non-footwear, 1: footwear)
    """
    if isinstance(labels, torch.Tensor):
        binary_labels = torch.zeros_like(labels)
        for footwear_class in FOOTWEAR_CLASSES:
            binary_labels[labels == footwear_class] = 1
    else:  # numpy array
        binary_labels = np.zeros_like(labels)
        for footwear_class in FOOTWEAR_CLASSES:
            binary_labels[labels == footwear_class] = 1
    
    return binary_labels


def get_binary_data_loaders(batch_size=256, data_root='data', num_workers=2):
    """
    Get data loaders for binary classification task
    
    Returns:
        train_loader, val_loader, test_loader (with binary labels)
    """
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Load datasets
    full_train_dataset = FashionMNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = FashionMNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True
    )
    
    # Convert to binary labels
    # Training data
    train_images = []
    train_labels = []
    for img, label in full_train_dataset:
        train_images.append(img)
        train_labels.append(label)
    
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    train_labels_binary = convert_to_binary_labels(train_labels)
    
    # Split into train and validation
    train_size = 50000
    train_dataset = TensorDataset(train_images[:train_size], train_labels_binary[:train_size])
    val_dataset = TensorDataset(train_images[train_size:], train_labels_binary[train_size:])
    
    # Test data
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img)
        test_labels.append(label)
    
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)
    test_labels_binary = convert_to_binary_labels(test_labels)
    
    test_dataset_binary = TensorDataset(test_images, test_labels_binary)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset_binary, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Binary Classification Task: Footwear vs Non-Footwear")
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset_binary)} samples")
    
    return train_loader, val_loader, test_loader


def get_binary_data_numpy(data_root='data', subset_size=None, seed=42):
    """
    Get numpy arrays for binary classification (for SVM)
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (with binary labels)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Load datasets
    full_train_dataset = FashionMNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root=data_root, train=False, transform=transform, download=True)
    
    # Convert to numpy and flatten
    X_train_full = []
    y_train_full = []
    for img, label in full_train_dataset:
        X_train_full.append(img.numpy().flatten())
        y_train_full.append(label)
    
    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)
    y_train_full_binary = convert_to_binary_labels(y_train_full)
    
    # Split
    train_size = 50000
    X_train = X_train_full[:train_size]
    y_train = y_train_full_binary[:train_size]
    X_val = X_train_full[train_size:]
    y_val = y_train_full_binary[train_size:]
    
    # Test data
    X_test = []
    y_test = []
    for img, label in test_dataset:
        X_test.append(img.numpy().flatten())
        y_test.append(label)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_test_binary = convert_to_binary_labels(y_test)
    
    # Optional subset
    if subset_size:
        indices = np.random.RandomState(seed).permutation(len(X_train))[:subset_size]
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    print(f"Binary Classification Task: Footwear vs Non-Footwear")
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test_binary
