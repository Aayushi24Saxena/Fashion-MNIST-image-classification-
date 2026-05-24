"""
Run Binary Classification Experiments

This script trains models on the binary classification task: Footwear vs Non-Footwear
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from auxiliary_tasks.binary_classification import get_binary_data_loaders, get_binary_data_numpy, CLASS_NAMES_BINARY
from models.cnn.model import CNN
from models.traditional.model import SimpleNN
from models.svm.model import SVMClassifier
from models.cnn.train import train_model as train_nn_model, evaluate as evaluate_nn
from models.svm.train import train_model as train_svm_model, evaluate as evaluate_svm
from models.cnn.evaluate import plot_confusion_matrix, generate_classification_report
import argparse


def train_binary_cnn(epochs=15, lr=0.001, batch_size=256):
    """Train CNN on binary classification task"""
    print("\n" + "="*70)
    print("TRAINING CNN - BINARY CLASSIFICATION (Footwear vs Non-Footwear)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Get data
    train_loader, val_loader, test_loader = get_binary_data_loaders(batch_size=batch_size)
    
    # Create model (2 classes instead of 10)
    model = CNN(num_classes=2, dropout_rate=0.3).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    config = {
        'epochs': epochs,
        'experiment_name': 'binary_cnn',
        'save_dir': 'model'
    }
    
    # Train
    results = train_nn_model(model, device, train_loader, val_loader, test_loader,
                            optimizer, criterion, scheduler, config, wandb_run=None)
    
    # Evaluate and save reports
    os.makedirs('images/binary_classification', exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(results['test_targets'], results['test_predictions'],
                         save_path='images/binary_classification/cnn_confusion_matrix.png',
                         class_names=CLASS_NAMES_BINARY)
    
    # Classification report (use binary class names)
    from sklearn.metrics import classification_report
    report = classification_report(results['test_targets'], results['test_predictions'],
                                  target_names=CLASS_NAMES_BINARY, digits=4)
    print("\n" + report)
    with open('images/binary_classification/cnn_classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n✅ Binary CNN - Test Accuracy: {results['test_accuracy']:.2f}%")
    return results


def train_binary_nn(epochs=15, lr=0.001, batch_size=128):
    """Train Traditional NN on binary classification task"""
    print("\n" + "="*70)
    print("TRAINING NN - BINARY CLASSIFICATION (Footwear vs Non-Footwear)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Get data
    train_loader, val_loader, test_loader = get_binary_data_loaders(batch_size=batch_size)
    
    # Create model (2 classes)
    model = SimpleNN(num_classes=2, dropout_rate=0.3).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    config = {
        'epochs': epochs,
        'experiment_name': 'binary_nn',
        'save_dir': 'model'
    }
    
    # Train
    results = train_nn_model(model, device, train_loader, val_loader, test_loader,
                            optimizer, criterion, scheduler, config, wandb_run=None)
    
    # Evaluate and save reports
    os.makedirs('images/binary_classification', exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(results['test_targets'], results['test_predictions'],
                         save_path='images/binary_classification/nn_confusion_matrix.png',
                         class_names=CLASS_NAMES_BINARY)
    
    # Classification report
    from sklearn.metrics import classification_report
    report = classification_report(results['test_targets'], results['test_predictions'],
                                  target_names=CLASS_NAMES_BINARY, digits=4)
    print("\n" + report)
    with open('images/binary_classification/nn_classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n✅ Binary NN - Test Accuracy: {results['test_accuracy']:.2f}%")
    return results


def train_binary_svm(kernel='rbf', C=1.0):
    """Train SVM on binary classification task"""
    print("\n" + "="*70)
    print(f"TRAINING SVM ({kernel.upper()}) - BINARY CLASSIFICATION")
    print("="*70)
    
    # Get data
    X_train, y_train, X_val, y_val, X_test, y_test = get_binary_data_numpy(subset_size=10000)
    
    # Create model
    model = SVMClassifier(kernel=kernel, C=C, max_iter=1000, random_state=42)
    
    # Train
    config = {'kernel': kernel, 'C': C, 'experiment_name': f'binary_svm_{kernel}'}
    results = train_svm_model(model, X_train, y_train, X_val, y_val, config)
    
    # Evaluate
    test_acc, test_pred = evaluate_svm(model, X_test, y_test)
    
    # Save reports
    os.makedirs('images/binary_classification', exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, test_pred,
                         save_path=f'images/binary_classification/svm_{kernel}_confusion_matrix.png',
                         class_names=CLASS_NAMES_BINARY)
    
    # Classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, test_pred, target_names=CLASS_NAMES_BINARY, digits=4)
    print("\n" + report)
    with open(f'images/binary_classification/svm_{kernel}_classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n✅ Binary SVM ({kernel}) - Test Accuracy: {test_acc:.2f}%")
    return {'test_accuracy': test_acc}


def main():
    parser = argparse.ArgumentParser(description='Binary Classification Experiments')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['cnn', 'nn', 'svm', 'all'],
                       help='Model to train')
    parser.add_argument('--svm-kernel', type=str, default='rbf',
                       choices=['linear', 'rbf', 'poly'],
                       help='SVM kernel type')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.model in ['cnn', 'all']:
        results['cnn'] = train_binary_cnn()
    
    if args.model in ['nn', 'all']:
        results['nn'] = train_binary_nn()
    
    if args.model in ['svm', 'all']:
        results['svm'] = train_binary_svm(kernel=args.svm_kernel)
    
    # Summary
    print("\n" + "="*70)
    print("BINARY CLASSIFICATION RESULTS SUMMARY")
    print("="*70)
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result['test_accuracy']:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
