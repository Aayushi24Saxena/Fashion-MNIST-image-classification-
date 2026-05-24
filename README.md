# Neural Network vs. CNN Comparison on Fashion-MNIST

This project compares the performance of a traditional Fully Connected Neural Network (NN) and a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset.

## Features

* **Direct Comparison:** Train and evaluate both an NN and a CNN on the same dataset.
* **Fashion-MNIST Dataset:** Uses the 28x28 grayscale Fashion-MNIST dataset.
* **Modular Design:** Separate experiments for Traditional NN, CNN, and SVM models.
* **Configurable:** Uses YAML configuration files for easy experiment management.

## Getting Started

### Prerequisites

* Python 3.6+
* PyTorch (`torch`, `torchvision`)
* Other dependencies: `pyyaml`, `wandb`, `tqdm`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run experiments using the `run_experiment.py` script for the desired model type, passing a configuration file.

**1. Convolutional Neural Network (CNN)**

```bash
python models/cnn/experiments/run_experiment.py --config models/cnn/configs/baseline.yaml
```

Available configs: `baseline.yaml`, `deeper_network.yaml`

**2. Traditional Neural Network**

```bash
python models/traditional/experiments/run_experiment.py --config models/traditional/configs/baseline.yaml
```

Available configs: `baseline.yaml`, `wide_network.yaml`

**3. Support Vector Machine (SVM)**

```bash
python models/svm/experiments/run_experiment.py --config models/svm/configs/linear_kernel.yaml
```

Available configs: `linear_kernel.yaml`, `poly_kernel.yaml`, `rbf_kernel.yaml`, `rbf_low_C.yaml`, `rbf_high_C.yaml`

## Project Structure

```
.
├── data/                         # Dataset storage
├── models/                       # Model implementations
│   ├── cnn/                      # CNN model, configs, and experiments
│   ├── traditional/              # Traditional NN model, configs, and experiments
│   └── svm/                      # SVM model, configs, and experiments
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```
