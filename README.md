# Neural Network from Scratch for MNIST Classification

This project involves building and training a neural network from scratch to classify images from the MNIST dataset. The network is implemented using basic Python and NumPy, with a focus on understanding the inner workings of neural networks without relying on high-level libraries like TensorFlow or PyTorch.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Implementation Details](#implementation-details)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [Future Work](#future-work)

## Introduction

The goal of this project is to demonstrate how to build a neural network from scratch, understand its components, and apply it to the classic MNIST digit classification problem. The project includes forward and backward propagation, dropout, L2 regularization, and gradient descent optimization.

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of 42,000 28x28 grayscale images of handwritten digits, split into 80% training examples and 20% test examples. Each image is labeled with the correct digit (0-9).

## Model Architecture

The neural network architecture used in this project is as follows:
- **Input Layer**: 784 units (28x28 pixels flattened)
- **Hidden Layers**: Two hidden layers with ReLU activation
  - First hidden layer: 128 units
  - Second hidden layer: 64 units
  - Third hidden layer: 32 units
  - Fourth hidden layer: 16 units
- **Output Layer**: 10 units with Softmax activation (one for each digit 0-9)

### Key Features
- **Dropout**: Applied during training to prevent overfitting.
- **L2 Regularization**: Implemented to penalize large weights and prevent overfitting.

## Implementation Details

The project is structured into a classe with different methods that handle different aspects of the neural network:
- **Initialization**: Weights and biases are initialized using random values.
- **Forward Propagation**: Computes activations and cache for backward propagation.
- **Backward Propagation**: Computes gradients and updates parameters.
- **Loss Calculation**: Uses categorical cross-entropy as the loss function.
- **Parameter Updates**: Parameters are updated using gradient descent.

## Training and Evaluation

### Hyperparameters
- **Learning Rate**: `0.1`
- **Iterations**: `1500`
- **Dropout Probability**: `0.75`
- **L2 Regularization Lambda**: `0.9`

### Training Process
1. **Forward Pass**: Compute the predicted output.
2. **Compute Loss**: Calculate the cost using cross-entropy loss.
3. **Backward Pass**: Compute gradients for all parameters.
4. **Parameter Update**: Adjust weights and biases based on computed gradients.

### Evaluation
- The model is evaluated on the test set using accuracy as the primary metric.

### Requirements
- Python 3.x
- NumPy
- Matplotlib
- Pandas

## Results

- The model achieves an accuracy of `96.6%` on the MNIST test set.
- Loss decreases steadily during training, indicating effective learning.

## Future Work

- **Improve Accuracy**: Experiment with deeper architectures, learning rate schedules, or alternative activation functions.
- **Add Visualization**: Visualize incorrect classifications to better understand model weaknesses.
- **Hyperparameter Tuning**: Use techniques like grid search or random search to find optimal hyperparameters.
