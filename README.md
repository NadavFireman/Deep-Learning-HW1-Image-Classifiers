# Deep Learning - Image Classifiers (HW1)

**Home Assignment 1 (M.Sc. Data Science, HIT). Three image classifiers built from scratch in PyTorch — multiclass SVM, Softmax, and a two-layer neural network — trained on CIFAR-10, with every analytic gradient verified by numeric gradient checking.**

## Overview
Each loss function and its gradient is implemented in both naive (loop-based) and fully vectorized form — no autograd — validated with centered finite-difference checks, and tuned through a hyperparameter search on a held-out validation split.

## Key Features
- **Multiclass SVM:** Hinge loss with analytic gradients, naive and vectorized.
- **Softmax Classifier:** Numerically stable cross-entropy, naive and vectorized.
- **Two-Layer Neural Network:** ReLU hidden layer — forward pass, manual backpropagation, SGD training, prediction and tuning.
- **Gradient Checking & Hyperparameter Search:** Every gradient verified against finite differences; learning rate and regularization tuned per model on the validation set.

## Results
CIFAR-10 test accuracy — SVM: **39.4%**, Softmax: **41.2%**, two-layer network: **56.7%** (validation bests: 39.7% / 41.0% / 57.4%).

## Repository Structure
- `linear_classifier.ipynb` / `linear_classifier.py`: SVM & Softmax — walkthrough notebook and implementation.
- `two_layer_net.ipynb` / `two_layer_net.py`: Two-layer network — walkthrough notebook and implementation.
- `eecs598/`: Course helper package — CIFAR-10 loading, gradient utilities, visualization and submission tooling.
- `Ex 1.pdf`: Original assignment instructions.

## Acknowledgment
Assignment scaffolding and the `eecs598` helper package are based on the University of Michigan's EECS 498/598 *Deep Learning for Computer Vision* course materials.

---
*Course: Deep Learning, M.Sc. Data Science, HIT · Python, PyTorch*
