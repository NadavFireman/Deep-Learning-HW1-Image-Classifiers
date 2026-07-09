# Deep Learning - Image Classifiers (HW1)
**Home Assignment 1 (M.Sc. Data Science, HIT). Implementing three image classifiers from scratch in PyTorch — a multiclass SVM, a Softmax classifier, and a two-layer neural network — trained on CIFAR-10, with fully vectorized gradient code verified against numeric gradient checking.**

## Overview
This assignment builds three classifiers from the ground up to compare their behavior and tradeoffs on the CIFAR-10 image-classification task. The emphasis is on implementing each loss function and its analytic gradient in vectorized form (rather than relying on autograd), validating correctness with numeric gradient checks, and tuning each model through a hyperparameter search on a held-out validation set.

## Key Features
- **Multiclass SVM Classifier:** Hinge-loss implemented in both naive (loop-based) and fully vectorized form, with analytic gradients.
- **Softmax Classifier:** Cross-entropy loss with a numerically stable softmax, in both naive and vectorized versions.
- **Two-Layer Neural Network:** A fully connected network with a ReLU hidden layer — forward pass, manual backpropagation, SGD training, prediction, and hyperparameter tuning.
- **Vectorized Gradients + Gradient Checking:** Every analytic gradient is verified against a centered finite-difference numeric approximation.
- **Hyperparameter Search:** Learning-rate and regularization tuning for each model, selecting the best configuration on the validation split.
- **Challenge Problem:** A hand-drawn-weights experiment — manually designing network weights, mapping a colormap image back to weight values, and running a manual forward pass to classify MNIST digits.

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch (tensors and training; core gradients implemented manually)
- **Dataset:** CIFAR-10 (with MNIST used in the challenge problem)
- **Environment:** Google Colab

## Repository Structure
- `linear_classifier.ipynb`: Walkthrough notebook for the SVM and Softmax classifiers, with results at each required cell.
- `linear_classifier.py`: Implementation of the SVM and Softmax losses, training loop, and hyperparameter search.
- `two_layer_net.ipynb`: Walkthrough notebook for the two-layer neural network, with results.
- `two_layer_net.py`: Implementation of the network's forward/backward pass, training, prediction, and tuning.
- `eecs598/`: Course helper package — CIFAR-10 loading and preprocessing (`data.py`), gradient utilities (`grad.py`), visualization and assignment helpers (`utils.py`, `a2_helpers.py`), the challenge problem (`a2_challenge_problem.py`), and submission tooling (`submit.py`).
- `Ex 1.pdf`: Original assignment instructions and requirements.

## Acknowledgment
The assignment scaffolding and helper package (`eecs598`) are based on the course materials from the University of Michigan's EECS 498/598 *Deep Learning for Computer Vision* course.
