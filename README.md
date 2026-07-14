# Deep Learning - Image Classifiers (HW1)

**Home Assignment 1 (Grade 95, M.Sc. Data Science, HIT). Three image classifiers built from scratch in PyTorch on CIFAR-10 — every loss and analytic gradient implemented in naive and fully vectorized form (no autograd) and verified by numeric gradient checking.**

## Key Features
- **Multiclass SVM:** Hinge loss with analytic gradients.
- **Softmax Classifier:** Numerically stable cross-entropy.
- **Two-Layer Neural Network:** ReLU hidden layer — forward pass, manual backpropagation, SGD training, and learning-rate/regularization search on a validation split.

## Results
CIFAR-10 test accuracy — SVM: **39.4%**, Softmax: **41.2%**, two-layer network: **56.7%**.

## Repository Structure
- `linear_classifier.ipynb`: Walkthrough notebook — SVM & Softmax, gradient checks, hyperparameter search and results.
- `linear_classifier.py`: Implementation of the SVM and Softmax losses, training loop and search functions.
- `two_layer_net.ipynb`: Walkthrough notebook — the two-layer network, from toy data to CIFAR-10.
- `two_layer_net.py`: Implementation of the network's forward/backward pass, training, prediction and tuning.
- `Ex 1.pdf`: Original assignment instructions.
- `eecs598/` — Course helper package:
  - `__init__.py`: Package initialization and seed utilities.
  - `data.py`: CIFAR-10 downloading, loading and preprocessing.
  - `grad.py`: Numeric gradient checking (finite differences, relative error).
  - `utils.py`: General assignment utilities.
  - `a2_helpers.py`: Toy-data generation and training-curve / weight visualization.
  - `a2_challenge_problem.py`: Scaffolding for the optional challenge problem.
  - `submit.py`: Submission packaging tooling.
