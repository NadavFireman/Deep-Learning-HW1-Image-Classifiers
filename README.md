# Deep Learning - Image Classifiers (HW1)

**Home Assignment 1 (M.Sc. Data Science, HIT). Three image classifiers built from scratch in PyTorch on CIFAR-10 — every loss and analytic gradient implemented in naive and fully vectorized form (no autograd) and verified by numeric gradient checking.**

## Key Features
- **Multiclass SVM:** Hinge loss with analytic gradients.
- **Softmax Classifier:** Numerically stable cross-entropy.
- **Two-Layer Neural Network:** ReLU hidden layer — forward pass, manual backpropagation, SGD training, and learning-rate/regularization search on a validation split.

## Results
CIFAR-10 test accuracy — SVM: **39.4%**, Softmax: **41.2%**, two-layer network: **56.7%**.

## Repository Structure
- `linear_classifier.ipynb` / `.py`: SVM & Softmax — notebook and implementation.
- `two_layer_net.ipynb` / `.py`: Two-layer network — notebook and implementation.
- `eecs598/`: Course helper package (CIFAR-10 loading, gradient and visualization utilities).
- `Ex 1.pdf`: Original assignment instructions.

## Acknowledgment
Scaffolding and the `eecs598` package are based on the University of Michigan's EECS 498/598 *Deep Learning for Computer Vision* course.

---
*Course: Deep Learning, M.Sc. Data Science, HIT · Python, PyTorch*
