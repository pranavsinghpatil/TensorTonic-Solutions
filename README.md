# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Pranav's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/pranavsingh.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Batch Shuffling & Mini-Batch Generator | Create shuffled mini-batches from NumPy feature and target arrays with reproducible ordering and final-batch handling. | https://www.tensortonic.com/problems/batch-generator |
| Bootstrap Mean & Confidence Interval | Estimate a sample mean and confidence interval through reproducible bootstrap resampling of numeric observations. | https://www.tensortonic.com/problems/bootstrap-mean |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Calculate Eigenvalues of a Matrix | Calculate the eigenvalues of a square matrix and return them in the format required by the numerical contract. | https://www.tensortonic.com/problems/eigenvalues |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Apply 4×4 Homogeneous Transform | Apply a 4x4 homogeneous transformation matrix to 3D points using rotation, translation, and homogeneous coordinates. | https://www.tensortonic.com/problems/homogeneous-transform |
| Impute Missing Values (mean/median) | Impute missing numeric values column-wise with either the mean or median while leaving observed values unchanged. | https://www.tensortonic.com/problems/impute-missing |
| K-Fold Split (Indices Only) | Generate deterministic K-fold train and validation index splits that use every sample exactly once for validation. | https://www.tensortonic.com/problems/kfold-split |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| One-Hot Encoding (Multi-class) | Convert multiclass integer labels into a NumPy one-hot matrix with one active column per sample. | https://www.tensortonic.com/problems/one-hot-encoding |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Poisson Probability Mass Function & Cumulative Distribution Function | Compute Poisson probability mass and cumulative probabilities for a nonnegative event count and rate. | https://www.tensortonic.com/problems/poisson-pmf-cdf |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Ridge Regression | Fit ridge regression with L2 regularization using the closed-form solution required by the problem. | https://www.tensortonic.com/problems/ridge-regression |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Stratified Train/Test Split | Split indices into train and test sets while approximately preserving the class distribution of each label. | https://www.tensortonic.com/problems/stratified-split |
| One-Sample t-Test | Compute a one-sample t-statistic in NumPy using the sample mean, Bessel-corrected deviation, and hypothesized mean. | https://www.tensortonic.com/problems/t-test-one-sample |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Estimate a Scalar Derivative | Estimate a scalar polynomial derivative with a forward finite difference using coefficients ordered by ascending power. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-finite-difference-derivative |
| Check a Product-Chain Gradient | Compare analytic gradients with forward-difference estimates for a two-operation scalar graph. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-gradient-check-product-chain |
| Apply a Gradient-Descent Step | Apply one NumPy gradient-descent update and compute the first-order predicted objective change without mutating inputs. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-gradient-descent-step |
| Measure Scalar Expression Partials | Estimate the three partial derivatives of a scalar expression by perturbing one input at a time. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-scalar-expression-partials |
| Build a Scalar Expression Graph | Build a scalar expression graph in forward order from leaf records and addition or multiplication operations. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l02-build-expression-graph |
| Trace a Reachable Expression Graph | Trace the ordered nodes and parent-to-child edges that contribute to one output in a scalar computation graph. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l02-trace-reachable-graph |
| Create an Addition Value Node | Create a scalar addition node that stores its forward value, operation type, and ordered parent identifiers. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l02-value-addition-node |
| Create a Multiplication Value Node | Create a scalar multiplication node that stores its forward value, operation type, and ordered parent identifiers. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l02-value-multiplication-node |
| Backpropagate Through a Scalar Neuron | Evaluate one tanh neuron and manually propagate an upstream gradient to its inputs, weights, and bias. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l04-neuron-backward |
| Evaluate a Scalar Neuron | Evaluate a scalar PyTorch tanh neuron from aligned inputs, weights, and bias using promoted floating-point types. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l04-neuron-forward |
| Gradient-Check a Scalar Neuron | Check a tanh neuron's analytic parameter gradients against one-parameter-at-a-time forward differences in float64. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l04-neuron-gradient-check |
| Differentiate a Tanh Activation | Evaluate scalar tanh and manually combine its local derivative with an upstream gradient. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l04-tanh-forward-backward |
| Apply Local Vector-Jacobian Rules | A reverse-mode autodiff engine moves an upstream scalar gradient through one operation at a time. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l05-local-vjp-rules |
| Topologically Sort a Computation DAG | Reverse-mode autodiff must process children before their parents during the backward pass. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l05-topological-sort |
| GELU | Implement exact GELU activation in CUDA with one thread per element and the device error-function intrinsic. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/gelu |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/leaky-relu |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/relu |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/sigmoid |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/swish |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/tanh |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/vector-subtract |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/pranavsingh)
<!-- tensortonic:end -->
