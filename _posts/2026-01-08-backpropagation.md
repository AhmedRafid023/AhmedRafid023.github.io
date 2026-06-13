---
layout: post
title: A Comprehensive Guide to Backpropagation
date: 2026-01-08
description: A complete mathematical derivation and numerical example of Backpropagation in a two-layer Neural Network.
tags: deep-learning math neural-networks
categories: [Deep Learning]
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

This note provides a comprehensive and self-contained overview of Backpropagation, the fundamental algorithm used to train multi-layer artificial neural networks. The mathematical mechanics of the algorithm are derived in detail using the chain rule of calculus to compute gradients of the loss function with respect to each trainable parameter. The exposition is structured around two main phases: the forward pass, where input data is transformed into predictions, and the backward pass, where error gradients are propagated from the output layer back through the network. A complete derivation is presented for a two-layer neural network with ReLU activation, including a worked numerical example to illustrate the full training step.

<br>

## Backpropagation in a Two-Layer Neural Network with ReLU

Backpropagation is the core algorithm used to train neural networks. It efficiently computes gradients of the loss function with respect to all model parameters by recursively applying the chain rule of calculus. These gradients are then used to update the parameters using gradient-based optimization methods such as gradient descent. This section presents a complete derivation of backpropagation for a simple two-layer neural network with a ReLU activation function.

{% include figure.liquid
  path="assets/img/post_images/backprop.png"
  class="img-fluid rounded z-depth-1"
  caption="Figure 1: Architecture of the two-layer neural network used in this derivation, illustrating the forward pass (blue arrows) and backpropagation of errors (red dashed arrows)."
%}

The network architecture is:

$$
\text{Input } (3) \;\rightarrow\; \text{Hidden } (5) \;\rightarrow\; \text{Output } (3)
$$

<br>

### Network Architecture

Let the input vector be:

$$
x \in \mathbb{R}^{3}
$$

The first linear layer is defined as:

$$
z = W_1 x + b_1, \quad W_1 \in \mathbb{R}^{5 \times 3}, \; b_1 \in \mathbb{R}^{5}
$$

A ReLU activation is applied:

$$
h = \text{ReLU}(z)
$$

The second linear layer is:

$$
y = W_2 h + b_2, \quad W_2 \in \mathbb{R}^{3 \times 5}, \; b_2 \in \mathbb{R}^{3}
$$

<br>

### ReLU Activation

The Rectified Linear Unit (ReLU) is defined element-wise as:

$$
\text{ReLU}(z_i) = \max(0, z_i)
$$

Its derivative is:

$$
\frac{d}{dz_i} \text{ReLU}(z_i) =
\begin{cases}
1 & \text{if } z_i > 0 \\
0 & \text{if } z_i \le 0
\end{cases}
$$

<br>

---

## Forward Pass Example

### Define the Input

Let the input vector be:

$$
x =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

**Shape:** $(3 \times 1)$

### First Linear Layer ($3 \to 5$)

#### Weights and Bias

Weight matrix $W_1$ ($5 \times 3$):

$$
W_1 =
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & 1 \\
1 & 1 & 0 \\
2 & 0 & 1 \\
0 & 2 & 1
\end{bmatrix}
$$

Bias $b_1$ ($5 \times 1$):

$$
b_1 =
\begin{bmatrix}
1 \\
0 \\
-1 \\
2 \\
1
\end{bmatrix}
$$

#### Linear Transformation

First, we calculate the linear pre-activation output $z = W_1 x + b_1$.

Compute the matrix-vector product $W_1 x$:

$$
W_1 x =
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & 1 \\
1 & 1 & 0 \\
2 & 0 & 1 \\
0 & 2 & 1
\end{bmatrix}
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
=
\begin{bmatrix}
7 \\
5 \\
3 \\
5 \\
7
\end{bmatrix}
$$

Add the bias vector $b_1$:

$$
z =
\begin{bmatrix}
7 \\
5 \\
3 \\
5 \\
7
\end{bmatrix}
+
\begin{bmatrix}
1 \\
0 \\
-1 \\
2 \\
1
\end{bmatrix}
=
\begin{bmatrix}
8 \\
5 \\
2 \\
7 \\
8
\end{bmatrix}
$$

<br>

#### ReLU Activation

Now apply the Rectified Linear Unit (ReLU) activation function element-wise to get the hidden state $h$:

$$
h = \text{ReLU}(z) = \max(0, z)
$$

Since all elements in $z$ are positive, the values remain unchanged:

$$
h = \text{ReLU}
\left(
\begin{bmatrix}
8 \\
5 \\
2 \\
7 \\
8
\end{bmatrix}
\right)
=
\begin{bmatrix}
8 \\
5 \\
2 \\
7 \\
8
\end{bmatrix}
$$

**Shape:** $(5 \times 1)$

### Second Linear Layer ($5 \to 3$)

#### Weights and Bias

Weight matrix $W_2$ ($3 \times 5$):

$$
W_2 =
\begin{bmatrix}
1 & 0 & 1 & 0 & 2 \\
0 & 1 & 0 & 1 & 1 \\
1 & 1 & 0 & 2 & 0
\end{bmatrix}
$$

Bias $b_2$ ($3 \times 1$):

$$
b_2 =
\begin{bmatrix}
0 \\
1 \\
-1
\end{bmatrix}
$$

#### Linear Transformation

The transformation is defined as:

$$
y = W_2 h + b_2
$$

$$
\begin{aligned}
W_2 h &=
\begin{bmatrix}
1 & 0 & 1 & 0 & 2 \\
0 & 1 & 0 & 1 & 1 \\
1 & 1 & 0 & 2 & 0
\end{bmatrix}
\begin{bmatrix}
8 \\
5 \\
2 \\
7 \\
8
\end{bmatrix} \\
&=
\begin{bmatrix}
26 \\
20 \\
27
\end{bmatrix}
\end{aligned}
$$

Now, add the bias vector $b_2$:

$$
y =
\begin{bmatrix}
26 \\
20 \\
27
\end{bmatrix}
+
\begin{bmatrix}
0 \\
1 \\
-1
\end{bmatrix}
=
\begin{bmatrix}
26 \\
21 \\
26
\end{bmatrix}
$$

### Final Output of the Network

$$
\boxed{
y =
\begin{bmatrix}
26 \\
21 \\
26
\end{bmatrix}
}
$$

**Shape:** $(3 \times 1)$

## <br>

## Loss Function

We use Mean Squared Error (MSE) loss:

$$
L = \frac{1}{2} \sum_{i=1}^{3} (y_i - t_i)^2
$$

where $t \in \mathbb{R}^{3}$ is the target vector.

### Gradient at the Output Layer

The gradient of the loss with respect to the output is:

$$
\frac{\partial L}{\partial y_i} = y_i - t_i
$$

In vector form:

$$
\delta^{(2)} = \frac{\partial L}{\partial y} = y - t
$$

### Gradients for the Second Layer

The output layer is:

$$
y = W_2 h + b_2
$$

#### Derivative with Respect to a Single Weight

For a single weight $W_{2_{ij}}$:

$$
y_i = \sum_{k=1}^{5} W_{2_{ik}} h_k + b_{2_i}
$$

Applying the chain rule:

$$
\frac{\partial L}{\partial W_{2_{ij}}}
=
\frac{\partial L}{\partial y_i}
\cdot
\frac{\partial y_i}{\partial W_{2_{ij}}}
=
\delta^{(2)}_i \cdot h_j
$$

#### Matrix Form

If we write this for all $i, j$:

$$
\frac{\partial L}{\partial W_2}
=
\begin{bmatrix}
\delta^{(2)}_1 h_1 & \delta^{(2)}_1 h_2 & \cdots & \delta^{(2)}_1 h_5 \\
\delta^{(2)}_2 h_1 & \delta^{(2)}_2 h_2 & \cdots & \delta^{(2)}_2 h_5 \\
\delta^{(2)}_3 h_1 & \delta^{(2)}_3 h_2 & \cdots & \delta^{(2)}_3 h_5
\end{bmatrix}
$$

This structure is exactly the definition of the **outer product** between the error vector $\delta^{(2)}$ and the transpose of the hidden vector $h^T$:

$$
\boxed{
\frac{\partial L}{\partial W_2} = \delta^{(2)} h^T
}
$$

#### Bias Gradient

$$
\boxed{
\frac{\partial L}{\partial b_2} = \delta^{(2)}
}
$$

<br>

### Backpropagating to the Hidden Layer

Each hidden neuron contributes to all output neurons. Using the multivariable chain rule:

$$
\frac{\partial L}{\partial h_j}
=
\sum_{i=1}^{3}
\frac{\partial L}{\partial y_i}
\cdot
\frac{\partial y_i}{\partial h_j}
=
\sum_{i=1}^{3}
\delta^{(2)}_i \cdot W_{2_{ij}}
$$

Stacking all $j$:

$$
\boxed{
\delta^{(1)} = \frac{\partial L}{\partial h} = W_2^T \delta^{(2)}
}
$$

### Backpropagation Through ReLU

Since $h = \text{ReLU}(z)$:

$$
\delta^{(z)} = \frac{\partial L}{\partial z} = \delta^{(1)} \odot \mathbb{I}(z > 0)
$$

where $\odot$ denotes element-wise multiplication and $\mathbb{I}(z > 0)$ is the indicator function.

### Gradients for the First Layer

The first layer is $z = W_1 x + b_1$.

**Weight Gradient:**

$$
\boxed{
\frac{\partial L}{\partial W_1} = \delta^{(z)} x^T
}
$$

**Bias Gradient:**

$$
\boxed{
\frac{\partial L}{\partial b_1} = \delta^{(z)}
}
$$

### Weight Update Rule

Using gradient descent with learning rate $\eta$:

$$
W_2 \leftarrow W_2 - \eta \, \delta^{(2)} h^T, \quad
b_2 \leftarrow b_2 - \eta \, \delta^{(2)}
$$

$$
W_1 \leftarrow W_1 - \eta \, \delta^{(z)} x^T, \quad
b_1 \leftarrow b_1 - \eta \, \delta^{(z)}
$$

<br>

---

## Worked Numerical Example (Backward Pass)

Let the target be:

$$
t =
\begin{bmatrix}
30 \\ 20 \\ 25
\end{bmatrix}
$$

**Loss:**

$$
L = \frac{1}{2}[(26-30)^2 + (21-20)^2 + (26-25)^2] = \frac{1}{2}[16 + 1 + 1] = 9
$$

**Output Error:**

$$
\delta^{(2)} = y - t =
\begin{bmatrix}
-4 \\ 1 \\ 1
\end{bmatrix}
$$

**Hidden Layer Error:**

$$
\delta^{(1)} = W_2^T \delta^{(2)} =
\begin{bmatrix}
-3 \\ 2 \\ -4 \\ 3 \\ -7
\end{bmatrix}
$$

Since all $z_i > 0$, $\delta^{(z)} = \delta^{(1)}$.

With these values, we can now update $W_1, b_1, W_2, b_2$ using the formulas derived above.

<br>

---

## The Vanishing and Exploding Gradient Problem

In the derivations above, we saw that the gradient for the first layer depends on the product of the weights and error signals from later layers:

$$
\delta^{(1)} = W_2^T \delta^{(2)}
$$

In a very deep neural network (many layers), the gradient at an early layer $l$ is calculated by repeatedly multiplying gradients through the Chain Rule:

$$
\frac{\partial L}{\partial W_l} \propto \underbrace{W_{L}^T \cdot D_{L} \cdot W_{L-1}^T \cdot D_{L-1} \dots}_{\text{Repeated Multiplication}}
$$

where $D$ represents the diagonal matrix of the derivative of the activation functions.

### Vanishing Gradients

The vanishing gradient problem occurs when the elements of the chain rule are smaller than 1 (e.g., small weights or activation derivatives $< 1$). As these small numbers are multiplied repeatedly during backpropagation from the output layer to the input layer, the gradient signal shrinks exponentially.

- **Cause:** Traditional activation functions like **Sigmoid** or **Tanh** have derivatives that are always less than 1 (max 0.25 for Sigmoid).
- **Effect:** The weights in the earlier layers of the network stop updating because their gradients become effectively zero. The network fails to learn features in the early layers.

### Exploding Gradients

Conversely, if the weights or derivatives are large ($> 1$), repeated multiplication causes the gradients to grow exponentially as they move backward.

- **Cause:** Poor weight initialization or lack of normalization.
- **Effect:** The weight updates become massive, causing the model parameters to oscillate wildly or diverge to `NaN` (Not a Number).

<br>

### Why ReLU Helps

In our derivation, we used the **ReLU** activation function.

$$
\frac{d}{dz} \text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}
$$

Because the derivative is exactly **1** for positive inputs, the gradient signal does not diminish as it passes through active neurons, regardless of network depth. This property makes ReLU (and its variants) essential for training deep networks, effectively solving the vanishing gradient problem for positive activation paths.
