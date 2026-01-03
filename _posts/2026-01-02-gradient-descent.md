---
layout: post
title: "Gradient Descent"
date: 2026-01-02
description: Understanding the Gradient vector, Partial Derivatives, and Backpropagation.
tags: math calculus deeplearning
categories: [Deep Learning]
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

Before we can understand how neural networks learn (via Gradient Descent), we must first understand the "Gradient" itself. In single-variable calculus, the derivative tells us the slope of a line. In deep learning, however, we deal with high-dimensional surfaces. This is the domain of **Multivariable Calculus**.

## 1. The Partial Derivative

Imagine a function $f(x, y)$ that depends on two variables. This forms a 3D surface (like a mountain terrain).
To find the slope, we cannot just ask "what is the slope?" because the slope depends on which direction we walk.

Instead, we ask: *"If we hold $y$ constant and only move $x$, how does $f$ change?"*

This is the **partial derivative** with respect to $x$, denoted as:

$$
\frac{\partial f}{\partial x}
$$

It treats all other variables as constants and measures the sensitivity of the function to just one variable.

<br>

## 2. The Gradient Vector ($\nabla$)

The **Gradient** is simply a vector that collects all these partial derivatives together. It is denoted by the symbol nabla ($\nabla$).

For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ with inputs $\mathbf{x} = [x_1, x_2, \dots, x_n]$, the gradient is:

$$
\nabla f(\mathbf{x}) = 
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

Even though $f$ outputs a single scalar value (like "Loss"), the gradient $\nabla f$ is a **vector** that lives in the input space.

<br>
<br>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <div id="gradient-plot" style="width: 100%; height: 500px;"></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
        fetch("{{ '/assets/json/gradient_surface.json' | relative_url }}")
            .then(response => response.json())
            .then(data => {
            Plotly.newPlot('gradient-plot', data.data, data.layout);
            });
        </script>
    </div>
</div>
<div class="caption">
    Figure 1: An interactive 3D "Loss Landscape". The Gradient vector at any point on this surface points in the direction of steepest ascent (uphill).
</div>

<br>
<br>

## 3. Geometric Interpretation (The Key Insight)

This is the most important concept for Machine Learning:

> **The Gradient vector $\nabla f(\mathbf{x})$ points in the direction of the steepest ascent.**

If you are standing on a mountain side (a point on the Loss landscape):
1.  The **magnitude** (length) of the vector tells you how steep the slope is.
2.  The **direction** of the vector points to the peak (uphill).

Because we want to minimize error (go to the bottom of the valley), we must move in the **opposite** direction of the gradient. This is why the update rule has a minus sign.

---

<br>

## 4. Analyzing the Plot: A Concrete Example

Let's look at the "Bowl" shape plotted above. Mathematically, this surface represents a simple convex function:

$$
J(x, y) = x^2 + y^2
$$

Imagine we initialize our weights at the point **$(3, 4)$**.
At this position, our "Loss" (height) is $3^2 + 4^2 = 25$. We are high up on the hill.

### Step 1: Calculate the Gradient
To find the slope, we compute the partial derivatives:

$$
\frac{\partial J}{\partial x} = 2x, \quad \frac{\partial J}{\partial y} = 2y
$$

Evaluating at our point $(3, 4)$:

$$
\nabla J(3, 4) = \begin{bmatrix} 2(3) \\ 2(4) \end{bmatrix} = \begin{bmatrix} 6 \\ 8 \end{bmatrix}
$$

### Step 2: What does this vector $[6, 8]$ mean?
1.  **Direction:** The vector points in the direction of positive $x$ and positive $y$. If you look at the plot, moving "outward" (away from the center) goes **uphill**. The gradient always points to the peak.
2.  **Magnitude:** The steepness is $\sqrt{6^2 + 8^2} = 10$. This is a steep slope!

<br>

## 5. Performing Gradient Descent

Since we want to minimize loss (go downhill), we must move in the **opposite** direction of the gradient.

Let's apply the update rule with a learning rate (step size) of $\alpha = 0.1$:

$$
\begin{aligned}
\begin{bmatrix} x_{new} \\ y_{new} \end{bmatrix} &= \begin{bmatrix} x_{old} \\ y_{old} \end{bmatrix} - \alpha \cdot \nabla J(3, 4) \\
&= \begin{bmatrix} 3 \\ 4 \end{bmatrix} - 0.1 \begin{bmatrix} 6 \\ 8 \end{bmatrix} \\
&= \begin{bmatrix} 3 - 0.6 \\ 4 - 0.8 \end{bmatrix} \\
&= \begin{bmatrix} 2.4 \\ 3.2 \end{bmatrix}
\end{aligned}
$$

### The Result
We moved from $(3, 4)$ to $(2.4, 3.2)$.
Did we improve? Let's check the new Loss:
$$
J(2.4, 3.2) = 2.4^2 + 3.2^2 = 5.76 + 10.24 = 16
$$
The Loss decreased from **25** to **16**. We have successfully taken one step towards the bottom of the bowl.

---

<br>

## 6. How it works in Deep Learning (Backpropagation)

In the example above, we calculated derivatives by hand because the function $x^2 + y^2$ is simple.
However, a Deep Neural Network is a massive composite function with millions of parameters:

$$
\hat{y} = \sigma(W_3 \cdot \sigma(W_2 \cdot \sigma(W_1 \cdot x)))
$$

We cannot derive a simple formula for the gradient of this monster. Instead, we use the **Chain Rule**. This process is called **Backpropagation**.

### The Chain Rule Visualized
Imagine a simple computational graph where input $x$ passes through a weight $w$ and bias $b$ to calculate a Loss $L$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
<pre style="line-height: 1.5; font-family: monospace; background: transparent; border: 1px solid rgba(127, 127, 127, 0.2); padding: 15px; border-radius: 5px;">
      [Input x] 
           \
            ( * ) ────> [ z = w·x ] ────> ( Loss L ) <──── [Target y]
           /
      [Weight w]

FORWARD:  Move Left to Right (Calculate Loss)
BACKWARD: Move Right to Left (Calculate Gradients)
</pre>
    </div>
</div>
<div class="caption">
    Figure 2: Text-based computational graph. Inputs x and w combine to form z, which is compared against target y to compute Loss.
</div>

Mathematically, if $y = f(u)$ and $u = g(x)$, then the change in $y$ caused by $x$ is the product of the individual changes:

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}
$$

### Deep Learning Example
Let's trace a single neuron with a squared error loss.
1.  **Forward Pass:** We compute the prediction and the error.
    * $z = w \cdot x$
    * $L = (z - y_{true})^2$

2.  **Backward Pass (Calculating Gradient):**
    We want to know: *How does changing the weight $w$ affect the Loss $L$?*
    
    We start from the Loss and work backward:

$$
\frac{\partial L}{\partial w} = \underbrace{\frac{\partial L}{\partial z}}_{\text{Step 1: Loss to Output}} \cdot \underbrace{\frac{\partial z}{\partial w}}_{\text{Step 2: Output to Weight}}
$$

Let's calculate the pieces:
* $\frac{\partial L}{\partial z} = 2(z - y_{true})$
* $\frac{\partial z}{\partial w} = x$

So the gradient for weight $w$ is:

$$
\nabla w = 2(w \cdot x - y_{true}) \cdot x
$$

In frameworks like PyTorch or TensorFlow, this "backward" multiplication happens automatically. It starts at the Loss function and propagates the error signal back through every layer, computing the gradient for every single parameter in the network efficiently.