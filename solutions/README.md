# Automatic Differentiation & Backpropagation From Scratch (NumPy Only)

This project builds a **minimal deep learning engine from first principles**, featuring:

- A `Tensor` class with **gradient tracking**
- **Reverse-mode automatic differentiation** (backpropagation)
- **Broadcasting-aware backward passes**
- **Matrix multiply, elementwise ops, reductions, reshaping, max, etc.**
- A fully working **MLP classifier trained on MNIST**
- A simple **SGD optimizer**

No PyTorch. No TensorFlow. No JAX.  
Just **NumPy** + the mathematics of backpropagation.

---

## Why This Project?

Most deep learning libraries abstract away how gradients are computed.  
This project reveals the **exact logic** behind:

- Graph construction
- Local derivative propagation
- Gradient accumulation
- Parameter updates

It is essentially a **tiny PyTorch**, built by hand.

---

## Core Architecture Overview

```
         Forward Pass                   Backward Pass
---------------------------------------------------------------
Tensor ----> Tensor ----> ... ----> Loss
                  \                   ↑
                   \                 /
                    ---- Graph ------
```

Each `Tensor` stores:
| Attribute | Meaning |
|----------|---------|
| `.data` | The underlying NumPy array |
| `.grad` | Gradient accumulated during backprop |
| `.recipe` | Information describing how this tensor was created |

Gradients are computed via **reverse traversal** of the graph.

---

## Math in One Page (Local Derivative Rules)

| Forward Operation | Gradient Rule |
|------------------|---------------|
| **y = log(x)** | ∂L/∂x = (∂L/∂y) · (1/x) |
| **z = x * y** | ∂L/∂x = g·y ; ∂L/∂y = g·x |
| **y = -x** | ∂L/∂x = -g |
| **y = exp(x)** | ∂L/∂x = g·exp(x) |
| **s = sum(x, dim)** | Re-insert reduced dim → broadcast gradient back to `x.shape` |
| **y = reshape(x)** | Reshape gradient back to original shape |
| **y = permute(x, axes)** | Apply inverse permutation to gradient |
| **m = maximum(x, y)** | Gradient flows to the max argument (tie rule: x≥y → x) |
| **Z = X @ Y** | ∂L/∂X = G @ Yᵀ ; ∂L/∂Y = Xᵀ @ G (where G = ∂L/∂Z) |

> This table is the **entire backprop engine** in compact form.

---

## Backpropagation: Algorithm Summary

```
def backprop(loss):
    loss.grad = 1
    for node in reverse_topological_order(loss):
        for each parent:
            parent.grad += parent_grad_contribution_from(node)
```

Key principles:

- **Reverse compute** from output → input
- **Accumulate gradients** when tensors branch
- **Un-broadcast** all gradients to match original shapes
- **Never overwrite, always add**

---

## Model Architecture (MLP)

```
Input (28×28) → Flatten → Linear(784 → 64) → ReLU
                                ↓
                          Linear(64 → 64) → ReLU
                                ↓
                           Linear(64 → 10) → Logits
```

---

## Stable Cross-Entropy 

```
shift = logits - max(logits)
log_softmax = shift - log(sum(exp(shift)))
loss = -log_softmax[range(batch), labels]
```

This prevents numerical overflow.

---

## Training Loop

```
for each epoch:
    zero_grad()
    logits = model(x)
    loss = cross_entropy(logits, y).mean()
    loss.backward()
    optimizer.step()
```

Optimizer:
```
param.data -= lr * param.grad
```

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | MNIST |
| Model | 3-layer MLP |
| Final Accuracy | **≈ 95%** |
| Training Stability | No exploding or vanishing grads |

---

## Key Insights Learned

| Insight | Explanation |
|--------|-------------|
| Backprop is local | Each op needs only **its** derivative rule. |
| Graph structure matters | Reverse order ensures correct gradient flow. |
| Reductions need care | Must re-expand gradients for `sum()`. |
| Broadcasting gradients must be undone | Ensure shapes align. |
| Numerical stability is essential | LogSumExp avoids NaNs. |

---
