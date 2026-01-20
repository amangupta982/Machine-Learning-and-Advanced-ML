# TensorFlow Fundamentals & Neural Networks

This folder contains **TensorFlow-based implementations** of core neural network concepts,  
focusing on **understanding internals rather than only using high-level APIs**.

The goal is to build a strong foundation in how neural networks operate under the hood
using TensorFlow primitives.

---

## 📌 Topics Covered

### 1. Dense Layers (Fully Connected Networks)
- Dense layer implemented **from scratch** using:
  - `tf.Variable`
  - `tf.matmul`
  - Bias addition
  - ReLU activation
- Dense layer implemented using **`tf.keras.layers.Dense`**
- Comparison between low-level and high-level TensorFlow APIs

📁 Folder:
---

## 🧠 Why This Folder Exists

Most tutorials directly jump to `model.fit()` and hide important details.

This folder focuses on:
- Understanding weight & bias initialization
- Forward propagation mechanics
- What Keras abstracts internally
- When to use low-level vs high-level APIs

---

## 📂 Folder Structure
├── Neural_Networks/
│   ├── dense_layer_from_scratch_tf.py
│   ├── dense_layer_usingKeras.py
│   ├── compare_dense_layer_scratch_vs_keras.py
│   └── README.md

---

## 🎯 Learning Outcome

After completing this section, you will:
- Understand how a dense layer computes outputs
- Be comfortable switching between raw TensorFlow and Keras
- Build neural networks step-by-step with clarity

---

## 🔜 Next Planned Topics

- Activation function comparisons
- Loss functions and optimizers
- Training loops and backpropagation
- Unit testing neural network components

---

## 📬 Maintainer

**Aman Gupta**  
Machine Learning Learner & Engineer-in-Progress