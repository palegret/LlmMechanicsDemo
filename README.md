# LLM Core Mechanics Demonstration

This repository contains a C# (.NET 10) architectural breakdown of the fundamental components governing Large Language Models (LLMs). The code is strictly procedural, intentionally avoiding LINQ and abstraction sugar, to maximize pedagogical clarity regarding the mathematical operations that underpin neural networks.

## Core Concepts

### 1. The Neuron (The Vibe Checker)

At its core, a neural network is a collection of artificial neurons. A single neuron acts as a simple decision-maker. It takes in numerical inputs, applies basic arithmetic, and produces a single output.

It makes decisions using three mechanisms:

- **Weights (Multipliers):** How much importance the neuron places on a specific input.
- **Bias (Starting Handicap):** An internal baseline adjustment independent of the inputs.
- **Activation Function:** A final rule applied to the result (e.g., ReLU, which converts any negative final score to zero).

### 2. Parameters vs. Hyperparameters

- **Trainable Parameters:** The internal weights and biases. These start as random noise and are automatically adjusted by the computer during training. (Analogy: The volume and equalization dials on a stereo).
- **Hyperparameters:** The architectural blueprint defined by engineers before the program runs, such as layer count, neuron count, and learning rate. (Analogy: The physical design and speaker configuration of the stereo).

### 3. Layer Stacking (The Chain Reaction)

A single neuron makes a linear decision. A **Neuron Layer** is a collection of neurons evaluating the exact same inputs simultaneously, but applying different weights to extract different meanings.

A **Deep Neural Network** chains these layers together sequentially. The output array (vector) of Layer 1 becomes the direct input array for Layer 2. This process allows the network to build highly abstract concepts from simple raw data.

### 4. Backpropagation (How the Model Learns)

Knowledge in an LLM is not stored in a database or text file; it exists entirely in the physical arrangement of weight and bias values. The network discovers these values through a process of trial and error:

1. **Forward Pass:** The network makes a guess using its current parameters.
2. **Loss Calculation:** The exact mathematical error is measured against the known correct answer.
3. **Backward Pass:** The error is multiplied by a small fraction (Learning Rate), and the internal weights are incrementally adjusted to reduce the error on the next attempt.

### 5. Persistence (Training vs. Inference)

Training requires immense computational loops to discover the correct parameters. Once discovered, the state is persisted by writing the raw, contiguous decimal values directly to a binary file. End-users bypass the expensive training loop by loading this binary memory dump back into RAM, enabling instant **Inference**. Because parameters are continuous memory blocks, relational databases (like SQLite) are architecturally inappropriate for weight storage.
