# nn-from-scratch
A minimal neural network library built from first principles, implementing automatic differentiation, neural layers, and training loops without relying on deep learning frameworks.

Framework-free: PyTorch/TensorFlow are not used for training.
PyTorch is only used in test_engine.py to verify gradient correctness.

This repository is a learning-focused implementation of a neural network stack:
- scalar-based autograd engine
- multi-layer perceptrons (MLPs)
- gradient descent training
- visualization of both computation graphs and network structure
- experiments on synthetic datasets

## Features

- **Autograd Engine**: Dynamic computation graph (Value)
- **Neural Network Components**: Neurons, Layers, and Multi-Layer Perceptrons (MLP)
- **Training**: Backpropagation with gradient descent
- **Visualisation**: Graphical representation of both the computation graph and neural networks
- **Tested**: Gradient checking against numerical differentiation / PyTorch


## Quick Start
```python
from micrograd.nn import MLP
from micrograd.engine import Value
from micrograd.visualisation import draw_nn

# Create a network: 2 inputs -> 16 hidden -> 1 output
model = MLP(2, [16, 16, 1])

# Forward pass
x = [Value(2.0), Value(3.0)]
output = model(x)

# Backward pass
output.backward()

# Example gradient calculated for the first param
print(model.parameters()[0].grad)

# Graph visualisation of NN
draw_nn(model).render('graph', view=True)
```

## Examples

| File                            | Description                                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `basic_usage.py`                | Minimal example: simple computation `L = (a+b)*c`, gradient backpropagation, and graph visualization |
| `binary_classification.py`      | Binary classification with `make_moons` dataset. Includes decision boundary visualization            |
| `xor_classifier.py`             | Synthetic XOR dataset (+1/-1 outputs), small MLP example                                             |
| `multi_class_classification.py` | Multi-class classification with softmax + NLL loss, synthetic dataset                                |


### Example: Binary Classification (Moons)
```python
# run from root folder (nn-from-scratch)
python examples.binary_classification
```
### Output snippet
Number of parameters: 337

step   0 | loss 1.2299 | acc 0.55
step  20 | loss 0.2351 | acc 0.95
step  40 | loss 0.1356 | acc 0.97
...
step 180 | loss 0.0365 | acc 1.00


## Project Structure
```
nn-from-scratch/
├── micrograd/
│   └── engine.py                 # Autograd engine (Value class)
│   └── nn.py                     # Neural network components (neuron, layer, mlp)
│   └── test_engine.py            # Tests for the engine's calculation of gradient
│   └── visualisation.py          # Graphviz graph which draws the created NN
├── examples/
│   └── basic_usage.py            # Simple example with a computation, gradient backpropagation and a graph to visualise it
│   └── binary_classification.py  # Example using an sklearn dataset for non-linear binary classification
│   └── linear_regression.py      # Basic (and very bad) linear regression example - showing how NN doesn't predict it well
│   └── xor_classifier.py         # Basic classifier using synthetic data and predicting +1 or -1 outputs
└── README.md
```

## How It Works

### 1. Autograd Engine (`engine.py`)

Value wraps a number and tracks operations to compute gradients via backpropagation
The Value class tracks:
- scalar data
- gradient
- parents in the computation graph (aptly called children in the class :) )
- backward function - which computes gradients using a directed graph (from last output backwards)

```python
a = Value(2.0)
b = Value(3.0)
# build computation graph
c = a * b + a ** 2
# compute gradients via backprop
c.backward()          

print(a.grad)  # dc/da = 7.0
print(b.grad)  # dc/db = 2.0
```

Supports: `+, -, *, /, **, tanh(), relu(), sigmoid(), exp(), log()`

### 2. Neural Networks (`nn.py`)

Simple structure for a NN:

- **Neuron**: Single neuron with weights, bias, and optional activation
- **Layer**: Collection of neurons
- **MLP**: Multi-layer perceptron (stack of layers)

Supports any selection of a number of layers and layer width (number of neurons in a layer). 
Hidden layers use non-linear activations (default added is tanh). Output layer is linear (logits).

### 3. Training Loop
```python
# 1. Forward pass
loss = sum((model(x) - y)**2 for x, y in data)

# 2. Backward pass
for p in model.parameters():
    p.grad = 0.0
loss.backward()

# 3. Update weights
for p in model.parameters():
    p.data -= learning_rate * p.grad
```

### 4. Visualisation
```python
# Neural Network Structure
draw_nn(model).render('graph', view=True)

# Computational Graph
draw_nn(loss).render('graph', view=True)
```

## Limitations

This is a learning project, not production code:
- Scalar based (slow)
- No GPU support
- No advanced optimizers (Adam, RMSprop)
- No convolutions or recurrent layers
- Minimal error handling
- Not optimized for speed

For real work, please use a real library - PyTorch, TensorFlow, or JAX!

## Inspiration

- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
- CS231n Stanford course

## Version

**v0.1.0** - Minimal working neural network with autograd

## License

GPL-3.0