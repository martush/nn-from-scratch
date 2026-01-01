# nn-from-scratch
A neural network library built from first principles, including automatic differentiation, neural layers, optimizers, and experiments on synthetic datasets.

The code does not use PyTorch/TensorFlow (PyTorch is only used to verify gradient calculations in test_engine.py)

# Neural Network from Scratch

A minimal neural network library built from first principles, implementing automatic differentiation and backpropagation without any ML frameworks.

## Features

- ✅ **Autograd Engine**: Automatic differentiation with dynamic computation graphs
- ✅ **Neural Network Components**: Neurons, Layers, and MLP
- ✅ **Training**: Backpropagation with gradient descent
- ✅ **Tested**: Gradient checking against numerical differentiation

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

# All gradients computed automatically!
print(model.parameters()[0].grad)

# Graph visualisation of NN
draw_nn(model).render('graph', view=True)

```

## Example: Binary Classification

Train a neural network to separate two classes (moons dataset):
```bash
python examples.binary_classification
```

Output:
```
Number of parameters: 337
Epoch 0, Loss: 100.234
Epoch 10, Loss: 45.123
...
Epoch 90, Loss: 2.456
```

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

The `Value` class wraps numbers and tracks operations:
```python
a = Value(2.0)
b = Value(3.0)
c = a * b + a ** 2    # Builds computation graph
c.backward()          # Computes gradients via backprop

print(a.grad)  # dc/da = 7.0
print(b.grad)  # dc/db = 2.0
```

Supports: `+, -, *, /, **, tanh(), relu(), exp(), log()`

### 2. Neural Networks (`nn.py`)

Simple object-oriented API:

- **Neuron**: Single neuron with weights and bias
- **Layer**: Collection of neurons
- **MLP**: Multi-layer perceptron (stack of layers)

### 3. Training Loop
```python
# 1. Forward pass
loss = sum((model(x) - y)**2 for x, y in data)

# 2. Backward pass
model.zero_grad()
loss.backward()

# 3. Update weights
for p in model.parameters():
    p.data -= learning_rate * p.grad
```

### 4. Visualisation
```python
draw_nn(model).render('graph', view=True)
```

## Limitations

This is a learning project, not production code:
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

MIT