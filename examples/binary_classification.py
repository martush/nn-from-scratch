import random
import math
import matplotlib.pyplot as plt
import numpy as np

from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.visualisation import draw_nn



def make_moons(n_samples=100, noise=0.1):
    """
    Using the sklearn datasets.
    This generates two interleaving half circles (moons) which creates a non-linear
    classification problem:
     - two curved clusters in 2D
     - each observation has coordinates (x,y) and a label (-1, 1)
     - plus some noise 
    
    Example visualisations here: https://scikit-learn.org/0.15/modules/generated/sklearn.datasets.make_moons.html 

    The moons are created so that they intertwine - making it impossible
    to separate them with a straight line
    """
    data = []
    half = n // 2

    for i in range(half):
        a = math.pi * i / half

        # upper moon
        x1 = math.cos(a) + random.uniform(-noise, noise)
        y1 = math.sin(a) + random.uniform(-noise, noise)
        data.append(([x1, y1], 1))

        # lower moon
        x2 = 1 - math.cos(a) + random.uniform(-noise, noise)
        y2 = 0.5 - math.sin(a) + random.uniform(-noise, noise)
        data.append(([x2, y2], -1))

    return data


def visualise_moons(data, model=None, filename='moons_plot.png'):
    """
    Plot the moons dataset and decision boundary using matplotlib.
    """
    
    # Separate data by class
    class_1 = [x for x, y in data if y == 1]
    class_minus_1 = [x for x, y in data if y == -1]
    
    # Plot data points
    plt.figure(figsize=(10, 6))
    
    if class_1:
        xs, ys = zip(*class_1)
        plt.scatter(xs, ys, c='blue', label='Class +1 (upper moon)', s=50, alpha=0.7)
    
    if class_minus_1:
        xs, ys = zip(*class_minus_1)
        plt.scatter(xs, ys, c='green', label='Class -1 (lower moon)', s=50, alpha=0.7)
    
    # Plot decision boundary after fitting the model
    if model:
        # Create a mesh grid
        x_min = min([x[0] for x, _ in data]) - 0.5
        x_max = max([x[0] for x, _ in data]) + 0.5
        y_min = min([x[1] for x, _ in data]) - 0.5
        y_max = max([x[1] for x, _ in data]) + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        # Get predictions for each point in the mesh
        Z = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x_val = [Value(xx[i, j]), Value(yy[i, j])]
                Z[i, j] = model(x_val).data
        
        # Plot decision boundary and regions
        plt.contourf(xx, yy, Z, levels=[-10, 0, 10], colors=['#ffcccc', '#ccccff'], alpha=0.3)
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Moons Dataset' + (' with Decision Boundary' if model else ''))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved as '{filename}'")
    plt.show()


# 1. Generate dataset
data = make_moons(n_samples=100)

# 2. Visualize the dataset
visualise_moons(data)

# 3. Create NN structure: 2 inputs, 2 hidden layers x 16 neurons each, 1 output
model = MLP(2, [16, 16, 1])
print(f"Number of parameters: {len(model.parameters())}")

# Training
def sign(x):
    '''Needed to calculate accuracy below'''
    return 1 if x > 0 else -1

xs = [p for p, y in data]
ys = [y for p, y in data]


for k in range(100):
    # forward pass
    ypred = [model(xi) for xi in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Count correct predictions
    correct = 0
    correct = sum(1 for ygt, yout in zip(y, ypred)
                    if sign(yout.data) == ygt
                  )

    accuracy = correct / len(data) * 100
    
    # Backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    
    # Update
    learning_rate = 0.01
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 10 == 0:
        print(f"Epoch {k:3d} | Loss: {loss.data:6.4f} | Accuracy: {accuracy:5.1f}%")

# Final evaluation
print("\n=== Final Evaluation ===")
correct = 0

for x, y in data:
    x = [Value(xi) for xi in x]
    y_pred = model(x)
    if (y_pred.data > 0 and y == 1) or (y_pred.data < 0 and y == -1):
        correct += 1

final_accuracy = correct / len(data) * 100
print(f"Final Accuracy: {final_accuracy:.1f}%")

# Test on specific points
print("\n=== Testing on sample points ===")
test_points = [
    ([0.5, 0.5], 1, "upper moon"),
    ([1.5, -0.5], -1, "lower moon"),
    ([0.0, 0.0], 1, "upper moon edge"),
    ([1.0, 0.5], -1, "lower moon edge"),
]

for x, expected, description in test_points:
    x_val = [Value(xi) for xi in x]
    y_pred = model(x_val)
    predicted_class = 1 if y_pred.data > 0 else -1
    status = "✓" if predicted_class == expected else "✗"
    print(f"{status} Point ({x[0]:.1f}, {x[1]:.1f}) [{description}]: "
          f"Predicted {predicted_class:+d}, Expected {expected:+d} "
          f"(raw: {y_pred.data:+.2f})")