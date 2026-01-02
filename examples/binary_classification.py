import math
import random
import numpy as np
import matplotlib.pyplot as plt

from micrograd.engine import Value
from micrograd.nn import MLP

# Data - from sklearn
from sklearn.datasets import make_moons, make_blobs

X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1

# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

# Generate dataset
data = make_moons(200)

# initialize a model 
model = MLP(2, [16, 16, 1]) # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))

X_list = X.tolist()
Y_list = y.tolist()

xs = [p[0] for p in X_list]
ys = [p[1] for p in X_list]
colors = ['red' if y == 1 else 'blue' for y in Y_list]

# Create NN architecture
model = MLP(2, [16, 16, 1])
print(f"Number of parameters: {len(model.parameters())}")
# 337

# Training
learning_rate = 0.05
steps = 100

for step in range(steps):
    # forward
    ypred = [model(xi) for xi in X_list]
    # Use mean loss (rather than MSE used previously)
    # Squared errors plus tanh (which was also used in initial run) produce flat gradients
    loss = sum((yout - ygt)**2 for ygt, yout in zip(Y_list, ypred)) / len(Y_list)


    # backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in model.parameters():
        p.data += -learning_rate * p.grad

    # accuracy
    correct = sum(
        1 for ygt, yout in zip(Y_list, ypred)
        if (yout.data > 0) == (ygt == 1)
    )
    accuracy = correct / len(Y_list)

    # Print stats every 20 steps
    if step % 20 == 0:
        print(f"step {step:3d} | loss {loss.data:.4f} | acc {accuracy:.2f}")

# ! Note initial run of this was wrong - it produces loss of 400 each step and accuracy of 0.5 consistently
# Upon investigating, main issue was the tanh activation which squishes the result (and to a smaller extent the MSE activation)
# This was fixed by adding an option to the neuron to not add an activation function and changing loss f to mean loss


# step   0 | loss 1.2299 | acc 0.55
# step  20 | loss 0.2351 | acc 0.95
# step  40 | loss 0.1356 | acc 0.97
# step  60 | loss 0.0973 | acc 0.98
# step  80 | loss 0.0771 | acc 0.99
# step 100 | loss 0.0639 | acc 0.99
# step 120 | loss 0.0543 | acc 0.99
# step 140 | loss 0.0470 | acc 0.99
# step 160 | loss 0.0412 | acc 1.00
# step 180 | loss 0.0365 | acc 1.00


# Visualisations

# initial data
plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20)
plt.show()

# predictions
pred_colors = [
    'red' if model(xi).data > 0 else 'blue'
    for xi in X_list
]

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=pred_colors, s=30)
plt.title("Model Predictions")
plt.show()


# # decision boundary
# h = 0.02
# x_min, x_max = min(xs) - 1, max(xs) + 1
# y_min, y_max = min(ys) - 1, max(ys) + 1

# xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, h),
#     np.arange(y_min, y_max, h)
# )

# Z = []
# for x, y in zip(xx.ravel(), yy.ravel()):
#     Z.append(model([x, y]).data)

# Z = np.array(Z).reshape(xx.shape)

# plt.figure(figsize=(6, 5))
# plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap='coolwarm')
# plt.scatter(xs, ys, c=colors, s=30)
# plt.title("Decision Boundary")
# plt.show()

# visualize decision boundary

h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
