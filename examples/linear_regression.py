from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.visualisation import draw_nn


# Data
xs = [
    ([0.0]),
    ([1.0]),
    ([2.0]), 
    ([3.0]),
    ([4.0]),
]

ys = [1.0, 3.0, 5.0, 7.0, 9.0]

# Function which generated ys is y = 2x + 1

# Simple setup - 1 input, 1 hidden, 1 output
linear_nn = MLP(1, [8, 1])

print(f"Number of parameters: {len(linear_nn.parameters())}")
# 2

# Training
for i in range(20):
    total_loss = Value(0)
    
    ypred = [linear_nn(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    #backward pass
    # flush gradients
    for p in linear_nn.parameters():
        p.grad = 0.0
    loss.backward()
    
    # update
    for p in linear_nn.parameters():
        # learning rate
        p.data += -0.01 * p.grad
    

    print(i, loss.data)

print(ys)
# [1.0, 3.0, 5.0, 7.0, 9.0]
print(ypred)
# [Value(data=0.5863891707782235), Value(data=0.9685442810771404), Value(data=0.9980432669602419), Value(data=0.9998799663859647), Value(data=0.9999926430193615)]


# Predictions suck - very far off from actual values and the loss function hits a limit when it reaches around 120 (even with 100 iterations, it moves just a little)
# This is in part because of the activation function - tanh squashes predictions between -1 and +1 

# # Test on new data
# print("\n=== Predicting new values ===")
# for x_new in [5.0, 6.0, 7.0]:
#     x_val = [Value(x_new)]
#     y_pred = model(x_val)
#     y_expected = 2 * x_new + 1
#     print(f"x={x_new:.1f} -> Predicted: {y_pred.data:.2f}, Expected: {y_expected:.1f}")
