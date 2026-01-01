from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.visualisation import draw_nn

# A very simple binary classifier

xs = [
    [3.0, 3.0, -1.0],
    [4.0, -1.0, 0.5],
    [0.5, 2.0, 1.0],
    [1.0, 1.0, -1.0],
]

# output either 1 or -1 for each of the 4 inputs
ys = [1.0, -1.0, -1.0, 1.0]


## NN setup
# 3 inputs (len of each x), 2 hidden layers of 4 neurons and 1 output
xor_nn = MLP(3, [4, 4, 1])

print(f"Number of params: {len(xor_nn.parameters())}")

print('Initial outputs: ')

ypred = [xor_nn(x) for x in xs]
print(ypred)

# Initial outputs are (this is just for a test run - outputs would be different each time)
[Value(data=-0.3161254682023705), 
 Value(data=-0.23901384112180174), 
 Value(data=-0.5991152466114962), 
 Value(data=-0.1788943179710747)]

# These are quite different from the needed +1/-1 outputs.
# Also need a single value which tells us how wrong we are - loss function

#MSE loss - mean squared error loss
# ygt: y ground truth
# yout: NN predicted y
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

loss.backward()

draw_nn(loss).render('graph', view=True)

print(xor_nn.layers[0].neurons[0].w[0].grad)
print(xor_nn.layers[0].neurons[0].w[0].data)

# Example of slight increment of params
# for p in loss.parameters():
#     p.data += -0.01 * p.grad

# After this loss function should have decreased a little
# ypred = [xor_nn(x) for x in xs]
# loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
# print(loss)

# Gradient Descent:

# 1. Forward pass 
# ypred = [f(x) for x in xs]
# loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

# 2. Backward pass
# loss.backward()

# 3. Nudge params
# for p in n.parameters():
#     p.data += -0.01 * p.grad

# An automated training loop - 60 iterations
for k in range(100):
    #forward pass
    ypred = [xor_nn(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    #backward pass
    # flush gradients
    for p in xor_nn.parameters():
        p.grad = 0.0
    loss.backward()
    
    # update
    for p in xor_nn.parameters():
        # learning rate
        p.data += -0.05 * p.grad

    if k % 10 == 0:  
        print(k, loss.data)

# 0 7.926970480695591
# 1 7.920279515652021
# 2 7.912206439504052
# 3 7.902274458946577
# 4 7.889761638452272
# 5 7.873522997067356
# 6 7.851635093577026
# 7 7.820621712309303
# 8 7.773569715416839
# 9 7.694827079623036
# 10 7.541829179046079
# 11 7.163475584061549
# 12 5.820095825803855
# 13 2.6429887808440813
# 14 1.5587403421829933
# 15 0.801690189734348
# 16 0.5290205997315451
# 17 0.2698509257986472
# 18 0.14349994688133805
# 19 0.11548061809523677
#...
# 99 0.006772140334220809

print(ypred)

# [Value(data=0.9768157861013156), 
#  Value(data=-0.9581580204896953), 
#  Value(data=-0.955308609278378), 
#  Value(data=0.9501345720348514)]