from math import exp
from value import Value


##################### Test Single Operations #############################

# Test if gradient estimation works correctly for addition (simplest case)
def test_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()

    assert a.grad == 1.0
    assert b.grad == 1.0
    print('Correct gradient estimation for addition')

test_add()


# Test if gradient estimation works correctly for multiplication
def test_mult():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()

    assert a.grad == 3.0 #b
    assert b.grad == 2.0 #a
    print('Correct gradient estimation for multiplication')

test_mult()


##################### Test Chain Operations #############################
def test_chain_rule():
    a = Value(5.0)
    b = Value(10.0)
    c = a * b + a
    c.backward()

    # Assert it returns the same as manually calculated derivative
    assert a.grad == b.data + 1
    assert b.grad == a.data
    print('Correct gradient estimation for chained derivatives')

test_chain_rule()


##################### Generalised checker #############################
# The below compares the gradient from the engine vs a generalised approximation
# Of a function derivative
# The gradients should be very close (however, will not be equal as one is an approximation)

def f(x):
    '''Quadratic function'''
    return 4*x**2 - 5*x + 5

def f2(x):
    '''Sigmoid function'''
    return exp(x) / (1 + exp(x))

def symmetrical_gradient(f, x, ):
    return (f(x+h) - f(x-h)) / (2*h)

h = 0.00000000000001

x = Value(4.0)

output = f(x)
output.backward()
engine_gradient = x.grad

calc_gradient = symmetrical_gradient(f, x.data)

diff = calc_gradient - engine_gradient

print('Quadratic Funtion')
print(f'Engine calculated gradient: {engine_gradient}')
print(f'Numerically approximated gradient: {calc_gradient}')
print(f'Difference: {diff}')

y = Value(5.0)
output2 = y.sigmoid()
output2.backward()
engine_gradient = y.grad

calc_gradient = symmetrical_gradient(f2, y.data)
diff = calc_gradient - engine_gradient

print('Sigmoid Function')
print(f'Engine calculated gradient: {engine_gradient}')
print(f'Numerically approximated gradient: {calc_gradient}')
print(f'Difference: {diff}')


##################### Test Against PyTorch #############################

# Softmax function
data = [0.0, 3.0, -2.0, 1.0]

def softmax(logits):
    counts = [x.exp() for x in logits]
    denom = sum(counts)
    return [c / denom for c in counts]

# Create a list of Values
logits = [Value(x) for x in data]

# this is the negative log likelihood loss function
probs = softmax(logits)
loss = -probs[3].log()
loss.backward()
print(loss.data)
my_grads = [x.grad for x in logits]

#Pytorch
import torch
import torch.nn.functional as F

t_logits = torch.tensor(data, requires_grad=True)
t_probs = F.softmax(t_logits, dim=0)
t_loss = -torch.log(t_probs[3])
t_loss.backward()

torch_grads = t_logits.grad.tolist()
