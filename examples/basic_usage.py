from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.visualisation import draw_nn


# Below is a simple output, based on multiplication and addition
# For this output, we generate all gradients and display them in a graph
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

L.backward()

draw_nn(L).render('graph', view=True)