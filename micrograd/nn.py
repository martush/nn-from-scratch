import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin, nonlin=True):
        # weight for each input data
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # Neuron bias
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
    
    def __call__(self, x):
        # w * x + b
        result = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Use tanh as an activation function
        #out = result.tanh()
        return result.tanh() if self.nonlin else result

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        #         params = []
        #         for neuron in self.neurons:
        #             ps = neuron.parameters()
        #             params.extend(ps)
        #         return params
        #shorter equivalent
        return [p for neuron in self.neurons for p in neuron.parameters()]


# class MLP:
#     '''Multi-Layer Perceptron - multiple layers of neurons'''

#     def __init__(self, nin, nouts):
#         sz = [nin] + nouts
#         #self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
#         self.layers = []

#         for i in range(len(nouts)):
#             # last layer: no activation
#             nonlin = i != len(nouts) - 1
#             self.layers.append(Layer(sz[i], sz[i+1], nonlin))


#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     def parameters(self):
        
#         return [p for layer in self.layers for p in layer.parameters()]
    

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            # last layer: no activation
            nonlin = i != len(nouts) - 1
            self.layers.append(Layer(sz[i], sz[i+1], nonlin))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]