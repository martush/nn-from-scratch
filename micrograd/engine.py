import math

class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 #default gradient
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        '''Provides a nicer looking expression. 
           Without it, we'd print out only the memory place allocation
        '''
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        '''
        Implementation of addition of the class numbers
        '''

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # When doing addition, gradient is always 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')
        
        def _backward():
            # When doing multiplication, gradient is the other value in the expression
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        '''
        Implementation of raising to the power of a number (other)
        '''
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            # Gradient of self to the power of other
            self.grad += other * self.data **(other-1) * out.grad
            
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other):
        '''
        Implementation in case of 2 * a where a is a Value - python swithches order
        So that self here is the other
        '''
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self, ), f'log{self.data}')

        def _backward():
            self.grad += self.data**-1 * out.grad

        out._backward = _backward
        return out
        
    def relu(self):

        x = self.data if self.data > 0 else 0
        out = Value(x, (self, ), 'relu')

        def _backward():
            gradient_addon = 1 if x > 0 else 0
            self.grad += gradient_addon * out.grad

        out._backward = _backward
        return out
    
    def sigmoid(self):
        return self.exp() / (1 + self.exp())

    def backward(self):
        # Topological ordering - so that they are ordered properly
        # From last output going backward
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward() 
