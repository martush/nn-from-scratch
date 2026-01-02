import random

from micrograd.engine import Value
from micrograd.nn import MLP


# Create data
def make_data(n_per_class=30):
    data = []

    # class 0
    for _ in range(n_per_class):
        x = random.gauss(-2, 0.5)
        y = random.gauss(0, 0.5)
        data.append(([x, y], 0))

    # class 1
    for _ in range(n_per_class):
        x = random.gauss(2, 0.5)
        y = random.gauss(0, 0.5)
        data.append(([x, y], 1))

    # class 2
    for _ in range(n_per_class):
        x = random.gauss(0, 0.5)
        y = random.gauss(2, 0.5)
        data.append(([x, y], 2))

    random.shuffle(data)
    return data

data = make_data()

# Define softmax function
def softmax(logits):
    exps = [l.exp() for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# Define NLL loss (negative log likelihood)
def nll_loss(probs, target_index):
    return -probs[target_index].log()
# This is the industry standard loss function used for classification
# Also mirrors how CrossEntropyLoss works in PyTorch.


# Define model
model = MLP(nin=2, nouts=[16, 16, 3]) #note 3 output classes

learning_rate = 0.05

for i in range(200):
    total_loss = Value(0.0)
    correct = 0

    for x, y in data:
        
        #forward pass
        # first run the NN architecture
        logits = model(x)
        # calculate probability
        probs = softmax(logits)
        # calculate loss
        loss = nll_loss(probs, y)

        total_loss += loss

        # accuracy
        pred_class = max(range(len(logits)), key=lambda i: logits[i].data)
        if pred_class == y:
            correct += 1

    # backward
    for p in model.parameters():
        p.grad = 0.0
    total_loss.backward()

    # update
    for p in model.parameters():
        p.data += -learning_rate * p.grad

    if i % 20 == 0:
        acc = correct / len(data)
        print(f"step {i:3d} | loss {total_loss.data:.4f} | acc {acc:.2f}")

# step   0 | loss 130.2927 | acc 0.33
# step  20 | loss 35.8288 | acc 1.00
# step  40 | loss 21.5924 | acc 1.00
# step  60 | loss 21.5824 | acc 1.00
# step  80 | loss 21.5778 | acc 1.00
# step 100 | loss 21.5749 | acc 1.00
# step 120 | loss 21.5729 | acc 1.00
# step 140 | loss 21.5714 | acc 1.00
# step 160 | loss 21.5703 | acc 1.00
# step 180 | loss 21.5693 | acc 1.00

