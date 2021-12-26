"""
Reads data in the format

0 a1 a2 a3 a4
b1 0 b2 b3 b4
c1 c2 0 c3 c4
...

where A_ij is the number of times player i won against player j
(Note, this program doesn't allow for draws.)
"""

import torch
import sys
import math

A = torch.tensor([list(map(float, line.split())) for line in sys.stdin])
N = len(A)
for i in range(N):
    A[i,i] = 1
P = A / (A + A.T)


elos = torch.zeros(N, requires_grad=True)
criterion = torch.nn.BCELoss(reduction='mean')

def train_lbgfs():
    optimizer = torch.optim.LBFGS([elos])
    def closure():
        optimizer.zero_grad()
        y_pred = torch.sigmoid(elos[..., None] - elos[..., None, :])
        loss = criterion(y_pred, P)
        loss.backward()
        return loss
    old_loss = None
    while True:
        new_loss = optimizer.step(closure)
        if old_loss is not None and torch.isclose(old_loss, new_loss):
            break
        old_loss = new_loss
    return old_loss.item()
train_lbgfs()


res = (elos - torch.min(elos)) * 400*math.log(10.)
for e in res:
    print(e.item())

