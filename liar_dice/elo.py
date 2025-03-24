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
import numpy as np
from scipy import optimize

scores = []
names = []
for line in sys.stdin:
    name, *vs = line.split()
    names.append(name)
    scores.append(list(map(float, vs)))
A = torch.tensor(scores, dtype=torch.float64)
N_GAMES = A.sum()  # total number of games played
N = len(A)
for i in range(N):
    A[i, i] = 1  # This just adds 0.5s to the diagonal of P.
P = A / (A + A.T)


def train_lbgfs(b):
    elos = torch.zeros(N, requires_grad=True, dtype=torch.float64)
    gen = torch.zeros(1, requires_grad=True, dtype=torch.float64)
    criterion = torch.nn.BCELoss(reduction="sum")
    optimizer = torch.optim.LBFGS([elos, gen])

    def closure():
        optimizer.zero_grad()
        diffs = elos[..., None] - elos[..., None, :]
        y_pred = torch.sigmoid(diffs)
        y_pred = y_pred * b + (1 - b) / 2.0
        loss = criterion(y_pred, P)
        loss.backward()
        return loss

    old_loss = None
    while True:
        new_loss = optimizer.step(closure)
        if old_loss is not None and torch.isclose(old_loss, new_loss, rtol=1e-8):
            break
        old_loss = new_loss
    return elos, old_loss.item()


entropy = -(P * torch.log(P)).sum().item()
for b in np.linspace(0, 1, 10):
    _, loss = train_lbgfs(b)
    print(b, loss - 2 * entropy, file=sys.stderr)

minimum = optimize.golden(lambda b: train_lbgfs(b)[1], brack=(0, 1), full_output=True)
b, loss, *_ = minimum
print("Best", b, loss - 2 * entropy, file=sys.stderr)

print(f"{entropy=}", file=sys.stderr)
minimum = [1]
print(minimum, file=sys.stderr)
b, *_ = minimum
elos, loss = train_lbgfs(b)
print(b, loss - 2 * entropy, file=sys.stderr)

res = (elos - torch.min(elos)) * 400 * math.log(10.0)
for name, e in zip(names, res):
    print(name, e.item())


################################################################################
# We can compute the variance of the MLE using the Hessian.
# But if we are computing the full Hessian anyway, we might as well just use
# the full Newton's method.
################################################################################


def train_newt():
    from torch.autograd.functional import hessian

    def likelihood(elos):
        p = torch.sigmoid(elos[..., None] - elos[..., None, :])
        return -(P * torch.log(p) + (1 - P) * torch.log(1 - p)).sum()

    x = torch.zeros(N, requires_grad=True)
    old_loss = None
    for _ in range(20):
        loss = likelihood(x)
        if old_loss is not None and torch.isclose(old_loss, loss, rtol=1e-8):
            break
        old_loss = loss
        loss.backward()
        H = hessian(likelihood, x)
        M = torch.inverse(H)
        with torch.no_grad():
            x -= M @ x.grad
            # x -= 1e-1 * x.grad
        x.grad.zero_()
    var = (-M).diagonal()
    return x, var


# elos, var = train_newt()
# scale = 400*math.log(10.)
# print(N_GAMES)
# std = (var / N_GAMES).sqrt() * scale
# res = (elos - torch.min(elos)) * scale
# for name, e, dev in zip(names, res, std):
#     print(name, e.item(), '±', dev.item())
