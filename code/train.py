# -*- coding: utf-8 -*-
import random
import torch
from torch import nn
import itertools
import math
from collections import Counter
import argparse
import os

from snyd import *

parser = argparse.ArgumentParser()
parser.add_argument("d1", type=int, help="Number of dice for player 1")
parser.add_argument("d2", type=int, help="Number of dice for player 2")
parser.add_argument("--sides", type=int, default=6, help="Number of sides on the dice")
parser.add_argument("--variant", type=str, default="normal", help="one of normal, joker, stairs")
parser.add_argument("--eps", type=float, default=1e-2, help="Added to regrets for exploration")
parser.add_argument("--layers", type=int, default=4, help="Number of fully connected layers")
parser.add_argument("--layer-size", type=int, default=100, help="Number of neurons per layer")
parser.add_argument("--lr", type=float, default=1e-3, help="LR = lr/t")
parser.add_argument("--w", type=float, default=1e-2, help="weight decay")
parser.add_argument("--path", type=str, default="model.pt", help="Where to save checkpoints")

args = parser.parse_args()


# Check if there is a model we should continue training
if os.path.isfile(args.path):
    checkpoint = torch.load(args.path)
    print(f"Using args from {args.path}")
    old_path = args.path
    args = checkpoint["args"]
    args.path = old_path
else:
    checkpoint = None

# Model : (private state, public state) -> value
D_PUB, D_PRI, *_ = calc_args(args.d1, args.d2, args.sides, args.variant)
model = NetConcat(D_PRI, D_PUB)
#model = Net(D_PRI, D_PUB)
#model = Net2(D_PRI, D_PUB)
game = Game(model, args.d1, args.d2, args.sides, args.variant)

if checkpoint is not None:
    print("Loading previous model for continued training")
    model.load_state_dict(checkpoint["model_state_dict"])


device = torch.device("cuda")
model.to(device)


@torch.no_grad()
def play(r1, r2, replay_buffer):
    privs = [game.make_priv(r1, 0).to(device),
            game.make_priv(r2, 1).to(device)]

    def play_inner(state):
        cur = game.get_cur(state)
        calls = game.get_calls(state)
        assert cur == len(calls) % 2

        if calls and calls[-1] == game.LIE_ACTION:
            prev_call = calls[-2] if len(calls) >= 2 else -1
            # If prev_call is good it mean we won (because our opponent called lie)
            res = 1 if game.evaluate_call(r1, r2, prev_call) else -1

        else:
            last_call = calls[-1] if calls else -1
            action = game.sample_action(privs[cur], state, last_call, args.eps)
            new_state = game.apply_action(state, action)
            # Just classic min/max stuff
            res = -play_inner(new_state)

        # Save the result from the perspective of both sides
        replay_buffer.append((privs[cur], state, res))
        replay_buffer.append((privs[1 - cur], state, -res))

        return res

    with torch.no_grad():
        state = game.make_state().to(device)
        play_inner(state)


def print_strategy(state):
    total_v = 0
    total_cnt = 0
    for r1, cnt in sorted(Counter(game.rolls(0)).items()):
        priv = game.make_priv(r1, 0).to(device)
        v = model(priv, state)
        rs = torch.tensor(game.make_regrets(priv, state, last_call=-1))
        if rs.sum() != 0:
            rs /= rs.sum()
        strat = []
        for action, prob in enumerate(rs):
            n, d = divmod(action, game.SIDES)
            n, d = n + 1, d + 1
            if d == 1:
                strat.append(f"{n}:")
            strat.append(f"{prob:.2f}")
        print(r1, f"{float(v):.4f}".rjust(7), f"({cnt})", " ".join(strat))
        total_v += v
        total_cnt += cnt
    print(f"Mean value: {total_v / total_cnt}")


class ReciLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=1, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ReciLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr / (self.last_epoch + 1)**self.gamma
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        return [base_lr / (self.last_epoch + 1)**self.gamma for base_lr in self.base_lrs]





def train():
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.w)
    scheduler = ReciLR(optimizer, gamma=.5)
    value_loss = torch.nn.MSELoss()
    all_rolls = list(itertools.product(game.rolls(0), game.rolls(1)))
    for t in range(100_000):
        replay_buffer = []

        BS = 100  # Number of rolls to include
        for r1, r2 in (all_rolls if len(all_rolls) <= BS else random.sample(all_rolls, BS)):
            play(r1, r2, replay_buffer)

        random.shuffle(replay_buffer)
        privs, states, y = zip(*replay_buffer)

        privs = torch.vstack(privs).to(device)
        states = torch.vstack(states).to(device)
        y = torch.tensor(y, dtype=torch.float).reshape(-1, 1).to(device)
        #privs.to(device)
        #states.to(device)
        #privs.to(device)

        y_pred = model(privs, states)

        # Compute and print loss
        loss = value_loss(y_pred, y)
        print(t, loss.item())

        if t % 5 == 0:
            with torch.no_grad():
                print_strategy(game.make_state().to(device))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (t + 1) % 10 == 0:
            print(f"Saving to {args.path}")
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                },
                args.path,
            )
        if (t+1)%1000 == 0:
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                },
                f'{args.path}.cp{t+1}',
            )



train()
