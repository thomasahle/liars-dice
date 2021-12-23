# -*- coding: utf-8 -*-
import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter
import argparse
import re

from snyd import *

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path of model")
args = parser.parse_args()

checkpoint = torch.load(args.path, map_location=torch.device('cpu'))
train_args = checkpoint["args"]

D_PUB, D_PRI, *_ = calc_args(
    train_args.d1, train_args.d2, train_args.sides, train_args.variant
)
model = NetCompBilin(D_PRI, D_PUB)
model.load_state_dict(checkpoint["model_state_dict"])
game = Game(model, train_args.d1, train_args.d2, train_args.sides, train_args.variant)


class Human:
    def get_action(self, state):
        last_call = game.get_last_call(state)
        while True:
            call = input('Your call [e.g. 24 for 2 fours, or "lie" to call a bluff]: ')
            if call == "lie":
                return game.LIE_ACTION
            elif m := re.match(r"(\d)(\d)", call):
                n, d = map(int, m.groups())
                action = (n - 1) * game.SIDES + (d - 1)
                if action <= last_call:
                    print(f"Can't make that call after {repr_action(last_call)}")
                elif action >= game.LIE_ACTION:
                    print(f"The largest call you can make is {repr_action(game.LIE_ACTION-1)}")
                else:
                    return action

    def __repr__(self):
        return "human"


class Robot:
    def __init__(self, priv):
        self.priv = priv

    def get_action(self, state):
        last_call = game.get_last_call(state)
        return game.sample_action(self.priv, state, last_call, eps=0)

    def __repr__(self):
        return "robot"


def repr_action(action):
    action = int(action)
    if action == -1:
        return "nothing"
    if action == game.LIE_ACTION:
        return "lie"
    n, d = divmod(action, game.SIDES)
    n, d = n + 1, d + 1
    return f"{n} {d}s"


while True:
    while (ans := input("Do you want to go first? [y/n/r] ")) not in ["y", "n", "r"]:
        pass

    r1 = random.choice(list(game.rolls(0)))
    r2 = random.choice(list(game.rolls(1)))
    privs = [game.make_priv(r1, 0), game.make_priv(r2, 1)]
    state = game.make_state()

    if ans == "y":
        print(f"> You rolled {r1}!")
        players = [Human(), Robot(privs[1])]
    elif ans == "n":
        print(f"> You rolled {r2}!")
        players = [Robot(privs[0]), Human()]
    elif ans == "r":
        players = [Robot(privs[0]), Robot(privs[1])]

    cur = 0
    while True:
        action = players[cur].get_action(state)
        print()
        print(f"> The {players[cur]} called {repr_action(action)}!")

        if action == game.LIE_ACTION:
            last_call = game.get_last_call(state)
            res = game.evaluate_call(r1, r2, last_call)
            print()
            print(f"> The rolls were {r1} and {r2}.")
            if res:
                print(f"> The call {repr_action(last_call)} was good!")
                print(f"> The {players[cur]} loses!")
            else:
                print(f"> The call {repr_action(last_call)} was a bluff!")
                print(f"> The {players[cur]} wins!")
            print()
            break

        state = game.apply_action(state, action)
        cur = 1 - cur
