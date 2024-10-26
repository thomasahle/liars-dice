"""
Contains the Game class, as well as various simple netural network architectures.
"""

import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter


class Net(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        hiddens = (100,) * 4

        # Bilinear can't be used inside nn.Sequantial
        # https://github.com/pytorch/pytorch/issues/37092
        self.layer0 = torch.nn.Bilinear(d_pri, d_pub, hiddens[0])

        layers = [torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer0(priv, pub)
        return self.seq(joined)


class NetConcat(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        hiddens = (500, 400, 300, 200, 100)

        layers = [torch.nn.Linear(d_pri + d_pub, hiddens[0]), torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        if len(priv.shape) == 1:
            joined = torch.cat((priv, pub), dim=0)
        else:
            joined = torch.cat((priv, pub), dim=1)
        return self.seq(joined)


class NetCompBilin(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        hiddens = (100,) * 4

        middle = 500
        self.layer_pri = torch.nn.Linear(d_pri, middle)
        self.layer_pub = torch.nn.Linear(d_pub, middle)

        layers = [torch.nn.ReLU(), torch.nn.Linear(middle, hiddens[0]), torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer_pri(priv) * self.layer_pub(pub)
        return self.seq(joined)


class Resid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        y = self.conv(y)


class Net3(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        def conv(channels, size):
            return nn.Sequential(
                nn.Conv1d(1, channels, kernel_size=size),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=size),
                nn.BatchNorm1d(channels),
            )

        # Bilinear can't be used inside nn.Sequantial
        # https://github.com/pytorch/pytorch/issues/37092
        self.layer0 = torch.nn.Bilinear(d_pri, d_pub, hiddens[0])

        layers = [torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer0(priv, pub)
        return self.seq(joined)


class Net2(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        channels = 20
        self.left = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=2),
            nn.ReLU(),
        )

        self.right = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=2),
            nn.ReLU(),
        )

        layer_size = 100
        self.bilin = torch.nn.Bilinear(d_pri, d_pub, layer_size)

        layers = [torch.nn.ReLU()]
        for i in range(3):
            layers += [torch.nn.Linear(layer_size, layer_size), torch.nn.ReLU()]
        layers += [torch.nn.Linear(layer_size, 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        if len(priv.shape) == 1:
            assert len(pub.shape) == 1
            priv = priv.unsqueeze(0)
            pub = pub.unsqueeze(0)
        # x = self.left(priv.unsqueeze(-2)).squeeze(1) + priv
        x = priv
        y = self.right(pub.unsqueeze(-2)).squeeze(1) + pub
        mixed = self.bilin(x, y)
        return self.seq(mixed)


def calc_args(d1, d2, sides, variant):
    # Maximum call is (D1+D2) 6s
    D_PUB = (d1 + d2) * sides
    if variant == "stairs":
        D_PUB = 2 * (d1 + d2) * sides

    # We also have a feature that means "game has been called"
    LIE_ACTION = D_PUB
    D_PUB += 1

    # Total number of actions in the game
    N_ACTIONS = D_PUB

    # Add two extra features to keep track of who's to play next
    CUR_INDEX = D_PUB
    D_PUB += 1

    # Duplicate for other player
    D_PUB_PER_PLAYER = D_PUB
    D_PUB *= 2

    # One player technically may have a smaller private space than the other,
    # but we just take the maximum for simplicity
    D_PRI = max(d1, d2) * sides

    # And then two features to describe from whos perspective we
    # are given the private information
    PRI_INDEX = D_PRI
    D_PRI += 2

    return D_PUB, D_PRI, N_ACTIONS, LIE_ACTION, CUR_INDEX, PRI_INDEX, D_PUB_PER_PLAYER


class Game:
    def __init__(self, model, d1, d2, sides, variant):
        self.model = model
        self.D1 = d1
        self.D2 = d2
        self.SIDES = sides
        self.VARIANT = variant

        (
            self.D_PUB,
            self.D_PRI,
            self.N_ACTIONS,
            self.LIE_ACTION,
            self.CUR_INDEX,
            self.PRI_INDEX,
            self.D_PUB_PER_PLAYER,
        ) = calc_args(d1, d2, sides, variant)

    def make_regrets(self, priv, state, last_call):
        """
        priv: Private state, including the perspective for the scores
        state: Public statte
        last_call: Last action taken by a player. Returned regrets will be for actions after this one.
        """

        if priv[self.PRI_INDEX] != state[self.CUR_INDEX]:
            print("Warning: Regrets are not with respect to current player")

        # Number of child nodes
        n_actions = self.N_ACTIONS - last_call - 1

        # One for the current state, and one for each child
        batch = state.repeat(n_actions + 1, 1)

        for i in range(n_actions):
            self._apply_action(batch[i + 1], i + last_call + 1)

        priv_batch = priv.repeat(n_actions + 1, 1)

        v, *vs = list(self.model(priv_batch, batch))
        return [max(vi - v, 0) for vi in vs]
        # The Hedge method
        # return [math.exp(10*(vi - v)) for vi in vs]

    def evaluate_call(self, r1, r2, last_call):
        # Players have rolled r1, and r2.
        # Previous actions are `state`
        # Player `caller` just called lie. (This is not included in last_call)
        # Returns True if the call is good, false otherwise

        # Calling lie immediately is an error, so we pretend the
        # last call was good to punish the player.
        if last_call == -1:
            return True

        n, d = divmod(last_call, self.SIDES)
        n, d = n + 1, d + 1  # (0, 0) means 1 of 1s

        cnt = Counter(r1 + r2)
        if self.VARIANT == "normal":
            actual = cnt[d]
        if self.VARIANT == "joker":
            actual = cnt[d] + cnt[1] if d != 1 else cnt[d]
        if self.VARIANT == "stairs":
            if all(r == i + 1 for r, i in zip(r1, range(self.SIDES))):
                actual += 2 * len(r1) - r1.count(d)
            if all(r == i + 1 for r, i in zip(r2, range(self.SIDES))):
                actual += 2 * len(r2) - r1.count(d)
        # print(f'{r1=}, {r2=}, {last_call=}, {(n, d)=}, {actual=}', actual >= n)
        return actual >= n

    def policy(self, priv, state, last_call, eps=0):
        regrets = self.make_regrets(priv, state, last_call)
        for i in range(len(regrets)):
            regrets[i] += eps
        if sum(regrets) <= 0:
            return [1 / len(regrets)] * len(regrets)
        else:
            s = sum(regrets)
            return [r / s for r in regrets]

    def sample_action(self, priv, state, last_call, eps):
        pi = self.policy(priv, state, last_call, eps)
        action = next(iter(torch.utils.data.WeightedRandomSampler(pi, num_samples=1)))
        return action + last_call + 1

    def apply_action(self, state, action):
        new_state = state.clone()
        self._apply_action(new_state, action)
        return new_state

    def _apply_action(self, state, action):
        # Inplace
        cur = self.get_cur(state)
        state[action + cur * self.D_PUB_PER_PLAYER] = 1
        state[self.CUR_INDEX + cur * self.D_PUB_PER_PLAYER] = 0
        state[self.CUR_INDEX + (1 - cur) * self.D_PUB_PER_PLAYER] = 1
        return state

    def make_priv(self, roll, player):
        assert player in [0, 1]
        priv = torch.zeros(self.D_PRI)
        priv[self.PRI_INDEX + player] = 1
        # New method inspired by Chinese poker paper
        cnt = Counter(roll)
        for face, c in cnt.items():
            for i in range(c):
                priv[(face - 1) * max(self.D1, self.D2) + i] = 1
        # Old encoding
        # for i, r in enumerate(roll):
        #     priv[i * self.SIDES + r - 1] = 1
        return priv

    def make_state(self):
        state = torch.zeros(self.D_PUB)
        state[self.CUR_INDEX] = 1
        return state

    def get_cur(self, state):
        # Player index in {0,1} is equal to one-hot encoding of player 2
        return 1 - int(state[self.CUR_INDEX])

    def rolls(self, player):
        assert player in [0, 1]
        n_faces = self.D1 if player == 0 else self.D2
        return [
            tuple(sorted(r))
            for r in itertools.product(range(1, self.SIDES + 1), repeat=n_faces)
        ]

    def get_calls(self, state):
        merged = (
            state[: self.CUR_INDEX]
            + state[self.D_PUB_PER_PLAYER : self.D_PUB_PER_PLAYER + self.CUR_INDEX]
        )
        return (merged == 1).nonzero(as_tuple=True)[0].tolist()

    def get_last_call(self, state):
        ids = self.get_calls(state)
        if not ids:
            return -1
        return int(ids[-1])
