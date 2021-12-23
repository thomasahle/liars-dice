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
from tqdm import tqdm

from snyd import *

class Robot:
    def __init__(self, priv, game):
        self.priv = priv
        self.game = game

    def get_action(self, state):
        last_call = self.game.get_last_call(state)
        return self.game.sample_action(self.priv, state, last_call, eps=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=str, nargs=2, help="Path of models")
    parser.add_argument("--d", type=int, nargs=2, default=(1,1), help="Number of dice for players")
    parser.add_argument("--sides", type=int, default=6, help="Number of sides on the dice")
    parser.add_argument("--variant", type=str, default="joker", help="one of normal, joker, stairs")
    parser.add_argument("--N", type=int, default=1000, help="Number of games to run")
    args = parser.parse_args()

    games = []
    for path in args.paths:
        if path.endswith('.onnx'):
            import onnxruntime as ort
            ort_sess = ort.InferenceSession(path)
            print(dir(ort_sess))
            print([o.name for o in ort_sess.get_inputs()])
            print([o.name for o in ort_sess.get_outputs()])
            def model(priv, pub):
                res = ort_sess.run(['value'], {'priv': priv, 'pub': pub})
                print(res)
                return res.value
        else:
            print('Loading', path)
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            train_args = checkpoint["args"]
            assert train_args.d1 == args.d[0]
            assert train_args.d2 == args.d[1]
            assert train_args.sides == args.sides
            assert train_args.variant == args.variant
            D_PUB, D_PRI, *_ = calc_args(
                train_args.d1, train_args.d2, train_args.sides, train_args.variant
            )
            model = NetCompBilin(D_PRI, D_PUB)
            model.load_state_dict(checkpoint["model_state_dict"])
        game = Game(model, args.d[0], args.d[1], args.sides, args.variant)
        games.append(game)
    game = games[0] # Just use the first model for the common things, like rolling and stuff

    N = args.N
    scores = [0, 0]
    for t in tqdm(range(N)):
        r1 = random.choice(list(game.rolls(0)))
        r2 = random.choice(list(game.rolls(1)))
        priv1 = game.make_priv(r1, 0)
        priv2 = game.make_priv(r2, 1)
        for flip in range(2):
            if not flip:
                players = [Robot(priv1, games[0]),
                           Robot(priv2, games[1])]
            else:
                players = [Robot(priv1, games[1]),
                           Robot(priv2, games[0])]

            state = game.make_state()
            cur = 0
            while True:
                action = players[cur].get_action(state)
                if action == game.LIE_ACTION:
                    last_call = game.get_last_call(state)
                    res = game.evaluate_call(r1, r2, last_call)
                    winner = 1-cur if res else cur
                    if not flip:
                        scores[winner] += 1
                    else:
                        scores[1 - winner] += 1
                    break
                state = game.apply_action(state, action)
                cur = 1 - cur

    # p1 wins with probability 1/(1 + 10^(Rdiff/400))
    # p1 ~ scores[0]/(2N)
    # N/s0 = (1 + 10^(Rd/400))
    # 400 * log10(2N/s0-1) = Rd

    # But also the variance in p1, which is
    # p1(1-p1)
    # So std = sqrt(p1(1-p1)*2/N)

    print('Results:', scores)
    p1 = scores[1] / (2*N)
    std = math.sqrt(p1 * (1-p1) / (2*N))
    mean_elo = 400 * math.log10(1/p1-1)
    elo_upper = 400 * math.log10(1/(p1-std)-1)
    elo_lower = 400 * math.log10(1/(p1+std)-1)
    print(f'{args.paths[0]} is about {mean_elo:.1f} stronger than {args.paths[1]} (between {elo_lower:.1f} and {elo_upper:.1f})')


if __name__ == '__main__':
    main()
