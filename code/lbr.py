# -*- coding: utf-8 -*-
import torch
import argparse

from snyd import Net, Game, calc_args
from collections import Counter

# in normal form games, given strategy matrix A for player 1 (hand -> action_probs)
# we will play a deterministic strategy b (hand -> action)

def lbr(game, roll, us, prune=0, prune_type='zero'):
    roll_cnt = Counter(game.rolls(1 - us))
    op_rolls = roll_cnt.keys()
    reach_probs = torch.tensor(list(roll_cnt.values()), dtype=torch.float32)
    reach_probs /= reach_probs.sum()
    op_privs = [game.make_priv(op_roll, 1 - us) for op_roll in op_rolls]

    def inner(state, reach_probs, likelihood):
        # basically simple max-chance algorithm.
        # possibly using our learned values for reducing the tree height
        # But I could start by making a version that just goes to the bottom

        calls = game.get_calls(state)
        cur = len(calls) % 2

        if likelihood < prune:
            #print('Pruning', likelihood)
            if prune_type == 'zero':
                return 0
            # I can return +1 or -1 here to get either an upper or lower
            # bound on exploitability
            if prune_type == 'upper':
                return 1
            if prune_type == 'lower':
                return -1
            if prune_type in ['us', 'avg']:
                # Even using our guess here, we will probably get a lower bound,
                # since we are replacing a "perfect" (best response) strategy
                # with whatever strategy the model suggests.
                # However it's not guaranteed to be (a lower bound), since the
                # value network could be overly optimistic.
                our_guess = game.model(game.make_priv(roll, us), state).item()
            if prune_type in ['them', 'avg']:
                op_guess = sum(prob * game.model(op_priv, state).item()
                               for prob, op_priv in zip(reach_probs, op_privs))
                # Remember opponent guess is relative to them, not us
                op_guess = -op_guess
            if prune_type == 'us':
                return our_guess
            if prune_type == 'them':
                return op_guess
            if prune_type == 'avg':
                return (our_guess + op_guess)/2
            assert False

        if calls and calls[-1] == game.LIE_ACTION:
            prev_call = calls[-2] if len(calls) >= 2 else -1
            res = 0
            # Take the average outcome over our opponents possible rolls
            for prob, op_roll in zip(reach_probs, op_rolls):
                r1, r2 = (roll, op_roll) if us == 0 else (op_roll, roll)
                correct = game.evaluate_call(r1, r2, prev_call)
                # If prev_call is good, and we are now, it mean we won
                # (because we made the all and our opponent called lie)
                if us == cur:
                    val = 1 if correct else -1
                else:
                    val = -1 if correct else 1
                res += prob * val
            return res

        last_call = calls[-1] if len(calls) >= 1 else -1

        # Max node
        if us == cur:
            best = -1
            for action in range(last_call + 1, game.N_ACTIONS):
                new_state = game.apply_action(state, action)
                val = inner(new_state, reach_probs, likelihood)
                best = max(best, val)
            return best

        # Chance node
        else:
            # Note: game.policy evaluates the model on this state as well
            # as all children, which means that in essense we compute the
            # model on the entire game tree, and not just the opponents nodes.

            policies = torch.vstack([
                torch.tensor(game.policy(op_priv, state, last_call))
                for op_priv in op_privs])

            score = 0
            for action in range(last_call + 1, game.N_ACTIONS):
                ai = action - last_call - 1
                pa = reach_probs @ policies[:, ai]
                if torch.isclose(pa, torch.tensor(0.0)):
                    continue
                # Bayes: P(R|A) = P(R)P(A|R)/P(A)
                # P(A) = P(R)P(A|R)/sum(P(A|r)P(r) for r in Rs)
                new_reach_probs = (reach_probs * policies[:, ai]) / pa
                new_state = game.apply_action(state, action)
                val = inner(new_state, new_reach_probs, likelihood*pa)
                score += pa * val
            return score

    return inner(game.make_state(), reach_probs, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path of model to test")
    parser.add_argument("--prune", type=float, default=0.0, help="Prune nodes with likelihood < prune")
    parser.add_argument("--type", type=str, default='zero', help="What should be used in place of the real value in pruned nodes?")
            #options=['upper', 'lower', 'zero', 'us', 'them', 'avg'])
    args = parser.parse_args()

    checkpoint = torch.load(args.path)
    train_args = checkpoint["args"]

    D_PUB, D_PRI, *_ = calc_args(
        train_args.d1, train_args.d2, train_args.sides, train_args.variant
    )
    model = Net(D_PRI, D_PUB)
    model.load_state_dict(checkpoint["model_state_dict"])
    game = Game(model, train_args.d1, train_args.d2, train_args.sides, train_args.variant)

    for player in range(2):
        print(f'Testing exploitability of player {1-player}')

        total_val = 0
        total_cnt = 0
        for roll, cnt in Counter(game.rolls(player)).items():
            print('Exploiting with roll', roll)
            total_val += cnt * lbr(game, roll, player, args.prune, args.type)
            total_cnt += cnt

        print(total_val / total_cnt)

if __name__ == '__main__':
    main()
