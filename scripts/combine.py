"""
Script for combining two ELO files
"""

import itertools
import sys
import re
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("paths", nargs=2, type=str)
args = parser.parse_args()


def read_path(path):
    with open(path) as file:
        d = {}
        for line in file:
            name, elo = line.split()
            cp = re.search(r"(\d+)$", name).group()
            d[int(cp)] = float(elo)
        return d


d1 = read_path(args.paths[0])
d2 = read_path(args.paths[1])

total_shift = 0
shared_keys = d1.keys() & d2.keys()
for k in shared_keys:
    total_shift += d1[k] - d2[k]
shift = total_shift / len(shared_keys)

for k in sorted(d1.keys() | d2.keys()):
    if k in d1 and k in d2:
        print(k, (d1[k] + d2[k] + shift) / 2)
    elif k in d1:
        print(k, d1[k])
    else:
        print(k, d2[k] + shift)
