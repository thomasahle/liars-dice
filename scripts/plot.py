"""
Script for making the ELO and Exploitability plots seen in the blog-post.
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
colors = sns.color_palette()

import numpy as np
import itertools
import sys
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("type", type=str, help="elo or lbr")
parser.add_argument("paths", type=str, nargs="*", help="Path of files")
args = parser.parse_args()

if args.type == "elo":

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    (ax1, ax2), (ax3, ax4) = axs
    fig.suptitle("Performance of models during training")

    data = []
    for path in args.paths:
        print(path)
        with open(path) as file:
            xys = []
            for line in file:
                name, elo = line.split()
                cp = re.search(r"(\d+)$", name).group()
                xys.append((int(cp) // 1000, float(elo)))
            xys.sort()
            data.append(tuple(xys))
    # medians = [sorted(xs)[1] for xs in zip(data)]

    labels = ["5v5", "4v4", "3v3", "2v2"]
    for l, ax, xyz in zip(labels, (ax1, ax2, ax3, ax4), data):
        xs = [x for x, y in xyz]
        ys = [y for x, y in xyz]
        ax.plot(xs, ys, label=f"{l} model")
        ax.legend()

    xs, ys = zip(*data[0])
    ax1.plot(
        xs,
        [ys[-1] - 28] * len(xs),
        label="Human players",
        linestyle="dashed",
        color=colors[1],
    )
    ax1.plot(xs, [ys[-1] - 43] * len(xs), linestyle="dashed", color=colors[1])
    ax1.legend()

    for ax in axs.flat:
        ax.set(xlabel="Iteration (in 1000s)", ylabel="ELO")

    for ax in fig.get_axes():
        ax.label_outer()

    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, right=1, top=0.95, bottom=0.05)

elif args.type == "elo1":

    plt.figure(figsize=(14, 7), dpi=100)
    plt.title("Performance of models during training")
    plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.08)

    labels = ["5v5", "4v4", "3v3", "2v2"]
    final_rating_total = 0
    max_iterations = 0
    for path, l in zip(args.paths, labels):
        print("Loading", path)
        xys = []
        with open(path) as file:
            for line in file:
                name, elo = line.split()
                cp = re.search(r"(\d+)$", name).group()
                xys.append((int(cp) // 1000, float(elo)))
        xys.sort()
        final_rating_total += xys[-1][1]
        max_iterations = max(max_iterations, xys[-1][0])

        xs, ys = zip(*xys)
        # xs = [x for x, y in xyz]
        # ys = [y for x, y in xyz]
        plt.plot(xs, ys, label=l)

    xs = [0, max_iterations]
    y = final_rating_total / 4
    plt.plot(
        xs,
        [y - 28] * len(xs),
        label="Human players",
        linestyle="dashed",
        color=colors[4],
    )
    plt.plot(xs, [y - 43] * len(xs), linestyle="dashed", color=colors[4])

    plt.legend()

    plt.xlabel("Iteration (in 1000s)")
    plt.ylabel("ELO")


elif args.type == "lbr":

    data = []
    for path in args.paths:
        with open(path) as file:
            xyzs = []
            for line in file:
                name, lbr1, lbr2 = line.split()
                cp = re.search(r"(\d+)$", name).group()
                xyzs.append((int(cp) // 1000, float(lbr1), float(lbr2)))
            xyzs.sort()
            data.append(xyzs)

    data = np.asarray(data)
    xs = data[0, :, 0]
    p1_med = np.median(data[:, :, 1], axis=0)
    p2_med = np.median(data[:, :, 2], axis=0)
    plt.plot(xs, p1_med, color=colors[0], label="Player 1")
    plt.plot(xs, p2_med, color=colors[2], label="Player 2")
    plt.plot(
        xs,
        [-7 / 327] * len(xs),
        label="Player 1, Nash",
        linestyle="dashed",
        color=colors[0],
    )
    plt.plot(
        xs,
        [7 / 327] * len(xs),
        label="Player 2, Nash",
        linestyle="dashed",
        color=colors[2],
    )

    for xyzs in data:
        xs, ys, zs = zip(*xyzs)
        plt.plot(xs, ys, color=colors[0], alpha=0.3)
        plt.plot(xs, zs, color=colors[2], alpha=0.3)

    plt.xlabel("Iteration (1000's)")
    plt.ylabel("Exploitability")
    plt.title("Exploitability in 1v1")
    plt.legend()
    plt.tight_layout()

elif args.type == "lbr12":

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(14, 7), dpi=100)
    # fig.figure(figsize=(14, 7), dpi=100)
    ax1, ax2 = axs

    data = []
    for path in args.paths:
        with open(path) as file:
            xyzs = []
            for line in file:
                name, lbr1, lbr2 = line.split()
                cp = re.search(r"(\d+)$", name).group()
                xyzs.append((int(cp) // 1000, float(lbr1), float(lbr2)))
            xyzs.sort()
            data.append(xyzs)

    ax1.set_title("Exploitability in 1v2")
    ax2.set_title("Exploitability in 2v1")

    data = np.asarray(data)
    xs = data[0, :, 0]
    for i, ax, val in zip(range(2), (ax1, ax2), (-5 / 54, 7 / 27)):
        ax.plot(xs, data[i, :, 2], color=colors[0], label="Player 1")
        ax.plot(xs, data[i, :, 1], color=colors[2], label="Player 2")
        ax.plot(
            xs,
            [-val] * len(xs),
            label="Player 1, Nash",
            linestyle="dashed",
            color=colors[0],
        )
        ax.plot(
            xs,
            [val] * len(xs),
            label="Player 2, Nash",
            linestyle="dashed",
            color=colors[2],
        )

    for ax in axs.flat:
        ax.set(xlabel="Iteration (1000's)", ylabel="Exploitability")
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.tight_layout()


plt.show()
