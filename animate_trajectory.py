#!/usr/bin/python3
# ============================================================
# ELEC 311 -- Replay an Exported Trajectory
#
# Reads the trajectory.csv that Traj_Obstacle_2DoF.py wrote and
# plays it back: the arm, the post, and the trace behind the hand.
#
#   python animate_trajectory.py
#   python animate_trajectory.py my_run.csv
#   python animate_trajectory.py my_run.csv --gif
#
# The animation is the reward, not the evidence. A path that looks
# fine at fifty frames a second can still be carrying a joint at
# 140 deg/s through the middle of it. The report is what you sign.
#
# This file reads. It does not compute. If the motion looks wrong,
# the trajectory is wrong -- go back and fix the waypoints.
# ============================================================

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

L1, L2 = 0.20, 0.15
REACH, INNER = L1 + L2, abs(L1 - L2)

BG = (8 / 255, 8 / 255, 15 / 255)
CY = (34 / 255, 211 / 255, 238 / 255)
MA = (220 / 255, 20 / 255, 150 / 255)
AM = (255 / 255, 176 / 255, 32 / 255)
DIM = (110 / 255, 117 / 255, 144 / 255)
LI = (158 / 255, 232 / 255, 75 / 255)
FG = (216 / 255, 220 / 255, 232 / 255)
plt.rcParams.update({
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": DIM, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "text.color": FG, "grid.color": (0.16, 0.17, 0.24),
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "axes.titlecolor": CY, "axes.spines.top": False, "axes.spines.right": False,
})


def load(path):
    """Read the CSV. The two comment lines carry the scene."""
    if not os.path.exists(path):
        sys.exit(f"\n  No file '{path}'.\n"
                 f"  Run Traj_Obstacle_2DoF.py first; it writes one.\n")

    obstacles, space, n_comments = [], "unknown", 0
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n_comments += 1
            if line.startswith("# obstacles"):
                body = line.split("# obstacles", 1)[1].strip()
                for chunk in filter(None, (c.strip() for c in body.split(";"))):
                    obstacles.append(tuple(float(v) for v in chunk.split(",")))
            elif line.startswith("# space"):
                space = line.split("# space", 1)[1].strip()

    # skip_header, not comments=: the scene lines are comma-separated too,
    # and genfromtxt will happily read one of them as the column names.
    d = np.genfromtxt(path, delimiter=",", names=True, skip_header=n_comments)
    if d.size == 0:
        sys.exit(f"\n  '{path}' has no samples in it.\n")
    return d, obstacles, space


def replay(path="trajectory.csv"):
    """
    Build the animation for one exported CSV and return it.

    Keep the returned object alive; if it is garbage collected the
    animation stops dead. The caller calls plt.show().
    """
    d, obstacles, space = load(path)

    t, x, y = d["t"], d["x"], d["y"]
    q1 = np.radians(d["q1_deg"])
    top = max(np.abs(d["dq1_dps"]).max(), np.abs(d["dq2_dps"]).max())
    print(f"\n  {path}: {len(t)} samples, {t[-1] - t[0]:.2f} s, SPACE = {space}")
    print(f"  peak joint speed in this file: {top:.1f} deg/s\n")

    f = plt.figure(figsize=(6.4, 6.4))
    ax = f.add_subplot(111, aspect="equal",
                       xlim=(-0.05, 0.40), ylim=(-0.05, 0.40))
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(REACH * np.cos(th), REACH * np.sin(th), color=DIM, lw=1)
    ax.plot(INNER * np.cos(th), INNER * np.sin(th), color=DIM, lw=1)
    for cx, cy, r in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color=MA, alpha=0.75, zorder=3))
        ax.add_patch(plt.Circle((cx, cy), r + 0.015, color=MA, fill=False,
                                ls="--", lw=1, zorder=3))

    trace, = ax.plot([], [], color=DIM, lw=1.2, zorder=2)
    arm, = ax.plot([], [], "o-", lw=3.5, color=CY, markerfacecolor=AM,
                   markersize=8, zorder=6)
    hand, = ax.plot([], [], "o", color=LI, markersize=9, zorder=7)
    clock = ax.text(0.02, 0.37, "", color=DIM, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"SPACE = {space}")

    # Aim for about 400 drawn frames whatever the sample count.
    step = max(1, len(t) // 400)
    idx = np.arange(0, len(t), step)

    def update(k):
        i = idx[k]
        ex, ey = L1 * np.cos(q1[i]), L1 * np.sin(q1[i])
        arm.set_data([0, ex, x[i]], [0, ey, y[i]])
        hand.set_data([x[i]], [y[i]])
        trace.set_data(x[:i], y[:i])
        clock.set_text(f"t = {t[i]:.2f} s")
        return arm, hand, trace, clock

    return FuncAnimation(f, update, frames=len(idx), interval=20,
                         blit=True, repeat=False)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "trajectory.csv"
    ani = replay(path)

    if "--gif" in sys.argv:
        out = os.path.splitext(path)[0] + ".gif"
        ani.save(out, writer=PillowWriter(fps=30))
        print(f"  wrote {out}\n")
    else:
        plt.show()
    print("=^..^=")


if __name__ == "__main__":
    main()