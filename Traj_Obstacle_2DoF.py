#!/usr/bin/python3
# ============================================================
# ELEC 311 -- Trajectory Design Around an Obstacle (2-DoF planar)
#
# ------------------------------------------------------------
# ROBOTICS, CYBORGS & ASSOCIATES -- MOTION PLANNING GROUP
#
# Spec RCA-SL-120:  no joint above 120 deg/s, anywhere, ever.
# Spec RCA-CL-015:  the hand stays at least 15 mm clear of every
#                   obstacle, for the whole motion.
#
# Two specs, and they pull in opposite directions. Going around
# the post costs you distance; covering that distance in the same
# time costs you joint speed. There is no setting that is free.
# ------------------------------------------------------------
#
#   python Traj_Obstacle_2DoF.py
#   python Traj_Obstacle_2DoF.py --animate
#
# Prints the report, draws the four panels, and writes the trajectory
# to trajectory.csv. With --animate it also replays it, by handing the
# exported file to animate_trajectory.py -- which you can equally run
# on its own, on any CSV you kept:
#
#   python animate_trajectory.py older_run.csv
#
# You change the block below. Nothing else.
# ============================================================

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ============================================================
# STUDENT CONFIGURATION -- this is the part you change
# ============================================================

# Waypoints. Three lists, same length, in order.
#   X[i], Y[i]  where the hand must be, in metres
#   T[i]        when it must be there, in seconds
#
# The first and last are the start and the goal. The hand is at
# rest at both. Everything in between is yours.
X = [0.30, 0.20, 0.10]
Y = [0.05, 0.05, 0.28]
T = [0.00, 1.00, 1.50]

# Obstacles: (x, y, radius) in metres. The post is not negotiable.
OBS = [(0.235, 0.175, 0.045)]

# Which space do you shape?
#   "task"   spline x(t) and y(t);  the hand path is what you drew
#   "joint"  spline q1(t) and q2(t); the joint speeds are what you drew
SPACE = "task"

# ============================================================
# DO NOT TOUCH ANYTHING BELOW THIS LINE
# ============================================================
# Everything from here down is the instrument: the kinematics, the
# spline construction, the clearance test, and the report. If you
# change it, your numbers stop being measurements and your report
# stops being evidence. Edit the block above instead.
# ============================================================

L1, L2 = 0.20, 0.15
REACH, INNER = L1 + L2, abs(L1 - L2)

SPEED_LIMIT = 120.0      # deg/s, RCA-SL-120
CLEARANCE_MIN = 0.015    # m,     RCA-CL-015
NSAMP = 2000

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
    "axes.titlecolor": CY, "lines.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------
# Kinematics -- the same 2-link model as every other week
# ---------------------------------------------------------
def fk(q1, q2):
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
                     L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])


def ik(x, y):
    """Closed form. One elbow branch, held for the whole motion."""
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        raise ValueError(f"({x:.3f}, {y:.3f}) is outside the workspace")
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2 = np.sqrt(1.0 - c2 * c2)
    q2 = -np.arctan2(s2, c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return q1, q2


def jacobian(q1, q2):
    return np.array([
        [-L1 * np.sin(q1) - L2 * np.sin(q1 + q2), -L2 * np.sin(q1 + q2)],
        [L1 * np.cos(q1) + L2 * np.cos(q1 + q2), L2 * np.cos(q1 + q2)]])


# ---------------------------------------------------------
# Build the trajectory, in whichever space was asked for
# ---------------------------------------------------------
def build():
    """Return t, hand position, joint angles, joint rates (deg/s)."""
    x_wp, y_wp, t_wp = np.array(X, float), np.array(Y, float), np.array(T, float)
    if not (len(x_wp) == len(y_wp) == len(t_wp)):
        sys.exit("  X, Y and T must be the same length.")
    if len(x_wp) < 2:
        sys.exit("  You need at least a start and a goal.")
    if np.any(np.diff(t_wp) <= 0):
        sys.exit("  T must increase. Two waypoints cannot share a time.")

    t = np.linspace(t_wp[0], t_wp[-1], NSAMP)

    if SPACE == "task":
        cx = CubicSpline(t_wp, x_wp, bc_type="clamped")
        cy = CubicSpline(t_wp, y_wp, bc_type="clamped")
        p = np.vstack([cx(t), cy(t)])
        dp = np.vstack([cx(t, 1), cy(t, 1)])

        q = np.zeros((2, len(t)))
        for i in range(len(t)):
            q[:, i] = ik(p[0, i], p[1, i])      # raises if it leaves the annulus
        # Joint rates from the Jacobian, not from differencing the IK.
        dq = np.zeros((2, len(t)))
        for i in range(len(t)):
            dq[:, i] = np.linalg.solve(jacobian(*q[:, i]), dp[:, i])

    elif SPACE == "joint":
        q_wp = np.array([ik(xi, yi) for xi, yi in zip(x_wp, y_wp)]).T
        c1 = CubicSpline(t_wp, q_wp[0], bc_type="clamped")
        c2 = CubicSpline(t_wp, q_wp[1], bc_type="clamped")
        q = np.vstack([c1(t), c2(t)])
        dq = np.vstack([c1(t, 1), c2(t, 1)])
        p = fk(q[0], q[1])

    else:
        sys.exit('  SPACE must be "task" or "joint".')

    return t, p, q, np.degrees(dq)


# ---------------------------------------------------------
# Naive obstacle test: a distance, nothing more
# ---------------------------------------------------------
def clearances(p):
    """
    Smallest gap between the hand and each obstacle, over the whole run.

    That is the entire obstacle model: a distance. Negative means the
    hand went through the post. There is no planner here, and no
    repulsion; you avoid the obstacle by choosing waypoints that do.
    """
    return [np.min(np.hypot(p[0] - cx_, p[1] - cy_) - r) for cx_, cy_, r in OBS]


# ---------------------------------------------------------
# Commissioning report
# ---------------------------------------------------------
def report(t, p, q, dq):
    gaps = clearances(p)
    worst = min(gaps) if gaps else float("inf")
    r = np.hypot(p[0], p[1])
    off = int(np.count_nonzero((r > REACH) | (r < INNER)))
    detJ = np.array([abs(np.linalg.det(jacobian(*q[:, i])))
                     for i in range(0, len(t), 5)])

    rows = [
        ("space shaped", SPACE, "", ""),
        ("waypoints", f"{len(X)}", "", ""),
        ("duration", f"{T[-1] - T[0]:.2f} s", "", ""),
        ("path length", f"{np.sum(np.hypot(np.diff(p[0]), np.diff(p[1])))*1000:.0f} mm",
         "", ""),
        ("samples off-workspace", f"{off} / {len(t)}", "0",
         "ok" if off == 0 else "OUTSIDE"),
        ("min |det J|", f"{detJ.min():.5f}", "", ""),
        ("MIN CLEARANCE", f"{worst*1000:.1f} mm",
         f"> {CLEARANCE_MIN*1000:.0f}",
         "PASS" if worst >= CLEARANCE_MIN else "FAIL"),
        ("max speed, joint 1", f"{np.abs(dq[0]).max():.1f} deg/s", "", ""),
        ("max speed, joint 2", f"{np.abs(dq[1]).max():.1f} deg/s", "", ""),
        ("MAX JOINT SPEED", f"{np.abs(dq).max():.1f} deg/s",
         f"< {SPEED_LIMIT:.0f}",
         "PASS" if np.abs(dq).max() <= SPEED_LIMIT else "FAIL"),
    ]

    w = 78
    print("\n" + "=" * w)
    print("  ROBOTICS, CYBORGS & ASSOCIATES -- COMMISSIONING REPORT")
    print("=" * w)
    print(f"  {'quantity':<24}{'measured':>16}{'spec':>14}{'verdict':>16}")
    print("  " + "-" * (w - 4))
    for name, val, spec, verdict in rows:
        print(f"  {name:<24}{val:>16}{spec:>14}{verdict:>16}")
    print("=" * w)
    ship = worst >= CLEARANCE_MIN and np.abs(dq).max() <= SPEED_LIMIT and off == 0
    print("  SHIP" if ship else "  DO NOT SHIP")
    print("=" * w + "\n")
    return ship


# ---------------------------------------------------------
# Plots: x vs t, y vs t, y vs x, and the joint speeds
# ---------------------------------------------------------
def draw_scene(ax):
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(REACH * np.cos(th), REACH * np.sin(th), color=DIM, lw=1)
    ax.plot(INNER * np.cos(th), INNER * np.sin(th), color=DIM, lw=1)
    for cx_, cy_, r in OBS:
        ax.add_patch(plt.Circle((cx_, cy_), r, color=MA, alpha=0.75, zorder=3))
        ax.add_patch(plt.Circle((cx_, cy_), r + CLEARANCE_MIN, color=MA,
                                fill=False, ls="--", lw=1, zorder=3))


def plots(t, p, q, dq):
    f, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].plot(t, p[0], color=CY)
    ax[0, 0].plot(T, X, "o", color=AM, ms=8)
    ax[0, 0].set_xlabel("t [s]"); ax[0, 0].set_ylabel("x [m]")
    ax[0, 0].set_title("x(t)")

    ax[0, 1].plot(t, p[1], color=MA)
    ax[0, 1].plot(T, Y, "o", color=AM, ms=8)
    ax[0, 1].set_xlabel("t [s]"); ax[0, 1].set_ylabel("y [m]")
    ax[0, 1].set_title("y(t)")

    draw_scene(ax[1, 0])
    ax[1, 0].plot(p[0], p[1], color=LI, zorder=4)
    ax[1, 0].plot(X, Y, "o", color=AM, ms=8, zorder=5)
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set_xlim(-0.02, 0.38); ax[1, 0].set_ylim(-0.02, 0.38)
    ax[1, 0].set_xlabel("x [m]"); ax[1, 0].set_ylabel("y [m]")
    ax[1, 0].set_title(f"the path  (SPACE = {SPACE})")

    ax[1, 1].plot(t, dq[0], color=CY, label="joint 1")
    ax[1, 1].plot(t, dq[1], color=LI, label="joint 2")
    ax[1, 1].axhline(SPEED_LIMIT, color=MA, lw=1.5)
    ax[1, 1].axhline(-SPEED_LIMIT, color=MA, lw=1.5)
    ax[1, 1].set_xlabel("t [s]"); ax[1, 1].set_ylabel("joint speed [deg/s]")
    ax[1, 1].set_title("RCA-SL-120"); ax[1, 1].legend(fontsize=9)

    f.tight_layout()


def export(t, p, q, dq, path="trajectory.csv"):
    """
    Write the trajectory out for the animator.

    Plain CSV, one row per sample, so you can open it and look at it.
    Columns: t, x, y, q1, q2, dq1, dq2 -- seconds, metres, degrees.

    The two comment lines carry the scene, so the animator can draw the
    post without being told about it separately.
    """
    data = np.column_stack([t, p[0], p[1],
                            np.degrees(q[0]), np.degrees(q[1]), dq[0], dq[1]])
    with open(path, "w") as fh:
        fh.write("# obstacles " + "; ".join(f"{a},{b},{r}" for a, b, r in OBS) + "\n")
        fh.write(f"# space {SPACE}\n")
        fh.write("t,x,y,q1_deg,q2_deg,dq1_dps,dq2_dps\n")
        np.savetxt(fh, data, delimiter=",", fmt="%.6f")
    print(f"  wrote {path}  ({len(t)} samples, SPACE = {SPACE})\n")


def main():
    try:
        t, p, q, dq = build()
    except ValueError as e:
        print(f"\n  Cannot build the trajectory: {e}")
        print("  A waypoint, or the curve between two of them, left the")
        print(f"  workspace annulus ({INNER:.2f} m to {REACH:.2f} m).\n")
        return
    report(t, p, q, dq)
    export(t, p, q, dq)
    plots(t, p, q, dq)

    keep = None
    if "--animate" in sys.argv:
        try:
            import animate_trajectory
        except ImportError:
            print("  --animate needs animate_trajectory.py in this folder.\n")
        else:
            # Held in a local so the animation is not garbage collected.
            keep = animate_trajectory.replay("trajectory.csv")

    for flag in (a for a in sys.argv[1:] if a.startswith("--")):
        if flag != "--animate":
            print(f"  (ignoring unknown option {flag})")

    plt.show()
    del keep


if __name__ == "__main__":
    main()