#!/usr/bin/env python3
import re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.lines import Line2D

# Optional smoothing
try:
    from scipy.signal import savgol_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -------------------- Style (paper-like) --------------------
def set_paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
    })


# -------------------- Data utils --------------------
def load_uc_csv(p: Path):
    df = pd.read_csv(p)
    if "step" not in df.columns or "value" not in df.columns:
        if len(df.columns) == 3:
            df.columns = ["wall_time", "step", "value"]
        else:
            raise ValueError(f"{p}: expected step,value columns")
    return df[["step", "value"]].dropna().sort_values("step")


def crop(df, lo, hi):
    return df[(df["step"] >= lo) & (df["step"] <= hi)]


def smooth_savgol(y, window=11, poly=2):
    if not HAVE_SCIPY or window <= 1 or len(y) < window:
        return y
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window, poly)


def clean_label(stem: str):
    """
    Converts:
    uc_out_tp2_pp1_dp2_sp1_z1
    ->
    TP: 2, PP: 1, DP: 2 (ZeRO-1), SP: 1
    """
    m = re.search(r"tp(\d+)_pp(\d+)_dp(\d+)_sp(\d+).*?_z(\d+)", stem)
    if not m:
        return stem
    tp, pp, dp, sp, z = m.groups()
    return f"TP: {tp}, PP: {pp}, DP: {dp} (ZeRO-{z}), SP: {sp}"

def ema(y, alpha=0.6):
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def plot_curve(ax, x, y, *, color, lw, alpha, zorder):
    y_s = ema(y, alpha=0.6)
    ax.plot(x, y_s, color=color, lw=lw, alpha=alpha, zorder=zorder)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--targets", nargs="+", required=True)

    ap.add_argument("--resume_step", type=int, default=100)
    ap.add_argument("--x_min", type=int, default=0)
    ap.add_argument("--x_max", type=int, default=200)

    ap.add_argument("--out", default="fig_single_to_many_paper.png")
    args = ap.parse_args()

    set_paper_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # ---------------- Source (dominant) ----------------
    sdf = crop(load_uc_csv(Path(args.source)), args.x_min, args.resume_step)
    plot_curve(
        ax,
        sdf["step"].to_numpy(),
        sdf["value"].to_numpy(),
        color="#1f77b4",
        lw=2.4,
        alpha=1.0,
        zorder=6,
    )

    # ---------------- Targets (overlapping) ----------------
    color_cycle = cycle([
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#9467bd", "#8c564b", "#7f7f7f"
    ])

    handles, labels = [], []

    for f in args.targets:
        df = crop(load_uc_csv(Path(f)), args.resume_step, args.x_max)
        if df.empty:
            continue

        c = next(color_cycle)
        plot_curve(
            ax,
            df["step"].to_numpy(),
            df["value"].to_numpy(),
            color=c,
            lw=1.6,
            alpha=0.85,
            zorder=4,
        )

        handles.append(Line2D([0], [0], color=c, lw=1.8))
        labels.append(clean_label(Path(f).stem))

    # ---------------- Resume marker ----------------
    ax.axvline(args.resume_step, color="#d62728", lw=2.0, ls="--", zorder=10)
    ax.text(
        args.resume_step + 2,
        7.15,
        "Resume Training",
        color="#d62728",
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )

    # ---------------- Region arrows ----------------
    y_arrow = 6.35
    ax.annotate("", (args.x_min + 2, y_arrow), (args.resume_step - 2, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=1.6))
    ax.text((args.x_min + args.resume_step) / 2, y_arrow + 0.2,
            "Single Source", ha="center", fontsize=14)

    ax.annotate("", (args.resume_step + 2, y_arrow), (args.x_max - 2, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=1.6))
    ax.text((args.resume_step + args.x_max) / 2, y_arrow + 0.2,
            "Multiple Targets (Overlapping)", ha="center", fontsize=14)

    # ---------------- Axes ----------------
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(6.2, 11.0)   # locked like reference
    ax.set_xlabel("Training Step")
    ax.set_ylabel("LM Loss")
    ax.set_title("(a) Single Source to Multiple Targets", pad=16)
    ax.grid(True)

    # ---------------- Legend (paper style) ----------------
    ax.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()