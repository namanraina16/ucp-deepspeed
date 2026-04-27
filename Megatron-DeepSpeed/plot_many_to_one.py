#!/usr/bin/env python3
import re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.lines import Line2D

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

def load_tb_csv(p: Path):
    df = pd.read_csv(p)
    if "step" not in df.columns or "value" not in df.columns:
        if len(df.columns) == 3:
            df.columns = ["wall_time", "step", "value"]
        else:
            raise ValueError(f"{p}: expected step,value columns (or 3-col TB csv).")
    df = df[["step", "value"]].dropna().sort_values("step")
    return df

def crop(df, lo, hi):
    return df[(df["step"] >= lo) & (df["step"] <= hi)]

def ema(y, alpha=0.6):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    out = np.zeros_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out

def plot_curve(ax, x, y, *, color, lw, alpha, zorder):
    ax.plot(x, ema(y, alpha=0.6), color=color, lw=lw, alpha=alpha, zorder=zorder)

def clean_label_default(stem: str):
    # stem should be the RUN name (folder), not the metric filename
    m = re.search(r"tp(\d+)_pp(\d+)_dp(\d+)_sp(\d+).*?_z(\d+)", stem)
    if m:
        tp, pp, dp, sp, z = m.groups()
        return f"TP: {tp}, PP: {pp}, DP: {dp} (ZeRO-{z}), SP: {sp}"
    return stem

def infer_run_name_from_metric_path(p: Path):
    # expects: .../<RUN_NAME>/<metric_file>.csv
    # e.g. csv_tb_clean/src_tp2_pp2_dp1_sp1_z1/lm-loss-training__lm_loss.csv
    if p.parent and p.parent.name:
        return p.parent.name
    return p.stem

def main():
    ap = argparse.ArgumentParser(description="Fig 7(b): multiple sources (overlap) -> single target after resume_step.")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="CSV paths for SOURCE runs (many). Plot only up to resume_step.")
    ap.add_argument("--target", required=True,
                    help="CSV path for SINGLE TARGET run. Plot only after resume_step.")

    ap.add_argument("--resume_step", type=int, default=200)
    ap.add_argument("--x_min", type=int, default=0)
    ap.add_argument("--x_max", type=int, default=300)

    ap.add_argument("--y_min", type=float, default=None)
    ap.add_argument("--y_max", type=float, default=None)

    ap.add_argument("--title", default="(b) Multiple Sources to Single Target")
    ap.add_argument("--out", default="fig7b_many_to_one.png")
    args = ap.parse_args()

    set_paper_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # ---- sources (overlap) ----
    src_color_cycle = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"])
    src_handles, src_labels = [], []
    plotted_any_source = False

    for f in args.sources:
        p = Path(f)
        if not p.exists():
            print(f"WARNING: missing source CSV, skipping: {p}")
            continue

        try:
            df = crop(load_tb_csv(p), args.x_min, args.resume_step)
        except Exception as e:
            print(f"WARNING: failed to read {p} ({e}), skipping.")
            continue

        if df.empty:
            print(f"WARNING: source empty in [{args.x_min}, {args.resume_step}], skipping: {p}")
            continue

        c = next(src_color_cycle)
        plot_curve(ax, df["step"].to_numpy(), df["value"].to_numpy(),
                   color=c, lw=1.8, alpha=0.95, zorder=4)

        run_name = infer_run_name_from_metric_path(p)
        src_handles.append(Line2D([0], [0], color=c, lw=2.0))
        src_labels.append(clean_label_default(run_name))
        plotted_any_source = True

    # ---- target (single) ----
    tgt_path = Path(args.target)
    plotted_target = False
    if not tgt_path.exists():
        print(f"WARNING: missing target CSV, skipping target: {tgt_path}")
    else:
        try:
            tdf = crop(load_tb_csv(tgt_path), args.resume_step, args.x_max)
        except Exception as e:
            print(f"WARNING: failed to read target {tgt_path} ({e}), skipping.")
            tdf = None

        if tdf is not None and not tdf.empty:
            plot_curve(ax, tdf["step"].to_numpy(), tdf["value"].to_numpy(),
                       color="black", lw=2.6, alpha=1.0, zorder=6)
            plotted_target = True

    if not plotted_any_source and not plotted_target:
        raise SystemExit("ERROR: nothing to plot (no valid sources and no valid target).")

    # ---- resume marker + arrows ----
    ax.axvline(args.resume_step, color="#d62728", lw=2.0, ls="--", zorder=10)

    ax.set_xlim(args.x_min, args.x_max)
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(args.y_min, args.y_max)
    ax.set_ylim(5,10)
    # if args.y_min is not None or args.y_max is not None:
    #     ax.set_ylim(args.y_min, args.y_max)

    y0, y1 = ax.get_ylim()
    ax.text(args.resume_step + 2, y0 + 0.02 * (y1 - y0),
            "Resume Training", color="#d62728", fontsize=14, fontweight="bold", va="bottom")

    y_arrow = y0 + 0.08 * (y1 - y0)
    ax.annotate("", (args.x_min + 2, y_arrow), (args.resume_step - 2, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=1.6))
    ax.text((args.x_min + args.resume_step) / 2, y_arrow + 0.03 * (y1 - y0),
            "Multiple Sources (Overlapping)", ha="center", fontsize=14)

    ax.annotate("", (args.resume_step + 2, y_arrow), (args.x_max - 2, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=1.6))
    ax.text((args.resume_step + args.x_max) / 2, y_arrow + 0.03 * (y1 - y0),
            "Single Target", ha="center", fontsize=14)

    # ---- axes ----
    ax.set_xlabel("Training Step")
    ax.set_ylabel("LM Loss")
    ax.set_title(args.title, pad=16)
    ax.grid(True)

    # legend: sources only (like Fig7b)
    if src_handles:
        ax.legend(src_handles, src_labels, loc="upper center", ncol=2,
                  frameon=False, handlelength=2.4, columnspacing=1.6)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
