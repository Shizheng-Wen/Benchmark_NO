#!/usr/bin/env python3
"""Plot scalability results produced by benchmark_scalability.

Generates four figures (paper‑ready) from a CSV created by
`benchmark_scalability()`.

Figures
-------
1. model_size_vs_throughput.<fmt>   – x: model size (M),  y: training throughput
2. model_size_vs_peakmem.<fmt>      – x: model size (M),  y: BS=1 peak memory
3. grid_vs_throughput.<fmt>         – x: input grid (#nodes),  y: training throughput
4. grid_vs_peakmem.<fmt>            – x: input grid (#nodes),  y: BS=1 peak memory

Command‑line
------------
usage: plot_scalability.py CSV [--outdir DIR] [--grid G] [--params P]
                              [--logx] [--logy] [--format {pdf,png}]

optional arguments:
  --logx              Use logarithmic x‑axis for all plots (default: linear).
  --format {pdf,png}  Output format/extension (default: pdf).

If `--grid` is not given, the largest grid is chosen for the *model‑size* plots.
Likewise, if `--params` is not given, the largest model size is chosen for the
*input‑grid* plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib style – try seaborn-paper, fall back gracefully
# ──────────────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-paper")
except OSError:
    plt.style.use("default")
    try:
        import seaborn as sns
        sns.set_theme(context="paper", style="whitegrid")
        print("[plot_scalability] seaborn-paper style not found; using seaborn whitegrid.")
    except ModuleNotFoundError:
        print("[plot_scalability] seaborn not installed; using matplotlib default style.")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "figure.dpi": 300,
})

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, fname: str, outdir: Path, fmt: str):
    path = outdir / f"{fname}.{fmt}"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[✓] Saved {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Plot scalability curves")
    p.add_argument("csv", type=Path, help="CSV produced by benchmark_scalability")
    p.add_argument("--outdir", type=Path, default=Path("figs"), help="output directory")
    p.add_argument("--grid", type=int, help="fix input grid when sweeping model size")
    p.add_argument("--params", type=float, help="fix model size when sweeping input grid")
    p.add_argument("--logx", action="store_true", help="use logarithmic x‑axis")
    p.add_argument("--logy", action="store_true", help="use logarithmic y‑axis")
    p.add_argument("--format", choices=["pdf", "png"], default="pdf",
                   help="figure file format (extension)")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = {"grid", "params_M", "trn_sps", "trn_bs1_peak_MB"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}.")

    # ── model‑size sweep ──────────────────────────────────────────────────
    grid_fix = args.grid if args.grid is not None else df["grid"].max()
    df_model = df[df["grid"] == grid_fix].sort_values("params_M")
    if df_model.empty:
        raise ValueError(f"No rows with grid == {grid_fix}.")

    fig1, ax1 = plt.subplots()
    ax1.plot(df_model["params_M"], df_model["trn_sps"], marker="o")
    ax1.set(xlabel="Model size (M parameters)", ylabel="Training throughput (samples/s)",
            title=f"Throughput vs Model Size (grid={grid_fix})")
    if args.logx:
        ax1.set_xscale("log")
    if args.logy:
        ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    _save(fig1, "model_size_vs_throughput", args.outdir, args.format)

    fig2, ax2 = plt.subplots()
    ax2.plot(df_model["params_M"], df_model["trn_bs1_peak_MB"], marker="o")
    ax2.set(xlabel="Model size (M parameters)", ylabel="Peak memory @ BS=1 (MB)",
            title=f"Peak Memory vs Model Size (grid={grid_fix})")
    if args.logx:
        ax2.set_xscale("log")
    if args.logy:
        ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    _save(fig2, "model_size_vs_peakmem", args.outdir, args.format)

    # ── input‑grid sweep ──────────────────────────────────────────────────
    params_fix = args.params if args.params is not None else df["params_M"].max()
    df_inp = df[df["params_M"] == params_fix].sort_values("grid")
    if df_inp.empty:
        raise ValueError(f"No rows with params_M == {params_fix}.")

    fig3, ax3 = plt.subplots()
    ax3.plot(df_inp["grid"], df_inp["trn_sps"], marker="o")
    ax3.set(xlabel="Input grid size (nodes)", ylabel="Training throughput (samples/s)",
            title=f"Throughput vs Input Size (model={params_fix}M)")
    if args.logx:
        ax3.set_xscale("log")
    if args.logy:
        ax3.set_yscale("log")
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5)
    _save(fig3, "grid_vs_throughput", args.outdir, args.format)

    fig4, ax4 = plt.subplots()
    ax4.plot(df_inp["grid"], df_inp["trn_bs1_peak_MB"], marker="o")
    ax4.set(xlabel="Input grid size (nodes)", ylabel="Peak memory @ BS=1 (MB)",
            title=f"Peak Memory vs Input Size (model={params_fix}M)")
    if args.logx:
        ax4.set_xscale("log")
    if args.logy:
        ax4.set_yscale("log")
    ax4.grid(True, which="both", linestyle="--", linewidth=0.5)
    _save(fig4, "grid_vs_peakmem", args.outdir, args.format)


# ─────────────────────────────────────────────────────────────
# multi-model comparison  --  draw all CSVs in a folder
# ─────────────────────────────────────────────────────────────
def _plot_multi(df, x, y, xlabel, ylabel, outdir, fname, *,
                logx=False, logy=False,fmt="pdf"):
    fig, ax = plt.subplots()
    for model_name, sub in df.groupby("model"):
        ax.plot(sub[x], sub[y], marker="o", label=model_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()
    _save(fig, fname, outdir, fmt)

def main_multi():
    """
    Example:
        python plot_scalability.py results --logx --format png
    """
    p = argparse.ArgumentParser(description="Plot multi-model scalability curves")
    p.add_argument("folder", type=Path, help="folder containing *.csv results")
    p.add_argument("--outdir", type=Path, default=Path("figs_multi"))
    p.add_argument("--logx", action="store_true", help="use log-x axis")
    p.add_argument("--logy", action="store_true", help="use log-y axis")
    p.add_argument("--format", choices=["pdf", "png"], default="pdf")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---------- collect all csv ----------
    dfs = []
    for csv in args.folder.glob("*.csv"):
        df = pd.read_csv(csv)
        if {"grid", "params_M", "trn_sps", "trn_bs1_peak_MB", "sweep"}.issubset(df.columns):
            df["model"] = csv.stem       # e.g. GNOT, Transolver
            dfs.append(df)
        else:
            print(f"[warn] {csv.name} skipped (missing columns)")
    if not dfs:
        raise ValueError(f"No valid CSV files in {args.folder}")

    df_all = pd.concat(dfs, ignore_index=True)

    # ── 1. model-scale curves (fixed grid) ──────────────────
    grid_fix = df_all["grid"].max()
    df_model = (df_all[
                       (df_all["sweep"] == "model_scale") &
                       df_all["trn_sps"].notna()])
    if not df_model.empty:
        _plot_multi(df_model, "params_M", "trn_sps",
                    "Model size (M parameters)", "Training throughput (samples/s)",
                    args.outdir, "model_vs_throughput",
                    logx=args.logx, logy=args.logy,fmt=args.format)

        _plot_multi(df_model, "params_M", "trn_bs1_peak_MB",
                    "Model size (M parameters)", "Peak mem @BS=1 (MB)",
                    args.outdir, "model_vs_peakmem",
                    logx=args.logx, logy=args.logy,fmt=args.format)
    else:
        print("[warn] no model_scale data to plot")

    # ── 2. input-grid curves (fixed model size) ─────────────
    params_fix = df_all["params_M"].max()
    df_grid = (df_all[
                      (df_all["sweep"] == "input_scale") &
                      df_all["trn_sps"].notna()])
    if not df_grid.empty:
        _plot_multi(df_grid, "grid", "trn_sps",
                    "Input grid size (nodes)", "Training throughput (samples/s)",
                    args.outdir, "grid_vs_throughput",
                    logx=True, logy=args.logy,fmt=args.format)   # grid 用 logx 更直观

        _plot_multi(df_grid, "grid", "trn_bs1_peak_MB",
                    "Input grid size (nodes)", "Peak mem @BS=1 (MB)",
                    args.outdir, "grid_vs_peakmem",
                    logx=True, logy=args.logy,fmt=args.format)
    else:
        print("[warn] no input_scale data to plot")


# ── entry point switch: single-csv vs folder mode ───────────
if __name__ == "__main__":
    import sys
    first = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if first and first.suffix == ".csv":
        main()          # existing single-file plotting
    else:
        main_multi()    # new multi-model plotting


