#!/usr/bin/env python
"""Presentation figures from scripts/evaluate.py (and optionally scripts/benchmark.py) output.

    python scripts/plot_results.py --results results_140m.json [--benchmark benchmark_140m.json] --out figs

Runs with the same arm name (different seeds) are pooled: points/bars show the mean, error bars the spread
(min-max over seeds) or a 95% Wilson interval over items. Figures:
  frontier.png     code loss vs cache bytes per token (KV + ICE): what each arm pays and gets
  coderecall.png   CodeRecall deciding-digit accuracy per category, ordered from "lookup right" to "lookup wrong"
  harm.png         misleading lookups: follow rate and loss on tokens where the lookup is wrong (two panels)
"""
import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference categorical palette (light mode), assigned per arm in a fixed order -- never by rank
ARM_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "C_notrust": "#eda100", "B_pm": "#e87ba4",
              "F": "#008300", "C_off": "#6250d6"}
OTHER = "#e34948"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CATEGORY_ORDER = ["copy_same", "override_same", "conflict_g1", "copy_diff", "conflict_g2", "conflict_g5",
                  "override_diff", "increment", "track"]          # lookup right -> lookup wrong

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--benchmark", default=None, help="benchmark.json of the SAME preset (adds ICE bytes/token)")
ap.add_argument("--out", default="figs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

res = json.load(open(a.results))
groups = defaultdict(list)                        # arm label -> list of result dicts (seeds)
for r in res.values():
    groups[r["arm"]].append(r)
arms = sorted(groups, key=lambda s: (list(ARM_COLORS).index(s) if s in ARM_COLORS else 99, s))
color = {arm: ARM_COLORS.get(arm, OTHER) for arm in arms}
bench = json.load(open(a.benchmark))["summary"] if a.benchmark else {}


def style(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def vals(arm, key):
    v = [r.get(key) for r in groups[arm]]
    return [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1. frontier: quality vs cache bytes per token ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
for arm in arms:
    if arm.endswith("_off"):
        continue
    loss = vals(arm, "code/all_loss")
    kv = vals(arm, "kv_bytes_per_token")
    if not loss or not kv:
        continue
    x = kv[0] + bench.get(arm, {}).get("ice_bytes_per_token", 0.0)
    y = float(np.mean(loss))
    yerr = [[y - min(loss)], [max(loss) - y]] if len(loss) > 1 else None
    ax.errorbar([x], [y], yerr=yerr, fmt="o", ms=8, color=color[arm], mec=SURFACE, mew=2, ecolor=color[arm],
                elinewidth=2, capsize=0, zorder=3)
    ax.annotate(f"{arm}  ({len(loss)} seed{'s' if len(loss) > 1 else ''})", (x, y), xytext=(8, 4),
                textcoords="offset points", color=INK, fontsize=8)
style(ax, "Code loss vs cache memory per token", "held-out code loss (lower is better)",
      "cache bytes per token: KV" + (" + ICE tables" if bench else "") + "  (less is cheaper)")
fig.tight_layout()
fig.savefig(os.path.join(a.out, "frontier.png"), dpi=200)
plt.close(fig)

# 2. CodeRecall per category ---------------------------------------------------------------------------------
cats = [c for c in CATEGORY_ORDER if any(k.startswith(f"coderecall/{c}@") for r in res.values() for k in r)]
if cats:
    plot_arms = [arm for arm in arms if any(k.endswith("/items_dec") for r in groups[arm] for k in r)]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * len(cats) * max(1, len(plot_arms)) * 0.32 + 2), 4.2),
                           facecolor=SURFACE)
    width = 0.8 / max(1, len(plot_arms))
    for i, arm in enumerate(plot_arms):
        means, lo, hi = [], [], []
        for c in cats:
            items = [v for r in groups[arm] for k, x in r.items()
                     if k.startswith(f"coderecall/{c}@") and k.endswith("/items_dec") for v in x]
            k_, n_ = sum(items), len(items)
            p = k_ / n_ if n_ else float("nan")
            l_, h_ = wilson(k_, n_)
            means.append(p)
            lo.append(p - l_)
            hi.append(h_ - p)
        xs = np.arange(len(cats)) + (i - (len(plot_arms) - 1) / 2) * width
        ax.bar(xs, means, width * 0.92, color=color[arm], label=arm, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([c.replace("_", "\n", 1) for c in cats], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_right = sum(c in ("copy_same", "override_same", "conflict_g1") for c in cats)
    ax.axvline(n_right - 0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text((n_right - 1) / 2, 1.08, "lookup right", ha="center", color=INK2, fontsize=8)
    ax.text((n_right + len(cats) - 1) / 2, 1.08, "lookup useless or wrong", ha="center", color=INK2, fontsize=8)
    style(ax, "CodeRecall: accuracy on the deciding digit (95% CI)", "accuracy")
    ax.legend(frameon=False, fontsize=8, ncol=min(6, len(plot_arms)), loc="upper right",
              bbox_to_anchor=(1.0, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "coderecall.png"), dpi=200)
    plt.close(fig)

# 3. harm: follow rate and loss on misleading lookups (two panels, one scale each) ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), facecolor=SURFACE)
for ax, key, title, ylabel in (
        (axes[0], "code/follow_wrong_lookup", "Follows a wrong lookup", "P(top-1 = wrong lookup token), code"),
        (axes[1], "code/lk_wrong_loss", "Loss where the lookup is wrong", "code loss on misleading-lookup tokens")):
    shown = [arm for arm in arms if vals(arm, key)]
    ys = [float(np.mean(vals(arm, key))) for arm in shown]
    err = [[y - min(vals(arm, key)) for arm, y in zip(shown, ys)], [max(vals(arm, key)) - y for arm, y in zip(shown, ys)]]
    ax.bar(range(len(shown)), ys, 0.6, color=[color[s] for s in shown], edgecolor=SURFACE, linewidth=1)
    ax.errorbar(range(len(shown)), ys, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
    for i, y in enumerate(ys):
        ax.text(i, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(shown, fontsize=8)
    style(ax, title, ylabel)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "harm.png"), dpi=200)
plt.close(fig)
print(f"wrote {a.out}/frontier.png, coderecall.png, harm.png")
