"""Loss-curve plotting (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

TRAIN_COLOR = "#2a78d6"   # categorical slot 1 (blue)
VALID_COLOR = "#eb6834"   # categorical slot 2 (orange)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def plot_loss_curves(history: dict, path: Path, title: str) -> None:
    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    series = [("Training loss", history["train_loss"], TRAIN_COLOR, "o"),
              ("Validation loss", history["val_loss"], VALID_COLOR, "s")]
    for label, values, color, marker in series:
        ax.plot(epochs, values, color=color, linewidth=2, marker=marker, markersize=4, label=label)
        ax.annotate(f"{values[-1]:.3f}", (epochs[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    best = min(range(len(epochs)), key=lambda i: history["val_loss"][i])
    ax.scatter([epochs[best]], [history["val_loss"][best]], s=90, facecolors="none",
               edgecolors=INK, linewidths=1.2, zorder=5, label=f"Best checkpoint (epoch {epochs[best]})")

    ax.set_title(title, fontsize=12, color=INK, loc="left")
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel("Label-smoothed cross-entropy (ε_ls = 0.1)", color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, labelcolor=INK)
    ax.set_xlim(epochs[0] - 0.3, epochs[-1] + max(1, len(epochs) * 0.08))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
