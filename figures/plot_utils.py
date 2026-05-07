"""Shared plot styling utilities.

Design reference: mentor paper figures (muted palette, horizontal-only grid,
top-anchored legend, hatched stacked bars).

Usage:
    from figures.plot_utils import apply_paper_style, make_fig, save_fig

Quick start:
    apply_paper_style()          # call once at top of script
    fig, ax = make_fig()
    ...
    save_fig(fig, "out/fig1.pdf")
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np


# ─── Color palettes ──────────────────────────────────────────────────────────

# Line-plot palette A — for method comparisons (4 conditions)
# matches: Mono/grey, Mixed/teal, Short/crimson, Constrained/gold
LINE_PALETTE_A = ["#9e9e9e", "#5bbcb0", "#8b2020", "#c89a1a"]

# Line-plot palette B — for quantitative sweeps (light → dark, 3 levels)
# matches: 25%/sage, 50%/teal, 75%/navy
LINE_PALETTE_B = ["#a8cfa0", "#3eada3", "#2a4a6b"]

# Bar-plot palette — stacked / grouped bars
BAR_PALETTE = ["#a0ccba", "#3eada3", "#4a6880"]   # light → dark
BAR_COLOR_SFT = "#b0bec5"    # kept from original for backward compat
BAR_COLOR_RL  = "#1a3a5c"

# Baseline / annotation lines
COLOR_CEILING  = "#888888"    # pass@k ceiling
COLOR_BASELINE = "#cccccc"

# Background
PAPER_BG = "#faf5e4"          # warm cream
WHITE_BG = "#ffffff"

# ─── rcParam presets ─────────────────────────────────────────────────────────

BAR_WIDTH      = 0.20
GROUP_SPACING  = 0.65
BAR_FONT_FAMILY = "STIXGeneral"   # matches mentor paper; good for math + regular text


def apply_paper_style(font_family="STIXGeneral", background="white"):
    """Apply the full paper aesthetic globally.

    Args:
        font_family: matplotlib font family
        background:  "white" (default) or "paper"/"cream"

    Call this ONCE at the top of every plotting script before any figure
    is created.  Subsequent make_fig() calls inherit these settings.
    """
    bg = WHITE_BG if str(background).lower() in {"white", "w"} else PAPER_BG
    plt.rcParams.update({
        # ── font ──
        "font.family":          font_family,
        "font.size":            11,
        "axes.titlesize":       12,
        "axes.titleweight":     "normal",
        "axes.labelsize":       11,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      10,

        # ── background ──
        "figure.facecolor":     bg,
        "axes.facecolor":       bg,
        "savefig.facecolor":    bg,

        # ── spines ──
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,
        "axes.linewidth":       0.6,
        "axes.edgecolor":       "#9a9070",   # warm grey spine

        # ── grid — horizontal only ──
        "axes.grid":            True,
        "axes.grid.axis":       "y",          # horizontal lines only
        "grid.color":           "#d8cfa8",
        "grid.linestyle":       "-",
        "grid.linewidth":       0.5,
        "grid.alpha":           1.0,

        # ── ticks ──
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.color":          "#5a5040",
        "ytick.color":          "#5a5040",

        # ── lines / markers ──
        "lines.linewidth":      1.6,
        "lines.markersize":     4,

        # ── color cycle ──
        "axes.prop_cycle":      plt.cycler(color=LINE_PALETTE_A),

        # ── legend ──
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#d8cfa8",
        "legend.facecolor":     bg,
        "legend.borderpad":     0.4,
        "legend.handlelength":  1.6,

        # ── output ──
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
    })


def apply_bar_rcparams():
    """Apply the standard bar-plot rcParams (calls apply_paper_style first)."""
    apply_paper_style()


# ─── Figure helpers ──────────────────────────────────────────────────────────

def make_fig(figsize=(8, 3.2), ncols=1, **kwargs):
    """Return (fig, axes) with paper-style figure size.

    For multi-panel figures pass ncols; axes is a flat list.
    For single panel, axes is a single Axes object.
    """
    if ncols > 1:
        fig, axes = plt.subplots(1, ncols, figsize=figsize, **kwargs)
        return fig, list(axes)
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def make_multipanel(ncols, figsize=None, sharey=False, **kwargs):
    """Standard multi-panel strip (like the 5-panel paper figures).

    Returns (fig, axes_list).
    """
    if figsize is None:
        figsize = (2.6 * ncols, 2.8)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=sharey, **kwargs)
    fig.subplots_adjust(wspace=0.35)
    return fig, list(axes)


def style_ax(ax, xlabel="", ylabel="", title=None,
             xticks=None, ylim=None, legend_loc="upper left",
             legend_ncol=1):
    """Apply standard axis styling in-place."""
    if title:
        ax.set_title(title, pad=4)
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc=legend_loc, ncol=legend_ncol)


def top_legend(fig, handles=None, labels=None, ncol=4,
               y=1.02, fontsize=7):
    """Place a shared legend above the figure (horizontal strip).

    If handles/labels are None, auto-collects from all axes.
    """
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
        # Remove per-axes legends
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg:
                leg.remove()
    legend_bg = plt.rcParams.get("figure.facecolor", PAPER_BG)
    return fig.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol,
                      fontsize=fontsize, frameon=True,
                      framealpha=0.9, edgecolor="#d8cfa8",
                      facecolor=legend_bg)


OVERLEAF_DIR = Path("/work/hdd/bbsg/twei2/rl/69e95abe7a34d3bbc17ab21e")
# Backwards-compat alias: previous name pointed directly at assets/figures.
OVERLEAF_FIGURES_DIR = OVERLEAF_DIR / "assets" / "figures"


def save_fig(fig, out_path, dpi=300, overleaf_figures_dir=None):
    """Tight-layout, save as PDF + PNG, print destination.

    Also mirrors the PDF into the Overleaf paper checkout's assets/figures/
    so it's ready to git push. Pass overleaf_figures_dir to override the
    default (the paper repo at OVERLEAF_DIR).
    """
    import shutil

    if overleaf_figures_dir is None:
        overleaf_figures_dir = OVERLEAF_DIR / "assets" / "figures"
    overleaf_figures_dir = Path(overleaf_figures_dir)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=150)
    print(f"Saved  {pdf_path}")
    print(f"       {png_path}")

    if overleaf_figures_dir.is_dir():
        dest = overleaf_figures_dir / pdf_path.name
        shutil.copyfile(pdf_path, dest)
        print(f"       {dest}  (overleaf)")
    else:
        print(f"       (overleaf dir {overleaf_figures_dir} not found, skipped)")


# ─── Curve helpers ───────────────────────────────────────────────────────────

def plot_curves(ax, xs, curves, palette=None):
    """Plot multiple curves onto ax.

    Args:
        ax:      matplotlib Axes
        xs:      shared x values
        curves:  list of (label, ys, marker) tuples
        palette: list of hex colors; if None uses current prop_cycle
    """
    colors = palette or [plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
                         for i in range(len(curves))]
    for (label, ys, marker), color in zip(curves, colors):
        ax.plot(xs, ys, marker=marker, label=label,
                color=color, markerfacecolor=color,
                markeredgewidth=0.5, markeredgecolor="white")


def add_ceiling_line(ax, y, label="pass@32 ceiling", color=COLOR_CEILING):
    """Draw a horizontal dashed ceiling / baseline line."""
    ax.axhline(y, ls="--", lw=1.0, color=color, alpha=0.75,
               label=label, zorder=0)


# ─── Bar helpers ─────────────────────────────────────────────────────────────

def draw_delta(ax, x_center, y_lo, y_hi, delta, color,
               bar_width=BAR_WIDTH):
    """Bracket + ±Npp annotation between two bars at x_center."""
    x_r   = x_center + bar_width / 2 + 0.08
    tick  = 0.015
    sign  = "+" if delta > 0 else ""
    ax.annotate("", xy=(x_r, y_hi), xytext=(x_r, y_lo),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2))
    ax.plot([x_r - tick, x_r + tick], [y_lo, y_lo], color=color, lw=1.2)
    ax.plot([x_r - tick, x_r + tick], [y_hi, y_hi], color=color, lw=1.2)
    ax.text(x_r + 0.04, (y_lo + y_hi) / 2,
            f"{sign}{delta}pp", color=color,
            va="center", ha="left", fontsize=8, fontweight="bold")


def grouped_bars(ax, labels, sft_values, rl_values, *,
                 title=None, ylim=(0, 110),
                 xlabel="Outcome", ylabel="Percentage (%)",
                 value_fmt="{:d}%",
                 bar_width=BAR_WIDTH, group_spacing=GROUP_SPACING,
                 color_sft=BAR_COLOR_SFT, color_rl=BAR_COLOR_RL):
    """Paired (post-SFT / post-RL) grouped bar chart."""
    x = np.arange(len(labels)) * group_spacing
    bars_sft = ax.bar(x - bar_width/2, sft_values, bar_width,
                      color=color_sft, label="post-SFT")
    bars_rl  = ax.bar(x + bar_width/2, rl_values,  bar_width,
                      color=color_rl,  label="post-RL")

    for bar in (*bars_sft, *bars_rl):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                value_fmt.format(int(bar.get_height())),
                ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_ylim(*ylim)
    ax.set_xlim(x[0] - 0.45, x[-1] + 0.45)
    if title:
        ax.set_title(title, pad=4)
    return x, bars_sft, bars_rl


# hatching constants — match derivation/enumeration encoding in paper
HATCH_DERIVATION  = ""        # solid fill
HATCH_ENUMERATION = "////"    # diagonal stripes


def stacked_bars(ax, labels, bottom_values, top_values, *,
                 bar_color, label_bottom="Derivation",
                 label_top="Enumeration",
                 hatch_bottom=HATCH_DERIVATION,
                 hatch_top=HATCH_ENUMERATION,
                 value_color=None,
                 bar_width=0.5):
    """Single-series stacked bar with derivation/enumeration split.

    bottom_values: solid portion (Derivation)
    top_values:    hatched portion (Enumeration)

    Returns (bars_bottom, bars_top).
    """
    x = np.arange(len(labels))
    vc = value_color or bar_color

    bars_b = ax.bar(x, bottom_values, bar_width,
                    color=bar_color, hatch=hatch_bottom,
                    label=label_bottom, edgecolor="white", linewidth=0.4)
    bars_t = ax.bar(x, top_values, bar_width,
                    bottom=bottom_values,
                    color=bar_color, hatch=hatch_top,
                    label=label_top, edgecolor="white", linewidth=0.4,
                    alpha=0.85)

    for i, (bv, tv) in enumerate(zip(bottom_values, top_values)):
        total = bv + tv
        ax.text(i, bv / 2, f"{int(bv)}",
                ha="center", va="center",
                fontsize=6.5, fontweight="bold",
                color=vc)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    return bars_b, bars_t


def add_sft_rl_legend(fig, color_sft=BAR_COLOR_SFT, color_rl=BAR_COLOR_RL,
                      loc="lower center", ncol=2,
                      bbox_to_anchor=(0.5, 0.01)):
    """Shared post-SFT / post-RL patch legend for multi-panel bar figures."""
    handles = [
        mpatches.Patch(color=color_sft, label="post-SFT"),
        mpatches.Patch(color=color_rl,  label="post-RL"),
    ]
    return fig.legend(handles=handles, loc=loc, ncol=ncol,
                      fontsize=7, frameon=True,
                      bbox_to_anchor=bbox_to_anchor,
                      facecolor=PAPER_BG, edgecolor="#d8cfa8")