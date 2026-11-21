"""Shared chart functions.

Used by train_model.py (to save PNGs into outputs/) and by app.py (to draw the
same charts live), so the plotting code exists in only one place.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIGH = "#2563EB"   # blue
LOW = "#F59E0B"    # amber
INK = "#1B2540"
MUTED = "#6B7694"
GRID = "#E3E8F2"

LABELS = {
    "Submission.Actions": "Submission",
    "Quiz.Actions": "Quiz",
    "Forum.Actions": "Forum",
    "Create.Actions": "Create",
    "Read.Actions": "Read",
    "Update.Actions": "Update",
    "Delete.Actions": "Delete",
    "Days.Logged.In": "Days logged in",
}


def theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })


def _fig(w=6.4, h=3.6):
    theme()
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def fig_class_balance(raw_counts, unique_counts):
    """Raw (as supplied) vs unique rows, by class."""
    fig, ax = _fig(6.0, 3.4)
    x = np.arange(2)
    w = 0.34
    hi = [raw_counts["High"], unique_counts["High"]]
    lo = [raw_counts["Low"], unique_counts["Low"]]
    b1 = ax.bar(x - w / 2, hi, w, color=HIGH, label="High")
    b2 = ax.bar(x + w / 2, lo, w, color=LOW, label="Low")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 25,
                    f"{int(b.get_height()):,}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["File as supplied\n(3,602 rows)", "Unique rows only\n(2,720 rows)"])
    ax.set_ylabel("Records")
    ax.set_ylim(0, max(hi + lo) * 1.15)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def fig_feature_dist(df, feature):
    """Histogram (log-spaced) + boxplot for one feature, split by rating."""
    theme()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    vals = df[feature]
    vmax = max(vals.max(), 1)
    # log1p scale so skewed counts remain readable
    bins = np.linspace(0, np.log1p(vmax), 28)
    for cls, col in (("High", HIGH), ("Low", LOW)):
        v = np.log1p(df.loc[df.rating == cls, feature])
        a1.hist(v, bins=bins, alpha=0.62, color=col, label=cls, edgecolor="white", linewidth=0.5)
    ticks = [t for t in [0, 1, 3, 10, 30, 100, 300, 1000, 10000, 100000] if t <= vmax]
    a1.set_xticks(np.log1p(ticks))
    a1.set_xticklabels([f"{t:,}" for t in ticks])
    a1.set_xlabel(f"{LABELS.get(feature, feature)} (log scale)")
    a1.set_ylabel("Records")
    a1.legend(frameon=False)
    data = [df.loc[df.rating == c, feature].values for c in ("High", "Low")]
    bp = a2.boxplot(data, tick_labels=["High", "Low"], patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.6), widths=0.55)
    for patch, col in zip(bp["boxes"], (HIGH, LOW)):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor(col)
    a2.set_ylabel("Count (outliers hidden)")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_heatmap(corr):
    theme()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    names = [LABELS.get(c, c) for c in corr.columns]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_yticklabels(names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def fig_confusion(cm, classes=("High", "Low")):
    theme()
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Predicted\n{c}" for c in classes])
    ax.set_yticklabels([f"Actual\n{c}" for c in classes])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=18,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else INK)
    fig.tight_layout()
    return fig


def fig_importance(imp):
    """imp: DataFrame with feature, perm_mean, perm_std."""
    imp = imp.sort_values("perm_mean")
    fig, ax = _fig(6.4, 3.8)
    y = np.arange(len(imp))
    ax.barh(y, imp["perm_mean"], xerr=imp["perm_std"], color=HIGH, alpha=0.85,
            error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(f, f) for f in imp["feature"]])
    ax.set_xlabel("Drop in test accuracy when the feature is shuffled")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_pdp(pdp, feature):
    """pdp: dict feature -> {'x': [...], 'p_high': [...]}"""
    d = pdp[feature]
    fig, ax = _fig(6.0, 3.4)
    x = np.array(d["x"], dtype=float)
    ax.plot(x, d["p_high"], color=HIGH, lw=2.4, marker="o", ms=3.5)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel(LABELS.get(feature, feature))
    ax.set_ylabel("Average P(High)")
    if x.max() / max(x.min(), 1) > 50:
        ax.set_xscale("symlog", linthresh=10)
    fig.tight_layout()
    return fig


def fig_calibration(cal):
    fig, ax = _fig(4.6, 3.9)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Perfectly calibrated")
    ax.plot(cal["predicted"], cal["observed"], color=HIGH, marker="o", lw=2, label="Model")
    ax.set_xlabel("Predicted P(High)")
    ax.set_ylabel("Share actually High")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def fig_profile(pctl_user, pctl_high, pctl_low):
    """Where the entered student sits (percentile of the training data) vs class medians."""
    theme()
    feats = list(pctl_user.keys())
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    y = np.arange(len(feats))[::-1]
    ax.hlines(y, 0, 100, color=GRID, lw=6, zorder=1)
    ax.scatter([pctl_high[f] for f in feats], y, s=70, color=HIGH, alpha=0.35, zorder=2,
               label="Typical High (median)")
    ax.scatter([pctl_low[f] for f in feats], y, s=70, color=LOW, alpha=0.45, zorder=2,
               label="Typical Low (median)")
    ax.scatter([pctl_user[f] for f in feats], y, s=95, color=INK, zorder=3,
               marker="D", label="This student")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(f, f) for f in feats])
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Percentile among the records in the supplied file")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3)
    fig.tight_layout()
    return fig


def fig_local(contrib):
    """contrib: dict feature -> change in P(High) vs replacing feature with typical value."""
    theme()
    items = sorted(contrib.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    vals = [v for _, v in items]
    ax.barh([LABELS.get(k, k) for k, _ in items], vals,
            color=[HIGH if v >= 0 else LOW for v in vals], alpha=0.9)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Change in P(High) caused by this value (vs. a typical value)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
