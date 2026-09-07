"""Experiment 1c — Extended Figures (Publication Quality)

Generates 6 figures with STIXGeneral (Times-like) font:
  1. Violin+box (EI_g + decomposition)
  2. Sweep comparison (EI_g vs MI, dual axis)
  3. Phase diagram (synthetic trajectories + real models)
  4. Normalized degradation curves
  5. MI vs EI_g scatter (discriminant validity)
  6. θ-permutation test (structural disruption)

Loads pre-computed data where available, computes new experiments inline.
"""
import json
import pathlib
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
CODE = ROOT.parent.parent
for p in (str(ROOT), str(ROOT.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from steps.eig_metric import compute_eig
from steps.exp1_synthetic_worlds import metric_mi_knn, _random_orth
from steps.exp1b_robustness import (
    gen_world, compute_one, preprocess, EIG_KW,
    WORLD_PARAMS, WORLD_TAGS, N, D, WC, SC,
)

# ── style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "legend.fontsize": 14,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
})
BBOX = dict(facecolor="white", alpha=0.88, edgecolor="none", pad=2.5)
C_EIG, C_MI, C_LVI, C_LM = "#3498db", "#e67e22", "#2ecc71", "#e74c3c"
RM_COLORS = {
    "Qwen-2.5-7B": "#1a5276",
    "Llama-3.1-8B": "#6c3483",
    "Mistral-v0.3": "#b9770e",
    "DeepSeek-R1": "#922b21",
}
EIG_CONST = (D / 2) * np.log(2 * np.pi * np.e)  # ≈ 8.513
L_THETA_I_LABEL = r"$\langle l(\theta)\rangle_I$"
PUB_AXIS_LABEL_FS = 28
PUB_TICK_FS = 25
PUB_LEGEND_FS = 22
PUB_ANNOT_FS = 23
PUB_PANEL_TAG_FS = 24
PUB_SECONDARY_FS = 25
FIG2_LEGEND_FS = 18
FIG5_LEGEND_FS = 16


# ── helpers ───────────────────────────────────────────────────
def _stats(reps, key):
    m = np.array([np.mean([r[key] for r in pt]) for pt in reps])
    s = np.array([np.std([r[key] for r in pt]) for pt in reps])
    return m, s


def _grad_line(ax, x, y, color, lw=3.0, zorder=2):
    """Draw a gradient line that fades in from light to full opacity."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    rgba = to_rgba(color)
    colors = [(*rgba[:3], 0.12 + 0.88 * i / max(n - 1, 1)) for i in range(n)]
    widths = np.linspace(max(lw * 0.4, 1.0), lw, n)
    lc = LineCollection(segs, colors=colors, linewidths=widths, zorder=zorder)
    ax.add_collection(lc)


def _add_bottom_panel_tag(ax, tag, y=-0.30, fontsize=16):
    ax.text(
        0.5, y, tag,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=fontsize, fontweight="bold",
    )


def _draw_axis_break(ax_top, ax_bot, size=0.015, color="#444444"):
    kwargs = dict(color=color, clip_on=False, lw=1.2)
    ax_top.plot((-size, +size), (-size, +size),
                transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - size, 1 + size), (-size, +size),
                transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-size, +size), (1 - size, 1 + size),
                transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - size, 1 + size), (1 - size, 1 + size),
                transform=ax_bot.transAxes, **kwargs)


def _compute_broken_ylims(data_groups):
    group_medians = np.array([np.median(vals) for vals in data_groups])
    order = np.argsort(group_medians)
    split = int(np.argmax(np.diff(group_medians[order]))) + 1
    lower_vals = np.concatenate([np.asarray(data_groups[i]) for i in order[:split]])
    upper_vals = np.concatenate([np.asarray(data_groups[i]) for i in order[split:]])

    lower_pad = max(np.ptp(lower_vals) * 0.12, 0.08)
    upper_pad = max(np.ptp(upper_vals) * 0.12, 0.08)
    lower_ylim = (lower_vals.min() - lower_pad, lower_vals.max() + lower_pad)
    upper_ylim = (upper_vals.min() - upper_pad, upper_vals.max() + upper_pad)
    return lower_ylim, upper_ylim


def _draw_violin_box(ax, data, pos, colors):
    vp = ax.violinplot(data, positions=pos, showmeans=False,
                       showmedians=False, showextrema=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.16)

    bp = ax.boxplot(data, positions=pos, widths=0.20,
                    patch_artist=True, zorder=5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.0)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color("#222222")
            ln.set_linewidth(1.0)
    for ln in bp["medians"]:
        ln.set_color("white")
        ln.set_linewidth(2.0)
    for ln in bp["fliers"]:
        ln.set_markerfacecolor("gray")
        ln.set_markersize(4)


def _place_pair_ylabel(fig, ax_top, ax_bot, ylabel, fontsize, x_shift=0.0):
    renderer = fig.canvas.get_renderer()
    label_boxes = []
    for ax in (ax_top, ax_bot):
        for tick in ax.get_yticklabels():
            if tick.get_text():
                bbox = tick.get_window_extent(renderer=renderer)
                if bbox.width > 0 and bbox.height > 0:
                    label_boxes.append(bbox)

    pair_center_y = 0.5 * (ax_bot.get_position().y0 + ax_top.get_position().y1)
    if not label_boxes:
        x_fig = ax_top.get_position().x0 - 0.015
    else:
        x0_disp = min(b.x0 for b in label_boxes)
        x0_fig = fig.transFigure.inverted().transform((x0_disp, 0))[0]
        x_fig = x0_fig - 0.012
    x_fig += x_shift

    fig.text(
        x_fig, pair_center_y, ylabel,
        rotation="vertical", va="center", ha="center",
        fontsize=fontsize,
    )


# ── data loading ──────────────────────────────────────────────
def load_exp1b():
    with (CODE / "output" / "exp1b_robustness.json").open() as f:
        return json.load(f)


def load_real_models():
    models = {}
    for name, dirn in [
        ("Qwen-2.5-7B", "output_qwen2.5-7b"),
        ("Llama-3.1-8B", "output_llama3.1-8b"),
        ("Mistral-v0.3", "output_mistral-7b-instruct-v0.3"),
        ("DeepSeek-R1", "output_deepseek-r1-distill-qwen-7b"),
    ]:
        p = CODE / dirn / "eig_metric_result.json"
        if p.exists():
            d = json.load(p.open())
            models[name] = dict(
                EI_g=d["EI_g"], l_mean=d["l_mean"],
                log_VI=float(np.log(max(d["VI"], 1e-30))),
            )
    return models


# ══════════════════════════════════════════════════════════════
#  New computations
# ══════════════════════════════════════════════════════════════

MECH_COLORS = {
    "noise": "#e74c3c", "twist": "#3498db",
    "collapse": "#2ecc71", "nonlinear": "#9b59b6",
}
MECH_LABELS = {
    "noise": "Noise-dom.", "twist": "Twist-dom.",
    "collapse": "Collapse-dom.", "nonlinear": "Nonlinear-dom.",
}
MECH_ORDER = ["noise", "twist", "collapse", "nonlinear"]


def _gen_world_ext(seed, noise=0.1, twist_scale=0.0, compress=1.0,
                   nonlinear=0.0):
    """World generator extended with nonlinear θ→y distortion."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D))
    theta = x @ (compress * np.eye(D)) + noise * rng.standard_normal((N, D))
    G = np.eye(D)
    if twist_scale > 0:
        R3 = _random_orth(3, rng)
        G[3:, 3:] = R3
    nv = np.full(D, noise)
    if twist_scale > 0:
        nv[3:] = twist_scale
    y_lin = theta @ G.T
    if nonlinear > 0:
        y_lin = y_lin + nonlinear * np.tanh(y_lin)
    y = y_lin + rng.standard_normal((N, D)) * nv
    return x, theta, y


def compute_scatter_wide():
    """Random sampling across full 4-D mechanism space."""
    print("  Computing wide mechanism sampling (500 pts) ...")
    rng = np.random.default_rng(9000)
    n_total = 500
    recs = []
    for i in range(n_total):
        noise = float(rng.uniform(0.05, 3.0))
        twist = float(rng.uniform(0.0, 2.0))
        compress = float(rng.uniform(0.01, 1.0))
        nonlin = float(rng.uniform(0.0, 2.0))
        dev = {
            "noise": (noise - 0.05) / 2.95,
            "twist": twist / 2.0,
            "collapse": (1.0 - compress) / 0.99,
            "nonlinear": nonlin / 2.0,
        }
        dominant = max(dev, key=dev.get)
        xr, tr, yr = _gen_world_ext(
            9000 + i, noise=noise, twist_scale=twist,
            compress=compress, nonlinear=nonlin)
        x, th, y, vx, vt = preprocess(xr, tr, yr)
        try:
            res = compute_eig(x=x, theta=th, y=y,
                              vol_x=vx, vol_theta=vt, **EIG_KW)
            eig_val = float(res["EI_g"])
            if np.isnan(eig_val):
                continue
        except Exception:
            continue
        mi = metric_mi_knn(x, y, k=7)
        recs.append(dict(
            EI_g=eig_val,
            MI_xy=float(mi),
            log_VI=float(np.log(max(res["VI"], 1e-30))),
            l_mean=float(res["l_mean"]),
            noise=noise, twist=twist,
            compress=compress, nonlinear=nonlin,
            mechanism=dominant,
        ))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_total}")
    return recs


def compute_perm_test():
    """θ-permutation test on a fixed World-A dataset."""
    print("  Computing θ-permutation test ...")
    from sklearn.neighbors import NearestNeighbors
    xr, tr, yr = gen_world(seed=42, **WORLD_PARAMS["A"])
    x, theta, y, vx, vt = preprocess(xr, tr, yr)

    res0 = compute_eig(x=x, theta=theta, y=y,
                        vol_x=vx, vol_theta=vt, **EIG_KW)
    eig0 = float(res0["EI_g"])
    print(f"    Original EI_g = {eig0:.3f}")

    results = {"original": eig0, "methods": {}}
    for mname in ("global", "local_50"):
        eigs = []
        for rep in range(20):
            rng = np.random.default_rng(8000 + rep)
            if mname == "global":
                theta_p = theta[rng.permutation(N)]
            else:
                nbrs = NearestNeighbors(n_neighbors=50).fit(x)
                _, idx = nbrs.kneighbors(x)
                theta_p = theta[[rng.choice(idx[i]) for i in range(N)]]

            res = compute_eig(x=x, theta=theta_p, y=y,
                              vol_x=vx, vol_theta=vt, **EIG_KW)
            eigs.append(float(res["EI_g"]))
        results["methods"][mname] = eigs
        ea = np.array(eigs)
        print(f"    Perm-{mname}: {ea.mean():.2f} +/- {ea.std():.2f}")
    return results


# ══════════════════════════════════════════════════════════════
#  Figure 1 — Violin + box
# ══════════════════════════════════════════════════════════════

def plot_fig1(bd):
    fig = plt.figure(figsize=(17.8, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.18],
                          hspace=0.05, wspace=0.30)
    keys = list(WORLD_PARAMS)
    colors = [MECH_COLORS[m] for m in MECH_ORDER]
    pos = np.arange(len(keys))
    world_labels = ["A", "B", "C", "D"]

    panels = [
        ("EI_g", r"$\mathrm{EI}_g$", "(a)"),
        ("log_VI", r"$\log V_I$", "(b)"),
        ("l_mean", L_THETA_I_LABEL, "(c)"),
    ]
    axes_pairs = []
    for col, (metric, ylabel, panel_tag) in enumerate(panels):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col], sharex=ax_top)
        axes_pairs.append((ax_top, ax_bot, ylabel))

        data = [np.asarray(bd[k][metric]) for k in keys]
        lower_ylim, upper_ylim = _compute_broken_ylims(data)

        for ax in (ax_top, ax_bot):
            _draw_violin_box(ax, data, pos, colors)
            rng_j = np.random.default_rng(42 + col)
            for i, d in enumerate(data):
                jit = rng_j.uniform(-0.12, 0.12, len(d))
                ax.scatter(pos[i] + jit, d, color=colors[i],
                           s=18, alpha=0.55, edgecolors="none", zorder=6)
            ax.tick_params(axis="y", labelsize=PUB_TICK_FS)
            ax.grid(axis="y", alpha=0.18, ls="--")

        ax_top.set_ylim(*upper_ylim)
        ax_bot.set_ylim(*lower_ylim)
        ax_top.spines.bottom.set_visible(False)
        ax_bot.spines.top.set_visible(False)
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_bot.tick_params(axis="x", labelsize=PUB_TICK_FS)
        ax_top.set_xlim(-0.55, len(keys) - 0.45)

        for i, vals in enumerate(data):
            mu = vals.mean()
            label = "<0.1" if metric == "l_mean" and abs(mu) < 0.1 else f"{mu:.1f}"
            target_ax = ax_top if vals.max() >= upper_ylim[0] else ax_bot
            target_ax.annotate(
                label,
                xy=(pos[i], vals.max()), xytext=(0, 9),
                textcoords="offset points", ha="center",
                fontsize=PUB_ANNOT_FS, color="#333333", fontweight="normal",
            )

        ax_bot.set_xticks(pos)
        ax_bot.set_xticklabels(world_labels, fontsize=PUB_TICK_FS,
                               rotation=0, ha="center")
        _draw_axis_break(ax_top, ax_bot)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.16,
                        wspace=0.28, hspace=0.05)
    fig.canvas.draw()
    ylabel_x_shifts = [0.0, 0.0, 0.006]
    for (ax_top, ax_bot, ylabel), x_shift in zip(axes_pairs, ylabel_x_shifts):
        _place_pair_ylabel(fig, ax_top, ax_bot, ylabel,
                           PUB_AXIS_LABEL_FS, x_shift=x_shift)
    return fig


# ══════════════════════════════════════════════════════════════
#  Figure 2 — Sweeps with MI baseline
# ══════════════════════════════════════════════════════════════

def plot_fig2(sw_json):
    sweep_names = ["noise", "twist", "collapse"]
    xlabels = {"noise": "noise level", "twist": "twist scale",
               "collapse": "compression"}
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.0))
    for ax, sn in zip(axes, sweep_names):
        sd = sw_json[sn]
        v = np.array(sd["vals"])
        eig_m, eig_s = np.array(sd["EI_g_mean"]), np.array(sd["EI_g_std"])
        mi_m, mi_s = np.array(sd["MI_xy_mean"]), np.array(sd["MI_xy_std"])
        lvi_m = np.array(sd["log_VI_mean"])
        lm_m = np.array(sd["l_mean_mean"])

        l1, = ax.plot(v, eig_m, color=C_EIG, lw=3.0,
                      label=r"$\mathrm{EI}_g$", zorder=5)
        ax.fill_between(v, eig_m - eig_s, eig_m + eig_s,
                        color=C_EIG, alpha=0.10)
        l2, = ax.plot(v, mi_m, color=C_MI, lw=2.5, ls="--",
                      label=r"$\mathrm{MI}(x,y)$", zorder=4)
        ax.fill_between(v, mi_m - mi_s, mi_m + mi_s,
                        color=C_MI, alpha=0.08)
        ax.set_xlabel(xlabels[sn], fontsize=PUB_AXIS_LABEL_FS)
        ax.set_ylabel("nats", fontsize=PUB_AXIS_LABEL_FS)
        ax.tick_params(axis="both", labelsize=PUB_TICK_FS)

        ax2 = ax.twinx()
        l3, = ax2.plot(v, lvi_m, color=C_LVI, lw=1.8, alpha=0.6,
                       label=r"$\log V_I$")
        l4, = ax2.plot(v, lm_m, color=C_LM, lw=1.8, ls="-.", alpha=0.6,
                       label=L_THETA_I_LABEL)
        ax2.set_ylabel("component (nats)", fontsize=PUB_SECONDARY_FS,
                       color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=PUB_TICK_FS)
        ax.legend(handles=[l1, l2, l3, l4], fontsize=FIG2_LEGEND_FS,
                  loc="best")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ══════════════════════════════════════════════════════════════
#  Figure 3 — Phase diagram + real models
# ══════════════════════════════════════════════════════════════

def plot_fig3(sw_json, bd, real_models):
    fig, (ax_d, ax_a) = plt.subplots(1, 2, figsize=(20, 9),
                                      gridspec_kw={"width_ratios": [1, 1]})
    # ── panel (a): Δ phase diagram (synthetic mechanisms) ──
    for dE in (-2, -5, -10, -15):
        xs = np.linspace(-22, 4, 80)
        ys = xs - dE
        m = (ys > -4) & (ys < 8)
        if m.any():
            ax_d.plot(xs[m], ys[m], color="#ccc", lw=0.7, ls=":", zorder=0)
            idx = np.where(m)[0]; mid = idx[len(idx) // 2]
            ax_d.text(xs[mid], ys[mid] + 0.18,
                      f"$\\Delta\\mathrm{{EI}}_g$={dE}",
                      fontsize=11, color="#aaa", rotation=45, ha="center")

    sweep_order = ["noise", "twist", "collapse"]
    markers = {"noise": "o", "twist": "s", "collapse": "D"}
    end_lab = {
        "noise": r"Noise $\uparrow$",
        "twist": r"Twist $\uparrow$",
        "collapse": r"Collapse $\uparrow$",
    }
    offsets = {"noise": (14, -22), "twist": (14, 6), "collapse": (14, 10)}

    for sn in sweep_order:
        sd = sw_json[sn]
        lvi = np.array(sd["log_VI_mean"])
        lm = np.array(sd["l_mean_mean"])
        dx, dy = lvi - lvi[0], lm - lm[0]
        c, mk = SC[sn], markers[sn]

        _grad_line(ax_d, dx, dy, c, lw=5.0, zorder=2)
        nn = len(dx)
        for i in [0, nn // 4, nn // 2, 3 * nn // 4, nn - 1]:
            al = 0.25 + 0.75 * i / max(nn - 1, 1)
            sz = 55 + 90 * i / max(nn - 1, 1)
            ax_d.scatter(dx[i], dy[i], color=c, s=sz, alpha=al,
                         marker=mk, edgecolors="white", linewidths=0.5,
                         zorder=3)
        if nn >= 2:
            ax_d.annotate("", xy=(dx[-1], dy[-1]),
                          xytext=(dx[-2], dy[-2]),
                          arrowprops=dict(arrowstyle="-|>", color=c,
                                          lw=3.5, mutation_scale=22),
                          zorder=4)
        ax_d.annotate(end_lab[sn], xy=(dx[-1], dy[-1]),
                      xytext=offsets[sn], textcoords="offset points",
                      fontsize=15, color=c, fontweight="bold",
                      bbox=BBOX, va="center")

    ax_d.scatter([0], [0], color="black", s=200, marker="*", zorder=5)
    ax_d.annotate("baseline", xy=(0, 0), xytext=(10, -18),
                  textcoords="offset points", fontsize=14, bbox=BBOX)
    ax_d.axhline(0, color="gray", lw=0.5)
    ax_d.axvline(0, color="gray", lw=0.5)
    ax_d.text(0.02, 0.97,
              r"$\leftarrow$ diversity loss"
              "          "
              r"$\uparrow$ fidelity loss",
              transform=ax_d.transAxes, fontsize=13, va="top",
              color="gray", style="italic")
    ax_d.set_xlabel(r"$\Delta\log(V\!I)$", fontsize=18)
    ax_d.set_ylabel(r"$\Delta\ell_{\mathrm{mean}}$", fontsize=18)
    ax_d.set_title("(a)  Mechanism decomposition\n(synthetic sweeps)",
                    fontsize=18)

    # ── panel (b): absolute space (synthetic worlds + real models) ──
    for eig_val in (5, 10, 15, 20):
        xs = np.linspace(8, 40, 80)
        ys = xs - eig_val - EIG_CONST
        m = (ys > -2) & (ys < 8)
        if m.any():
            ax_a.plot(xs[m], ys[m], color="#ddd", lw=0.7, ls=":", zorder=0)
            idx = np.where(m)[0]; mid = idx[len(idx) // 2]
            ax_a.text(xs[mid] + 0.3, ys[mid] + 0.18,
                      f"$\\mathrm{{EI}}_g$={eig_val}",
                      fontsize=11, color="#aaa", rotation=45, ha="center")

    for w in ("A", "B", "C", "D"):
        lvi_m = np.mean(bd[w]["log_VI"])
        lm_m = np.mean(bd[w]["l_mean"])
        ax_a.scatter(lvi_m, lm_m, color=WC[w], s=150, marker="o",
                     edgecolors="white", linewidths=1.0, zorder=4)
        ax_a.annotate(f"World {w}", xy=(lvi_m, lm_m),
                      xytext=(10, 8), textcoords="offset points",
                      fontsize=14, color=WC[w], fontweight="bold", bbox=BBOX)

    rm_markers = {
        "Qwen-2.5-7B": "P",
        "Llama-3.1-8B": "D",
        "Mistral-v0.3": "^",
        "DeepSeek-R1": "X",
    }
    rm_offsets = {
        "Qwen-2.5-7B": (10, 10),
        "Llama-3.1-8B": (10, -20),
        "Mistral-v0.3": (-92, 10),
        "DeepSeek-R1": (10, -16),
    }
    for name, vals in real_models.items():
        mc = RM_COLORS.get(name, "black")
        ax_a.scatter(vals["log_VI"], vals["l_mean"], color=mc, s=250,
                     marker=rm_markers.get(name, "o"), edgecolors="white",
                     linewidths=1.0, zorder=5)
        ax_a.annotate(
            f'{name}\n$\\mathrm{{EI}}_g$={vals["EI_g"]:.1f}',
            xy=(vals["log_VI"], vals["l_mean"]),
            xytext=rm_offsets.get(name, (10, 10)), textcoords="offset points",
            fontsize=14, color=mc, fontweight="bold", bbox=BBOX,
        )

    ax_a.set_xlabel(r"$\log(V\!I)$  (diversity)", fontsize=18)
    ax_a.set_ylabel(r"$\ell_{\mathrm{mean}}$  (fidelity loss)", fontsize=18)
    ax_a.set_title("(b)  Synthetic worlds + real LLMs\n(absolute coordinates)",
                    fontsize=18)

    fig.suptitle(
        r"$\mathrm{EI}_g$ Phase Diagram: "
        r"$\mathrm{EI}_g = \log(V\!I) - \ell_{\mathrm{mean}} - C$"
        r"$\;\Rightarrow\;$ orthogonal diversity & fidelity axes",
        fontsize=21, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ══════════════════════════════════════════════════════════════
#  Figure 4 — Normalized degradation curves
# ══════════════════════════════════════════════════════════════

def plot_fig4(sw_json):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sweep_order = ["noise", "twist", "collapse"]
    labels = {
        "noise": r"Noise ($\sigma$: 0.05$\to$3)",
        "twist": r"Twist ($s$: 0$\to$2)",
        "collapse": r"Collapse ($c$: 1$\to$0.01)",
    }

    for sn in sweep_order:
        sd = sw_json[sn]
        eig_m = np.array(sd["EI_g_mean"])
        eig_s = np.array(sd["EI_g_std"])
        e0 = eig_m[0]
        norm_m = (eig_m - e0) / abs(e0)
        norm_s = eig_s / abs(e0)

        frac = np.linspace(0, 1, len(eig_m))
        c = SC[sn]
        ax.plot(frac, norm_m, color=c, lw=3.0, label=labels[sn])
        ax.fill_between(frac, norm_m - norm_s, norm_m + norm_s,
                        color=c, alpha=0.12)

    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Degradation progress  (0 = baseline, 1 = maximum)",
                  fontsize=16)
    ax.set_ylabel(
        r"$(\mathrm{EI}_g - \mathrm{EI}_g^{(0)}) \,/\,"
        r" |\mathrm{EI}_g^{(0)}|$",
        fontsize=17,
    )
    ax.set_title(
        r"Normalized $\mathrm{EI}_g$ drop under three degradation mechanisms",
        fontsize=19,
    )
    ax.legend(fontsize=15, loc="lower left")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  Figure 5 — MI vs EI_g: wide sampling + conditional slices
# ══════════════════════════════════════════════════════════════

def _cond_strip(ax, values, mechs, mech_order):
    """Violin + box + jittered strip for a conditional slice."""
    data_by = {}
    for m in mech_order:
        mask = mechs == m
        if mask.sum() >= 3:
            data_by[m] = values[mask]
    valid = [m for m in mech_order if m in data_by]
    if not valid:
        return
    pos = np.arange(len(valid))
    cols = [MECH_COLORS[m] for m in valid]
    bdata = [data_by[m] for m in valid]
    vp = ax.violinplot(bdata, positions=pos, showmeans=False,
                       showmedians=False, showextrema=False)
    for body, c in zip(vp["bodies"], cols):
        body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.20)
    bp = ax.boxplot(bdata, positions=pos, widths=0.24,
                    patch_artist=True, zorder=4)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.6)
        p.set_edgecolor("k"); p.set_linewidth(1.0)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color("k"); ln.set_linewidth(1.0)
    for ln in bp["medians"]:
        ln.set_color("white"); ln.set_linewidth(2.0)
    for ln in bp["fliers"]:
        ln.set_markerfacecolor("gray"); ln.set_markersize(4)
    rng_j = np.random.default_rng(42)
    for i, (m, d) in enumerate(zip(valid, bdata)):
        jit = rng_j.uniform(-0.12, 0.12, len(d))
        ax.scatter(pos[i] + jit, d, color=MECH_COLORS[m],
                   s=18, alpha=0.55, edgecolors="none", zorder=5)
        ax.annotate(f"$n$={len(d)}\n$\\mu$={d.mean():.1f}",
                    xy=(pos[i], d.max()), xytext=(0, 14),
                    textcoords="offset points", ha="center",
                    fontsize=PUB_ANNOT_FS, color=MECH_COLORS[m],
                    fontweight="bold", bbox=BBOX)
    ax.set_xticks(pos)
    ax.set_xticklabels([MECH_LABELS[m] for m in valid], fontsize=PUB_TICK_FS,
                       rotation=12, ha="center", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=PUB_TICK_FS)
    ax.tick_params(axis="x", pad=6)


def plot_fig5(recs):
    """MI vs EI_g with random mechanism sampling + conditional slices."""
    fig = plt.figure(figsize=(22, 6.7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.08, 1.16, 1.16], wspace=0.22)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    mi = np.array([r["MI_xy"] for r in recs])
    eig = np.array([r["EI_g"] for r in recs])
    mech = np.array([r["mechanism"] for r in recs])

    # ── (a) global scatter ──
    for m in MECH_ORDER:
        mask = mech == m
        ax_a.scatter(mi[mask], eig[mask], color=MECH_COLORS[m], s=42,
                     alpha=0.55, edgecolors="white", linewidths=0.3,
                     label=MECH_LABELS[m], zorder=3)

    z = np.polyfit(mi, eig, 1)
    xs = np.linspace(mi.min(), mi.max(), 50)
    ax_a.plot(xs, np.polyval(z, xs), "k--", lw=1.5, alpha=0.4, zorder=1)

    from scipy.stats import spearmanr, pearsonr
    rho_s, _ = spearmanr(mi, eig)
    rho_p, _ = pearsonr(mi, eig)
    ax_a.text(0.03, 0.97,
              f"Spearman $\\rho$ = {rho_s:.2f}\n"
              f"Pearson  $r$ = {rho_p:.2f}\n"
              f"$N$ = {len(recs)}",
              transform=ax_a.transAxes, fontsize=PUB_ANNOT_FS,
              va="top", bbox=BBOX)

    mi_lo, mi_hi = np.percentile(mi, [35, 65])
    eig_lo, eig_hi = np.percentile(eig, [35, 65])
    ax_a.axvspan(mi_lo, mi_hi, color="#f0e68c", alpha=0.25, zorder=0)
    ax_a.axhspan(eig_lo, eig_hi, color="#add8e6", alpha=0.20, zorder=0)

    ax_a.set_xlabel(r"$\mathrm{MI}(x,y)$  (nats)", fontsize=PUB_AXIS_LABEL_FS)
    ax_a.set_ylabel(r"$\mathrm{EI}_g$  (nats)", fontsize=PUB_AXIS_LABEL_FS)
    ax_a.tick_params(axis="both", labelsize=PUB_TICK_FS)
    ax_a.legend(
        fontsize=FIG5_LEGEND_FS,
        loc="lower right",
        ncol=2,
        columnspacing=0.65,
        handletextpad=0.35,
        labelspacing=0.25,
        borderpad=0.30,
    )

    # ── (b) fix MI band → EI_g by mechanism ──
    mi_mask = (mi >= mi_lo) & (mi <= mi_hi)
    _cond_strip(ax_b, eig[mi_mask], mech[mi_mask], MECH_ORDER)
    ax_b.set_ylabel(r"$\mathrm{EI}_g$  (nats)", fontsize=PUB_AXIS_LABEL_FS)
    ax_b.tick_params(axis="both", labelsize=PUB_TICK_FS)

    # ── (c) fix EI_g band → MI by mechanism ──
    eig_mask = (eig >= eig_lo) & (eig <= eig_hi)
    _cond_strip(ax_c, mi[eig_mask], mech[eig_mask], MECH_ORDER)
    ax_c.set_ylabel(r"$\mathrm{MI}(x,y)$  (nats)", fontsize=PUB_AXIS_LABEL_FS)
    ax_c.tick_params(axis="both", labelsize=PUB_TICK_FS)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ══════════════════════════════════════════════════════════════
#  Figure 6 — θ-permutation test
# ══════════════════════════════════════════════════════════════

def plot_fig6(perm_res, bd):
    fig, ax = plt.subplots(figsize=(10, 7))

    orig_vals = bd["A"]["EI_g"]
    perm_global = perm_res["methods"]["global"]
    perm_local = perm_res["methods"]["local_50"]

    data = [orig_vals, perm_local, perm_global]
    labels = [
        r"Original" + "\n(20 seeds)",
        r"Perm-$\theta$-local" + "\n($k$=50)",
        r"Perm-$\theta$-global",
    ]
    cols = [WC["A"], "#f39c12", "#e74c3c"]
    pos = np.arange(len(data))

    vp = ax.violinplot(data, positions=pos,
                       showmeans=False, showmedians=False,
                       showextrema=False)
    for body, c in zip(vp["bodies"], cols):
        body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.22)
    bp = ax.boxplot(data, positions=pos, widths=0.22,
                    patch_artist=True, zorder=5)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.7)
        p.set_edgecolor("k"); p.set_linewidth(1.0)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color("k"); ln.set_linewidth(1.0)
    for ln in bp["medians"]:
        ln.set_color("white"); ln.set_linewidth(2.0)
    for ln in bp["fliers"]:
        ln.set_markerfacecolor("gray"); ln.set_markersize(5)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=15)

    for i, (d, c) in enumerate(zip(data, cols)):
        a = np.array(d)
        ax.annotate(f"{a.mean():.1f} $\\pm$ {a.std():.2f}",
                    xy=(i, a.max()), xytext=(0, 14),
                    textcoords="offset points", ha="center",
                    fontsize=14, color=c, fontweight="bold", bbox=BBOX)

    ax.set_ylabel(r"$\mathrm{EI}_g$  (nats)", fontsize=18)
    ax.set_title(
        r"$\theta$-Permutation Test: disrupting internal representations "
        r"$\Rightarrow$ $\mathrm{EI}_g$ drops",
        fontsize=19,
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  Figure 7 — Appendix: conditional independence analysis
# ══════════════════════════════════════════════════════════════

def plot_fig7(recs):
    """Residual analysis + narrow-bin stratification (appendix)."""
    from scipy.stats import f_oneway

    mi = np.array([r["MI_xy"] for r in recs])
    eig = np.array([r["EI_g"] for r in recs])
    mech = np.array([r["mechanism"] for r in recs])

    # quadratic regression EI_g ~ poly(MI, 2)
    z = np.polyfit(mi, eig, 2)
    eig_pred = np.polyval(z, mi)
    resid = eig - eig_pred

    fig = plt.figure(figsize=(22, 7.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.30)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── (a) residual scatter ──
    for m in MECH_ORDER:
        mask = mech == m
        ax_a.scatter(mi[mask], resid[mask], color=MECH_COLORS[m], s=32,
                     alpha=0.50, edgecolors="none", label=MECH_LABELS[m],
                     zorder=3)
    ax_a.axhline(0, color="gray", lw=0.8, ls=":")
    ax_a.set_xlabel(r"$\mathrm{MI}(x,y)$  (nats)", fontsize=17)
    ax_a.set_ylabel(r"Residual $\mathrm{EI}_g - \hat{\mathrm{EI}}_g$", fontsize=16)
    ax_a.set_title("(a) Regression residuals\n"
                    r"($\hat{\mathrm{EI}}_g$ = quadratic fit on MI)",
                    fontsize=17)
    ax_a.legend(fontsize=11, loc="lower right", ncol=2)

    # ── (b) residual violin by mechanism + ANOVA ──
    grp = {m: resid[mech == m] for m in MECH_ORDER}
    F_val, p_anova = f_oneway(*[grp[m] for m in MECH_ORDER])
    ss_b = sum(len(grp[m]) * (grp[m].mean() - resid.mean()) ** 2
               for m in MECH_ORDER)
    ss_t = np.sum((resid - resid.mean()) ** 2)
    eta_sq = ss_b / ss_t

    bdata = [grp[m] for m in MECH_ORDER]
    cols = [MECH_COLORS[m] for m in MECH_ORDER]
    pos = np.arange(len(MECH_ORDER))

    vp = ax_b.violinplot(bdata, positions=pos, showmeans=False,
                         showmedians=False, showextrema=False)
    for body, c in zip(vp["bodies"], cols):
        body.set_facecolor(c); body.set_edgecolor(c); body.set_alpha(0.20)
    bp = ax_b.boxplot(bdata, positions=pos, widths=0.24,
                      patch_artist=True, zorder=4)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.6)
        p.set_edgecolor("k"); p.set_linewidth(1.0)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color("k"); ln.set_linewidth(1.0)
    for ln in bp["medians"]:
        ln.set_color("white"); ln.set_linewidth(2.0)
    for ln in bp["fliers"]:
        ln.set_markerfacecolor("gray"); ln.set_markersize(4)

    rng_j = np.random.default_rng(77)
    for i, (m, d) in enumerate(zip(MECH_ORDER, bdata)):
        jit = rng_j.uniform(-0.12, 0.12, len(d))
        ax_b.scatter(pos[i] + jit, d, color=MECH_COLORS[m],
                     s=14, alpha=0.45, edgecolors="none", zorder=5)
        ax_b.annotate(f"$\\mu$={d.mean():.2f}\n$n$={len(d)}",
                      xy=(pos[i], d.max()), xytext=(0, 12),
                      textcoords="offset points", ha="center",
                      fontsize=12, color=MECH_COLORS[m],
                      fontweight="bold", bbox=BBOX)

    ax_b.axhline(0, color="gray", lw=0.8, ls=":")
    ax_b.set_xticks(pos)
    ax_b.set_xticklabels([MECH_LABELS[m] for m in MECH_ORDER],
                         fontsize=12, rotation=12, ha="right")
    ax_b.set_ylabel(r"Residual (nats)", fontsize=16)
    ax_b.set_title("(b) Residual by mechanism\n(ANOVA test)", fontsize=17)

    ax_b.text(0.03, 0.03,
              f"$F$ = {F_val:.1f},  $p$ = {p_anova:.1e}\n"
              f"Partial $\\eta^2$ = {eta_sq:.3f}",
              transform=ax_b.transAxes, fontsize=14, va="bottom",
              bbox=BBOX, fontweight="bold")

    # ── (c) narrow-bin stratification ──
    n_bins = 6
    mi_edges = np.percentile(mi, np.linspace(0, 100, n_bins + 1))
    mi_edges[0] -= 0.01
    mi_edges[-1] += 0.01
    bin_idx = np.digitize(mi, mi_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_centers = 0.5 * (mi_edges[:-1] + mi_edges[1:])
    bin_labels = [f"[{mi_edges[i]:.2f},\n{mi_edges[i+1]:.2f})"
                  for i in range(n_bins)]

    for m in MECH_ORDER:
        means, stds, ns = [], [], []
        for b in range(n_bins):
            mask = (bin_idx == b) & (mech == m)
            n_pts = mask.sum()
            if n_pts >= 2:
                means.append(eig[mask].mean())
                stds.append(eig[mask].std() / np.sqrt(n_pts))
                ns.append(n_pts)
            else:
                means.append(np.nan)
                stds.append(0)
                ns.append(0)
        means = np.array(means)
        stds = np.array(stds)
        valid = ~np.isnan(means)
        ax_c.plot(bin_centers[valid], means[valid], color=MECH_COLORS[m],
                  marker="o", lw=2.5, ms=8, label=MECH_LABELS[m], zorder=3)
        ax_c.fill_between(bin_centers[valid],
                          means[valid] - stds[valid],
                          means[valid] + stds[valid],
                          color=MECH_COLORS[m], alpha=0.12)

    ax_c.set_xlabel(r"$\mathrm{MI}(x,y)$ bin center (nats)", fontsize=17)
    ax_c.set_ylabel(r"Mean $\mathrm{EI}_g$ (nats)", fontsize=17)
    ax_c.set_title("(c) Stratified by MI quantile bin\n"
                    r"(mean $\pm$ SE, $\approx$83 pts/bin)",
                    fontsize=17)
    ax_c.legend(fontsize=12, loc="upper left")

    fig.suptitle(
        r"Appendix: $\mathrm{EI}_g$ carries mechanism-specific "
        r"information beyond $\mathrm{MI}(x,y)$",
        fontsize=20, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    print(f"    ANOVA: F={F_val:.1f}, p={p_anova:.2e}, "
          f"partial eta^2={eta_sq:.4f}")
    return fig


# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

def main():
    out = CODE / "output"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("Loading pre-computed data ...")
    d1b = load_exp1b()
    bd = d1b["boxplot"]
    sw = d1b["sweeps"]
    rm = load_real_models()
    print(f"  Real models: {list(rm.keys())}")
    for name, vals in rm.items():
        print(f"    {name}: EI_g={vals['EI_g']:.2f}  "
              f"log_VI={vals['log_VI']:.2f}  l_mean={vals['l_mean']:.2f}")

    # ── new computations ──
    t0 = time.time()
    wide = compute_scatter_wide()
    print(f"  Wide sampling: {len(wide)} pts ({time.time()-t0:.0f}s)")

    t0 = time.time()
    perm = compute_perm_test()
    print(f"  Perm test done ({time.time()-t0:.0f}s)")

    # ── figures ──
    print("\nGenerating figures ...")
    fig_defs = [
        (plot_fig1, (bd,), "fig1_worlds"),
        (plot_fig2, (sw,), "fig2_sweeps"),
        (plot_fig3, (sw, bd, rm), "fig3_phase"),
        (plot_fig4, (sw,), "fig4_normalized"),
        (plot_fig5, (wide,), "fig5_scatter"),
        (plot_fig6, (perm, bd), "fig6_permutation"),
        (plot_fig7, (wide,), "fig7_appendix"),
    ]
    n_figs = len(fig_defs)
    for i, (fn, args, name) in enumerate(fig_defs, 1):
        f = fn(*args)
        p = figs / f"exp1c_{name}.png"
        f.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(f)
        print(f"  [{i}/{n_figs}] {p.name}")

    # ── save JSON ──
    save = {
        "scatter_wide": wide,
        "permutation_test": perm,
        "real_models": rm,
    }
    jp = out / "exp1c_extended.json"
    with jp.open("w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  JSON -> {jp}")
    print("  Done.")


if __name__ == "__main__":
    main()
