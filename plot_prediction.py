import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
FIG_DIR_ROLLING = os.path.join(FIG_DIR, "rolling_forecast")
FIG_DIR_ABLATION = os.path.join(FIG_DIR, "ablation")
for _d in (FIG_DIR, FIG_DIR_ROLLING, FIG_DIR_ABLATION):
    os.makedirs(_d, exist_ok=True)

COL_WIDTH = 3.5
FULL_WIDTH = 7.16
COL_SPAN = 1
FIG_WIDTH = COL_WIDTH if COL_SPAN == 1 else FULL_WIDTH
FIG_HEIGHT = 2.0
FIG_DPI = 600
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "legend.fontsize": 5,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "lines.linewidth": 0.7,
    }
)

SEQ_LEN = 16
SEED = 1  # seed used for the forecast overlay
WINDOW = 150  # consecutive forecast steps drawn
XMAX = 150  # fixed right edge of the rolling-plot x-axis
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
PRED_LENS = [1, 5, 16, 32, 48]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MODELS_ORDER = [
    "MTSF-ANO-comb3local",
    "MTSF-ANO-comb5local",
    "LSTM",
    "QuLTSF",
    "MTSF-PZ",
    "QFWP",
    "QLSTM",
    "DLinear",
    "NLinear",
    "Linear",
]
CHANNEL = -1  # target variable (last column)
DENORM = False  # keep standardized values on the y-axis

# Raw CSVs
DATASET_CSV = {
    "ETTh1": "dataset/ETT-small/ETTh1.csv",
    "ETTh2": "dataset/ETT-small/ETTh2.csv",
    "ETTm1": "dataset/ETT-small/ETTm1.csv",
    "ETTm2": "dataset/ETT-small/ETTm2.csv",
}
_DENORM_CACHE = {}


def target_stats(dataset):
    """(mean, std) of the target column over the train split (mirrors data_provider's
    StandardScaler), for inverting standardized values: real = std * x + mean."""
    if dataset in _DENORM_CACHE:
        return _DENORM_CACHE[dataset]
    rel = DATASET_CSV.get(dataset)
    if rel is None:
        return (0.0, 1.0)
    df = pd.read_csv(os.path.join(ROOT, rel))
    col = df.columns[-1]
    if dataset in ("ETTh1", "ETTh2"):
        n_train = 12 * 30 * 24
    elif dataset in ("ETTm1", "ETTm2"):
        n_train = 12 * 30 * 24 * 4
    else:
        n_train = int(len(df) * 0.7)
    s = df[col].values[:n_train].astype(float)
    mean, std = float(s.mean()), float(s.std(ddof=0))  # population std (sklearn)
    _DENORM_CACHE[dataset] = (mean, std if std else 1.0)
    return _DENORM_CACHE[dataset]


# Per-model line colors; ground truth stays neutral gray.
COLOR = {
    "GT": "#6f6f6f",
    "DLinear": "#1F30E8",
    "LSTM": "#FF7F00",
    "QuLTSF": "#4DAF4A",
    "QFWP": "#F781BF",
    "QLSTM": "#984EA3",
    "NLinear": "#4EC1EA",
    "Linear": "#999999",
    "MTSF-PZ": "#9D2727",
    "MTSF-ANO-comb3local": "#E41A1C",
    "MTSF-ANO-comb5local": "#BC05F4",
    "MTSF-ANO-CI-comb3local": "#1B9E77",
    "MTSF-ANO-CI-comb5local": "#A6761D",
    "MTSF-ANO-bestk": "#E41A1C",
    "MTSF-ANO-CI-bestk": "#1B9E77",
}

# Legend labels that differ from the result-dir model name.
LEGEND_ALIAS = {
    "MTSF-ANO-bestk": "MTSF-ANO",
    "MTSF-ANO-CI-bestk": "MTSF-ANO-CI",
}


def disp_name(m):
    if m in LEGEND_ALIAS:
        return LEGEND_ALIAS[m]
    mobj = re.fullmatch(r"(MTSF-ANO(?:-CI)?)-comb(\d+)local", m)
    return f"{mobj.group(1)} ({mobj.group(2)}-local)" if mobj else m


def find_setting(impl, dataset, pred_len, seed=SEED):
    """Run directory results/<impl>/<dataset>_L<L>_H<pred_len>_s<seed>, or None."""
    hits = glob.glob(os.path.join(RESULTS, impl, f"{dataset}_L*_H{pred_len}_s{seed}"))
    return hits[0] if hits else None


def load_npy(setting_dir, name):
    try:
        return np.load(os.path.join(setting_dir, name))
    except FileNotFoundError:
        return None


def collect_preds(dataset, pred_len):
    gt_full = None
    out = []
    for m in MODELS_ORDER:
        d = find_setting(m, dataset, pred_len)
        if not d:
            continue
        p = load_npy(d, "pred.npy")  # [n, pred_len, C]
        if p is None:
            continue
        if gt_full is None:
            t = load_npy(d, "true.npy")  # identical GT across models
            if t is not None:
                gt_full = t[:, 0, CHANNEL]
        out.append((m, p[:, :, CHANNEL]))
    return gt_full, out


PROPOSED = ["MTSF-ANO-comb3local", "MTSF-ANO-comb5local"]
BASELINES = ["LSTM", "QuLTSF", "MTSF-PZ", "QFWP", "QLSTM", "DLinear"]
ALWAYS_SHOW = ["MTSF-PZ"]  # baselines drawn even when outside the top-3


def seed_mses(model, dataset, pred_len):
    out = {}
    for seed in SEEDS:
        d = find_setting(model, dataset, pred_len, seed=seed)
        if not d:
            continue
        fp = os.path.join(d, "metrics.npz")
        if os.path.isfile(fp):
            z = np.load(fp)
            if "mse" in z.files:
                out[seed] = float(z["mse"])
    return out


def mean_mse(model, dataset, pred_len):
    mses = list(seed_mses(model, dataset, pred_len).values())
    return float(np.mean(mses)) if mses else None


def _agg_metrics(model, ds, pl):
    """(mse, mae, rmse, n_seeds) averaged over available seeds, or None."""
    mses, maes, rmses = [], [], []
    for seed in SEEDS:
        d = find_setting(model, ds, pl, seed=seed)
        if not d:
            continue
        fp = os.path.join(d, "metrics.npz")
        if not os.path.isfile(fp):
            continue
        z = np.load(fp)
        if "mse" in z.files:
            mses.append(float(z["mse"]))
        if "mae" in z.files:
            maes.append(float(z["mae"]))
        if "rmse" in z.files:
            rmses.append(float(z["rmse"]))
        elif "mse" in z.files:
            rmses.append(float(np.sqrt(z["mse"])))
    if not mses:
        return None
    return (
        float(np.mean(mses)),
        float(np.mean(maes)),
        float(np.mean(rmses)),
        len(mses),
    )


def mean_mse_full(model, dataset, pred_len, n_seeds=None):
    n_seeds = n_seeds if n_seeds is not None else len(SEEDS)
    d = seed_mses(model, dataset, pred_len)
    return float(np.mean(list(d.values()))) if len(d) >= n_seeds else None


def best_models(dataset, pred_len, n_base=3):
    def rank(pool):
        scored = [
            (m, mse)
            for m in pool
            if (mse := mean_mse(m, dataset, pred_len)) is not None
        ]
        scored.sort(key=lambda t: t[1])
        return [m for m, _ in scored]

    keep = rank(PROPOSED)[:1] + rank(BASELINES)[:n_base]
    for m in rank(ALWAYS_SHOW):
        if m not in keep:
            keep.append(m)
    return keep


def _finalize(fig, ax, out_path):
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    handles, labels = ax.get_legend_handles_labels()
    is_top = [l == "Ground truth" or l.startswith("MTSF-ANO") for l in labels]
    order = [i for i, t in enumerate(is_top) if t] + [
        i for i, t in enumerate(is_top) if not t
    ]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    legend_artists = [
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            frameon=False,
            handlelength=1.4,
            columnspacing=1.0,
            handletextpad=0.4,
        )
    ]
    ax.grid(alpha=0.25, lw=0.3)
    fig.tight_layout(pad=0.2)
    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        bbox_extra_artists=legend_artists,
    )
    plt.close(fig)
    return out_path


def plot_one(dataset, pred_len, horizon=None, suffix=""):
    horizon = (pred_len - 1) if horizon is None else horizon  # default: hardest step
    gt_full, preds = collect_preds(dataset, pred_len)
    if gt_full is None or not preds:
        return None
    keep = best_models(dataset, pred_len)
    preds = [(m, p) for m, p in preds if m in keep]
    h = horizon
    n = len(gt_full)
    mean, std = target_stats(dataset) if DENORM else (0.0, 1.0)

    def dn(a):  # standardized -> real units
        return np.asarray(a) * std + mean

    W = min(WINDOW, n - SEQ_LEN - pred_len + 1)  # forecast points that fit
    start = max(SEQ_LEN, (n - W) // 2)
    start = min(start, n - W - pred_len + 1)

    gt_seg = dn(gt_full[start - SEQ_LEN : start + W + pred_len - 1])
    x_gt = np.arange(len(gt_seg))
    fc_samples = np.arange(start, start + W)
    x_fc = np.arange(SEQ_LEN + h, SEQ_LEN + h + W)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.axvline(
        SEQ_LEN, color="tab:blue", ls=":", lw=0.75, ymin=0.06, ymax=0.94, zorder=1
    )
    if h > 0:  # forecast-start line (coincides with L for H=1)
        ax.axvline(
            SEQ_LEN + h,
            color="tab:red",
            ls=":",
            lw=0.75,
            ymin=0.06,
            ymax=0.94,
            zorder=1,
        )
    ax.plot(x_gt, gt_seg, color=COLOR["GT"], lw=0.75, label="Ground truth", zorder=3)
    for m, p in preds:
        y = dn(p[fc_samples, h])
        ax.plot(
            x_fc,
            y,
            color=COLOR.get(m, None),
            lw=0.75,
            alpha=0.85,
            label=disp_name(m),
            zorder=2,
        )

    ax.set_xlim(0, XMAX)
    # mark x=L and x=L+h, dropping default ticks that crowd x=L
    marks = {SEQ_LEN} | ({SEQ_LEN + h} if h > 0 else set())
    base = [t for t in ax.get_xticks() if 0 <= t <= XMAX]
    xt = [t for t in base if abs(t - SEQ_LEN) >= 8 or t in marks]
    ax.set_xticks(sorted(set(xt) | marks))
    out = os.path.join(
        FIG_DIR_ROLLING, f"{dataset}_L{SEQ_LEN}_h{pred_len}_s{SEED}{suffix}.png"
    )
    return _finalize(fig, ax, out)


def _ablation_heatmap(
    data,
    row_labels,
    col_labels,
    out_path,
    fmt="{:.0f}",
    cmap="viridis",
    cbar_label="",
    col_norm=False,
    best="max",
    vmin=None,
    vmax=None,
    row_group_at=None,
    col_group_at=None,
    col_super=None,
    cell_w=0.62,
    cell_h=0.42,
    ax=None,
    annot=None,
    add_cbar=True,
    title=None,
    ylabel=None,
    xticks_top=False,
    super_y=-0.95,
    best_cells=None,
    bold_best=True,
):
    data = np.asarray(data, dtype=float)
    nr, nc = data.shape
    masked = np.ma.masked_invalid(data)
    if annot is not None:
        annot = np.asarray(annot, dtype=object)

    if col_norm:
        cval = np.full_like(data, np.nan)
        for j in range(nc):
            col = data[:, j]
            finite = col[np.isfinite(col)]
            if finite.size:
                lo, hi = finite.min(), finite.max()
                span = (hi - lo) or 1.0
                norm = (col - lo) / span
                cval[:, j] = norm if best == "max" else 1.0 - norm  # best = brightest
        color = np.ma.masked_invalid(cval)
        cvmin, cvmax = 0.0, 1.0
    else:
        color = masked
        cvmin, cvmax = vmin, vmax

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(cell_w * nc + 1.4, cell_h * nr + 1.2))
    else:
        fig = ax.figure
    im = ax.imshow(color, cmap=cmap, aspect="auto", vmin=cvmin, vmax=cvmax)

    best_rc = set()  # cells to box+bold
    if best_cells is not None:
        best_rc = set(best_cells)
    elif best in ("max", "min"):
        for j in range(nc):
            col = masked[:, j]
            if col.count() == 0:
                continue
            i = int(np.ma.argmax(col) if best == "max" else np.ma.argmin(col))
            best_rc.add((i, j))

    for i in range(nr):
        for j in range(nc):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(
                    j,
                    i,
                    annot[i, j] if annot is not None else "-",
                    ha="center",
                    va="center",
                    color="#999999",
                    fontsize=6,
                )
                continue
            is_best = (i, j) in best_rc
            s = annot[i, j] if annot is not None else fmt.format(v)
            r, g, b, _ = im.to_rgba(color[i, j])  # text color from cell luminance
            txt_c = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
            ax.text(
                j,
                i,
                s,
                ha="center",
                va="center",
                color=txt_c,
                fontweight="bold" if (is_best and bold_best) else "normal",
                fontsize=6,
            )
            if is_best:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        lw=1.2,
                        zorder=5,
                    )
                )

    ax.set_xticks(range(nc))
    ax.set_yticks(range(nr))
    ax.set_xticklabels(col_labels, fontsize=5)
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.tick_params(length=0)
    if xticks_top:
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=6)
    if title:
        ax.set_title(title, fontsize=7, fontweight="bold", pad=22 if col_super else 8)

    # cell gridlines + heavier group-boundary rules
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    for r in row_group_at or []:
        ax.axhline(r - 0.5, color="black", lw=1.1)
    for c in col_group_at or []:
        ax.axvline(c - 0.5, color="black", lw=1.1)

    if col_super:
        for label, c0, c1 in col_super:
            ax.text(
                (c0 + c1) / 2.0,
                super_y,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
            )

    if not add_cbar:
        return im

    # arrow the bar when the scale is clipped past the true range
    has = masked.count() > 0
    under = not col_norm and vmin is not None and has and masked.min() < vmin
    over = not col_norm and vmax is not None and has and masked.max() > vmax
    extend = (
        "both" if under and over else "min" if under else "max" if over else "neither"
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend=extend)
    cb.set_label(cbar_label, fontsize=5)
    cb.ax.tick_params(labelsize=4)
    if col_norm:
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["worst", "best"])

    if not own_fig:
        return im

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


# Ablation arms are pinned to ETTh1.
ABLATION_DATASET = "ETTh1"


def _arm_name(k, layer, scaling, entanglement):
    name = f"MTSF-ANO-comb{k}local"
    if layer != 3:
        name += f"-L{layer}"
    if not scaling:
        name += "-fixscale"
    if entanglement == "none":
        name += "-noent"
    return name


_LOCALITY_KS = [1, 2, 3, 4, 5, 6, 7]
_LOCALITY_HORIZONS = [1, 5, 16, 32, 48]
_LOCALITY_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def _locality_rel_matrix(dataset):
    data = np.full((1 + len(_LOCALITY_KS), len(_LOCALITY_HORIZONS)), np.nan)
    for j, H in enumerate(_LOCALITY_HORIZONS):
        base = mean_mse_full("MTSF-PZ", dataset, H)
        if base is None:  # no full MTSF-PZ run -> can't rescale this column
            continue
        data[0, j] = 0.0  # MTSF-PZ row -> 0%
        for i, k in enumerate(_LOCALITY_KS):
            mse = mean_mse_full(f"MTSF-ANO-comb{k}local", dataset, H)
            if mse is not None:
                data[i + 1, j] = (
                    100.0 * (base - mse) / base
                )  # % MSE reduction vs MTSF-PZ
    return data


def _plot_locality_heatmap(data, out_path, cbar_label, title=None):
    finite = data[np.isfinite(data)]
    floor = float(np.nanpercentile(finite, 10)) if finite.size else None
    return _ablation_heatmap(
        data,
        row_labels=[disp_name("MTSF-PZ")] + [f"$k={k}$" for k in _LOCALITY_KS],
        col_labels=[f"$H={h}$" for h in _LOCALITY_HORIZONS],
        out_path=out_path,
        fmt="{:.1f}%",
        cmap="YlGnBu",
        cbar_label=cbar_label,
        col_norm=False,
        best="max",
        vmin=floor,
        vmax=None,
        row_group_at=[1],
        title=title,
    )


def plot_ablation_locality(dataset=ABLATION_DATASET, out_path=None):
    data = _locality_rel_matrix(dataset)
    return _plot_locality_heatmap(
        data,
        out_path or os.path.join(FIG_DIR_ABLATION, f"{dataset}_ablation_locality.png"),
        "rel. improvement vs. MTSF-PZ (%)",
    )


def plot_ablation_locality_avg(datasets=None, out_path=None):
    datasets = datasets or _LOCALITY_DATASETS
    mats = [_locality_rel_matrix(ds) for ds in datasets]
    data = np.full_like(mats[0], np.nan)
    for idx in np.ndindex(data.shape):
        vals = [m[idx] for m in mats]
        if all(np.isfinite(v) for v in vals):
            data[idx] = float(np.mean(vals))
    return _plot_locality_heatmap(
        data,
        out_path or os.path.join(FIG_DIR_ABLATION, "ablation_locality.png"),
        "mean rel. improvement vs. MTSF-PZ (%)",
        title="Locality ablation · mean over " + ", ".join(datasets),
    )


_ARCH_KS = [3, 5]
_ARCH_LAYERS = [1, 3, 5]  # VQC depth -> rows ("Layer")
# (column label, ENTANGLEMENT arm): w/ CNOT = entangled circuit, w/o CNOT = ENTANGLEMENT=none.
_ARCH_ENTS = [("w/ CNOT", "cnot"), ("w/o CNOT", "none")]
_ARCH_HORIZONS = [1, 48]  # one panel per H


def plot_ablation_architecture(dataset=ABLATION_DATASET, out_path=None):
    out_path = out_path or os.path.join(
        FIG_DIR_ABLATION, f"{dataset}_ablation_architecture.png"
    )
    col_specs = [
        (k, ent_label, ent_val) for k in _ARCH_KS for ent_label, ent_val in _ARCH_ENTS
    ]
    n_ent = len(_ARCH_ENTS)

    fig, axes = plt.subplots(
        len(_ARCH_HORIZONS), 1, figsize=(FIG_WIDTH, 1.7 * len(_ARCH_HORIZONS) + 0.4)
    )
    axes = np.atleast_1d(axes)
    for ax, H in zip(axes, _ARCH_HORIZONS):
        color = np.full((len(_ARCH_LAYERS), len(col_specs)), np.nan)  # min of the pair
        annot = np.empty_like(color, dtype=object)
        pairs = {}  # (i, j) -> (MSE w/ scaling, MSE w/o scaling)
        for i, L in enumerate(_ARCH_LAYERS):
            for j, (k, _ent_label, ent_val) in enumerate(col_specs):
                vw = mean_mse_full(_arm_name(k, L, True, ent_val), dataset, H)
                vo = mean_mse_full(_arm_name(k, L, False, ent_val), dataset, H)
                pairs[(i, j)] = (vw, vo)
                p = [v for v in (vw, vo) if v is not None]
                if p:
                    color[i, j] = min(p)

        # Boldface convention from the table: within each (layer, k) block (two cells =
        # four numbers) bold only the single minimum value (ties all bold) via mathtext,
        # and box the cell(s) holding it.
        best_cells = set()
        for i in range(len(_ARCH_LAYERS)):
            for g in range(len(_ARCH_KS)):
                js = [g * n_ent + e for e in range(n_ent)]
                vals = [round(v, 3) for j in js for v in pairs[(i, j)] if v is not None]
                mn = min(vals) if vals else None

                def num(v, mn=mn):
                    if v is None:
                        return "-"
                    s = f"{v:.3f}"
                    return (
                        rf"$\mathbf{{{s}}}$"
                        if mn is not None and round(v, 3) == mn
                        else s
                    )

                for j in js:
                    vw, vo = pairs[(i, j)]
                    annot[i, j] = f"{num(vw)} | {num(vo)}"
                    if mn is not None and any(
                        v is not None and round(v, 3) == mn for v in (vw, vo)
                    ):
                        best_cells.add((i, j))

        _ablation_heatmap(
            color,
            ax=ax,
            annot=annot,
            add_cbar=True,
            best_cells=best_cells,
            bold_best=False,
            row_labels=[str(L) for L in _ARCH_LAYERS],
            col_labels=[ent_label for _, ent_label, _ in col_specs],
            out_path=None,
            cmap="YlGnBu",
            cbar_label="MSE (better of w/ scaling, w/o scaling)",
            col_norm=True,
            best="min",
            col_group_at=[n_ent],
            col_super=[
                (f"$k={k:d}$", g * n_ent, (g + 1) * n_ent - 1)
                for g, k in enumerate(_ARCH_KS)
            ],
            title=f"$H={H}$",
            ylabel="Layer",
            xticks_top=True,
            super_y=-1.35,
        )

    fig.tight_layout(h_pad=2.0)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


# Lookback ablation (MSE vs L on ETTh1, H=48). L=16 runs live in results/<model>/,
# the others in results/<model>-lookback<L>/.
ABLATION_LOOKBACK_DATASET = "ETTh1"
ABLATION_LOOKBACK_H = 48
ABLATION_LOOKBACK_LS = [8, 16, 24, 32, 48, 72, 96]
ABLATION_LOOKBACK_CI = ["MTSF-ANO-CI-comb3local", "MTSF-ANO-CI-comb5local"]
ABLATION_LOOKBACK_KS = [3, 5]


def _lookback_impl(model, L):
    """Results directory name of `model` at lookback L."""
    return model if L == 16 else f"{model}-lookback{L}"


def _lookback_xy(spec, dataset, H, Ls, pos):
    xs, ys = [], []
    for L in Ls:
        if isinstance(spec, tuple):  # ('bestk', family) -> min over available k
            cand = [
                mean_mse_full(_lookback_impl(f"{spec[1]}-comb{k}local", L), dataset, H)
                for k in ABLATION_LOOKBACK_KS
            ]
            mse = min((v for v in cand if v is not None), default=None)
        else:
            mse = mean_mse_full(_lookback_impl(spec, L), dataset, H)
        if mse is not None:
            xs.append(pos[L])
            ys.append(mse)
    return xs, ys


def plot_ablation_lookback(
    dataset=ABLATION_LOOKBACK_DATASET,
    H=ABLATION_LOOKBACK_H,
    Ls=None,
    best_k=False,
    out_path=None,
):
    Ls = ABLATION_LOOKBACK_LS if Ls is None else Ls
    baselines = [m for m in MODELS_ORDER if not m.startswith("MTSF-ANO-comb")]
    if best_k:
        # One curve per ANO family: the best k at each L.
        series = [
            ("MTSF-ANO-bestk", ("bestk", "MTSF-ANO")),
            ("MTSF-ANO-CI-bestk", ("bestk", "MTSF-ANO-CI")),
        ] + [(m, m) for m in baselines]
    else:
        series = [(m, m) for m in MODELS_ORDER + ABLATION_LOOKBACK_CI]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]
    pos = {L: i for i, L in enumerate(Ls)}  # L -> equal-spaced x index

    fig, (ax, axz) = plt.subplots(
        2,
        1,
        figsize=(FIG_WIDTH, 3.3),
        gridspec_kw={"hspace": 0.30, "height_ratios": [1, 0.72]},
    )
    drawn = 0
    all_y = []
    for mi, (key, spec) in enumerate(series):
        xs, ys = _lookback_xy(spec, dataset, H, Ls, pos)
        if not xs:
            continue
        drawn += 1
        all_y.extend(ys)
        mk, col = markers[mi % len(markers)], COLOR.get(key, None)
        ax.plot(
            xs,
            ys,
            marker=mk,
            markersize=2.3,
            color=col,
            lw=0.8,
            markeredgewidth=0.0,
            label=disp_name(key),
        )  # full range (top)
        axz.plot(
            xs, ys, marker=mk, markersize=3.2, color=col, lw=1.0, markeredgewidth=0.0
        )  # same curve, zoomed
    if drawn == 0:
        plt.close(fig)
        print("plot_ablation_lookback: no completed arms found, skipped")
        return None

    ymin, ymax = min(all_y), max(all_y)
    yspan = ymax - ymin
    zoom_Ls = [L for L in Ls if L >= 24]
    lm, rm = 0.15, 0.42  # asymmetric top-panel x-margins so the funnel splays right
    pad_x = 0.2  # box padding so its edges miss the markers
    rx0, rx1 = pos[zoom_Ls[0]], pos[zoom_Ls[-1]]
    bx0, bx1 = rx0 - pad_x, rx1 + pad_x
    # zoom band hand-tuned to ETTh1's converged low-MSE cluster (so it isn't squashed)
    ymin_z, ymax_z = 0.34, 0.395
    ax.set_xlim(-lm, (len(Ls) - 1) + rm)
    ax.set_ylim(ymin - 0.05 * yspan, ymax + 0.08 * yspan)
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([str(L) for L in Ls])
    axz.set_xlim(bx0, bx1)
    axz.set_ylim(ymin_z, ymax_z)
    axz.set_xticks([pos[L] for L in zoom_Ls])
    axz.set_xticklabels([str(L) for L in zoom_Ls])
    axz.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for a in (ax, axz):
        a.yaxis.set_major_locator(MaxNLocator(nbins=4))
        a.tick_params(length=2, pad=1.5)
        a.grid(alpha=0.22, lw=0.3)
        for side in ("top", "right"):
            a.spines[side].set_visible(True)
    axz.set_yticks([0.34, 0.35, 0.36, 0.37, 0.38, 0.39])

    # zoom funnel: red box on the top panel + two connectors to the zoom panel corners
    zoom_ec = "#d62728"
    ax.add_patch(
        plt.Rectangle(
            (bx0, ymin_z),
            bx1 - bx0,
            ymax_z - ymin_z,
            facecolor="none",
            edgecolor=zoom_ec,
            lw=0.6,
            zorder=6,
        )
    )
    for xr in (bx0, bx1):
        fig.add_artist(
            ConnectionPatch(
                xyA=(xr, ymin_z),
                coordsA=ax.transData,
                xyB=(xr, ymax_z),
                coordsB=axz.transData,
                color=zoom_ec,
                lw=0.6,
                alpha=0.9,
                zorder=6,
            )
        )

    # legend: MTSF-ANO first, baselines after; two evenly split rows above the top axes
    handles, labels = ax.get_legend_handles_labels()
    is_top = [l.startswith("MTSF-ANO") for l in labels]
    order = [i for i, t in enumerate(is_top) if t] + [
        i for i, t in enumerate(is_top) if not t
    ]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    ncol = (len(labels) + 1) // 2
    legend_artists = [
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=ncol,
            frameon=False,
            handlelength=1.2,
            columnspacing=0.9,
            handletextpad=0.35,
            labelspacing=0.25,
            borderaxespad=0.1,
            markerscale=1.5,
        )
    ]
    tag = "_bestk" if best_k else ""
    out = out_path or os.path.join(
        FIG_DIR_ABLATION, f"{dataset}_ablation_lookback{tag}.png"
    )
    fig.tight_layout(pad=0.1)
    fig.savefig(
        out,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        bbox_extra_artists=legend_artists,
    )
    plt.close(fig)
    print(f"wrote {out}")
    return out


def write_res_txt(out_path=None):
    """Write mean metrics over seeds per (model, dataset, pred_len) to res.txt."""
    out_path = out_path or os.path.join(ROOT, "res.txt")
    rows = {
        (m, ds, pl): _agg_metrics(m, ds, pl)
        for m in MODELS_ORDER
        for ds in DATASETS
        for pl in PRED_LENS
    }

    lines = [
        (
            "# values are mean over available seeds; "
            "columns: model<TAB>dataset<TAB>pred_len<TAB>mse<TAB>mae<TAB>rmse"
        ),
        "",
        "# Best model per (dataset, pred_len) ranked by MSE",
    ]
    for ds in DATASETS:
        for pl in PRED_LENS:
            ranked = sorted(
                ((m, rows[(m, ds, pl)]) for m in MODELS_ORDER if rows.get((m, ds, pl))),
                key=lambda t: t[1][0],
            )[:4]
            if not ranked:
                lines.append(f"# {ds} T={pl}: (no results)")
                continue
            parts = [
                f"{i + 1}. {m} mse={v[0]:.3f} mae={v[1]:.3f}"
                for i, (m, v) in enumerate(ranked)
            ]
            lines.append(f"# {ds} T={pl}: " + "  |  ".join(parts))
    lines.append("")

    for m in MODELS_ORDER:
        for ds in DATASETS:
            for pl in PRED_LENS:
                v = rows.get((m, ds, pl))
                if v:
                    mse, mae, rmse, nseeds = v
                    lines.append(
                        f"{m}\t{ds}\t{pl}\t{mse:.3f}\t{mae:.3f}\t{rmse:.3f}\t(n={nseeds})"
                    )
                else:
                    lines.append(f"{m}\t{ds}\t{pl}\tNA\tNA\tNA")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


_ANO_KS = [1, 2, 3, 4, 5, 6, 7]


def write_res_best_txt(out_path=None):
    out_path = out_path or os.path.join(ROOT, "res_best.txt")
    ano_arms = [f"MTSF-ANO-comb{k}local" for k in _ANO_KS]
    others = [m for m in MODELS_ORDER if not m.startswith("MTSF-ANO-comb")]

    rows = {}
    winner = {}  # (ds, pl) -> best MTSF-ANO arm
    for ds in DATASETS:
        for pl in PRED_LENS:
            best = None
            for m in ano_arms:
                v = _agg_metrics(m, ds, pl)
                if v and (best is None or v[0] < best[1][0]):
                    best = (m, v)
            if best is not None:
                winner[(ds, pl)] = best[0]
                rows[("MTSF-ANO-best", ds, pl)] = best[1]
            for m in others:
                rows[(m, ds, pl)] = _agg_metrics(m, ds, pl)

    order = ["MTSF-ANO-best"] + others
    lines = [
        (
            "# MTSF-ANO locality family collapsed to its best (min mean-MSE) default-L3 arm "
            "per (dataset, pred_len); other models shown individually."
        ),
        "# columns: model<TAB>dataset<TAB>pred_len<TAB>mse<TAB>mae<TAB>rmse",
        "",
        "# Best model per (dataset, pred_len) ranked by MSE",
    ]
    for ds in DATASETS:
        for pl in PRED_LENS:
            ranked = sorted(
                ((m, rows[(m, ds, pl)]) for m in order if rows.get((m, ds, pl))),
                key=lambda t: t[1][0],
            )[:4]
            if not ranked:
                lines.append(f"# {ds} T={pl}: (no results)")
                continue
            parts = [
                f"{i + 1}. {m} mse={v[0]:.3f} mae={v[1]:.3f}"
                for i, (m, v) in enumerate(ranked)
            ]
            lines.append(f"# {ds} T={pl}: " + "  |  ".join(parts))
    lines.append("")
    lines.append("# best MTSF-ANO locality arm per (dataset, pred_len)")
    for ds in DATASETS:
        for pl in PRED_LENS:
            w = winner.get((ds, pl))
            lines.append(f"# {ds} T={pl}: " + (w if w else "(none)"))
    lines.append("")

    for m in order:
        for ds in DATASETS:
            for pl in PRED_LENS:
                v = rows.get((m, ds, pl))
                tag = (
                    f"  [{winner[(ds, pl)]}]"
                    if m == "MTSF-ANO-best" and (ds, pl) in winner
                    else ""
                )
                if v:
                    mse, mae, rmse, nseeds = v
                    lines.append(
                        f"{m}\t{ds}\t{pl}\t{mse:.3f}\t{mae:.3f}\t{rmse:.3f}\t(n={nseeds}){tag}"
                    )
                else:
                    lines.append(f"{m}\t{ds}\t{pl}\tNA\tNA\tNA")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    written = []
    for ds in DATASETS:
        for pl in PRED_LENS:
            f = plot_one(ds, pl)
            if f:
                written.append(f)
    for ds in _LOCALITY_DATASETS:
        written.append(plot_ablation_locality(ds))
    written.append(plot_ablation_locality_avg())
    written.append(plot_ablation_architecture())
    written.append(plot_ablation_lookback())
    written.append(plot_ablation_lookback(best_k=True))
    for f in written:
        print(f)
    print(f"Total figures: {len(written)}")
    write_res_txt()
    write_res_best_txt()
