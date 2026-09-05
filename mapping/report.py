"""Merge per-tile statistics of a colorization run into out/<tag>/report.md + PNG plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics
from .config import OUT_DIR

VARIANTS = ("med", "nt", "nt_noocc")
VARIANT_LABEL = {"med": "median top-5 (product)", "nt": "nearest in time, occlusion", "nt_noocc": "nearest in time, no occlusion"}


def load_run(tag: str, out_dir: Path = OUT_DIR) -> tuple[dict, dict, dict]:
    stats_dir = Path(out_dir) / tag / "stats"
    metas = {p.stem[:3]: json.loads(p.read_text()) for p in stats_dir.glob("*_meta.json")}
    hists = {}
    for v in VARIANTS + tuple(f"{x}_76" for x in VARIANTS):
        h = metrics.StrataHist()
        per_tile = {}
        for p in sorted(stats_dir.glob(f"*_{v}.npz")):
            t = metrics.StrataHist.load(p)
            h.merge(t)
            per_tile[p.stem[:3]] = t
        hists[v] = (h, per_tile)
    samples = []
    for p in sorted(stats_dir.glob("*_sample.npz")):
        with np.load(p) as z:
            samples.append({k: z[k] for k in z.files})
    sample = {k: np.concatenate([s[k] for s in samples]) for k in samples[0]} if samples else {}
    return metas, hists, sample


def _fmt(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "| – |" * 1
    return f"{s['median']:.2f} | {s['mad']:.2f} | {s['p25']:.2f} | {s['p75']:.2f} | {s['p90']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | {s['pct_gt_20']:.1f} % | {s['pct_lt_5']:.1f} %"


def _table_by(h: metrics.StrataHist, family: str, labels, min_n: int = 1000) -> str:
    rows = ["| stratum | n | median | MAD | P25 | P75 | P90 | P95 | P99 | >20 | <5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, lab in enumerate(labels):
        s = metrics.hist_stats(h.h[family][i])
        if s["n"] < min_n:
            continue
        rows.append(f"| {lab} | {s['n']:,} | {_fmt(s)} |")
    return "\n".join(rows)


def _plot_hist(hists, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    centres = (np.arange(metrics.N_DE_BINS) + 0.5) * metrics.DE_BIN
    for v in VARIANTS:
        c = hists[v][0].total_hist()
        if c.sum():
            ax.plot(centres, c / c.sum() / metrics.DE_BIN, label=VARIANT_LABEL[v])
    ax.set_xscale("log")
    ax.set_xlim(0.3, 100)
    ax.set_xlabel("ΔE00 vs TerraScan")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_family(hists, family: str, x, xlabel: str, path: Path, min_n: int = 1000):
    fig, ax = plt.subplots(figsize=(8, 4))
    for v in VARIANTS:
        h = hists[v][0]
        med = []
        p90 = []
        for i in range(len(x)):
            s = metrics.hist_stats(h.h[family][i])
            med.append(s["median"] if s["n"] >= min_n else np.nan)
            p90.append(s["p90"] if s["n"] >= min_n else np.nan)
        ax.plot(x, med, "-o", ms=3, label=f"{VARIANT_LABEL[v]} median")
        ax.plot(x, p90, ":", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ΔE00 (solid median, dotted P90)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(tag: str, out_dir: Path = OUT_DIR) -> Path:
    metas, hists, sample = load_run(tag, out_dir)
    rd = Path(out_dir) / tag
    n_total = sum(m["n"] for m in metas.values())
    lines = [f"# Colorization run `{tag}`", "", f"Tiles: {len(metas)}, points: {n_total:,}, wall time Σ {sum(m['seconds'] for m in metas.values())/60:.1f} min (per-tile, before parallelism).", ""]

    lines += ["## Coverage", "", "| variant | coloured share |", "|---|---|"]
    for v in VARIANTS:
        cov = sum(m["coverage"][v] * m["n"] for m in metas.values()) / n_total
        lines.append(f"| {VARIANT_LABEL[v]} | {100*cov:.2f} % |")
    lines.append("")

    lines += ["## ΔE00 vs TerraScan RGB (coloured points only)", "", "| variant | n | median | MAD | P25 | P75 | P90 | P95 | P99 | >20 | <5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        s = metrics.hist_stats(hists[v][0].total_hist())
        lines.append(f"| {VARIANT_LABEL[v]} | {s['n']:,} | {_fmt(s)} |")
    lines += ["", "CIE76 (comparable with the pilot's 6.08):", "", "| variant | median | P25 | P75 |", "|---|---|---|---|"]
    for v in VARIANTS:
        s = metrics.hist_stats(hists[v + "_76"][0].total_hist())
        lines.append(f"| {VARIANT_LABEL[v]} | {s.get('median', float('nan')):.2f} | {s.get('p25', float('nan')):.2f} | {s.get('p75', float('nan')):.2f} |")
    s_med = metrics.hist_stats(hists["med"][0].total_hist())
    ok = s_med["median"] < 5 and s_med["pct_gt_20"] < 5
    lines += ["", f"**M1 target (median ΔE < 5 and >20 share < 5 %) for the product: {'MET' if ok else 'NOT MET'}** (median {s_med['median']:.2f}, >20: {s_med['pct_gt_20']:.1f} %).", ""]

    dist_labels = [f"{a:g}–{b:g} m" for a, b in zip(metrics.DIST_EDGES[:-1], metrics.DIST_EDGES[1:])]
    dist_labels[-1] = "≥40 m or no nearest-in-time sample"
    grad_labels = [f"{a:g}–{b:g}" for a, b in zip(metrics.GRAD_EDGES[:-1], metrics.GRAD_EDGES[1:])]
    el_labels = [f"{a:+d}…{b:+d}°" for a, b in zip(metrics.EL_EDGES[:-1], metrics.EL_EDGES[1:])]
    for v in ("med", "nt"):
        h = hists[v][0]
        lines += [f"## Strata — {VARIANT_LABEL[v]}", "", "### by classification", "", _table_by(h, "class", [str(i) for i in range(256)]), "",
                  "### by camera distance (nearest-in-time frame)", "", _table_by(h, "dist", dist_labels), "",
                  "### by image gradient at the sampled pixel (low = radiometry only, high = geometry)", "", _table_by(h, "grad", grad_labels), "",
                  "### by elevation", "", _table_by(h, "elev", el_labels), "",
                  "### by number of fused views", "", _table_by(h, "n_views", [str(i) for i in range(32)]), ""]
    # gradient ratio diagnostic
    h = hists["med"][0]
    lo = metrics.hist_stats(h.h["grad"][0:2].sum(0))
    hi = metrics.hist_stats(h.h["grad"][7:].sum(0))
    if lo["n"] and hi["n"]:
        lines += [f"Gradient diagnostic (product): smooth pixels median {lo['median']:.2f} (radiometric floor) vs edge pixels {hi['median']:.2f} → ratio {hi['median']/max(lo['median'],1e-6):.2f} (pilot: 2.64).", ""]

    lines += ["## Per tile (product median ΔE00 / >20 %)", "", "| tile | n | frames | s | median | >20 | coverage |", "|---|---|---|---|---|---|---|"]
    for name in sorted(metas):
        s = metrics.hist_stats(hists["med"][1][name].total_hist()) if name in hists["med"][1] else {"n": 0}
        m = metas[name]
        lines.append(f"| {name} | {m['n']:,} | {m['n_frames']} | {m['seconds']:.0f} | {s.get('median', float('nan')):.2f} | {s.get('pct_gt_20', float('nan')):.1f} % | {100*m['coverage']['med']:.1f} % |")
    lines.append("")

    # plots
    _plot_hist(hists, rd / "hist_de00.png")
    _plot_family(hists, "dist", (metrics.DIST_EDGES[:-1] + np.minimum(metrics.DIST_EDGES[1:], 60)) / 2, "camera distance [m]", rd / "de_vs_dist.png")
    _plot_family(hists, "grad", np.arange(len(grad_labels)), "image gradient band", rd / "de_vs_grad.png")
    _plot_family(hists, "elev", (metrics.EL_EDGES[:-1] + metrics.EL_EDGES[1:]) / 2, "elevation [deg]", rd / "de_vs_elev.png")
    _plot_family(hists, "az", np.arange(0, 360, metrics.AZ_STEP) + metrics.AZ_STEP / 2, "azimuth in image [deg] (0 = driving direction)", rd / "de_vs_az.png")
    # per-frame median (nt variant) reveals bad frames / sync
    h = hists["nt"][0].h["frame"]
    fr_med = np.array([metrics.hist_stats(h[i])["median"] if h[i].sum() >= 200 else np.nan for i in range(h.shape[0])])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(fr_med, ".", ms=3)
    ax.set_xlabel("frame index (time order)")
    ax.set_ylabel("median ΔE00 (nt)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rd / "de_per_frame.png", dpi=120)
    plt.close(fig)
    lines += ["## Plots", "", "![](hist_de00.png)", "![](de_vs_dist.png)", "![](de_vs_grad.png)", "![](de_vs_elev.png)", "![](de_vs_az.png)", "![](de_per_frame.png)", ""]

    out = rd / "report.md"
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    import sys

    print(write_report(sys.argv[1] if len(sys.argv) > 1 else "run"))
