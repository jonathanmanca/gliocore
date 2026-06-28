"""
docs/make_figures.py — Rigenera in modo deterministico le figure del benchmark
dai risultati per-caso in docs/results/brats_MRI_<Model>.json.

    python docs/make_figures.py

Produce (in docs/):
  fig1_ari_boxplot.png       — distribuzione ARI per metodo
  fig2_metrics_ci.png        — ARI / Dice TC / Dice ET con CI bootstrap 95%
  fig3_accuracy_runtime.png  — accuratezza (ARI) vs runtime
  fig4_mrf_speedup.png       — runtime MRF-EM prima/dopo la vettorizzazione

Le figure sono riproducibili (bootstrap con seed fisso); numeri in tabella,
file in results/ e figure restano allineati.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

METHOD_ORDER = ["Threshold", "FCM", "GMM", "MRF-EM", "Hierarchical"]
METHOD_COLORS = {
    "Threshold":    "#B0BEC5",   # blu-grigio (baseline)
    "FCM":          "#F39C12",   # arancio
    "GMM":          "#3498DB",   # blu
    "MRF-EM":       "#9B59B6",   # viola
    "Hierarchical": "#E74C3C",   # rosso — metodo proposto (in evidenza)
}
# Colori per le tre metriche (fig2)
METRIC_COLORS = {"ARI": "#34495E", "Dice TC": "#16A085", "Dice ET": "#E67E22"}


def _clean(v):
    return [float(x) for x in v if x is not None and x == x and x != float("inf")]


def load(results_dir):
    out = {}
    for m in METHOD_ORDER:
        p = results_dir / f"brats_MRI_{m}.json"
        if p.exists():
            out[m] = json.load(open(p, encoding="utf-8"))
    if not out:
        raise SystemExit(f"Nessun risultato in {results_dir}")
    return out


def med_ci(vals, nboot, seed):
    v = np.array(_clean(vals), float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(nboot, len(v)), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def collect(results, nboot, seed):
    s = {}
    for m, rows in results.items():
        ok = [r for r in rows if not r.get("error")]
        s[m] = {
            "ari":  [r["ari"] for r in ok],
            "tc":   _clean([r["dice_tc"] for r in ok if r.get("gt_has_tc", True)]),
            "et":   _clean([r["dice_et"] for r in ok if r.get("gt_has_et", True)]),
            "rt":   _clean([r["runtime_s"] for r in ok]),
        }
        s[m]["ari_ci"] = med_ci(s[m]["ari"], nboot, seed)
        s[m]["tc_ci"]  = med_ci(s[m]["tc"],  nboot, seed)
        s[m]["et_ci"]  = med_ci(s[m]["et"],  nboot, seed)
        s[m]["rt_mean"] = float(np.mean(s[m]["rt"])) if s[m]["rt"] else float("nan")
        s[m]["rt_med"]  = float(np.median(s[m]["rt"])) if s[m]["rt"] else float("nan")
    return s


def _mods(s):
    return [m for m in METHOD_ORDER if m in s]


def fig1_boxplot(s, out):
    models = _mods(s)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bp = ax.boxplot([s[m]["ari"] for m in models], patch_artist=True, showfliers=False,
                    widths=0.6, medianprops=dict(color="#222222", linewidth=2))
    ax.set_xticks(range(1, len(models) + 1)); ax.set_xticklabels(models)
    for b, m in zip(bp["boxes"], models):
        b.set_facecolor(METHOD_COLORS[m]); b.set_alpha(0.9); b.set_edgecolor("#444444")
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#666666")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Structural agreement per method — BraTS 2024 (271 cases)")
    ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig2_metrics(s, out):
    models = _mods(s)
    metrics = [("ari_ci", "ARI"), ("tc_ci", "Dice TC"), ("et_ci", "Dice ET")]
    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, (key, lab) in enumerate(metrics):
        med = [s[m][key][0] for m in models]
        lo  = [s[m][key][0] - s[m][key][1] for m in models]
        hi  = [s[m][key][2] - s[m][key][0] for m in models]
        ax.bar(x + (i - 1) * w, med, w, yerr=[lo, hi], capsize=3,
               color=METRIC_COLORS[lab], alpha=0.92, label=lab,
               error_kw=dict(ecolor="#555555", lw=1))
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Score (median, 95% bootstrap CI)")
    ax.set_title("Segmentation metrics by method — BraTS 2024")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, 0.9)
    ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig3_acc_runtime(s, out, stat="mean"):
    models = _mods(s)
    key = "rt_mean" if stat == "mean" else "rt_med"
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for m in models:
        xx, yy = s[m][key], s[m]["ari_ci"][0]
        ax.scatter(xx, yy, s=460, color=METHOD_COLORS[m], edgecolor="white",
                   linewidth=2, zorder=3)
        ax.annotate(m, (xx, yy), xytext=(15, 9), textcoords="offset points",
                    fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel(f"Runtime per case ({stat}, seconds — log scale)")
    ax.set_ylabel("Median ARI")
    ax.set_title("Accuracy vs computational cost — 271 cases")
    ax.text(0.02, 0.95, "Better: high and to the left", transform=ax.transAxes,
            fontsize=11, style="italic", color="#888888", va="top")
    ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)   # solo orizzontale
    ax.margins(x=0.22, y=0.14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig4_speedup(s, out, repo, stat="mean"):
    """Runtime MRF-EM prima/dopo la vettorizzazione dell'ICM."""
    key = "rt_mean" if stat == "mean" else "rt_med"
    new = s["MRF-EM"][key]
    # runtime "prima" = versione precedente (loop ICM) recuperata da git
    try:
        old_json = subprocess.run(
            ["git", "show", "HEAD:docs/results/brats_MRI_MRF-EM.json"],
            cwd=repo, capture_output=True, text=True).stdout
        ort = _clean([r["runtime_s"] for r in json.loads(old_json) if not r.get("error")])
        old = float(np.mean(ort)) if stat == "mean" else float(np.median(ort))
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    bars = ax.bar(["Before\n(scalar ICM loop)", "After\n(vectorized ICM)"],
                  [old, new], width=0.55,
                  color=["#BDC3C7", METHOD_COLORS["MRF-EM"]], edgecolor="#444444")
    for b, val in zip(bars, [old, new]):
        ax.text(b.get_x() + b.get_width() / 2, val + max(old, new) * 0.02,
                f"{val:.1f} s", ha="center", fontweight="bold")
    ax.set_ylabel(f"Runtime per case ({stat}, seconds)")
    ax.set_title(f"MRF-EM vectorization — ~{old/new:.1f}× faster")
    ax.set_ylim(0, old * 1.18)
    ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return True


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=here / "results")
    ap.add_argument("--out-dir", type=Path, default=here)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runtime-stat", choices=["mean", "median"], default="mean")
    a = ap.parse_args()

    s = collect(load(a.results_dir), a.n_boot, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig1_boxplot(s, a.out_dir / "fig1_ari_boxplot.png")
    fig2_metrics(s, a.out_dir / "fig2_metrics_ci.png")
    fig3_acc_runtime(s, a.out_dir / "fig3_accuracy_runtime.png", a.runtime_stat)
    ok4 = fig4_speedup(s, a.out_dir / "fig4_mrf_speedup.png", str(repo), a.runtime_stat)
    figs = "fig1, fig2, fig3" + (", fig4" if ok4 else "")
    print(f"Salvate ({figs}) in {a.out_dir}  [runtime={a.runtime_stat}]")
    for m in _mods(s):
        c = s[m]
        rt = c["rt_mean"] if a.runtime_stat == "mean" else c["rt_med"]
        print(f"  {m:13s} ARI {c['ari_ci'][0]:.2f}  DiceET {c['et_ci'][0]:.2f}  runtime {rt:.1f}s")


if __name__ == "__main__":
    main()
