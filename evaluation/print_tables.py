#!/usr/bin/env python
"""
Print LaTeX (or Markdown) results tables from the eval_ssl.py results store.

Needs only numpy -- no torch, no GPU, no model. Runs anywhere you can copy results.pkl.

Examples
  # everything (summary + per-dataset depth / camera / point-map tables) to stdout
  python print_tables.py results/ssl_eval/results.pkl

  # only the main table, fixed row / column order, saved to a file for Overleaf
  python print_tables.py results.pkl --tables summary --models MAE CroCo MuM Muskie Vista -o tables.tex

  # choose metrics per task (any key listed by --list_metrics)
  python print_tables.py results.pkl --metrics "depth=abs_rel,delta1;camera=auc30,auc5,rra15"
  # camera metrics: rra/rta @ 5, 15, 20, 30 and auc @ 3, 5, 15, 20, 30 (records made before the
  # extra thresholds show '--' until `eval_ssl.py --recompute`)

  # quick look in the terminal
  python print_tables.py results.pkl --markdown

  # one .tex per table + CSV/JSON mirrors into a folder (e.g. your Overleaf project's tables/)
  python print_tables.py results.pkl --models MAE CroCo MuM Muskie Vista --out_dir paper/tables

  # a compilable document (paste into Overleaf as main.tex to preview)
  python print_tables.py results.pkl --standalone -o preview.tex

Only the latest COMPLETE record of every (model, dataset) is used; flagged runs (partial,
debug, NaN metrics, ...) are skipped unless --allow_incomplete. A warning is printed to
stderr whenever the selected records were produced with different evaluation settings.
LaTeX needs: \\usepackage{booktabs} \\usepackage{graphicx}
"""



import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import numpy as np


def warn(msg):
    print(f"[print_tables] WARNING: {msg}", file=sys.stderr)


# =============================================================================
# Metric catalog: every metric the store holds, with label, direction and format
# =============================================================================
def M(key, label, better, scale=1.0, fmt="{:.3f}", md=None):
    return dict(key=key, label=label, better=better, scale=scale, fmt=fmt, md=md or key)


CATALOG = {
    "depth": {
        "abs_rel": M("abs_rel", r"AbsRel$\downarrow$", "min", md="AbsRel↓"),
        "sq_rel": M("sq_rel", r"SqRel$\downarrow$", "min", md="SqRel↓"),
        "rmse": M("rmse", r"RMSE$\downarrow$", "min", md="RMSE↓"),
        "log_rmse": M("log_rmse", r"RMSE$_{\log}\!\downarrow$", "min", md="logRMSE↓"),
        "delta1": M("delta1", r"$\delta_1\!\uparrow$", "max", 100, "{:.1f}", "δ1↑"),
        "delta2": M("delta2", r"$\delta_2\!\uparrow$", "max", 100, "{:.1f}", "δ2↑"),
        "delta3": M("delta3", r"$\delta_3\!\uparrow$", "max", 100, "{:.1f}", "δ3↑"),
    },
    "camera": {
        **{f"{m.lower()}{t}": M(f"{m.lower()}{t}", rf"{m}@{t}$\uparrow$", "max", 100, "{:.1f}", f"{m}@{t}↑")
           for t in (5, 15, 20, 30) for m in ("RRA", "RTA")},
        **{f"auc{t}": M(f"auc{t}", rf"AUC@{t}$\uparrow$", "max", 100, "{:.1f}", f"AUC@{t}↑")
           for t in (30, 20, 15, 5, 3)},
        "maa30": M("maa30", r"mAA@30$\uparrow$", "max", 100, "{:.1f}", "mAA30↑"),
    },
    "pointmap": {
        "acc": M("acc", r"Acc$\downarrow$", "min", md="Acc↓"),
        "acc_med": M("acc_med", r"Acc$_{\mathrm{med}}\!\downarrow$", "min", md="AccMed↓"),
        "comp": M("comp", r"Comp$\downarrow$", "min", md="Comp↓"),
        "comp_med": M("comp_med", r"Comp$_{\mathrm{med}}\!\downarrow$", "min", md="CompMed↓"),
        "nc": M("nc", r"NC$\uparrow$", "max", md="NC↑"),
        "nc_med": M("nc_med", r"NC$_{\mathrm{med}}\!\uparrow$", "max", md="NCMed↑"),
        "comp_ratio": M("comp_ratio", r"CR$\uparrow$", "max", 100, "{:.1f}", "CR↑"),
    },
}


DEFAULT_METRICS = {
    "depth": ["abs_rel", "delta1"],
    "camera": ["rra5", "rta5", "rra15", "rta15", "rra20", "rta20", "rra30", "rta30",
               "auc3", "auc5", "auc15", "auc20", "auc30"],
    "pointmap": ["acc", "comp", "nc"],
}
TASK_TITLE = {"depth": "Depth", "camera": "Camera pose", "pointmap": "Point map"}
TASK_CAPTION = {
    "depth": r"Multi-view depth estimation after scale alignment.",
    "camera": r"Relative camera pose estimation (\%). Metrics are averaged over scenes.",
    "pointmap": r"Point map reconstruction after Sim(3) alignment to ground truth.",
}
DISPLAY = {
    "Habitat_HM3D": "HM3D", "Matterport3D": "MP3D", "Arkitscenes": "ARKitScenes", "Mapfree": "Map-free",
    "DL3DV": "DL3DV", "MegaDepth": "MegaDepth", "ScanNet": "ScanNet", "Hypersim": "Hypersim",
    "Taskonomy": "Taskonomy", "Replica": "Replica",
}
AVG = "__avg__"
PER_MODEL_SETTINGS = {"model_class", "model_kwargs"}  # differ per model by design


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def display_name(ds, latex=True):
    if ds == AVG:
        return "Avg."
    base = ds.replace("_NVS", "")
    return DISPLAY.get(base, tex(base) if latex else base)


# =============================================================================
# Reading the store
# =============================================================================
def load_store(path):
    for p in (path, path + ".bak"):
        if osp.exists(p):
            try:
                with open(p, "rb") as f:
                    store = pickle.load(f)
                if p != path:
                    warn(f"{path} unreadable -> using {p}")
                return store
            except Exception as e:  # noqa: BLE001
                warn(f"could not read {p}: {e}")
    raise SystemExit(f"No readable results store at {path}")


def select_records(store, allow_incomplete=False):
    """Latest complete record per (model, dataset) -> ({model: {dataset: metrics}}, [records])."""
    results, chosen = defaultdict(dict), []
    for model, per_ds in store["runs"].items():
        for ds, hist in per_ds.items():
            rec = next((r for r in reversed(hist) if r["complete"] or allow_incomplete), None)
            if rec is None:
                warn(f"{model}/{ds}: no complete record -> left out (latest errors: {hist[-1]['errors']})")
                continue
            results[model][ds] = rec["metrics"]
            chosen.append(rec)
    return results, chosen


def check_consistency(chosen, strict=False):
    problems = []
    keys = sorted({k for r in chosen for k in r["settings"]} - PER_MODEL_SETTINGS)
    for k in keys:
        vals = defaultdict(list)
        for r in chosen:
            vals[repr(r["settings"].get(k))].append(f"{r['model']}/{r['dataset']}")
        if len(vals) > 1:
            problems.append(f"setting '{k}' differs: " + "; ".join(
                f"{v} <- {', '.join(w[:4])}" + (" ..." if len(w) > 4 else "") for v, w in vals.items()))
    pm = {r.get("pm_cameras") for r in chosen if r.get("pm_cameras")}
    if len(pm) > 1:
        problems.append(f"point maps used different cameras across records: {sorted(pm)}")
    flagged = [f"{r['model']}/{r['dataset']}" for r in chosen if not r["complete"]]
    if flagged:
        problems.append(f"tables include flagged (incomplete) records: {flagged}")
    shas = {r.get("code_sha") for r in chosen}
    if len(shas) > 1:
        warn(f"records come from {len(shas)} versions of eval_ssl.py {sorted(map(str, shas))}; "
             "run `eval_ssl.py --recompute` to put them on one version")
    for p in problems:
        warn("CONSISTENCY: " + p)
    if problems and strict:
        raise SystemExit("Inconsistent records (see warnings above); fix them or drop --strict.")


# =============================================================================
# Table building
# =============================================================================
def get_value(results, model, ds, task, key, datasets):
    if ds == AVG:  # average only when every dataset is present
        vals = [get_value(results, model, d, task, key, datasets) for d in datasets]
        return float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
    v = results.get(model, {}).get(ds, {}).get(task, {}).get(key)
    return None if v is None or not np.isfinite(v) else float(v)


def column_ranks(values, spec):
    """{model: rank} on the DISPLAYED (rounded) values, so visual ties share a rank."""
    shown = {m: float(spec["fmt"].format(v * spec["scale"])) for m, v in values.items() if v is not None}
    uniq = sorted(set(shown.values()), reverse=(spec["better"] == "max"))
    return {m: uniq.index(s) for m, s in shown.items()} if len(shown) >= 2 else {}


def fmt_cell(v, spec, rank, latex=True):
    if v is None:
        return "--"
    s = spec["fmt"].format(v * spec["scale"])
    if rank == 0:
        return rf"\textbf{{{s}}}" if latex else f"**{s}**"
    if rank == 1:
        return rf"\underline{{{s}}}" if latex else f"_{s}_"
    return s


def resolve_metrics(spec_str):
    chosen = {t: list(v) for t, v in DEFAULT_METRICS.items()}
    if spec_str:
        for part in spec_str.split(";"):
            if not part.strip():
                continue
            task, keys = part.split("=")
            task = task.strip()
            if task not in CATALOG:
                raise SystemExit(f"unknown task '{task}' (choose from {list(CATALOG)})")
            keys = [k.strip() for k in keys.split(",") if k.strip()]
            bad = [k for k in keys if k not in CATALOG[task]]
            if bad:
                raise SystemExit(f"unknown {task} metric(s) {bad}; available: {list(CATALOG[task])}")
            chosen[task] = keys
    return {t: [CATALOG[t][k] for k in keys] for t, keys in chosen.items()}


def task_table(results, task, specs, models, datasets, per_block):
    cols = datasets + [AVG]
    n_blocks = -(-len(cols) // per_block)
    size = -(-len(cols) // n_blocks)  # balanced blocks, so Avg. is never alone
    blocks = [cols[i:i + size] for i in range(0, len(cols), size)]
    width = max(len(b) for b in blocks)
    ncol = width * len(specs)

    L = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{TASK_CAPTION[task]}}}", rf"\label{{tab:ssl_{task}}}",
         r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{l{'c' * ncol}}}", r"\toprule"]
    for bi, block in enumerate(blocks):
        if bi > 0:
            L.append(r"\midrule")
        pad = (width - len(block)) * len(specs)
        head = [rf"\multicolumn{{{len(specs)}}}{{c}}{{{display_name(d)}}}" for d in block] + [""] * pad
        L.append(" & " + " & ".join(head) + r" \\")
        L.append(" ".join(rf"\cmidrule(lr){{{2 + i * len(specs)}-{1 + (i + 1) * len(specs)}}}"
                          for i in range(len(block))))
        L.append("Method & " + " & ".join([s["label"] for _ in block for s in specs] + [""] * pad) + r" \\")
        L.append(r"\midrule")
        ranks = {(d, s["key"]): column_ranks(
            {m: get_value(results, m, d, task, s["key"], datasets) for m in models}, s)
            for d in block for s in specs}
        for m in models:
            cells = [fmt_cell(get_value(results, m, d, task, s["key"], datasets), s, ranks[(d, s["key"])].get(m))
                     for d in block for s in specs]
            L.append(tex(m) + " & " + " & ".join(cells + [""] * pad) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def summary_table(results, task_specs, models, datasets):
    tasks = list(task_specs)
    ncol = sum(len(task_specs[t]) for t in tasks)
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{SSL backbone comparison, averaged over {len(datasets)} datasets. "
         r"Best in \textbf{bold}, second \underline{underlined}.}",
         r"\label{tab:ssl_summary}", r"\resizebox{\linewidth}{!}{%",
         rf"\begin{{tabular}}{{l{'c' * ncol}}}", r"\toprule"]
    head, rules, c = [], [], 2
    for t in tasks:
        n = len(task_specs[t])
        head.append(rf"\multicolumn{{{n}}}{{c}}{{{TASK_TITLE[t]}}}")
        rules.append(rf"\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    L.append(" & " + " & ".join(head) + r" \\")
    L.append(" ".join(rules))
    L.append("Method & " + " & ".join(s["label"] for t in tasks for s in task_specs[t]) + r" \\")
    L.append(r"\midrule")
    ranks = {(t, s["key"]): column_ranks({m: get_value(results, m, AVG, t, s["key"], datasets) for m in models}, s)
             for t in tasks for s in task_specs[t]}
    for m in models:
        cells = [fmt_cell(get_value(results, m, AVG, t, s["key"], datasets), s, ranks[(t, s["key"])].get(m))
                 for t in tasks for s in task_specs[t]]
        L.append(tex(m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def markdown_table(results, task, specs, models, datasets):
    """Terminal-friendly: one small table per metric, rows = datasets (+ Avg.), columns = models.
    Best per ROW is **bold**, second _underlined_."""
    L = [f"### {TASK_TITLE[task]}"]
    for s in specs:
        L += ["", f"**{s['md']}**", "", "| Dataset | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
        for d in datasets + [AVG]:
            vals = {m: get_value(results, m, d, task, s["key"], datasets) for m in models}
            ranks = column_ranks(vals, s)
            cells = [fmt_cell(vals[m], s, ranks.get(m), latex=False) for m in models]
            L.append(f"| {display_name(d, latex=False)} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def order(available, requested, what):
    if not requested:
        return sorted(available)
    missing = [x for x in requested if x not in available]
    if missing:
        warn(f"{what} not in the store (skipped): {missing}")
    return [x for x in requested if x in available]


def build(results, chosen, tables=("summary", "depth", "camera", "pointmap"), models=None, datasets=None,
          metrics=None, per_block=6, markdown=False):
    """Returns {table_name: text}."""
    task_specs = resolve_metrics(metrics)
    model_list = order(results.keys(), models, "models")
    ds_list = order({d for r in results.values() for d in r}, datasets, "datasets")
    present = [t for t in ("depth", "camera", "pointmap")
               if any(t in results[m].get(d, {}) for m in model_list for d in ds_list)]
    task_specs = {t: task_specs[t] for t in present}

    missing = [(m, d) for m in model_list for d in ds_list if d not in results[m]]
    if missing:
        warn(f"missing (model, dataset) pairs -> shown as '--' and excluded from Avg.: {missing}")

    out = {}
    if markdown:
        for t in present:
            if t in tables or "summary" in tables:
                out[t] = markdown_table(results, t, task_specs[t], model_list, ds_list)
        return out
    if "summary" in tables and present:
        out["summary"] = summary_table(results, task_specs, model_list, ds_list)
    for t in present:
        if t in tables:
            out[t] = task_table(results, t, task_specs[t], model_list, ds_list, per_block)
    return out


def write_all(store_path, out_dir, models=None, datasets=None, metrics=None, per_block=6,
              allow_incomplete=False, strict=False):
    """Writes summary/depth/camera/pointmap .tex, all_tables.tex, results_latest.json and
    results_long.csv to out_dir (--out_dir). Returns the summary text."""
    store = load_store(store_path)
    results, chosen = select_records(store, allow_incomplete)
    if not results:
        warn(f"no usable results in {store_path}")
        return ""
    check_consistency(chosen, strict)
    tables = build(results, chosen, models=models, datasets=datasets, metrics=metrics, per_block=per_block)
    os.makedirs(out_dir, exist_ok=True)
    for name, text in tables.items():
        with open(osp.join(out_dir, f"{name}.tex"), "w") as f:
            f.write(text + "\n")
    with open(osp.join(out_dir, "all_tables.tex"), "w") as f:
        f.write("% Generated from results.pkl -- requires \\usepackage{booktabs,graphicx}\n"
                + "\n\n".join(tables.values()) + "\n")
    with open(osp.join(out_dir, "results_latest.json"), "w") as f:
        json.dump({f"{r['model']}/{r['dataset']}": r for r in chosen}, f, indent=2, default=str)
    with open(osp.join(out_dir, "results_long.csv"), "w") as f:
        f.write("model,dataset,task,metric,value\n")
        for m in sorted(results):
            for d, met in sorted(results[m].items()):
                for t in ("depth", "camera", "pointmap"):
                    for k, v in met.get(t, {}).items():
                        f.write(f"{m},{d},{t},{k},{v}\n")
    return tables.get("summary", "")


STANDALONE_HEAD = r"""\documentclass{article}
\usepackage[margin=1.5cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}
"""


def main():
    p = argparse.ArgumentParser(description="Print LaTeX tables from an eval_ssl.py results.pkl",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("pkl", help="Path to results.pkl (or a results_dir containing it)")
    p.add_argument("--tables", default="summary,depth,camera,pointmap",
                   help="Comma-separated subset of: summary, depth, camera, pointmap")
    p.add_argument("--models", nargs="*", default=None, help="Row order (default: alphabetical)")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Column order / subset, as stored (e.g. Hypersim_NVS ScanNet_NVS)")
    p.add_argument("--metrics", default=None,
                   help='Per-task metric columns, e.g. "depth=abs_rel,delta1;camera=auc30,rra15"')
    p.add_argument("--datasets_per_block", type=int, default=6)
    p.add_argument("--markdown", action="store_true", help="Markdown instead of LaTeX (quick terminal view)")
    p.add_argument("--standalone", action="store_true", help="Wrap in a compilable LaTeX document")
    p.add_argument("--allow_incomplete", action="store_true", help="Also use flagged records")
    p.add_argument("--strict", action="store_true", help="Fail if records used different settings")
    p.add_argument("--list_metrics", action="store_true", help="List available metric keys and exit")
    p.add_argument("--list", action="store_true", help="List models and datasets in the store and exit")
    p.add_argument("-o", "--output", default=None, help="Write to this file instead of stdout")
    p.add_argument("--out_dir", default=None,
                   help="Write summary/depth/camera/pointmap .tex, all_tables.tex, results_latest.json "
                        "and results_long.csv into this folder")
    args = p.parse_args()

    if args.list_metrics:
        for t, ms in CATALOG.items():
            print(f"{t:<9} " + ", ".join(f"{k}{' (default)' if k in DEFAULT_METRICS[t] else ''}" for k in ms))
        return

    path = osp.join(args.pkl, "results.pkl") if osp.isdir(args.pkl) else args.pkl
    if args.out_dir:
        write_all(path, args.out_dir, args.models, args.datasets, args.metrics, args.datasets_per_block,
                  args.allow_incomplete, args.strict)
        print(f"[print_tables] wrote tables to {args.out_dir}", file=sys.stderr)
        return
    store = load_store(path)
    results, chosen = select_records(store, args.allow_incomplete)
    if args.list:
        print("models:  ", " ".join(sorted(results)))
        print("datasets:", " ".join(sorted({d for r in results.values() for d in r})))
        return
    if not results:
        raise SystemExit("No usable (complete) records in the store; try --allow_incomplete.")
    check_consistency(chosen, args.strict)

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    out = build(results, chosen, tables, args.models, args.datasets, args.metrics,
                args.datasets_per_block, args.markdown)
    text = "\n\n".join(out.values())
    if args.standalone and not args.markdown:
        text = STANDALONE_HEAD + text + "\n\\end{document}"
    elif not args.markdown:
        text = "% Generated by print_tables.py -- requires \\usepackage{booktabs,graphicx}\n" + text

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"[print_tables] wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()