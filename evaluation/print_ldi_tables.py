#!/usr/bin/env python
"""
Print LaTeX (or Markdown) results tables from the eval_ldi.py results store (results_ldi.pkl).

Companion of print_tables.py: same table style (datasets as column blocks + Avg., best per column in bold,
second underlined), same options. Needs only numpy -- no torch, no GPU, no model.

Tables (--tables, comma-separated; default: main)
  main        ONE table with everything: datasets as row blocks + Avg., column groups
                Visible depth (AbsRel, delta1) | Hidden depth (AbsRel, delta1) | Occupancy (hidden mIoU, mDice)
                | Chamfer / F-score (CD L0, CD hidden, CD all, F@10cm hidden) | Occluded geometry (P, R, F1 @10cm)
  visible     visible depth (layer 0) details                       (rows with GT layer 0 are left out)
  hidden      hidden-layer depth, pooled and per layer
  occupancy   IoU / Dice of the layer masks from the predicted stop index (the only stop-index metrics)
  chamfer     per-view Chamfer / F-score, predicted points selected by the GT layer mask (as LaRI)
  occlusion   occluded geometry (all views fused): P / R / F1, whole-scene Chamfer / F-score
  The detail tables use datasets as column blocks + Avg.

Examples
  python print_ldi_tables.py results/ldi_eval/results_ldi.pkl
  python print_ldi_tables.py results/ldi_eval --models gt_l0 mae mum muskie poincar3 vista vistav2 -o ldi.tex
  python print_ldi_tables.py results_ldi.pkl --tables main,occupancy,chamfer --markdown
  python print_ldi_tables.py results_ldi.pkl --metrics "main=vis_abs_rel,miou_hid,cd_hidden,occ_f1@10cm"
  python print_ldi_tables.py results_ldi.pkl --rename "vistav2=Vista-v2" "mum_oracleL0=MuM (GT L0)"
  python print_ldi_tables.py results_ldi.pkl --out_dir paper/tables      # one .tex per table + CSV
  python print_ldi_tables.py results_ldi.pkl --list_metrics

Only the latest COMPLETE record of every (model, dataset) is used (--allow_incomplete to include flagged ones).
Avg. is shown only for models evaluated on every selected dataset.
LaTeX needs: \\usepackage{booktabs} \\usepackage{graphicx}
"""

import argparse
import csv
import json
import os
import os.path as osp
import pickle
import re
import sys
from collections import defaultdict

import numpy as np

TAUS = ("2cm", "5cm", "10cm", "20cm")
AVG = "__avg__"


def warn(msg):
    print(f"[print_ldi_tables] WARNING: {msg}", file=sys.stderr)


# =============================================================================
# Metric catalog (keys are the flat metric names stored by eval_ldi.py)
# =============================================================================
def M(key, label, better, scale=1.0, fmt="{:.3f}", md=None):
    return dict(key=key, label=label, better=better, scale=scale, fmt=fmt, md=md or key)


PCT = dict(scale=100, fmt="{:.1f}")          # fractions shown as percentages

LAYERS = ("L0", "L1", "L2", "L3")
CATALOG = {
    "visible": {
        "abs_rel": M("vis_abs_rel", r"AbsRel$\downarrow$", "min", md="AbsRel↓"),
        "sq_rel": M("vis_sq_rel", r"SqRel$\downarrow$", "min", md="SqRel↓"),
        "rmse": M("vis_rmse", r"RMSE$\downarrow$", "min", md="RMSE↓"),
        "log_rmse": M("vis_log_rmse", r"RMSE$_{\log}\!\downarrow$", "min", md="logRMSE↓"),
        "d1": M("vis_d1", r"$\delta_1\!\uparrow$", "max", md="δ1↑", **PCT),
        "d2": M("vis_d2", r"$\delta_2\!\uparrow$", "max", md="δ2↑", **PCT),
        "d3": M("vis_d3", r"$\delta_3\!\uparrow$", "max", md="δ3↑", **PCT),
    },
    "hidden": {
        "abs_rel": M("abs_rel_hidden", r"AbsRel$\downarrow$", "min", md="AbsRel hid↓"),
        "d1": M("d1_hidden", r"$\delta_1\!\uparrow$", "max", md="δ1 hid↑", **PCT),
        **{f"{n}_{k}": M(f"{key}_{k}", rf"{lab}$_{{{k}}}\!{arrow}$", b, md=f"{n} {k}", **(PCT if b == "max" else {}))
           for k in LAYERS[1:]
           for n, key, lab, b, arrow in (("abs_rel", "abs_rel", "AbsRel", "min", r"\downarrow"),
                                         ("d1", "d1", r"$\delta_1$", "max", r"\uparrow"))},
    },
    "occupancy": {
        "miou_hid": M("stop_miou_hidden", r"mIoU$\uparrow$", "max", md="mIoU hid↑", **PCT),
        "mdice_hid": M("stop_mdice_hidden", r"mDice$\uparrow$", "max", md="mDice hid↑", **PCT),
        "miou": M("stop_miou", r"mIoU$_{\mathrm{all}}\!\uparrow$", "max", md="mIoU all↑", **PCT),
        "mdice": M("stop_mdice", r"mDice$_{\mathrm{all}}\!\uparrow$", "max", md="mDice all↑", **PCT),
        **{f"{w}_{k}": M(f"stop_{w}_{k}", rf"{lab}$_{{{k}}}\!\uparrow$", "max", md=f"{lab} {k}↑", **PCT)
           for k in LAYERS for w, lab in (("iou", "IoU"), ("dice", "Dice"))},
    },
    "chamfer": {
        **{f"cd_{k}": M(f"cd_{k}_gtmask", rf"CD$_{{{k}}}\!\downarrow$", "min", 100, "{:.2f}", f"CD {k} cm↓")
           for k in LAYERS},
        "cd_hidden": M("cd_hidden_gtmask", r"CD$_{\mathrm{hid}}\!\downarrow$", "min", 100, "{:.2f}", "CD hid cm↓"),
        "cd_all": M("cd_all_gtmask", r"CD$_{\mathrm{all}}\!\downarrow$", "min", 100, "{:.2f}", "CD all cm↓"),
        **{f"f@{t}_{w}": M(f"fscore@{t}_{w}_gtmask", rf"F$_{{\mathrm{{{w[:3]}}}}}$@{t}$\uparrow$", "max",
                           md=f"F {w}@{t}↑", **PCT)
           for t in TAUS for w in ("hidden", "all")},
    },
    "occlusion": {
        **{f"occ_{w}@{t}": M(f"occ_{key}@{t}", rf"{lab}@{t}$\uparrow$", "max", md=f"Occ{lab}@{t}↑", **PCT)
           for t in TAUS for w, key, lab in (("p", "precision", "P"), ("r", "recall", "R"), ("f1", "f1", "F1"))},
        "scene_cd": M("chamfer", r"CD$_{\mathrm{scene}}\!\downarrow$", "min", 100, "{:.2f}", "Scene CD cm↓"),
        **{f"scene_f@{t}": M(f"f_all@{t}", rf"F$_{{\mathrm{{scene}}}}$@{t}$\uparrow$", "max", md=f"Scene F@{t}↑",
                             **PCT) for t in TAUS},
    },
}
# the ONE main table: (group title, table, metric) in column order
MAIN_COLUMNS = [("Visible depth", "visible", "abs_rel"), ("Visible depth", "visible", "d1"),
                ("Hidden depth", "hidden", "abs_rel"), ("Hidden depth", "hidden", "d1"),
                ("Occupancy", "occupancy", "miou_hid"), ("Occupancy", "occupancy", "mdice_hid"),
                ("Chamfer / F-score", "chamfer", "cd_L0"), ("Chamfer / F-score", "chamfer", "cd_hidden"),
                ("Chamfer / F-score", "chamfer", "cd_all"), ("Chamfer / F-score", "chamfer", "f@10cm_hidden"),
                ("Occluded geometry", "occlusion", "occ_p@10cm"), ("Occluded geometry", "occlusion", "occ_r@10cm"),
                ("Occluded geometry", "occlusion", "occ_f1@10cm")]
MAIN_CAPTION = (r"Layered depth probing. Depth after per-view median scale alignment; hidden = layers 1--3. "
                r"Occupancy: IoU / Dice (\%) of the hidden-layer masks from the predicted stop index. "
                r"Chamfer distance (cm) and F-score (\%): per view, predicted points selected by the GT layer mask. "
                r"Occluded geometry (\%): all views fused, GT points no input view sees, at 10\,cm. "
                r"Best in \textbf{bold}, second \underline{underlined}.")
DEFAULT_METRICS = {
    "visible": ["abs_rel", "rmse", "d1"],
    "hidden": ["abs_rel", "d1", "abs_rel_L1", "abs_rel_L2", "abs_rel_L3"],
    "occupancy": ["miou_hid", "mdice_hid", "iou_L1", "iou_L2", "iou_L3"],
    "chamfer": ["cd_L0", "cd_L1", "cd_L2", "cd_L3", "cd_hidden", "cd_all", "f@10cm_hidden"],
    "occlusion": ["occ_p@10cm", "occ_r@10cm", "occ_f1@10cm", "scene_cd"],
}
TABLE_TITLE = {"visible": "Visible depth", "hidden": "Hidden depth", "occupancy": "Occupancy",
               "chamfer": "Chamfer / F-score", "occlusion": "Occluded geometry"}
TABLE_CAPTION = {
    "visible": r"Visible depth (layer 0). Models whose layer 0 is the ground truth are omitted.",
    "hidden": r"Hidden-layer depth on pixels where the GT layer exists (layers 1--3 pooled, and per layer).",
    "occupancy": r"Occupancy: IoU / Dice (\%) of the layer masks from the predicted stop index (layer $k$ exists iff "
                 r"stop $> k$); hidden = mean over layers 1--3.",
    "chamfer": r"Per-view Chamfer distance (cm) and F-score (\%), predicted points selected by the GT layer mask.",
    "occlusion": r"Occluded geometry (all views fused, after scale alignment): precision / recall / F1 (\%) of the "
                 r"GT points no input view sees, and whole-scene Chamfer distance (cm).",
}
TABLE_ORDER = ["visible", "hidden", "occupancy", "chamfer", "occlusion"]
DISPLAY_NAMES = {"mae": "MAE", "mum": "MuM", "mum_v2": "MuM-v2", "muskie": "Muskie", "muskie_v2": "Muskie-v2",
                 "vista": "Vista", "vistav2": "Vista-v2", "poincar3": "Poincar3", "gt_l0": "GT layer 0 only"}
PER_MODEL_SETTINGS = {"model_class", "layer0"}       # differ per model by design


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def dataset_label(ds, latex=True):
    if ds == AVG:
        return "Avg."
    name, _, split = ds.partition("/")
    name = re.sub(r"_LDI$", "", name)
    label = name if split in ("", "test") else f"{name} ({split})"
    return tex(label) if latex else label


# =============================================================================
# Store
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
    raise SystemExit(f"No readable LDI results store at {path}")


def uses_gt_layer0(rec):
    """Records whose layer 0 IS the GT (gt_l0 reference / --layer0 gt oracle)."""
    s = rec.get("settings", {})
    return s.get("model_class") == "gt_l0" or s.get("layer0") == "gt"


def _f(p, r):
    if np.isnan(p) or np.isnan(r):
        return float("nan")
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def stop_mask_scores(conf):
    """IoU / Dice of the layer masks from the stop index (layer k exists iff class > k), from the stop confusion
    (rows = GT). Same definitions as eval_ldi.stop_mask_scores, so older records can be filled in."""
    conf = np.asarray(conf, dtype=np.float64)
    out, iou, dice = {}, {}, {}
    for k in range(conf.shape[0] - 1):
        tp, fp, fn = conf[k + 1:, k + 1:].sum(), conf[:k + 1, k + 1:].sum(), conf[k + 1:, :k + 1].sum()
        if tp + fn == 0:
            continue
        iou[k], dice[k] = tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)
        out[f"stop_iou_L{k}"], out[f"stop_dice_L{k}"] = float(iou[k]), float(dice[k])
    for name, ks in (("", list(iou)), ("_hidden", [k for k in iou if k >= 1])):
        if ks:
            out[f"stop_miou{name}"] = float(np.mean([iou[k] for k in ks]))
            out[f"stop_mdice{name}"] = float(np.mean([dice[k] for k in ks]))
    return out


def select_records(store, allow_incomplete=False):
    """Latest complete record per (model, dataset) -> ({model: {dataset: record}}, [records]).
    Records without the IoU / Dice metrics get them from the stored stop confusion (GT stop 0 row dropped)."""
    chosen, flat, backfilled = defaultdict(dict), [], []
    for model, per_ds in store["runs"].items():
        for ds, hist in per_ds.items():
            rec = next((r for r in reversed(hist) if r["complete"] or allow_incomplete), None)
            if rec is None:
                warn(f"{model}/{ds}: no complete record -> left out (latest errors: {hist[-1].get('errors')})")
                continue
            if "stop_miou" not in rec["metrics"] and rec.get("stop_confusion") is not None:
                conf = np.array(rec["stop_confusion"], dtype=np.float64)
                conf[0, :] = 0                                    # drop GT stop 0 (missing GT depth)
                rec = dict(rec, metrics={**rec["metrics"], **stop_mask_scores(conf)})
                backfilled.append(f"{model}/{ds}")
            chosen[model][ds] = rec
            flat.append(rec)
    if backfilled:
        warn(f"IoU / Dice recomputed from the stored stop confusion for {backfilled}")
    return chosen, flat


def check_consistency(flat, strict=False):
    problems = []
    keys = sorted({k for r in flat for k in r["settings"]} - PER_MODEL_SETTINGS)
    for k in keys:
        vals = defaultdict(list)
        for r in flat:
            vals[repr(r["settings"].get(k))].append(f"{r['model']}/{r['dataset']}")
        if len(vals) > 1:
            problems.append(f"setting '{k}' differs: " + "; ".join(
                f"{v} <- {', '.join(w[:4])}" + (" ..." if len(w) > 4 else "") for v, w in vals.items()))
    flagged = [f"{r['model']}/{r['dataset']}" for r in flat if not r["complete"]]
    if flagged:
        problems.append(f"tables include flagged (incomplete) records: {flagged}")
    no_vis = sorted({r["model"] for r in flat if not uses_gt_layer0(r) and "vis_abs_rel" not in r["metrics"]})
    if no_vis:
        warn(f"no visible-depth metrics for {no_vis}: re-evaluate them with eval_ldi.py --overwrite")
    for ds in sorted({r["dataset"] for r in flat}):
        px = {r["model"]: r["metrics"].get("vis_num_pixels") for r in flat
              if r["dataset"] == ds and not uses_gt_layer0(r) and r["metrics"].get("vis_num_pixels")}
        if len(set(px.values())) > 1:
            warn(f"{ds}: visible-depth pixel counts differ across models {px} "
                 "(some model predicts no depth on part of the GT pixels)")
    shas = {r.get("code_sha") for r in flat}
    if len(shas) > 1:
        warn(f"records come from {len(shas)} versions of eval_ldi.py {sorted(map(str, shas))}")
    for p in problems:
        warn("CONSISTENCY: " + p)
    if problems and strict:
        raise SystemExit("Inconsistent records (see warnings above); fix them or drop --strict.")


# =============================================================================
# Tables
# =============================================================================
def get_value(chosen, model, ds, key, datasets):
    if ds == AVG:  # average only when every dataset is present
        vals = [get_value(chosen, model, d, key, datasets) for d in datasets]
        return float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
    rec = chosen.get(model, {}).get(ds)
    v = None if rec is None else rec["metrics"].get(key)
    return None if v is None or not np.isfinite(v) else float(v)


def column_ranks(values, spec):
    """{model: rank} on the DISPLAYED (rounded) values, so visual ties share a rank."""
    shown = {m: float(spec["fmt"].format(v * spec["scale"])) for m, v in values.items() if v is not None}
    uniq = sorted(set(shown.values()), reverse=(spec["better"] == "max"))
    return {m: uniq.index(s) for m, s in shown.items()} if len(uniq) >= 2 else {}


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
    """-> ({table: [spec]}, main columns or None). "main=k1,k2" picks main-table columns from any table."""
    chosen = {t: list(v) for t, v in DEFAULT_METRICS.items()}
    main_cols = None
    if spec_str:
        for part in spec_str.split(";"):
            if not part.strip():
                continue
            table, keys = part.split("=")
            table = table.strip()
            if table == "main":
                lookup = {k: (TABLE_TITLE[t], t, k) for t in TABLE_ORDER for k in CATALOG[t]}
                lookup.update({f"{t}.{k}": (TABLE_TITLE[t], t, k) for t in TABLE_ORDER for k in CATALOG[t]})
                lookup.update({spec["key"]: (TABLE_TITLE[t], t, k) for t in TABLE_ORDER for k, spec in CATALOG[t].items()})
                keys = [k.strip() for k in keys.split(",") if k.strip()]
                bad = [k for k in keys if k not in lookup]
                if bad:
                    raise SystemExit(f"unknown main metric(s) {bad}; use table.metric, e.g. hidden.abs_rel")
                main_cols = [lookup[k] for k in keys]
                continue
            if table not in CATALOG:
                raise SystemExit(f"unknown table '{table}' (choose from {list(CATALOG)})")
            keys = [k.strip() for k in keys.split(",") if k.strip()]
            bad = [k for k in keys if k not in CATALOG[table]]
            if bad:
                raise SystemExit(f"unknown {table} metric(s) {bad}; available: {list(CATALOG[table])}")
            chosen[table] = keys
    return {t: [CATALOG[t][k] for k in keys] for t, keys in chosen.items()}, main_cols


def rows_for(table, models, chosen):
    """The visible-depth table leaves out models whose layer 0 is the GT (they would score perfectly)."""
    if table != "visible":
        return models
    return [m for m in models if not any(uses_gt_layer0(r) for r in chosen.get(m, {}).values())]


def task_table(chosen, table, specs, models, datasets, per_block, names):
    models = rows_for(table, models, chosen)
    cols = datasets + [AVG]
    n_blocks = -(-len(cols) // per_block)
    size = -(-len(cols) // n_blocks)  # balanced blocks, so Avg. is never alone
    blocks = [cols[i:i + size] for i in range(0, len(cols), size)]
    width = max(len(b) for b in blocks)
    ncol = width * len(specs)

    L = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{TABLE_CAPTION[table]}}}",
         rf"\label{{tab:ldi_{table}}}", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{l{'c' * ncol}}}",
         r"\toprule"]
    for bi, block in enumerate(blocks):
        if bi > 0:
            L.append(r"\midrule")
        pad = (width - len(block)) * len(specs)
        head = [rf"\multicolumn{{{len(specs)}}}{{c}}{{{dataset_label(d)}}}" for d in block] + [""] * pad
        L.append(" & " + " & ".join(head) + r" \\")
        L.append(" ".join(rf"\cmidrule(lr){{{2 + i * len(specs)}-{1 + (i + 1) * len(specs)}}}"
                          for i in range(len(block))))
        L.append("Method & " + " & ".join([s["label"] for _ in block for s in specs] + [""] * pad) + r" \\")
        L.append(r"\midrule")
        ranks = {(d, s["key"]): column_ranks({m: get_value(chosen, m, d, s["key"], datasets) for m in models}, s)
                 for d in block for s in specs}
        for m in models:
            cells = [fmt_cell(get_value(chosen, m, d, s["key"], datasets), s, ranks[(d, s["key"])].get(m))
                     for d in block for s in specs]
            L.append(names(m) + " & " + " & ".join(cells + [""] * pad) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def main_value(chosen, m, d, t, spec, datasets):
    if t == "visible" and m not in rows_for("visible", [m], chosen):
        return None                                               # layer 0 IS the GT: not a real number
    return get_value(chosen, m, d, spec["key"], datasets)


def main_specs(cols=None):
    cols = cols or MAIN_COLUMNS
    return [(g, t, CATALOG[t][k]) for g, t, k in cols]


def main_table(chosen, models, datasets, names, cols=None, markdown=False):
    """ONE table: datasets (+ Avg.) as row blocks, metric groups as columns."""
    specs = main_specs(cols)
    blocks = datasets + ([AVG] if len(datasets) > 1 else [])
    if markdown:
        L = ["### LDI probing (all metrics)"]
        for d in blocks:
            L += ["", f"**{dataset_label(d, latex=False)}**", "",
                  "| Method | " + " | ".join(f"{g}: {s['md']}" for g, _, s in specs) + " |",
                  "|---|" + "---:|" * len(specs)]
            ranks = [column_ranks({m: main_value(chosen, m, d, t, s, datasets) for m in models}, s)
                     for _, t, s in specs]
            for m in models:
                L.append(f"| {names(m)} | " + " | ".join(
                    fmt_cell(main_value(chosen, m, d, t, s, datasets), s, r.get(m), latex=False)
                    for (_, t, s), r in zip(specs, ranks)) + " |")
        return "\n".join(L)
    groups = []
    for g, _, _ in specs:
        if groups and groups[-1][0] == g:
            groups[-1][1] += 1
        else:
            groups.append([g, 1])
    head, rules, c = [], [], 2
    for g, n in groups:
        head.append(rf"\multicolumn{{{n}}}{{c}}{{{g}}}")
        rules.append(rf"\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    L = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{MAIN_CAPTION}}}", r"\label{tab:ldi_main}",
         r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{l{'c' * len(specs)}}}", r"\toprule",
         " & " + " & ".join(head) + r" \\", " ".join(rules),
         "Method & " + " & ".join(s["label"] for _, _, s in specs) + r" \\"]
    for d in blocks:
        L += [r"\midrule", rf"\multicolumn{{{len(specs) + 1}}}{{l}}{{\textit{{{dataset_label(d)}}}}} \\"]
        ranks = [column_ranks({m: main_value(chosen, m, d, t, s, datasets) for m in models}, s) for _, t, s in specs]
        for m in models:
            L.append(names(m) + " & " + " & ".join(
                fmt_cell(main_value(chosen, m, d, t, s, datasets), s, r.get(m))
                for (_, t, s), r in zip(specs, ranks)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def markdown_table(chosen, table, specs, models, datasets, names):
    """Terminal-friendly: one small table per metric, rows = datasets (+ Avg.), columns = models."""
    models = rows_for(table, models, chosen)
    L = [f"### {TABLE_TITLE[table]}"]
    for s in specs:
        L += ["", f"**{s['md']}**", "", "| Dataset | " + " | ".join(names(m) for m in models) + " |",
              "|---|" + "---:|" * len(models)]
        for d in datasets + [AVG]:
            vals = {m: get_value(chosen, m, d, s["key"], datasets) for m in models}
            ranks = column_ranks(vals, s)
            L.append(f"| {dataset_label(d, latex=False)} | "
                     + " | ".join(fmt_cell(vals[m], s, ranks.get(m), latex=False) for m in models) + " |")
    return "\n".join(L)


def order(available, requested, what):
    if not requested:
        return sorted(available)
    missing = [x for x in requested if x not in available]
    if missing:
        warn(f"{what} not in the store (skipped): {missing}")
    return [x for x in requested if x in available]


def make_namer(chosen, renames, latex=True):
    def name(m):
        if m in renames:
            return renames[m]
        rec = next(iter(chosen.get(m, {}).values()), None)
        base = (rec or {}).get("display_name") or DISPLAY_NAMES.get(m, m)
        if rec and rec["settings"].get("layer0") == "gt" and m != "gt_l0":
            base += " (GT L0)"
        return tex(base) if latex else base
    return name


def build(chosen, tables=("main",), models=None, datasets=None, metrics=None, per_block=6,
          markdown=False, renames=None):
    """Returns {table_name: text}."""
    specs, main_cols = resolve_metrics(metrics)
    model_list = order(chosen.keys(), models, "models")
    ds_list = order({d for r in chosen.values() for d in r}, datasets, "datasets")
    missing = [(m, d) for m in model_list for d in ds_list if d not in chosen[m]]
    if missing:
        warn(f"missing (model, dataset) pairs -> shown as '--' and no Avg. for that model: {missing}")
    present = [t for t in TABLE_ORDER
               if any(get_value(chosen, m, d, s["key"], ds_list) is not None
                      for s in specs[t] for m in model_list for d in ds_list)]
    names = make_namer(chosen, renames or {}, latex=not markdown)

    out = {}
    if "main" in tables:
        out["main"] = main_table(chosen, model_list, ds_list, names, main_cols, markdown)
    for t in present:
        if t in tables:
            out[t] = (markdown_table(chosen, t, specs[t], model_list, ds_list, names) if markdown else
                      task_table(chosen, t, specs[t], model_list, ds_list, per_block, names))
    return out


STANDALONE_HEAD = r"""\documentclass{article}
\usepackage[margin=1.5cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}
"""


def write_dir(out_dir, tables, chosen):
    os.makedirs(out_dir, exist_ok=True)
    for name, text in tables.items():
        with open(osp.join(out_dir, f"ldi_{name}.tex"), "w") as f:
            f.write(text + "\n")
    with open(osp.join(out_dir, "ldi_all_tables.tex"), "w") as f:
        f.write("% Generated by print_ldi_tables.py -- requires \\usepackage{booktabs,graphicx}\n"
                + "\n\n".join(tables.values()) + "\n")
    with open(osp.join(out_dir, "ldi_results_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "metric", "value"])
        for m in sorted(chosen):
            for d, rec in sorted(chosen[m].items()):
                for k, v in sorted(rec["metrics"].items()):
                    w.writerow([m, d, k, v])
    with open(osp.join(out_dir, "ldi_results_latest.json"), "w") as f:
        json.dump({f"{m}/{d}": {k: v for k, v in rec.items() if k not in ("per_sample", "stop_confusion")}
                   for m, per in chosen.items() for d, rec in per.items()}, f, indent=2, default=str)


def main():
    p = argparse.ArgumentParser(description="Print LaTeX tables from an eval_ldi.py results_ldi.pkl",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("pkl", help="Path to results_ldi.pkl (or a results_dir containing it)")
    p.add_argument("--tables", default="main",
                   help=f"Comma-separated subset of: main, {', '.join(TABLE_ORDER)}, or all")
    p.add_argument("--models", nargs="*", default=None, help="Row order (default: alphabetical)")
    p.add_argument("--datasets", nargs="*", default=None, help="Column order / subset, as stored")
    p.add_argument("--metrics", default=None,
                   help='Columns, e.g. "main=visible.abs_rel,occupancy.miou_hid;occlusion=occ_f1@10cm"')
    p.add_argument("--rename", nargs="*", default=[], help='Row names, e.g. "vistav2=Vista-v2"')
    p.add_argument("--datasets_per_block", type=int, default=6)
    p.add_argument("--markdown", action="store_true", help="Markdown instead of LaTeX (quick terminal view)")
    p.add_argument("--standalone", action="store_true", help="Wrap in a compilable LaTeX document")
    p.add_argument("--allow_incomplete", action="store_true", help="Also use flagged records")
    p.add_argument("--strict", action="store_true", help="Fail if records used different settings")
    p.add_argument("--list_metrics", action="store_true", help="List metric keys per table and exit")
    p.add_argument("--list", action="store_true", help="List models and datasets in the store and exit")
    p.add_argument("-o", "--output", default=None, help="Write to this file instead of stdout")
    p.add_argument("--out_dir", default=None, help="Write one .tex per table + all tables + CSV/JSON here")
    args = p.parse_args()

    if args.list_metrics:
        print("main       " + ", ".join(f"{t}.{k}" for _, t, k in MAIN_COLUMNS) + "  (default columns)")
        for t, ms in CATALOG.items():
            print(f"{t:<10} " + ", ".join(f"{k}{' (default)' if k in DEFAULT_METRICS[t] else ''}" for k in ms))
        return

    path = osp.join(args.pkl, "results_ldi.pkl") if osp.isdir(args.pkl) else args.pkl
    store = load_store(path)
    chosen, flat = select_records(store, args.allow_incomplete)
    if args.list:
        print("models:  ", " ".join(sorted(chosen)))
        print("datasets:", " ".join(sorted({d for r in chosen.values() for d in r})))
        return
    if not chosen:
        raise SystemExit("No usable (complete) records in the store; try --allow_incomplete.")
    check_consistency(flat, args.strict)

    renames = dict(r.split("=", 1) for r in args.rename)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    if tables == ["all"]:
        tables = ["main", *TABLE_ORDER]
    bad = [t for t in tables if t not in ("main", *TABLE_ORDER)]
    if bad:
        raise SystemExit(f"unknown table(s) {bad}; choose from main, {', '.join(TABLE_ORDER)}, all")
    out = build(chosen, tables, args.models, args.datasets, args.metrics, args.datasets_per_block,
                args.markdown, renames)
    if not out:
        raise SystemExit("Nothing to print for the selected tables / models / datasets.")

    if args.out_dir and not args.markdown:
        write_dir(args.out_dir, out, chosen)
        print(f"[print_ldi_tables] wrote tables to {args.out_dir}", file=sys.stderr)
        return
    text = "\n\n".join(out.values())
    if args.standalone and not args.markdown:
        text = STANDALONE_HEAD + text + "\n\\end{document}"
    elif not args.markdown:
        text = "% Generated by print_ldi_tables.py -- requires \\usepackage{booktabs,graphicx}\n" + text
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"[print_ldi_tables] wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()