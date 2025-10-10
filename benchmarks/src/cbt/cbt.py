# benchmarks/tasks/cbt.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.cache import try_load, save, stable_hash, config_signature, cleanup_cache
from src.utils.io import load_yaml_file
from src.metrics import compute_binary, plot_confusion
from src.cbt.openai_cbt import predict_stage1, predict_stage2, predict_stage3, predict_stage_with_config, parse_decision
from src.openai.client_openai import chat_complete_many

def _load_prompts_for_cbt(root, prompts_path):
    """
    Load prompts YAML once. If prompts_path is None, load default 'configs/cbt_prompts.yaml'.
    """
    from src.utils.io import load_yaml_file
    # Use provided prompts_path or fall back to default
    path = prompts_path if prompts_path else "cbt/prompts_cbt.yaml"
    return load_yaml_file(root / path)

def _run_cbt_parallel(df, cols, cfg, fig_dir, table_dir, json_dir, logger,
                      cache_dir, cfg_sig, prompts_path,
                      all_trues, all_preds, all_pairs, use_cache, num_rows):
    """
    Parallel path with real async IO via chat_complete_many:
    - Load system prompts once.
    - For each stage, build a batch of uncached items and call chat_complete_many with parallel=True.
    - Respect caching, fill results, and update tracking containers.
    """
    logger.info(f"[CBT][Parallel] Running real async path on {num_rows} rows...")

    # Resolve prompts once (system prompts per stage)
    root = Path(__file__).resolve().parent.parent
    prompts = _load_prompts_for_cbt(root, prompts_path)
    reasoner = prompts.get("reasoner", {})
    sys_stage1 = reasoner.get("stage1")
    sys_stage2 = reasoner.get("stage2")
    sys_stage3 = reasoner.get("stage3")

    # Read OpenAI params and parallel controls
    api_base = cfg.get("api_base")
    model = cfg.get("model")
    effort = cfg.get("effort", "low")
    timeout_seconds = cfg.get("timeout_seconds")
    max_batch = int(cfg.get("parallel_max_batch", 1))

    # ---- Stage 1 batch ----
    logger.info("[CBT][Parallel] Building Stage1 batch (skip cached)...")
    items_s1 = []
    map_idx_s1 = []  # map batch index -> row index
    keys_s1 = []
    s1_labels = []  # ground truth labels in order processed

    for i in range(num_rows):
        row = df.iloc[i]
        stmt = str(row[cols["statement"]])
        s1_text = str(row[cols["stage1_text"]])
        lab1 = row[cols["stage1_label"]]
        if pd.isna(lab1):
            logger.warning(f"[CBT][Parallel][Stage1] Missing label at row {i}; skipping.")
            continue
        s1_lab = int(lab1)
        k1 = f"cbt::s1::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}"
        cached1 = try_load(cache_dir, k1) if use_cache else None
        if cached1 is not None:
            all_trues["stage1"].append(s1_lab)
            all_preds["stage1"].append(int(cached1["label"]))
            all_pairs["stage1"].append({
                "idx": i,
                "stage": "stage1",
                "y_true": s1_lab,
                "y_pred": int(cached1["label"]),
                "statement": stmt,
                "candidate": s1_text,
                "raw": cached1["raw"]
            })
        else:
            payload = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text};"'
            items_s1.append({"system_content": sys_stage1, "user_content": payload})
            map_idx_s1.append(i)
            keys_s1.append(k1)
        if i % 200 == 0 and i > 0:
            logger.info(f"[CBT][Parallel][Stage1] Scanned {i}/{num_rows} rows...")

    logger.info(f"[CBT][Parallel][Stage1] Cached={len(all_pairs['stage1'])}, ToRequest={len(items_s1)}")
    if len(items_s1) > 0:
        logger.info(f"[CBT][Parallel][Stage1] Dispatching {len(items_s1)} requests with max_batch={max_batch}")
        raws = chat_complete_many(api_base, model, items_s1, effort, timeout_seconds, True, max_batch)
        for j, raw in enumerate(raws):
            i = map_idx_s1[j]
            row = df.iloc[i]
            stmt = str(row[cols["statement"]])
            s1_text = str(row[cols["stage1_text"]]); s1_lab = int(row[cols["stage1_label"]])
            label = parse_decision(raw)
            r1 = {"label": label, "raw": raw}
            if use_cache and cfg["save_intermediate"]:
                save(cfg["cache_dir"], keys_s1[j], r1)
            all_trues["stage1"].append(s1_lab)
            all_preds["stage1"].append(int(r1["label"]))
            all_pairs["stage1"].append({
                "idx": i,
                "stage": "stage1",
                "y_true": s1_lab,
                "y_pred": int(r1["label"]),
                "statement": stmt,
                "candidate": s1_text,
                "raw": r1["raw"]
            })

    # ---- Stage 2 batch ----
    logger.info("[CBT][Parallel] Building Stage2 batch (skip cached)...")
    items_s2 = []
    map_idx_s2 = []
    keys_s2 = []
    for i in range(num_rows):
        row = df.iloc[i]
        stmt = str(row[cols["statement"]])
        s1_text = str(row[cols["stage1_text"]])
        s2_text = str(row[cols["stage2_text"]])
        lab2 = row[cols["stage2_label"]]
        if pd.isna(lab2):
            logger.warning(f"[CBT][Parallel][Stage2] Missing label at row {i}; skipping.")
            continue
        s2_lab = int(lab2)
        k2 = f"cbt::s2::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}::{stable_hash(s2_text)}"
        cached2 = try_load(cache_dir, k2) if use_cache else None
        if cached2 is not None:
            all_trues["stage2"].append(s2_lab)
            all_preds["stage2"].append(int(cached2["label"]))
            all_pairs["stage2"].append({
                "idx": i,
                "stage": "stage2",
                "y_true": s2_lab,
                "y_pred": int(cached2["label"]),
                "statement": stmt,
                "candidate": s2_text,
                "raw": cached2["raw"]
            })
        else:
            payload = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text}; CHALLENGE: {s2_text};"'
            items_s2.append({"system_content": sys_stage2, "user_content": payload})
            map_idx_s2.append(i)
            keys_s2.append(k2)
        if i % 200 == 0 and i > 0:
            logger.info(f"[CBT][Parallel][Stage2] Scanned {i}/{num_rows} rows...")

    logger.info(f"[CBT][Parallel][Stage2] Cached={len(all_pairs['stage2'])}, ToRequest={len(items_s2)}")
    if len(items_s2) > 0:
        logger.info(f"[CBT][Parallel][Stage2] Dispatching {len(items_s2)} requests with max_batch={max_batch}")
        raws = chat_complete_many(api_base, model, items_s2, effort, timeout_seconds, True, max_batch)
        for j, raw in enumerate(raws):
            i = map_idx_s2[j]
            row = df.iloc[i]
            stmt = str(row[cols["statement"]])
            s1_text = str(row[cols["stage1_text"]]); s2_text = str(row[cols["stage2_text"]]); s2_lab = int(row[cols["stage2_label"]])
            label = parse_decision(raw)
            r2 = {"label": label, "raw": raw}
            if use_cache and cfg["save_intermediate"]:
                save(cfg["cache_dir"], keys_s2[j], r2)
            all_trues["stage2"].append(s2_lab)
            all_preds["stage2"].append(int(r2["label"]))
            all_pairs["stage2"].append({
                "idx": i,
                "stage": "stage2",
                "y_true": s2_lab,
                "y_pred": int(r2["label"]),
                "statement": stmt,
                "candidate": s2_text,
                "raw": r2["raw"]
            })

    # ---- Stage 3 batch ----
    logger.info("[CBT][Parallel] Building Stage3 batch (skip cached)...")
    items_s3 = []
    map_idx_s3 = []
    keys_s3 = []
    for i in range(num_rows):
        row = df.iloc[i]
        stmt = str(row[cols["statement"]])
        s1_text = str(row[cols["stage1_text"]])
        s2_text = str(row[cols["stage2_text"]])
        s3_text = str(row[cols["stage3_text"]])
        lab3 = row[cols["stage3_label"]]
        if pd.isna(lab3):
            logger.warning(f"[CBT][Parallel][Stage3] Missing label at row {i}; skipping.")
            continue
        s3_lab = int(lab3)
        k3 = f"cbt::s3::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}::{stable_hash(s2_text)}::{stable_hash(s3_text)}"
        cached3 = try_load(cache_dir, k3) if use_cache else None
        if cached3 is not None:
            all_trues["stage3"].append(s3_lab)
            all_preds["stage3"].append(int(cached3["label"]))
            all_pairs["stage3"].append({
                "idx": i,
                "stage": "stage3",
                "y_true": s3_lab,
                "y_pred": int(cached3["label"]),
                "statement": stmt,
                "candidate": s3_text,
                "raw": cached3["raw"]
            })
        else:
            payload = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text}; CHALLENGE: {s2_text}; REFRAME: {s3_text};"'
            items_s3.append({"system_content": sys_stage3, "user_content": payload})
            map_idx_s3.append(i)
            keys_s3.append(k3)
        if i % 200 == 0 and i > 0:
            logger.info(f"[CBT][Parallel][Stage3] Scanned {i}/{num_rows} rows...")

    logger.info(f"[CBT][Parallel][Stage3] Cached={len(all_pairs['stage3'])}, ToRequest={len(items_s3)}")
    if len(items_s3) > 0:
        logger.info(f"[CBT][Parallel][Stage3] Dispatching {len(items_s3)} requests with max_batch={max_batch}")
        raws = chat_complete_many(api_base, model, items_s3, effort, timeout_seconds, True, max_batch)
        for j, raw in enumerate(raws):
            i = map_idx_s3[j]
            row = df.iloc[i]
            stmt = str(row[cols["statement"]])
            s3_text = str(row[cols["stage3_text"]]); s3_lab = int(row[cols["stage3_label"]])
            label = parse_decision(raw)
            r3 = {"label": label, "raw": raw}
            if use_cache and cfg["save_intermediate"]:
                save(cfg["cache_dir"], keys_s3[j], r3)
            all_trues["stage3"].append(s3_lab)
            all_preds["stage3"].append(int(r3["label"]))
            all_pairs["stage3"].append({
                "idx": i,
                "stage": "stage3",
                "y_true": s3_lab,
                "y_pred": int(r3["label"]),
                "statement": stmt,
                "candidate": s3_text,
                "raw": r3["raw"]
            })

    logger.info("[CBT][Parallel] Completed all stages with async batches.")

def _run_cbt_sequential(df, cols, cfg, fig_dir, table_dir, json_dir, logger,
                        cache_dir, cfg_sig, prompts_path,
                        all_trues, all_preds, all_pairs, use_cache, num_rows):
    """
    Sequential path:
    - Iterate rows and perform per-stage predictions.
    - Honor caching, populate tracking containers, and save intermediates when configured.
    """
    logger.info(f"[CBT][Sequential] Running on {num_rows} rows...")
    for i in tqdm(range(num_rows), desc="CBT"):
        row = df.iloc[i]
        # Extract statement and per-stage texts from the row
        stmt = str(row[cols["statement"]])
        s1_text = str(row[cols["stage1_text"]])
        s2_text = str(row[cols["stage2_text"]])
        s3_text = str(row[cols["stage3_text"]])

        # ---- Stage 1: UNHELPFUL_THOUGHTS ----
        lab1 = row[cols["stage1_label"]]
        if pd.isna(lab1):
            logger.warning(f"[CBT][Sequential][Stage1] Missing label at row {i}; skipping stage1.")
        else:
            s1_lab = int(lab1)
            # Build a unique cache key for this input
            k1 = f"cbt::s1::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}"
            cached1 = try_load(cache_dir, k1) if use_cache else None
            r1 = cached1 or predict_stage1(stmt, s1_text)
            # Save result to cache if not already cached and saving is enabled
            if use_cache and cfg["save_intermediate"] and cached1 is None:
                save(cfg["cache_dir"], k1, r1)
            # Record ground truth, prediction, and detailed info for this example
            all_trues["stage1"].append(s1_lab)
            all_preds["stage1"].append(int(r1["label"]))
            all_pairs["stage1"].append({
                "idx": i,
                "stage": "stage1",
                "y_true": s1_lab,
                "y_pred": int(r1["label"]),
                "statement": stmt,
                "candidate": s1_text,
                "raw": r1["raw"]
            })
        if i % 50 == 0:
            logger.info(f"[CBT][Sequential] Processed {i+1}/{num_rows} rows (stage1)")

        # ---- Stage 2: CHALLENGE ----
        lab2 = row[cols["stage2_label"]]
        if pd.isna(lab2):
            logger.warning(f"[CBT][Sequential][Stage2] Missing label at row {i}; skipping stage2.")
        else:
            s2_lab = int(lab2)
            # Build a unique cache key for this input
            k2 = f"cbt::s2::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}::{stable_hash(s2_text)}"
            cached2 = try_load(cache_dir, k2) if use_cache else None
            r2 = cached2 or predict_stage2(stmt, s1_text, s2_text)
            # Save result to cache if not already cached and saving is enabled
            if use_cache and cfg["save_intermediate"] and cached2 is None:
                save(cfg["cache_dir"], k2, r2)
            # Record ground truth, prediction, and detailed info for this example
            all_trues["stage2"].append(s2_lab)
            all_preds["stage2"].append(int(r2["label"]))
            all_pairs["stage2"].append({
                "idx": i,
                "stage": "stage2",
                "y_true": s2_lab,
                "y_pred": int(r2["label"]),
                "statement": stmt,
                "candidate": s2_text,
                "raw": r2["raw"]
            })
        if i % 50 == 0:
            logger.info(f"[CBT][Sequential] Processed {i+1}/{num_rows} rows (stage2)")

        # ---- Stage 3: REFRAME ----
        lab3 = row[cols["stage3_label"]]
        if pd.isna(lab3):
            logger.warning(f"[CBT][Sequential][Stage3] Missing label at row {i}; skipping stage3.")
        else:
            s3_lab = int(lab3)
            # Build a unique cache key for this input
            k3 = f"cbt::s3::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}::{stable_hash(s2_text)}::{stable_hash(s3_text)}"
            cached3 = try_load(cache_dir, k3) if use_cache else None
            r3 = cached3 or predict_stage3(stmt, s1_text, s2_text, s3_text)
            # Save result to cache if not already cached and saving is enabled
            if use_cache and cfg["save_intermediate"] and cached3 is None:
                save(cfg["cache_dir"], k3, r3)
            # Record ground truth, prediction, and detailed info for this example
            all_trues["stage3"].append(s3_lab)
            all_preds["stage3"].append(int(r3["label"]))
            all_pairs["stage3"].append({
                "idx": i,
                "stage": "stage3",
                "y_true": s3_lab,
                "y_pred": int(r3["label"]),
                "statement": stmt,
                "candidate": s3_text,
                "raw": r3["raw"]
            })
        if i % 50 == 0:
            logger.info(f"[CBT][Sequential] Processed {i+1}/{num_rows} rows (stage3)")

    logger.info(f"[CBT][Sequential] Completed all {num_rows} rows.")

def run_cbt(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
    """
    Evaluate CBT reasoner correctness per stage using agent's prompts.
    - Reads dataset row-wise.
    - For each stage: calls the agent bridge to obtain label and raw output.
    - Aggregates per-stage and overall metrics and writes confusion matrices.
    """
    # Prepare cache directory for storing intermediate results
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[CBT] Cache directory prepared at {cache_dir}")
    # Optional: clean up old cache entries to avoid infinite growth
    max_age_days = cfg.get("cache_max_age_days", None)
    if max_age_days is not None:
        logger.info(f"[CBT] Cleaning up cache entries older than {max_age_days} days...")
        cleanup_cache(cache_dir, max_age_days)

    # Initialize containers for ground truth labels, predictions, and detailed pairs for each stage
    all_trues = {"stage1": [], "stage2": [], "stage3": []}
    all_preds = {"stage1": [], "stage2": [], "stage3": []}
    all_pairs = {"stage1": [], "stage2": [], "stage3": []}

    # Optional: path to custom prompts YAML for agent
    prompts_path = cfg.get("prompts_path", None)

    # Merge OpenAI config so GPT params are available when prompts_path is used
    root = Path(__file__).resolve().parent.parent
    openai_cfg = load_yaml_file(root / "openai" / "config_openai.yaml")
    cfg = {**openai_cfg, **cfg}

    # Parallel controls
    parallel = bool(cfg.get("parallel", False))
    parallel_max_batch = int(cfg.get("parallel_max_batch", 1))

    # Limit examples
    max_examples = cfg.get("max_examples", None)
    num_rows = len(df) if max_examples is None else min(int(max_examples), len(df))

    logger.info(f"[CBT] Starting run: {num_rows} rows, parallel={parallel}, batch={parallel_max_batch}, prompts_path={prompts_path}")

    # Build config signature
    cfg_sig = config_signature(cfg)
    use_cache = bool(cfg.get("use_cache", True))

    if parallel:
        logger.info("[CBT] Using parallel execution path.")
        _run_cbt_parallel(
            df, cols, cfg, fig_dir, table_dir, json_dir, logger,
            cache_dir, cfg_sig, prompts_path,
            all_trues, all_preds, all_pairs, use_cache, num_rows
        )
    else:
        logger.info("[CBT] Using sequential execution path.")
        _run_cbt_sequential(
            df, cols, cfg, fig_dir, table_dir, json_dir, logger,
            cache_dir, cfg_sig, prompts_path,
            all_trues, all_preds, all_pairs, use_cache, num_rows
        )

    # ---- Compute metrics and save results ----
    import json as _json, numpy as _np
    out = {}

    logger.info("[CBT] Computing metrics and saving results...")

    # Compute and save metrics, confusion matrices, and detailed pairs for each stage
    for st in ["stage1", "stage2", "stage3"]:
        # Compute binary classification metrics for this stage
        mt = compute_binary(all_trues[st], all_preds[st])
        out[st] = mt
        logger.info(f"[CBT] {st}: accuracy={mt['accuracy']:.3f}, precision={mt['precision_pos1']:.3f}, recall={mt['recall_pos1']:.3f}, f1={mt['f1_pos1']:.3f}")
        # Plot and save confusion matrix for this stage
        plot_confusion(all_trues[st], all_preds[st], [0, 1], f"Confusion - {st}", str(Path(fig_dir) / f"cm_{st}.png"))
        logger.debug(f"[CBT] Saved confusion matrix for {st} to {Path(fig_dir) / f'cm_{st}.png'}")
        # Save detailed prediction pairs for this stage
        pd.DataFrame(all_pairs[st]).to_csv(Path(table_dir) / f"pairs_{st}.csv", index=False)
        logger.debug(f"[CBT] Saved pairs for {st} to {Path(table_dir) / f'pairs_{st}.csv'}")

    # ---- Compute overall metrics across all stages ----
    y_true_all = all_trues["stage1"] + all_trues["stage2"] + all_trues["stage3"]
    y_pred_all = all_preds["stage1"] + all_preds["stage2"] + all_preds["stage3"]
    mt_all = compute_binary(y_true_all, y_pred_all)
    out["overall"] = mt_all
    logger.info(f"[CBT] overall: accuracy={mt_all['accuracy']:.3f}, precision={mt_all['precision_pos1']:.3f}, recall={mt_all['recall_pos1']:.3f}, f1={mt_all['f1_pos1']:.3f}")
    # Plot and save overall confusion matrix
    plot_confusion(y_true_all, y_pred_all, [0, 1], "Confusion - overall", str(Path(fig_dir) / "cm_overall.png"))
    logger.debug(f"[CBT] Saved overall confusion matrix to {Path(fig_dir) / 'cm_overall.png'}")

    # Save summary metrics as JSON 
    out_compact = {
        scope: {k: v for k, v in metrics.items() if k != "report"}
        for scope, metrics in out.items()
    }
    with open(Path(json_dir) / "summary.json", "w") as f:
        _json.dump(out_compact, f, ensure_ascii=False, indent=2)
    logger.info(f"[CBT] Saved summary metrics (compact) to {Path(json_dir) / 'summary.json'}")

    # Save summary metrics as CSV (for each stage and overall)
    pd.DataFrame([
        {
            "scope": k,
            "accuracy": out[k]["accuracy"],
            "precision_pos1": out[k]["precision_pos1"],
            "recall_pos1": out[k]["recall_pos1"],
            "f1_pos1": out[k]["f1_pos1"],
        }
        for k in ["stage1", "stage2", "stage3", "overall"]
    ]).to_csv(Path(table_dir) / "metrics.csv", index=False)
    logger.info(f"[CBT] Saved metrics CSV to {Path(table_dir) / 'metrics.csv'}")