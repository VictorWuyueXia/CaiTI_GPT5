# benchmarks/tasks/cbt.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.cache import try_load, save, stable_hash, config_signature, cleanup_cache
from utils.io import load_yaml_file
from eval.metrics import compute_binary, plot_confusion
from agent_bridge.cbt import predict_stage1, predict_stage2, predict_stage3, predict_stage_with_config

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
    # Optional: clean up old cache entries to avoid infinite growth
    max_age_days = cfg.get("cache_max_age_days", None)
    if max_age_days is not None:
        cleanup_cache(cache_dir, max_age_days)

    # Initialize containers for ground truth labels, predictions, and detailed pairs for each stage
    all_trues = {"stage1": [], "stage2": [], "stage3": []}
    all_preds = {"stage1": [], "stage2": [], "stage3": []}
    all_pairs = {"stage1": [], "stage2": [], "stage3": []}

    # Optional: path to custom prompts YAML for agent
    prompts_path = cfg.get("prompts_path", None)

    # Merge OpenAI config so GPT params are available when prompts_path is used
    root = Path(__file__).resolve().parent.parent  # benchmarks/
    openai_cfg = load_yaml_file(root / "configs" / "openai_config.yaml")
    cfg = {**openai_cfg, **cfg}

    # Limit examples for dev speed if configured
    max_examples = cfg.get("max_examples", None)
    num_rows = len(df) if max_examples is None else min(int(max_examples), len(df))

    # Build a stable configuration signature used in cache keys
    cfg_sig = config_signature(cfg)
    use_cache = bool(cfg.get("use_cache", True))

    # Iterate over each row in the dataset
    for i in tqdm(range(num_rows), desc="CBT"):
        row = df.iloc[i]
        # Extract statement and per-stage texts/labels from the row
        stmt = str(row[cols["statement"]])
        s1_text = str(row[cols["stage1_text"]]); s1_lab = int(row[cols["stage1_label"]])
        s2_text = str(row[cols["stage2_text"]]); s2_lab = int(row[cols["stage2_label"]])
        s3_text = str(row[cols["stage3_text"]]); s3_lab = int(row[cols["stage3_label"]])

        # ---- Stage 1: UNHELPFUL_THOUGHTS ----
        # Build a unique cache key for this input
        k1 = f"cbt::s1::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}"
        cached1 = try_load(cache_dir, k1) if use_cache else None
        if prompts_path:
            # If using custom prompts, build the payload and call the generic stage predictor
            payload1 = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text};"'
            r1 = cached1 or predict_stage_with_config(
                "stage1",
                payload1,
                prompts_yaml_path=prompts_path,
                api_base=cfg.get("api_base"),
                model=cfg.get("model"),
                effort=cfg.get("effort"),
                timeout_seconds=cfg.get("timeout_seconds"),
            )
        else:
            # Otherwise, use the default stage1 predictor
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

        # ---- Stage 2: CHALLENGE ----
        # Build a unique cache key for this input
        k2 = f"cbt::s2::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}::{stable_hash(s2_text)}"
        cached2 = try_load(cache_dir, k2) if use_cache else None
        if prompts_path:
            # If using custom prompts, build the payload and call the generic stage predictor
            payload2 = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text}; CHALLENGE: {s2_text};"'
            r2 = cached2 or predict_stage_with_config(
                "stage2",
                payload2,
                prompts_yaml_path=prompts_path,
                api_base=cfg.get("api_base"),
                model=cfg.get("model"),
                effort=cfg.get("effort"),
                timeout_seconds=cfg.get("timeout_seconds"),
            )
        else:
            # Otherwise, use the default stage2 predictor
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

        # ---- Stage 3: REFRAME ----
        # Build a unique cache key for this input
        k3 = f"cbt::s3::{cfg_sig}::{stable_hash(stmt)}::{stable_hash(s1_text)}::{stable_hash(s2_text)}::{stable_hash(s3_text)}"
        cached3 = try_load(cache_dir, k3) if use_cache else None
        if prompts_path:
            # If using custom prompts, build the payload and call the generic stage predictor
            payload3 = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text}; CHALLENGE: {s2_text}; REFRAME: {s3_text};"'
            r3 = cached3 or predict_stage_with_config(
                "stage3",
                payload3,
                prompts_yaml_path=prompts_path,
                api_base=cfg.get("api_base"),
                model=cfg.get("model"),
                effort=cfg.get("effort"),
                timeout_seconds=cfg.get("timeout_seconds"),
            )
        else:
            # Otherwise, use the default stage3 predictor
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

    # ---- Compute metrics and save results ----
    import json as _json, numpy as _np
    out = {}

    # Compute and save metrics, confusion matrices, and detailed pairs for each stage
    for st in ["stage1", "stage2", "stage3"]:
        # Compute binary classification metrics for this stage
        mt = compute_binary(all_trues[st], all_preds[st])
        out[st] = mt
        # Plot and save confusion matrix for this stage
        plot_confusion(all_trues[st], all_preds[st], [0, 1], f"Confusion - {st}", str(Path(fig_dir) / f"cm_{st}.png"))
        # Save detailed prediction pairs for this stage
        pd.DataFrame(all_pairs[st]).to_csv(Path(table_dir) / f"pairs_{st}.csv", index=False)

    # ---- Compute overall metrics across all stages ----
    y_true_all = all_trues["stage1"] + all_trues["stage2"] + all_trues["stage3"]
    y_pred_all = all_preds["stage1"] + all_preds["stage2"] + all_preds["stage3"]
    mt_all = compute_binary(y_true_all, y_pred_all)
    out["overall"] = mt_all
    # Plot and save overall confusion matrix
    plot_confusion(y_true_all, y_pred_all, [0, 1], "Confusion - overall", str(Path(fig_dir) / "cm_overall.png"))

    # Save summary metrics as JSON
    with open(Path(json_dir) / "summary.json", "w") as f:
        _json.dump(out, f, ensure_ascii=False, indent=2)

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