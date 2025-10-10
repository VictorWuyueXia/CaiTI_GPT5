import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.utils.cache import try_load, save, stable_hash, config_signature, cleanup_cache
from src.utils.io import load_yaml_file
from src.metrics import compute_binary, plot_confusion
from src.rv.openai_rv import predict_reasoner
from src.openai.client_openai import chat_complete_many


def _run_rv_parallel(df, cols, cfg, fig_dir, table_dir, json_dir, logger,
                    cache_dir, cfg_sig, prompts_path,
                    all_trues, all_preds, all_pairs, use_cache, num_rows):
    """
    Parallel execution path for RV evaluation.
    Processes rows asynchronously in batches for improved performance.
    """
    logger.info(f"[RV][Parallel] Running real async path on {num_rows} rows...")

    # Load merged cfg (so GPT params are available)
    # Combine OpenAI config, RV config, and runtime config with precedence: runtime > RV > OpenAI
    root = Path(__file__).resolve().parent.parent
    rv_cfg = load_yaml_file(root / "rv" / "config_rv.yaml")
    openai_cfg = load_yaml_file(root / "openai" / "config_openai.yaml")
    merged_cfg = {**openai_cfg, **rv_cfg, **cfg}

    # Determine prompts path: use custom path if provided, otherwise use RV config default
    prompts_path = cfg.get("prompts_path", rv_cfg.get("prompts_path", "rv/prompts_rv.yaml"))
    logger.info(f"[RV][Parallel] Using prompts_path={prompts_path}")

    # Extract API configuration parameters from merged config
    api_base = merged_cfg.get("api_base")
    model = merged_cfg.get("model")
    effort = merged_cfg.get("effort", "low")
    timeout_seconds = merged_cfg.get("timeout_seconds")
    max_batch = int(merged_cfg.get("parallel_max_batch", 1))

    # Build batch for parallel processing
    items = []      # List of payloads to send to LLM
    map_idx = []    # Maps batch index back to original row index
    keys = []       # Cache keys for each request

    # Process each row to build the batch
    for i in range(num_rows):
        row = df.iloc[i]
        topic = str(row[cols["topic"]])
        original = str(row[cols["original"]])
        follow_up = str(row[cols["follow_up"]])
        lab = row[cols["label"]]
        
        # Skip rows with missing labels
        if pd.isna(lab):
            logger.warning(f"[RV][Parallel] Missing label at row {i}; skipping.")
            continue
        
        y_true = int(lab)
        # Generate unique cache key based on input data and config signature
        k = f"rv::reasoner::{cfg_sig}::{stable_hash(topic)}::{stable_hash(original)}::{stable_hash(follow_up)}"
        
        # Check if result is already cached
        cached = try_load(cache_dir, k) if use_cache else None
        if cached is not None:
            # Use cached result if available
            all_trues.append(y_true)
            all_preds.append(int(cached["label"]))
            all_pairs.append({
                "idx": i,
                "y_true": y_true,
                "y_pred": int(cached["label"]),
                "topic": topic,
                "original": original,
                "follow_up": follow_up,
                "raw": cached["raw"],
            })
        else:
            # Prepare payload for LLM request if not cached
            payload = {
                "system_content": load_yaml_file(Path(__file__).resolve().parent.parent / prompts_path)["reasoner"],
                "user_content": f'{{"Topic": {topic!r}, "Original Response": {original!r}, "Follow Up Response": {follow_up!r}}}',
            }
            items.append(payload)
            map_idx.append(i)
            keys.append(k)
        
        # Log progress periodically
        if i % 200 == 0 and i > 0:
            logger.info(f"[RV][Parallel] Scanned {i}/{num_rows} rows...")

    # Log cache statistics
    logger.info(f"[RV][Parallel] Cached={len(all_pairs)}, ToRequest={len(items)}")
    
    # Process uncached items in parallel batches
    if len(items) > 0:
        logger.info(f"[RV][Parallel] Dispatching {len(items)} requests with max_batch={max_batch}")
        
        # Send batch requests to LLM
        raws = chat_complete_many(api_base, model, items, effort, timeout_seconds, True, max_batch)
        
        # Import here to avoid circular imports
        from src.cbt.openai_cbt import parse_decision
        
        # Process each response from the batch
        for j, raw in enumerate(raws):
            i = map_idx[j]
            row = df.iloc[i]
            topic = str(row[cols["topic"]])
            original = str(row[cols["original"]])
            follow_up = str(row[cols["follow_up"]])
            y_true = int(row[cols["label"]])
            
            # Parse decision from LLM response
            label = parse_decision(raw)
            r = {"label": label, "raw": raw}
            
            # Save to cache if enabled
            if use_cache and cfg["save_intermediate"]:
                save(cfg["cache_dir"], keys[j], r)
            
            # Store results
            all_trues.append(y_true)
            all_preds.append(int(r["label"]))
            all_pairs.append({
                "idx": i,
                "y_true": y_true,
                "y_pred": int(r["label"]),
                "topic": topic,
                "original": original,
                "follow_up": follow_up,
                "raw": r["raw"],
            })

    logger.info("[RV][Parallel] Completed async batch.")


def _run_rv_sequential(df, cols, cfg, fig_dir, table_dir, json_dir, logger,
                      cache_dir, cfg_sig, prompts_path,
                      all_trues, all_preds, all_pairs, use_cache, num_rows):
    """
    Sequential execution path for RV evaluation.
    Processes rows one by one with progress tracking.
    """
    logger.info(f"[RV][Sequential] Running on {num_rows} rows...")
    
    # Process each row sequentially with progress bar
    for i in tqdm(range(num_rows), desc="RV"):
        row = df.iloc[i]
        topic = str(row[cols["topic"]])
        original = str(row[cols["original"]])
        follow_up = str(row[cols["follow_up"]])

        # Check for missing labels
        lab = row[cols["label"]]
        if pd.isna(lab):
            logger.warning(f"[RV][Sequential] Missing label at row {i}; skipping.")
            continue
        
        y_true = int(lab)

        # Generate unique cache key
        k = f"rv::reasoner::{cfg_sig}::{stable_hash(topic)}::{stable_hash(original)}::{stable_hash(follow_up)}"
        
        # Check cache first
        cached = try_load(cache_dir, k) if use_cache else None
        if cached is None:
            # Call RV reasoner if not cached
            r = predict_reasoner(topic, original, follow_up)
            # Save to cache if enabled
            if use_cache and cfg["save_intermediate"]:
                save(cfg["cache_dir"], k, r)
        else:
            # Use cached result
            r = cached

        # Store results for metrics calculation
        all_trues.append(y_true)
        all_preds.append(int(r["label"]))
        all_pairs.append({
            "idx": i,
            "y_true": y_true,
            "y_pred": int(r["label"]),
            "topic": topic,
            "original": original,
            "follow_up": follow_up,
            "raw": r["raw"],
        })
        
        # Log progress periodically
        if i % 50 == 0:
            logger.info(f"[RV][Sequential] Processed {i+1}/{num_rows} rows")

    logger.info(f"[RV][Sequential] Completed all {num_rows} rows.")


def run_rv(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
    """
    Evaluate RV reasoner correctness: whether follow-up is related to the topic/original response.
    - Read dataset row-wise
    - For each row: call RV reasoner to obtain label and raw output
    - Aggregate metrics and write confusion matrix and tables
    """
    # Prepare cache
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[RV] Cache directory prepared at {cache_dir}")

    # Optional: clean up old cache entries to avoid infinite growth
    max_age_days = cfg.get("cache_max_age_days", None)
    if max_age_days is not None:
        logger.info(f"[RV] Cleaning up cache entries older than {max_age_days} days...")
        cleanup_cache(cache_dir, max_age_days)

    # Initialize containers for ground truth labels, predictions, and detailed pairs
    all_trues = []
    all_preds = []
    all_pairs = []

    # Optional: path to custom prompts YAML for agent
    prompts_path = cfg.get("prompts_path", None)

    # Merge OpenAI config (align with CBT)
    root = Path(__file__).resolve().parent.parent
    openai_cfg = load_yaml_file(root / "openai" / "config_openai.yaml")
    cfg = {**openai_cfg, **cfg}

    # Parallel controls
    parallel = bool(cfg.get("parallel", False))
    parallel_max_batch = int(cfg.get("parallel_max_batch", 1))

    # Limit examples
    max_examples = cfg.get("max_examples", None)
    num_rows = len(df) if max_examples is None else min(int(max_examples), len(df))
    logger.info(f"[RV] Starting run: {num_rows} rows, parallel={parallel}, batch={parallel_max_batch}, prompts_path={prompts_path}")

    # Build config signature
    cfg_sig = config_signature(cfg)
    use_cache = bool(cfg.get("use_cache", True))

    if parallel:
        logger.info("[RV] Using parallel execution path.")
        _run_rv_parallel(
            df, cols, cfg, fig_dir, table_dir, json_dir, logger,
            cache_dir, cfg_sig, prompts_path,
            all_trues, all_preds, all_pairs, use_cache, num_rows
        )
    else:
        logger.info("[RV] Using sequential execution path.")
        _run_rv_sequential(
            df, cols, cfg, fig_dir, table_dir, json_dir, logger,
            cache_dir, cfg_sig, prompts_path,
            all_trues, all_preds, all_pairs, use_cache, num_rows
        )

    # Metrics & reports
    out = {}
    logger.info("[RV] Computing metrics and saving results...")
    mt = compute_binary(all_trues, all_preds)
    out["rv"] = mt
    logger.info(f"[RV] accuracy={mt['accuracy']:.3f}, precision={mt['precision_pos1']:.3f}, recall={mt['recall_pos1']:.3f}, f1={mt['f1_pos1']:.3f}")
    plot_confusion(all_trues, all_preds, [0, 1], "Confusion - rv", str(Path(fig_dir) / "cm_rv.png"))

    # Save artifacts
    pd.DataFrame(all_pairs).to_csv(Path(table_dir) / "pairs_rv.csv", index=False)

    # Save summary JSON/CSV (compact like CBT)
    out_compact = {k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in out.items()}
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(json_dir) / "summary.json", "w") as f:
        json.dump(out_compact, f, ensure_ascii=False, indent=2)

    pd.DataFrame([
        {
            "scope": "rv",
            "accuracy": out["rv"]["accuracy"],
            "precision_pos1": out["rv"]["precision_pos1"],
            "recall_pos1": out["rv"]["recall_pos1"],
            "f1_pos1": out["rv"]["f1_pos1"],
        }
    ]).to_csv(Path(table_dir) / "metrics.csv", index=False)
    logger.info(f"[RV] Saved metrics CSV to {Path(table_dir) / 'metrics.csv'}")


