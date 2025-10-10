import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Import utility functions for cache, file IO, metrics, and LLM client (OpenAI)
from src.utils.cache import try_load, save, stable_hash, config_signature, cleanup_cache
from src.utils.io import load_yaml_file
from src.metrics import compute_multiclass, plot_confusion
from src.openai.client_openai import chat_complete_many, chat_complete_jsonless

# List of valid/canonical output dimension classes for this RA-General task
_GENERAL_DIMENSIONS = ["Yes", "No", "Stop", "Question", "Maybe"]


def _parse_dimension_from_raw(raw_text):
	"""
	Parse a dimension label from free-form model output.
	Target labels: Yes, No, Stop, Question, Maybe (case-insensitive).
	Accepts formats like:
	- "Yes, 0" or "No, 1"
	- JSON-like: {"res": "Yes, 0"} or {"dimension": "Yes", "score": 0}
	- Bare token: "Yes" / "No" / ...
	Returns canonical-cased label or None if not found.
	"""
	# Coerce input to string and trim any whitespace
	s = str(raw_text).strip()
	
	# Try parsing as JSON object (if output is JSON-like)
	if s.startswith("{") and s.endswith("}"):
		try:
			# Parse string as dict
			obj = json.loads(s)
			kl = {str(k).lower(): v for k, v in obj.items()}
			# Check if "res" key is present, e.g. {"res": "Yes, 0"}
			if "res" in kl:
				m = re.search(r"\b(Yes|No|Stop|Question|Maybe)\b", str(kl["res"]), flags=re.IGNORECASE)
				if m:
					return m.group(1).capitalize()
			# Check if "dimension" key is present, e.g. {"dimension": "Yes"}
			if "dimension" in kl:
				val = str(kl["dimension"]).strip()
				m = re.match(r"^(Yes|No|Stop|Question|Maybe)$", val, flags=re.IGNORECASE)
				if m:
					return m.group(1).capitalize()
		except Exception:
			# Fail soft on parsing errors: return None below if not found
			pass
	# If it's not valid JSON, scan the plain text output for any dimension word
	m = re.search(r"\b(Yes|No|Stop|Question|Maybe)\b", s, flags=re.IGNORECASE)
	if m:
		return m.group(1).capitalize()
	return None  # No valid code found


def _run_ra_general_sequential(
		df, cols, cfg, fig_dir, table_dir, json_dir, logger,
		cache_dir, cfg_sig, prompts_path,
		all_trues, all_preds, all_pairs, use_cache, num_rows
	):
	"""
	Sequential execution path for RA-General evaluation.
	Predict DIMENSION from Response using the agent-identical system prompt.
	This is used when 'parallel' mode is disabled.
	"""
	logger.info(f"[RA_GENERAL][Sequential] Running on {num_rows} rows...")

	# Load prompt yaml file (only once)
	root = Path(__file__).resolve().parent.parent
	prompts = load_yaml_file(root / prompts_path) if prompts_path else {"analyzer": None}
	sys_prompt = prompts.get("analyzer")
	
	# Extract OpenAI API settings from config
	api_base = cfg.get("api_base")
	model = cfg.get("model")
	effort = cfg.get("effort", "low")
	timeout_seconds = cfg.get("timeout_seconds")

	# Loop through rows of dataset and process responses
	for i in tqdm(range(num_rows), desc="RAG"):
		row = df.iloc[i]
		gold_dim = str(row[cols["dimension"]])     # Ground-truth label
		response = str(row[cols["response"]])      # Text response to classify
		
		# Cache key for this example
		k = f"rag::analyzer::{cfg_sig}::{stable_hash(response)}"
		# Try to load previous result from cache, if caching enabled
		cached = try_load(cache_dir, k) if use_cache else None
		if cached is None:
			# Not cached, send prompt to the LLM
			raw = chat_complete_jsonless(api_base, model, sys_prompt, response, effort, timeout_seconds)
			pred_dim = _parse_dimension_from_raw(raw)     # Extract predicted label
			if pred_dim is None:
				pred_dim = "Unknown"  # Mark as Unknown if extraction fails
			r = {"label": pred_dim, "raw": raw}
			# Save result to cache if caching is on and enabled in config
			if use_cache and cfg["save_intermediate"]:
				save(cfg["cache_dir"], k, r)
		else:
			# Use cached result
			r = cached

		# Accumulate results for metrics and reporting
		all_trues.append(gold_dim)
		all_preds.append(str(r["label"]))
		all_pairs.append({
			"idx": i,
			"y_true": gold_dim,
			"y_pred": str(r["label"]),
			"response": response,
			"raw": r["raw"],
		})
		if i % 50 == 0:
			logger.info(f"[RA_GENERAL][Sequential] Processed {i+1}/{num_rows} rows")

	logger.info(f"[RA_GENERAL][Sequential] Completed all {num_rows} rows.")


def _run_ra_general_parallel(
		df, cols, cfg, fig_dir, table_dir, json_dir, logger,
		cache_dir, cfg_sig, prompts_path,
		all_trues, all_preds, all_pairs, use_cache, num_rows
	):
	"""
	Parallel execution path using async batch requests.
	Uses async batching to process many requests more efficiently if enabled.
	"""
	logger.info(f"[RA_GENERAL][Parallel] Running real async path on {num_rows} rows...")
	root = Path(__file__).resolve().parent.parent

	# Default location of prompts if not explicitly set
	default_prompts_path = "ra_general/prompts_ra_general.yaml"
	prompts_path = cfg.get("prompts_path", default_prompts_path)
	logger.info(f"[RA_GENERAL][Parallel] Using prompts_path={prompts_path}")
	prompts = load_yaml_file(root / prompts_path)
	sys_prompt = prompts.get("analyzer")

	# Gather OpenAI/LLM config
	api_base = cfg.get("api_base")
	model = cfg.get("model")
	effort = cfg.get("effort", "low")
	timeout_seconds = cfg.get("timeout_seconds")
	max_batch = int(cfg.get("parallel_max_batch", 1))  # how many async requests to send at once

	# Lists used to build batch requests and map responses
	items = []      # items to query in parallel (not in cache)
	map_idx = []    # indexes of each item
	keys = []       # corresponding cache keys

	# Loop through the data, prepare for batch submission, use cache if available
	from tqdm import tqdm as _tqdm
	with _tqdm(total=num_rows, desc="RAG-Scan", smoothing=0.1, mininterval=0.2, dynamic_ncols=True, ascii=True, leave=False) as pbar:
		for i in range(num_rows):
			row = df.iloc[i]
			gold_dim = str(row[cols["dimension"]])
			response = str(row[cols["response"]])

			k = f"rag::analyzer::{cfg_sig}::{stable_hash(response)}"
			cached = try_load(cache_dir, k) if use_cache else None
			if cached is not None:
				# Immediately use and aggregate cached result, don't submit for async
				all_trues.append(gold_dim)
				all_preds.append(str(cached["label"]))
				all_pairs.append({
					"idx": i,
					"y_true": gold_dim,
					"y_pred": str(cached["label"]),
					"response": response,
					"raw": cached["raw"],
				})
			else:
				# Prepare batch input for LLM
				items.append({
					"system_content": sys_prompt,
					"user_content": response,
				})
				map_idx.append(i)
				keys.append(k)

			pbar.update(1)

	logger.info(f"[RA_GENERAL][Parallel] Cached={len(all_pairs)}, ToRequest={len(items)}")

	if len(items) > 0:
		# If any items weren't cached, submit them in an async batch to the LLM
		logger.info(f"[RA_GENERAL][Parallel] Dispatching {len(items)} requests with max_batch={max_batch}")
		raws = chat_complete_many(api_base, model, items, effort, timeout_seconds, True, max_batch)
		for j, raw in enumerate(raws):
			i = map_idx[j]
			row = df.iloc[i]
			gold_dim = str(row[cols["dimension"]])
			response = str(row[cols["response"]])
			pred_dim = _parse_dimension_from_raw(raw)
			if pred_dim is None:
				pred_dim = "Unknown"
			r = {"label": pred_dim, "raw": raw}
			# Save the inference result to cache
			if use_cache and cfg["save_intermediate"]:
				save(cfg["cache_dir"], keys[j], r)
			# Accumulate metrics, reporting info
			all_trues.append(gold_dim)
			all_preds.append(str(r["label"]))
			all_pairs.append({
				"idx": i,
				"y_true": gold_dim,
				"y_pred": str(r["label"]),
				"response": response,
				"raw": r["raw"],
			})

	logger.info("[RA_GENERAL][Parallel] Completed async batch.")


def run_ra_general(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
	"""
	Evaluate general response analyzer: predict DIMENSION for a given Response.
	- Reads dataset (Dimension, Response)
	- Calls analyzer prompt (identical to agent's INIT_ASKER_SYSTEM_PROMPT_V2)
	- Aggregates multiclass metrics and writes confusion matrix and tables
	"""
	# Prepare and create directory for cache if needed
	cache_dir = Path(cfg["cache_dir"]) 
	cache_dir.mkdir(parents=True, exist_ok=True)
	logger.info(f"[RA_GENERAL] Cache directory prepared at {cache_dir}")

	# Optionally clean up old cache files based on max_age_days in config
	max_age_days = cfg.get("cache_max_age_days", None)
	if max_age_days is not None:
		logger.info(f"[RA_GENERAL] Cleaning up cache entries older than {max_age_days} days...")
		cleanup_cache(cache_dir, max_age_days)

	all_trues = []   # List of gold (actual) labels for all examples
	all_preds = []   # List of predicted labels
	all_pairs = []   # List of dicts with details per example for final csv

	# Load merged config (OpenAI config overlays) for GPT parameters
	root = Path(__file__).resolve().parent.parent
	openai_cfg = load_yaml_file(root / "openai" / "config_openai.yaml")
	cfg = {**openai_cfg, **cfg}

	# Determine evaluation settings (sequential/parallel, etc)
	parallel = bool(cfg.get("parallel", False))
	parallel_max_batch = int(cfg.get("parallel_max_batch", 1))
	max_examples = cfg.get("max_examples", None)
	num_rows = len(df) if max_examples is None else min(int(max_examples), len(df))
	logger.info(f"[RA_GENERAL] Starting run: {num_rows} rows, parallel={parallel}, batch={parallel_max_batch}, prompts_path={cfg.get('prompts_path')}")

	cfg_sig = config_signature(cfg)         # Used for unique cache key
	use_cache = bool(cfg.get("use_cache", True))
	prompts_path = cfg.get("prompts_path", None)

	# Choose parallel or sequential by config, delegate to underlying processor
	if parallel:
		logger.info("[RA_GENERAL] Using parallel execution path.")
		_run_ra_general_parallel(
			df, cols, cfg, fig_dir, table_dir, json_dir, logger,
			cache_dir, cfg_sig, prompts_path,
			all_trues, all_preds, all_pairs, use_cache, num_rows
		)
	else:
		logger.info("[RA_GENERAL] Using sequential execution path.")
		_run_ra_general_sequential(
			df, cols, cfg, fig_dir, table_dir, json_dir, logger,
			cache_dir, cfg_sig, prompts_path,
			all_trues, all_preds, all_pairs, use_cache, num_rows
		)

	# Metrics & reports
	logger.info("[RA_GENERAL] Computing metrics and saving results...")
	# Compose all possible class labels (including Unknown as "failure" class)
	labels = _GENERAL_DIMENSIONS + ["Unknown"]
	
	# Compute metrics with all data
	mt = compute_multiclass(all_trues, all_preds)
	logger.info(
		f"[RA_GENERAL] accuracy={mt['accuracy']:.3f}, "
		f"precision={mt['precision_macro']:.3f}, "
		f"recall={mt['recall_macro']:.3f}, "
		f"f1={mt['f1_macro']:.3f}"
	)
	# Save confusion matrix chart
	plot_confusion(all_trues, all_preds, labels, "Confusion - ra_general", str(Path(fig_dir) / "cm_ra_general.png"))

	# Save all pairs/results as CSV for auditing
	pd.DataFrame(all_pairs).to_csv(Path(table_dir) / "pairs_ra_general.csv", index=False)

	# Save summary metrics JSON (drop 'report' which is a pandas object)
	out_compact = {"ra_general": {k: v for k, v in mt.items() if k != "report"}}
	Path(json_dir).mkdir(parents=True, exist_ok=True)
	with open(Path(json_dir) / "summary.json", "w") as f:
		json.dump(out_compact, f, ensure_ascii=False, indent=2)

	# Save main metrics as a single-row CSV for summary table
	pd.DataFrame([
		{
			"scope": "ra_general",
			"accuracy": mt["accuracy"],
			"precision_macro": mt["precision_macro"],
			"recall_macro": mt["recall_macro"],
			"f1_macro": mt["f1_macro"],
		}
	]).to_csv(Path(table_dir) / "metrics.csv", index=False)
	logger.info(f"[RA_GENERAL] Saved metrics CSV to {Path(table_dir) / 'metrics.csv'}")


