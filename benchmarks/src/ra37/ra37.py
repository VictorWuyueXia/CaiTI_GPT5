import json
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.utils.cache import try_load, save, stable_hash, config_signature, cleanup_cache
from src.utils.io import load_yaml_file
from src.metrics import compute_multiclass, plot_confusion
from src.openai.client_openai import chat_complete_many, chat_complete_jsonless
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def _parse_score_from_raw(raw_text):
	"""
	Parse a score in {0,1,2} from a free-form model output.
	Accepts formats like:
	- "3_talk, 1"
	- "DLA_3_talk, 2"
	- JSON-like: {"res": "3_talk, 0"} or {"dimension": "3_talk", "score": 1}
	Falls back to searching the first standalone digit 0/1/2.
	Returns int in {0,1,2} or None if not found.
	"""
	s = str(raw_text).strip()
	# Try to parse answer as JSON object if possible
	if s.startswith("{") and s.endswith("}"):
		try:
			obj = json.loads(s)
			# Lowercase all keys for robustness
			d = {str(k).lower(): v for k, v in obj.items()}
			if "score" in d:
				val = d["score"]
				# Check if valid score in [0, 1, 2]
				if isinstance(val, int) and 0 <= val <= 2:
					return val
			if "res" in d:
				# Try to find score trailing in a known separated format like "3_talk, 1"
				m = re.search(r"[,:\-\s]([0-2])\s*$", str(d["res"]))
				if m:
					return int(m.group(1))
		except Exception:
			pass
	# Try to find score at the end of string after common separators
	m = re.search(r"[,:\-\s]([0-2])\s*$", s)
	if m:
		return int(m.group(1))
	# Try to find *any* standalone 0/1/2 in text (fallback)
	m = re.search(r"\b([0-2])\b", s)
	if m:
		return int(m.group(1))
	# Return None if score could not be found
	return None


def _extract_raw_label_token(raw_text):
	"""
	Extract the dimension label token directly from raw LLM output (without altering raw).
	Rules:
	- If JSON-like and has 'dimension', use it.
	- Else if JSON-like and has 'res', take the text before the first comma.
	- Otherwise, take the substring before the first comma from plain text.
	- Strip optional prefix 'DLA_' if present.
	Returns the token string (e.g., '3_talk').
	"""
	s = str(raw_text).strip()
	token = None
	if s.startswith("{") and s.endswith("}"):
		try:
			obj = json.loads(s)
			kl = {str(k).lower(): v for k, v in obj.items()}
			if "dimension" in kl:
				token = str(kl["dimension"]).strip()
			elif "res" in kl:
				token = str(kl["res"]).split(",", 1)[0].strip()
		except Exception:
			token = None
	if token is None:
		token = s.split(",", 1)[0].strip()
	# Remove optional leading 'DLA_'
	if token.upper().startswith("DLA_"):
		token = token.split("_", 1)[1]
	return token


def _parse_dimension_label_direct(raw_text, allowed_lower_to_canonical):
	"""
	Parse dimension label directly from raw output.
	- Extract token via _extract_raw_label_token
	- Lowercase it and look up in allowed set (lower->canonical)
	- If not found, return the fallback 'NA, 99'
	"""
	tok = _extract_raw_label_token(raw_text)
	if tok is None:
		return "NA, 99"
	key = tok.strip().lower()
	return allowed_lower_to_canonical.get(key, "NA, 99")


def _plot_confusion_large(y_true, y_pred, labels, title, out_path, figsize_w, figsize_h, dpi, tick_fontsize, ann_fontsize, tick_rotation):
	"""
	Plot and save a large confusion matrix with controllable font sizes for readability.
	All parameters are required to avoid hidden defaults.
	"""
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=dpi)
	im = ax.imshow(cm, cmap="Blues")
	ax.set_title(title)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_xticks(range(len(labels)))
	ax.set_yticks(range(len(labels)))
	ax.set_xticklabels([str(x) for x in labels], rotation=tick_rotation, fontsize=tick_fontsize)
	ax.set_yticklabels([str(x) for x in labels], fontsize=tick_fontsize)
	for (i, j), v in np.ndenumerate(cm):
		ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=ann_fontsize)
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	fig.tight_layout()
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)


def _run_ra37_sequential(df, cols, cfg, fig_dir, table_dir, json_dir, logger,
						  cache_dir, cfg_sig, prompts_path,
						  all_trues, all_preds, all_pairs, use_cache, num_rows,
						  dim_norm_map):
	"""
	Sequential execution path for RA37 evaluation.
	Processes rows one by one with progress tracking.
	"""
	logger.info(f"[RA37][Sequential] Running on {num_rows} rows...")

	# Load prompts (system prompt for the LLM, typically comes from YAML, can be None for raw)
	root = Path(__file__).resolve().parent.parent
	prompts = load_yaml_file(root / prompts_path) if prompts_path else {"analyzer": None}
	sys_prompt = prompts.get("analyzer")

	api_base = cfg.get("api_base")
	model = cfg.get("model")
	effort = cfg.get("effort", "low")
	timeout_seconds = cfg.get("timeout_seconds")

	# Loop over each data row, sequentially running model and evaluating responses
	for i in tqdm(range(num_rows), desc="RA37"):
		row = df.iloc[i]
		dim_label = str(row[cols["dimension_label"]])
		response = str(row[cols["response"]])
		lab = row[cols["score"]]
		if pd.isna(lab):
			logger.warning(f"[RA37][Sequential] Missing score at row {i}; skipping.")
			continue
		y_true = int(lab)

		# Generate a unique cache key for each (dimension, response) pair and model config
		k = f"ra37::analyzer::{cfg_sig}::{stable_hash(dim_label)}::{stable_hash(response)}"
		cached = try_load(cache_dir, k) if use_cache else None

		if cached is None:
			# No cache: call model
			# Call LLM using system/user content; aim for same experience as agent invocation
			raw = chat_complete_jsonless(api_base, model, sys_prompt, response, effort, timeout_seconds)
			label = _parse_score_from_raw(raw)
			if label is None:
				logger.warning(f"[RA37][Sequential] Failed to parse score; default -1 at row {i}")
				label = -1
			# Parse dimension label directly; fallback to 'NA, 99'
			dim_pred = _parse_dimension_label_direct(raw, dim_norm_map)
			r = {"label": label, "raw": raw, "dim_pred": dim_pred}
			# Save intermediate output to cache if enabled
			if use_cache and cfg["save_intermediate"]:
				save(cfg["cache_dir"], k, r)
		else:
			# Use previously cached result
			r = cached
			# Backfill dim_pred from raw if absent in older cache entries
			if isinstance(r, dict) and r.get("dim_pred") is None:
				try:
					r["dim_pred"] = _parse_dimension_label_direct(r.get("raw"), dim_norm_map)
				except Exception:
					pass

		# Track gold and predicted labels for reporting
		all_trues.append(y_true)
		all_preds.append(int(r["label"]))
		all_pairs.append({
			"idx": i,
			"y_true": y_true,
			"y_pred": int(r["label"]),
			"dimension_label": dim_label,
			"response": response,
			"raw": r["raw"],
			"dim_pred": r.get("dim_pred") if isinstance(r, dict) else None,
		})

	logger.info(f"[RA37][Sequential] Completed all {num_rows} rows.")


def _run_ra37_parallel(df, cols, cfg, fig_dir, table_dir, json_dir, logger,
						cache_dir, cfg_sig, prompts_path,
						all_trues, all_preds, all_pairs, use_cache, num_rows,
						dim_norm_map):
	"""
	Parallel execution path for RA37 evaluation using async batch requests.
	Likely much faster than sequential for large datasets.
	"""
	logger.info(f"[RA37][Parallel] Running real async path on {num_rows} rows...")

	# Resolve prompts (prefer path in config, fall back to default path)
	root = Path(__file__).resolve().parent.parent
	default_prompts_path = "ra37/prompts_ra37.yaml"
	prompts_path = cfg.get("prompts_path", default_prompts_path)
	logger.info(f"[RA37][Parallel] Using prompts_path={prompts_path}")
	prompts = load_yaml_file(root / prompts_path)
	sys_prompt = prompts.get("analyzer")

	api_base = cfg.get("api_base")
	model = cfg.get("model")
	effort = cfg.get("effort", "low")
	timeout_seconds = cfg.get("timeout_seconds")
	max_batch = int(cfg.get("parallel_max_batch", 1))

	# For requests not cached, save (system,user) items to feed to async API
	items = []
	map_idx = []  # Indices of original rows
	keys = []     # Corresponding cache keys

	for i in range(num_rows):
		row = df.iloc[i]
		dim_label = str(row[cols["dimension_label"]])
		response = str(row[cols["response"]])
		lab = row[cols["score"]]
		if pd.isna(lab):
			logger.warning(f"[RA37][Parallel] Missing score at row {i}; skipping.")
			continue
		y_true = int(lab)

		k = f"ra37::analyzer::{cfg_sig}::{stable_hash(dim_label)}::{stable_hash(response)}"
		cached = try_load(cache_dir, k) if use_cache else None
		if cached is not None:
			# If answer already cached, use result right away
			all_trues.append(y_true)
			all_preds.append(int(cached["label"]))
			all_pairs.append({
				"idx": i,
				"y_true": y_true,
				"y_pred": int(cached["label"]),
				"dimension_label": dim_label,
				"response": response,
				"raw": cached["raw"],
				"dim_pred": cached.get("dim_pred") if isinstance(cached, dict) else _parse_dimension_label_direct(cached.get("raw"), dim_norm_map),
			})
		else:
			# Otherwise queue request for batch call later
			items.append({
				"system_content": sys_prompt,
				"user_content": response,
			})
			map_idx.append(i)
			keys.append(k)

		# Log scan progress periodically (every 200 examples)
		if i % 200 == 0 and i > 0:
			logger.info(f"[RA37][Parallel] Scanned {i}/{num_rows} rows...")

	# Summary: report how many were cached, and how many need API calls
	logger.info(f"[RA37][Parallel] Cached={len(all_pairs)}, ToRequest={len(items)}")

	if len(items) > 0:
		logger.info(f"[RA37][Parallel] Dispatching {len(items)} requests with max_batch={max_batch}")
		# Asynchronously query the LLM for all requests, batching as configured
		raws = chat_complete_many(api_base, model, items, effort, timeout_seconds, True, max_batch)
		for j, raw in enumerate(raws):
			i = map_idx[j]
			row = df.iloc[i]
			dim_label = str(row[cols["dimension_label"]])
			response = str(row[cols["response"]])
			y_true = int(row[cols["score"]])
			label = _parse_score_from_raw(raw)
			if label is None:
				logger.warning(f"[RA37][Parallel] Failed to parse score; default -1 at row {i}")
				label = -1
			# Parse dimension prediction directly with fallback
			dim_pred = _parse_dimension_label_direct(raw, dim_norm_map)
			r = {"label": label, "raw": raw, "dim_pred": dim_pred}
			# Persist result to cache for future reuse (if allowed)
			if use_cache and cfg["save_intermediate"]:
				save(cfg["cache_dir"], keys[j], r)
			# Append to final result collections
			all_trues.append(y_true)
			all_preds.append(int(r["label"]))
			all_pairs.append({
				"idx": i,
				"y_true": y_true,
				"y_pred": int(r["label"]),
				"dimension_label": dim_label,
				"response": response,
				"raw": r["raw"],
				"dim_pred": r.get("dim_pred"),
			})

	logger.info("[RA37][Parallel] Completed async batch.")


def run_ra37(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
	"""
	Evaluate the Response Analyzer across 37 dimensions: predict SCORE for a given response.
	- Reads dataset row-wise (Dimension, Dimension Label, Response) with true Score
	- Calls analyzer prompt (identical text to agent) to obtain predicted score
	- Aggregates multiclass metrics and writes confusion matrix and tables
	"""
	# Prepare and ensure cache directory exists
	cache_dir = Path(cfg["cache_dir"])
	cache_dir.mkdir(parents=True, exist_ok=True)
	logger.info(f"[RA37] Cache directory prepared at {cache_dir}")
	max_age_days = cfg.get("cache_max_age_days", None)
	# Optionally clean up very old cache entries to avoid stale responses / disk bloat
	if max_age_days is not None:
		logger.info(f"[RA37] Cleaning up cache entries older than {max_age_days} days...")
		cleanup_cache(cache_dir, max_age_days)

	all_trues = []  # All gold scores (for metrics)
	all_preds = []  # All predicted scores
	all_pairs = []  # For CSV/JSON, include additional fields per example

	# Merge OpenAI config in so GPT params are visible, even if omitted by user manually
	root = Path(__file__).resolve().parent.parent
	openai_cfg = load_yaml_file(root / "openai" / "config_openai.yaml")
	cfg = {**openai_cfg, **cfg}

	parallel = bool(cfg.get("parallel", False))
	parallel_max_batch = int(cfg.get("parallel_max_batch", 1))
	max_examples = cfg.get("max_examples", None)
	# Determine number of rows, optionally limit with max_examples for debugging
	num_rows = len(df) if max_examples is None else min(int(max_examples), len(df))
	logger.info(f"[RA37] Starting run: {num_rows} rows, parallel={parallel}, batch={parallel_max_batch}, prompts_path={cfg.get('prompts_path')}")

	cfg_sig = config_signature(cfg)  # Unique signature hash for config, for cache keys, etc.
	use_cache = bool(cfg.get("use_cache", True))
	prompts_path = cfg.get("prompts_path", None)

	# Build allowed label set (lower->canonical) from dataset, plus NA, 99
	allowed = {}
	seen = set()
	for v in df[cols["dimension_label"]].astype(str).tolist():
		vv = v.strip()
		if vv in seen:
			continue
		seen.add(vv)
		allowed[vv.strip().lower()] = vv
	# Include fallback as 38th class
	allowed["na, 99"] = "NA, 99"
	dim_norm_map = allowed
	labels_dim = list(allowed.values())

	# Launch appropriate execution path: parallel is recommended for large runs with sufficient quota
	if parallel:
		logger.info("[RA37] Using parallel execution path.")
		_run_ra37_parallel(
			df, cols, cfg, fig_dir, table_dir, json_dir, logger,
			cache_dir, cfg_sig, prompts_path,
			all_trues, all_preds, all_pairs, use_cache, num_rows,
			dim_norm_map
		)
	else:
		logger.info("[RA37] Using sequential execution path.")
		_run_ra37_sequential(
			df, cols, cfg, fig_dir, table_dir, json_dir, logger,
			cache_dir, cfg_sig, prompts_path,
			all_trues, all_preds, all_pairs, use_cache, num_rows,
			dim_norm_map
		)

	# Metrics & reports
	logger.info("[RA37] Computing metrics and saving results...")
	# Drop any predictions that could not be parsed (-1 means model output was not a recognizable class label)
	flt = [(t, p, i) for i, (t, p) in enumerate(zip(all_trues, all_preds)) if p in (0, 1, 2)]
	if len(flt) != len(all_trues):
		logger.warning(f"[RA37] Dropped {len(all_trues) - len(flt)} unparsable predictions from metrics")
	y_true = [t for t, p, _ in flt]
	y_pred = [p for t, p, _ in flt]
	# Compute macro-averaged multiclass metrics (accuracy, precision, recall, f1) for SCORE
	score_mt = compute_multiclass(y_true, y_pred)
	logger.info(
		f"[RA37][Score] accuracy={score_mt['accuracy']:.3f}, precision={score_mt['precision_macro']:.3f}, "
		f"recall={score_mt['recall_macro']:.3f}, f1={score_mt['f1_macro']:.3f}"
	)
	# Visualize (and save) confusion matrix PNG to disk (score-level 3-class)
	plot_confusion(y_true, y_pred, [0, 1, 2], "Confusion - ra37 (score)", str(Path(fig_dir) / "cm_ra37.png"))
	logger.debug(f"[RA37] Saved score confusion matrix to {Path(fig_dir) / 'cm_ra37.png'}")

	# Build dimension-level metrics and confusion matrix (37+1 expected with NA, 99)
	# dim_norm_map: lower->canonical (now includes 'NA, 99')
	y_true_dim = []
	y_pred_dim = []
	for p in all_pairs:
		pred = p.get("dim_pred")
		if pred is None:
			continue
		# True label canonical is directly from dataset column
		true_canon = str(p.get("dimension_label")).strip()
		# Pred is already canonical or 'NA, 99' from parser
		pred_canon = str(pred).strip()
		if true_canon is None or pred_canon is None:
			continue
		y_true_dim.append(true_canon)
		y_pred_dim.append(pred_canon)
	if len(y_true_dim) > 0:
		dim_mt = compute_multiclass(y_true_dim, y_pred_dim)
		logger.info(
			f"[RA37][Dimension] accuracy={dim_mt['accuracy']:.3f}, precision={dim_mt['precision_macro']:.3f}, "
			f"recall={dim_mt['recall_macro']:.3f}, f1={dim_mt['f1_macro']:.3f}"
		)
		# Larger canvas, smaller tick/annotation fonts for readability
		_plot_confusion_large(
			y_true_dim, y_pred_dim, labels_dim,
			"Confusion - ra37 (dimensions)", str(Path(fig_dir) / "cm_ra37_dim.png"),
			figsize_w=22, figsize_h=22, dpi=220, tick_fontsize=6, ann_fontsize=5, tick_rotation=60
		)
		logger.debug(f"[RA37] Saved dimension confusion matrix to {Path(fig_dir) / 'cm_ra37_dim.png'}")
	else:
		# If we have no dimension predictions, still create empty metrics placeholder
		dim_mt = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0, "report": {}}

	# Save all scored pairs (with gold, pred, question text, etc.) to disk as CSV for in-depth analysis
	pairs_df = pd.DataFrame(all_pairs)
	pairs_df.to_csv(Path(table_dir) / "pairs_ra37.csv", index=False)

	# Save summary statistics to disk as compact JSON and as a CSV table
	out_compact = {
		"ra37": {k: v for k, v in score_mt.items() if k != "report"},
		"ra37_dim": {k: v for k, v in dim_mt.items() if k != "report"}
	}
	Path(json_dir).mkdir(parents=True, exist_ok=True)
	with open(Path(json_dir) / "summary.json", "w") as f:
		json.dump(out_compact, f, ensure_ascii=False, indent=2)

	pd.DataFrame([
		{
			"scope": "ra37",
			"accuracy": score_mt["accuracy"],
			"precision_macro": score_mt["precision_macro"],
			"recall_macro": score_mt["recall_macro"],
			"f1_macro": score_mt["f1_macro"],
		},
		{
			"scope": "ra37_dim",
			"accuracy": dim_mt["accuracy"],
			"precision_macro": dim_mt["precision_macro"],
			"recall_macro": dim_mt["recall_macro"],
			"f1_macro": dim_mt["f1_macro"],
		}
	]).to_csv(Path(table_dir) / "metrics.csv", index=False)
	logger.info(f"[RA37] Saved metrics CSV to {Path(table_dir) / 'metrics.csv'}")

