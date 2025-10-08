# benchmarks/tasks/ra37.py
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.cache import try_load, save, stable_hash, config_signature, cleanup_cache
from eval.metrics import compute_multiclass, plot_confusion
from agent_bridge.analyzer import predict_analyzer

def run_ra37(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
	cache_dir = Path(cfg["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
	# Optional cleanup
	max_age_days = cfg.get("cache_max_age_days", None)
	if max_age_days is not None:
		cleanup_cache(cache_dir, max_age_days)
	gold_dims, pred_dims, gold_scores, pred_scores, rows = [], [], [], [], []
	# Limit examples for dev speed if configured
	max_examples = cfg.get("max_examples", None)
	num_rows = len(df) if max_examples is None else min(int(max_examples), len(df))
	cfg_sig = config_signature(cfg)
	use_cache = bool(cfg.get("use_cache", True))

	for i in tqdm(range(num_rows), desc="RA37"):
		row = df.iloc[i]
		text = str(row[cols["user_input"]])
		g_dim = str(row[cols["gold_dimension"]])
		g_score = int(row[cols["gold_score"]])

		key = f"ra37::{cfg_sig}::{stable_hash(text)}"
		cached = try_load(cache_dir, key) if use_cache else None
		resp = cached or predict_analyzer(text)
		if use_cache and cfg["save_intermediate"] and cached is None:
			save(cfg["cache_dir"], key, resp)
		gold_dims.append(g_dim); pred_dims.append(resp["dimension"])
		gold_scores.append(g_score); pred_scores.append(int(resp["score"]))
		rows.append({"idx": i, "text": text, "gold_dim": g_dim, "pred_dim": resp["dimension"], "gold_score": g_score, "pred_score": int(resp["score"]), "raw": resp["raw"]})

	labels_dim = sorted(list(set(gold_dims + pred_dims)))
	labels_score = sorted(list(set(gold_scores + pred_scores)))

	mt_dim = compute_multiclass(gold_dims, pred_dims)
	mt_score = compute_multiclass(gold_scores, pred_scores)

	plot_confusion(gold_dims, pred_dims, labels_dim, "Confusion - RA Dimension", str(Path(fig_dir)/"cm_ra_dim.png"))
	plot_confusion(gold_scores, pred_scores, labels_score, "Confusion - RA Score", str(Path(fig_dir)/"cm_ra_score.png"))

	pd.DataFrame(rows).to_csv(Path(table_dir)/"pairs_ra37.csv", index=False)
	with open(Path(json_dir)/"summary.json","w") as f:
		json.dump({"dimension": mt_dim, "score": mt_score}, f, ensure_ascii=False, indent=2)

	pd.DataFrame([
		{"scope":"ra_dimension","accuracy": mt_dim["accuracy"], "precision_macro": mt_dim["precision_macro"], "recall_macro": mt_dim["recall_macro"], "f1_macro": mt_dim["f1_macro"]},
		{"scope":"ra_score","accuracy": mt_score["accuracy"], "precision_macro": mt_score["precision_macro"], "recall_macro": mt_score["recall_macro"], "f1_macro": mt_score["f1_macro"]},
	]).to_csv(Path(table_dir)/"metrics.csv", index=False)