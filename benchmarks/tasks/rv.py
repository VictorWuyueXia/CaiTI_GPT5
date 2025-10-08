# benchmarks/tasks/rv.py
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.cache import try_load, save, stable_hash, config_signature, cleanup_cache
from eval.metrics import compute_binary, plot_confusion
from agent_bridge.rv import predict_rv

def run_rv(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
	cache_dir = Path(cfg["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
	# Optional cleanup
	max_age_days = cfg.get("cache_max_age_days", None)
	if max_age_days is not None:
		cleanup_cache(cache_dir, max_age_days)
	y_true, y_pred, rows = [], [], []
	# Limit examples for dev speed if configured
	max_examples = cfg.get("max_examples", None)
	num_rows = len(df) if max_examples is None else min(int(max_examples), len(df))
	cfg_sig = config_signature(cfg)
	use_cache = bool(cfg.get("use_cache", True))

	for i in tqdm(range(num_rows), desc="RV"):
		row = df.iloc[i]
		topic = str(row[cols["topic"]]) if cols["topic"] else ""
		orig_q = str(row[cols["original_question"]]) if cols["original_question"] else ""
		orig_r = str(row[cols["original_response"]])
		fu = str(row[cols["follow_up"]])
		label = int(row[cols["label"]])

		key = f"rv::{cfg_sig}::{stable_hash(topic)}::{stable_hash(orig_q)}::{stable_hash(orig_r)}::{stable_hash(fu)}"
		cached = try_load(cache_dir, key) if use_cache else None
		resp = cached or predict_rv(topic, orig_q, orig_r, fu)
		if use_cache and cfg["save_intermediate"] and cached is None:
			save(cfg["cache_dir"], key, resp)
		y_true.append(label); y_pred.append(int(resp["label"]))
		rows.append({"idx": i, "y_true": label, "y_pred": int(resp["label"]), "topic": topic, "original": orig_r, "follow_up": fu, "raw": resp["raw"]})

	mt = compute_binary(y_true, y_pred)
	plot_confusion(y_true, y_pred, [0,1], "Confusion - rv", str(Path(fig_dir)/"cm_rv.png"))
	pd.DataFrame(rows).to_csv(Path(table_dir)/"pairs_rv.csv", index=False)
	with open(Path(json_dir)/"summary.json","w") as f:
		json.dump(mt, f, ensure_ascii=False, indent=2)
	pd.DataFrame([{
		"scope":"rv",
		"accuracy": mt["accuracy"],
		"precision_pos1": mt["precision_pos1"],
		"recall_pos1": mt["recall_pos1"],
		"f1_pos1": mt["f1_pos1"],
	}]).to_csv(Path(table_dir)/"metrics.csv", index=False)