# benchmarks/tasks/ra37.py
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.cache import try_load, save
from eval.metrics import compute_multiclass, plot_confusion
from agent_bridge.analyzer import predict_analyzer

def run_ra37(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
	cache_dir = Path(cfg["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
	gold_dims, pred_dims, gold_scores, pred_scores, rows = [], [], [], [], []

	for i in tqdm(range(len(df)), desc="RA37"):
		row = df.iloc[i]
		text = str(row[cols["user_input"]])
		g_dim = str(row[cols["gold_dimension"]])
		g_score = int(row[cols["gold_score"]])

		key = f"ra37::{hash(text)}"
		resp = try_load(cache_dir, key) or predict_analyzer(text); save(cfg["cache_dir"], key, resp) if try_load(cache_dir, key) is None and cfg["save_intermediate"] else None
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