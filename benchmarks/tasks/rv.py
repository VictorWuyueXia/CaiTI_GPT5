# benchmarks/tasks/rv.py
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.cache import try_load, save
from eval.metrics import compute_binary, plot_confusion
from agent_bridge.rv import predict_rv

def run_rv(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
	cache_dir = Path(cfg["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
	y_true, y_pred, rows = [], [], []

	for i in tqdm(range(len(df)), desc="RV"):
		row = df.iloc[i]
		topic = str(row[cols["topic"]]) if cols["topic"] else ""
		orig_q = str(row[cols["original_question"]]) if cols["original_question"] else ""
		orig_r = str(row[cols["original_response"]])
		fu = str(row[cols["follow_up"]])
		label = int(row[cols["label"]])

		key = f"rv::{hash(topic)}::{hash(orig_q)}::{hash(orig_r)}::{hash(fu)}"
		resp = try_load(cache_dir, key) or predict_rv(topic, orig_q, orig_r, fu); save(cfg["cache_dir"], key, resp) if try_load(cache_dir, key) is None and cfg["save_intermediate"] else None
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