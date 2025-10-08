# benchmarks/tasks/cbt.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.cache import try_load, save
from eval.metrics import compute_binary, plot_confusion
from agent_bridge.cbt import predict_stage1, predict_stage2, predict_stage3, predict_stage_with_config

def run_cbt(df, cols, cfg, fig_dir, table_dir, json_dir, logger):
    """
    Evaluate CBT reasoner correctness per stage using agent's prompts.
    - Reads dataset row-wise.
    - For each stage: calls the agent bridge to obtain label and raw output.
    - Aggregates per-stage and overall metrics and writes confusion matrices.
    """
    cache_dir = Path(cfg["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
    all_trues = {"stage1": [], "stage2": [], "stage3": []}
    all_preds = {"stage1": [], "stage2": [], "stage3": []}
    all_pairs = {"stage1": [], "stage2": [], "stage3": []}
    prompts_path = cfg.get("prompts_path", None)

	for i in tqdm(range(len(df)), desc="CBT"):
		row = df.iloc[i]
		stmt = str(row[cols["statement"]])
		s1_text = str(row[cols["stage1_text"]]); s1_lab = int(row[cols["stage1_label"]])
		s2_text = str(row[cols["stage2_text"]]); s2_lab = int(row[cols["stage2_label"]])
		s3_text = str(row[cols["stage3_text"]]); s3_lab = int(row[cols["stage3_label"]])

        # stage1
        k1 = f"cbt::s1::{hash(stmt)}::{hash(s1_text)}"
        if prompts_path:
            payload1 = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text};"'
            r1 = try_load(cache_dir, k1) or predict_stage_with_config(
                "stage1",
                payload1,
                prompts_yaml_path=prompts_path,
                api_base=cfg.get("api_base"),
                model=cfg.get("model"),
                temperature=cfg.get("temperature"),
                max_tokens=cfg.get("max_tokens"),
                timeout_seconds=cfg.get("timeout_seconds"),
            )
        else:
            r1 = try_load(cache_dir, k1) or predict_stage1(stmt, s1_text)
        save(cfg["cache_dir"], k1, r1) if try_load(cache_dir, k1) is None and cfg["save_intermediate"] else None
		all_trues["stage1"].append(s1_lab); all_preds["stage1"].append(int(r1["label"]))
		all_pairs["stage1"].append({"idx": i, "stage":"stage1", "y_true": s1_lab, "y_pred": int(r1["label"]), "statement": stmt, "candidate": s1_text, "raw": r1["raw"]})

        # stage2
        k2 = f"cbt::s2::{hash(stmt)}::{hash(s1_text)}::{hash(s2_text)}"
        if prompts_path:
            payload2 = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text}; CHALLENGE: {s2_text};"'
            r2 = try_load(cache_dir, k2) or predict_stage_with_config(
                "stage2",
                payload2,
                prompts_yaml_path=prompts_path,
                api_base=cfg.get("api_base"),
                model=cfg.get("model"),
                temperature=cfg.get("temperature"),
                max_tokens=cfg.get("max_tokens"),
                timeout_seconds=cfg.get("timeout_seconds"),
            )
        else:
            r2 = try_load(cache_dir, k2) or predict_stage2(stmt, s1_text, s2_text)
        save(cfg["cache_dir"], k2, r2) if try_load(cache_dir, k2) is None and cfg["save_intermediate"] else None
		all_trues["stage2"].append(s2_lab); all_preds["stage2"].append(int(r2["label"]))
		all_pairs["stage2"].append({"idx": i, "stage":"stage2", "y_true": s2_lab, "y_pred": int(r2["label"]), "statement": stmt, "candidate": s2_text, "raw": r2["raw"]})

        # stage3
        k3 = f"cbt::s3::{hash(stmt)}::{hash(s1_text)}::{hash(s2_text)}::{hash(s3_text)}"
        if prompts_path:
            payload3 = f'"STATEMENT: {stmt}; UNHELPFUL_THOUGHTS: {s1_text}; CHALLENGE: {s2_text}; REFRAME: {s3_text};"'
            r3 = try_load(cache_dir, k3) or predict_stage_with_config(
                "stage3",
                payload3,
                prompts_yaml_path=prompts_path,
                api_base=cfg.get("api_base"),
                model=cfg.get("model"),
                temperature=cfg.get("temperature"),
                max_tokens=cfg.get("max_tokens"),
                timeout_seconds=cfg.get("timeout_seconds"),
            )
        else:
            r3 = try_load(cache_dir, k3) or predict_stage3(stmt, s1_text, s2_text, s3_text)
        save(cfg["cache_dir"], k3, r3) if try_load(cache_dir, k3) is None and cfg["save_intermediate"] else None
		all_trues["stage3"].append(s3_lab); all_preds["stage3"].append(int(r3["label"]))
		all_pairs["stage3"].append({"idx": i, "stage":"stage3", "y_true": s3_lab, "y_pred": int(r3["label"]), "statement": stmt, "candidate": s3_text, "raw": r3["raw"]})

	# per-stage metrics + overall
	import json as _json, numpy as _np
	out = {}
	for st in ["stage1","stage2","stage3"]:
		mt = compute_binary(all_trues[st], all_preds[st])
		out[st] = mt
		plot_confusion(all_trues[st], all_preds[st], [0,1], f"Confusion - {st}", str(Path(fig_dir)/f"cm_{st}.png"))
		pd.DataFrame(all_pairs[st]).to_csv(Path(table_dir)/f"pairs_{st}.csv", index=False)

	# overall
	y_true_all = all_trues["stage1"] + all_trues["stage2"] + all_trues["stage3"]
	y_pred_all = all_preds["stage1"] + all_preds["stage2"] + all_preds["stage3"]
	mt_all = compute_binary(y_true_all, y_pred_all)
	out["overall"] = mt_all
	plot_confusion(y_true_all, y_pred_all, [0,1], "Confusion - overall", str(Path(fig_dir)/"cm_overall.png"))

	with open(Path(json_dir)/"summary.json","w") as f:
		_json.dump(out, f, ensure_ascii=False, indent=2)
	pd.DataFrame([
		{"scope":k, **{
			"accuracy": out[k]["accuracy"],
			"precision_pos1": out[k]["precision_pos1"],
			"recall_pos1": out[k]["recall_pos1"],
			"f1_pos1": out[k]["f1_pos1"],
		}} for k in ["stage1","stage2","stage3","overall"]
	]).to_csv(Path(table_dir)/"metrics.csv", index=False)