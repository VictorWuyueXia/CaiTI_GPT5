from re import L
import argparse, sys, time
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd

# Set up paths for importing project modules
CUR = Path(__file__).resolve().parent
ROOT = CUR
print(f"ROOT: {ROOT}")
# sys.path.insert(0, str(ROOT))
# sys.path.insert(0, str(ROOT.parent / "agent"))  # import agent/src/*

# Import utility and task-specific functions
from src.utils.log import build_logger, ensure_dir
from src.utils.io import load_csv_with_columns, load_yaml_file
from src.cbt.cbt import run_cbt
from src.rv.rv import run_rv
from src.ra37.ra37 import run_ra37
from src.ra_general.ra_general import run_ra_general

def now_tag():
	"""Return a timestamp string for unique run IDs."""
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _merge_general_into_task_cfg(task_cfg, root_cfg):
	"""Merge general keys from root config into task config without overriding task-specific fields."""
	merged = dict(task_cfg)
	for k in ("output_root", "cache_dir", "save_intermediate", "max_examples", "use_cache", "cache_max_age_days", "parallel", "parallel_max_batch"):
		if k in root_cfg:
			merged[k] = root_cfg[k]
	return merged

def main():
	"""
	Run evaluation pipeline for a specified task. Central config is fixed.

	Usage:
	------
	python run_eval.py --task <task_name>

	Arguments:
		--task   : Name of the task to run. Supported: cbt (rv, ra37 TBD). Defaults to cbt.

	Outputs:
		- Results, figures, tables, and logs will be saved to the output directories specified in the config.
	"""

	run_id = now_tag()
 
	# Load central config (fixed path)
	central_cfg_path = ROOT / "config.yaml"
	root_cfg = load_yaml_file(central_cfg_path)
 
	# Parse task
	p = argparse.ArgumentParser()
	p.add_argument("--task", type=str, required=False, help="Task to run: cbt, rv, ra37, ra_general", default="cbt")
	args = p.parse_args()

	# Resolve task and task-specific config path
	task = args.task.lower()
	if task == "cbt":
		task_cfg_rel = root_cfg.get("cbt_config", "src/cbt/config_cbt.yaml")
	elif task == "rv":
		task_cfg_rel = root_cfg.get("rv_config", "src/rv/config_rv.yaml")
	elif task == "ra37":
		task_cfg_rel = root_cfg.get("ra37_config", "src/ra37/config_ra37.yaml")
	elif task == "ra_general":
		task_cfg_rel = root_cfg.get("ra_general_config", "src/ra_general/config_ra_general.yaml")
	else:
		raise ValueError(f"Unsupported or unimplemented task: {task}")
 
	# Load logger
	log_dir = root_cfg.get("log_dir", "logs")
	logger = build_logger(log_dir, f"{task}_{run_id}")
	logger.info(f"Task: {args.task}")
	
	# Load task config
	task_cfg_path = task_cfg_rel if Path(task_cfg_rel).is_absolute() else (ROOT / task_cfg_rel)
	task_cfg = load_yaml_file(task_cfg_path)
	logger.info(f"Task config: {task_cfg}")

	# Merge general settings and OpenAI settings
	cfg = _merge_general_into_task_cfg(task_cfg, root_cfg)
	openai_cfg = load_yaml_file(ROOT / "src" / "openai" / "config_openai.yaml")
	cfg = {**openai_cfg, **cfg}

	# Prepare output directories
	out_root = Path(cfg.get("output_root", "reports"))
	out_dir = out_root / f"{task}_{run_id}"
	fig_dir = out_dir / "figs"
	table_dir = out_dir / "tables"
	json_dir = out_dir / "json"
	for d in [out_dir, fig_dir, table_dir, json_dir, log_dir]:
		ensure_dir(d)
	logger.info(f"Output directories: {out_dir}, {fig_dir}, {table_dir}, {json_dir}, {log_dir}")

	# Load dataset and columns from task config
	df, cols = load_csv_with_columns(ROOT / cfg["dataset_path"], cfg["columns"])
	logger.info(f"Task={task} rows={len(df)} dataset={cfg['dataset_path']}")

	# Dispatch
	if task == "cbt":
		run_cbt(df, cols, cfg, str(fig_dir), str(table_dir), str(json_dir), logger)
	elif task == "rv":
		run_rv(df, cols, cfg, str(fig_dir), str(table_dir), str(json_dir), logger)
	elif task == "ra37":
		run_ra37(df, cols, cfg, str(fig_dir), str(table_dir), str(json_dir), logger)
	elif task == "ra_general":
		run_ra_general(df, cols, cfg, str(fig_dir), str(table_dir), str(json_dir), logger)
	else:
		raise ValueError(f"Unknown task: {task}")

	logger.info("Done.")

if __name__ == "__main__":
	main()