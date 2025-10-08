# benchmarks/scripts/run_eval.py

import argparse, sys, time
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd

# Set up paths for importing project modules
CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "agent"))  # import agent/src/*

# Import utility and task-specific functions
from utils.log import build_logger, ensure_dir
from utils.io import load_csv_with_columns
from tasks.cbt import run_cbt
from tasks.rv import run_rv
from tasks.ra37 import run_ra37

def now_tag():
	"""Return a timestamp string for unique run IDs."""
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def main():
	# Parse command-line arguments for task and config file
	p = argparse.ArgumentParser()
	p.add_argument("--task", type=str, required=True)
	p.add_argument("--config", type=str, required=True)
	args = p.parse_args()

	# Load configuration YAML
	with open(args.config, "r") as f:
		cfg = yaml.safe_load(f)

	task = cfg["task"]
	run_id = now_tag()  # Unique run identifier
	out_root = Path(cfg["output_root"])
	# Define output directories for this run
	out_dir = out_root / f"{task}_{run_id}"
	fig_dir = out_dir / "figs"
	table_dir = out_dir / "tables"
	json_dir = out_dir / "json"
	log_dir = ROOT / "logs"
	# Ensure all output directories exist
	for d in [out_dir, fig_dir, table_dir, json_dir, log_dir]:
		ensure_dir(d)

	# Set up logger for this run
	logger = build_logger(log_dir, f"{task}_{run_id}")

	# Load dataset and column mapping
	df, cols = load_csv_with_columns(ROOT / cfg["dataset_path"], cfg["columns"])
	logger.info(f"Task={task} rows={len(df)} dataset={cfg['dataset_path']}")

	# Dispatch to the appropriate task runner
	if task == "cbt":
		run_cbt(df, cols, cfg, str(fig_dir), str(table_dir), str(json_dir), logger)
	elif task == "rv":
		run_rv(df, cols, cfg, str(fig_dir), str(table_dir), str(json_dir), logger)
	elif task == "ra37":
		run_ra37(df, cols, cfg, str(fig_dir), str(table_dir), str(json_dir), logger)
	else:
		raise ValueError("Unknown task")

	logger.info("Done.")

if __name__ == "__main__":
	main()