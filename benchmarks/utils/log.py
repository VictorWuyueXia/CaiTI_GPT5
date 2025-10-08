# benchmarks/utils/log.py
import logging, sys
from pathlib import Path

def ensure_dir(p):
	Path(p).mkdir(parents=True, exist_ok=True)

def build_logger(log_dir, name):
	ensure_dir(log_dir)
	log_path = Path(log_dir) / f"{name}.log"
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	fh = logging.FileHandler(str(log_path))
	fh.setFormatter(fmt)
	ch = logging.StreamHandler(sys.stdout)
	ch.setFormatter(fmt)
	logger.handlers = []
	logger.addHandler(fh)
	logger.addHandler(ch)
	return logger