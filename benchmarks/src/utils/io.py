# benchmarks/utils/io.py
import pandas as pd
import yaml
from pathlib import Path

def load_yaml_file(yaml_path):
	"""
	Load a YAML file and return the parsed Python object.
	Raise if the file does not exist or content is not a mapping/sequence as expected.
	"""
	p = Path(yaml_path)
	if not p.exists():
		raise FileNotFoundError(f"YAML not found: {yaml_path}")
	with open(p, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)

def load_csv_with_columns(csv_path, columns_map):
	df = pd.read_csv(csv_path)
	for k, v in columns_map.items():
		if v and v not in df.columns:
			raise ValueError(f"Missing column: {v}")
	return df, columns_map