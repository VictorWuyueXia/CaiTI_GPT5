# benchmarks/utils/cache.py
import json
from pathlib import Path

def key_to_path(cache_dir, key):
	return Path(cache_dir) / (key.replace("::","__") + ".json")

def try_load(cache_dir, key):
	fp = key_to_path(cache_dir, key)
	if fp.exists():
		with open(fp, "r") as f:
			return json.load(f)
	return None

def save(cache_dir, key, obj):
	fp = key_to_path(cache_dir, key)
	fp.parent.mkdir(parents=True, exist_ok=True)
	with open(fp, "w") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)