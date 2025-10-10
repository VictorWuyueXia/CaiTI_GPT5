# benchmarks/utils/cache.py
import json
import hashlib
import time
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

def stable_hash(text):
	"""Return a stable short hash for a given text using SHA256."""
	return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]

def config_signature(cfg):
	"""Build a stable signature string from config fields that affect model outputs.

	Includes: model, effort, api_base, timeout_seconds, prompts_path (if any).
	This function accepts a merged cfg (CBT cfg overlaid on OpenAI cfg) or a CBT-only cfg.
	"""
	parts = [
		str(cfg.get("model")),
		str(cfg.get("effort")),
		str(cfg.get("api_base")),
		str(cfg.get("timeout_seconds")),
		str(cfg.get("prompts_path")),
	]
	joined = "|".join(parts)
	return stable_hash(joined)

def cleanup_cache(cache_dir, max_age_days):
	"""Delete cache files older than max_age_days in the given cache_dir."""
	root = Path(cache_dir)
	if not root.exists():
		return
	now_ts = time.time()
	max_age_seconds = float(max_age_days) * 86400.0
	for fp in root.rglob("*.json"):
		st = fp.stat()
		age = now_ts - st.st_mtime
		if age > max_age_seconds:
			fp.unlink()