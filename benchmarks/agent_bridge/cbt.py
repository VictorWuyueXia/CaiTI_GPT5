# benchmarks/agent_bridge/cbt.py
from typing import Dict
from benchmarks.utils.io import load_yaml_file
from benchmarks.llm.openai_client import chat_complete_jsonless

def parse_decision(text):
	"""
	Parse a reasoner output to an integer label {0,1}.
	Expected patterns include 'DECISION: 0' or 'DECISION: 1'.
	Fallback scans the string for first 0/1 when the prefix is missing.
	"""
	for tok in ("0","1"):
		if f"DECISION: {tok}" in text:
			return int(tok)
	for ch in text:
		if ch in ("0","1"):
			return int(ch)
	raise ValueError("Cannot parse DECISION 0/1")

def predict_stage1(statement, unhelpful_thoughts):
	"""
	Invoke agent's Stage1 reasoner to classify whether UNHELPFUL_THOUGHTS is acceptable.
	Returns a dict with integer label and raw text for traceability.
	"""
	# standalone mode expects external caller to supply prompts via predict_stage_with_config
	# Here we keep a minimal placeholder raising if called directly without prompts.
	raise RuntimeError("predict_stage1 requires prompts mode in standalone benchmarks. Use predict_stage_with_config.")
	return {"label": parse_decision(resp), "raw": resp}

def predict_stage2(statement, unhelpful_thoughts, challenge):
	"""
	Invoke agent's Stage2 reasoner to classify whether CHALLENGE is acceptable.
	"""
	raise RuntimeError("predict_stage2 requires prompts mode in standalone benchmarks. Use predict_stage_with_config.")

def predict_stage3(statement, unhelpful_thoughts, challenge, reframe):
	"""
	Invoke agent's Stage3 reasoner to classify whether REFRAME is acceptable.
	"""
	raise RuntimeError("predict_stage3 requires prompts mode in standalone benchmarks. Use predict_stage_with_config.")

def predict_stage_with_config(stage, payload, prompts_yaml_path=None, api_base=None, model=None, temperature=None, max_tokens=None, timeout_seconds=None):
	"""
	Generic entry that allows overriding the system prompt via YAML for maintenance only.
	If prompts_yaml_path is provided, we will load the exact copy and call agent's _chat_complete
	with that system prompt to keep behavior identical. Otherwise fallback to stageN_reasoner.
	This function keeps interfaces minimal for future RV/Analyzer reuse.
	"""
	if prompts_yaml_path:
		prompts = load_yaml_file(prompts_yaml_path)
		reasoner = prompts.get("reasoner", {})
		if stage == "stage1":
			sys_prompt = reasoner.get("stage1")
			raw = chat_complete_jsonless(api_base, model, sys_prompt, payload, temperature, max_tokens, timeout_seconds)
			return {"raw": raw, "label": parse_decision(raw)}
		elif stage == "stage2":
			sys_prompt = reasoner.get("stage2")
			raw = chat_complete_jsonless(api_base, model, sys_prompt, payload, temperature, max_tokens, timeout_seconds)
			return {"raw": raw, "label": parse_decision(raw)}
		elif stage == "stage3":
			sys_prompt = reasoner.get("stage3")
			raw = chat_complete_jsonless(api_base, model, sys_prompt, payload, temperature, max_tokens, timeout_seconds)
			return {"raw": raw, "label": parse_decision(raw)}
		else:
			raise ValueError("Unknown stage")
	else:
		raise ValueError("prompts_yaml_path must be provided for generic call")