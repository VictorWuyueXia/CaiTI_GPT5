# benchmarks/agent_bridge/cbt.py
from typing import Dict
from pathlib import Path
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

def _predict_stage(stage, sys_prompt, payload, cfg):
	raw = chat_complete_jsonless(
		cfg.get("api_base"),
		cfg.get("model"),
		sys_prompt,
		payload,
		cfg.get("effort", "low"),
		cfg.get("timeout_seconds"),
	)
	return {"label": parse_decision(raw), "raw": raw}

def _get_cfg_and_prompts():
	root = Path(__file__).resolve().parent.parent  # benchmarks/
	# Load CBT-specific config and OpenAI config, then merge so GPT params are available
	cbt_cfg = load_yaml_file(root / "configs" / "cbt.yaml")
	openai_cfg = load_yaml_file(root / "configs" / "openai_config.yaml")
	cfg = {**openai_cfg, **cbt_cfg}
	prompts = load_yaml_file(root / "configs" / "cbt_prompts.yaml")
	return cfg, prompts

def predict_stage1(statement, unhelpful_thoughts):
	"""
	Invoke agent's Stage1 reasoner to classify whether UNHELPFUL_THOUGHTS is acceptable.
	Returns a dict with integer label and raw text for traceability.
	"""
	cfg, prompts = _get_cfg_and_prompts()
	reasoner = prompts.get("reasoner", {})
	sys_prompt = reasoner.get("stage1")
	payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'
	return _predict_stage("stage1", sys_prompt, payload, cfg)

def predict_stage2(statement, unhelpful_thoughts, challenge):
	"""
	Invoke agent's Stage2 reasoner to classify whether CHALLENGE is acceptable.
	"""
	cfg, prompts = _get_cfg_and_prompts()
	reasoner = prompts.get("reasoner", {})
	sys_prompt = reasoner.get("stage2")
	payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge};"'
	return _predict_stage("stage2", sys_prompt, payload, cfg)

def predict_stage3(statement, unhelpful_thoughts, challenge, reframe):
	"""
	Invoke agent's Stage3 reasoner to classify whether REFRAME is acceptable.
	"""
	cfg, prompts = _get_cfg_and_prompts()
	reasoner = prompts.get("reasoner", {})
	sys_prompt = reasoner.get("stage3")
	payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge}; REFRAME: {reframe};"'
	return _predict_stage("stage3", sys_prompt, payload, cfg)

def predict_stage_with_config(stage, payload, prompts_yaml_path=None, api_base=None, model=None, effort=None, timeout_seconds=None):
	"""
	Generic entry that allows overriding the system prompt via YAML for maintenance only.
	If prompts_yaml_path is provided, we will load the exact copy and call agent's _chat_complete
	with that system prompt to keep behavior identical. Otherwise fallback to stageN_reasoner.
	This function keeps interfaces minimal for future RV/Analyzer reuse.
	"""
	if not prompts_yaml_path:
		raise ValueError("prompts_yaml_path must be provided for generic call")
	prompts = load_yaml_file(prompts_yaml_path)
	reasoner = prompts.get("reasoner", {})
	sys_prompt = reasoner.get(stage)
	if sys_prompt is None:
		raise ValueError(f"Unknown stage: {stage}")
	raw = chat_complete_jsonless(
		api_base, model, sys_prompt, payload, effort or "low", timeout_seconds
	)
	return {"raw": raw, "label": parse_decision(raw)}