# benchmarks/agent_bridge/cbt.py
from typing import Dict
from pathlib import Path
from utils.io import load_yaml_file
from LLM_bridge.openai_client import chat_complete_jsonless
import logging

# Set up logger for this module
logger = logging.getLogger("LLM_bridge.cbt")

def parse_decision(text):
    """
    Parse a reasoner output to an integer label {0,1}.
    Expected patterns include 'DECISION: 0' or 'DECISION: 1'.
    Fallback scans the string for first 0/1 when the prefix is missing.
    """
    logger.debug(f"[parse_decision] Parsing decision from text: {text!r}")
    for tok in ("0", "1"):
        if f"DECISION: {tok}" in text:
            logger.debug(f"[parse_decision] Found explicit DECISION: {tok}")
            return int(tok)
    for ch in text:
        if ch in ("0", "1"):
            logger.debug(f"[parse_decision] Fallback: found digit {ch}")
            return int(ch)
    logger.error(f"[parse_decision] Could not parse DECISION from: {text!r}")
    raise ValueError("Cannot parse DECISION 0/1")

def _predict_stage(stage, sys_prompt, payload, cfg):
    """
    Internal helper to call the LLM for a given stage.
    """
    logger.info(f"[_predict_stage] Invoking LLM for {stage} with model={cfg.get('model')}, api_base={cfg.get('api_base')}")
    logger.debug(f"[_predict_stage] System prompt: {sys_prompt}")
    logger.debug(f"[_predict_stage] Payload: {payload}")
    raw = chat_complete_jsonless(
        cfg.get("api_base"),
        cfg.get("model"),
        sys_prompt,
        payload,
        cfg.get("effort", "low"),
        cfg.get("timeout_seconds"),
    )
    logger.debug(f"[_predict_stage] Raw LLM output: {raw!r}")
    label = parse_decision(raw)
    logger.info(f"[_predict_stage] Parsed label: {label}")
    return {"label": label, "raw": raw}

def _get_cfg_and_prompts():
    """
    Load and merge configuration and prompt files for CBT and OpenAI.
    """
    root = Path(__file__).resolve().parent.parent  # benchmarks/
    logger.info("[_get_cfg_and_prompts] Loading CBT and OpenAI configs")
    # Load CBT-specific config and OpenAI config, then merge so GPT params are available
    cbt_cfg = load_yaml_file(root / "configs" / "cbt.yaml")
    openai_cfg = load_yaml_file(root / "configs" / "openai_config.yaml")
    cfg = {**openai_cfg, **cbt_cfg}
    # Resolve prompts path from CBT config to honor new centralized config setup
    prompts_path = cbt_cfg.get("prompts_path", "configs/cbt_prompts.yaml")
    logger.info(f"[_get_cfg_and_prompts] Loading prompts from {prompts_path}")
    prompts = load_yaml_file(root / prompts_path)
    return cfg, prompts

def predict_stage1(statement, unhelpful_thoughts):
    """
    Invoke agent's Stage1 reasoner to classify whether UNHELPFUL_THOUGHTS is acceptable.
    Returns a dict with integer label and raw text for traceability.
    """
    logger.info("[predict_stage1] Running Stage 1 prediction")
    cfg, prompts = _get_cfg_and_prompts()
    reasoner = prompts.get("reasoner", {})
    sys_prompt = reasoner.get("stage1")
    if sys_prompt is None:
        logger.error("[predict_stage1] No system prompt found for stage1")
        raise ValueError("Missing system prompt for stage1")
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'
    logger.debug(f"[predict_stage1] Payload: {payload}")
    return _predict_stage("stage1", sys_prompt, payload, cfg)

def predict_stage2(statement, unhelpful_thoughts, challenge):
    """
    Invoke agent's Stage2 reasoner to classify whether CHALLENGE is acceptable.
    """
    logger.info("[predict_stage2] Running Stage 2 prediction")
    cfg, prompts = _get_cfg_and_prompts()
    reasoner = prompts.get("reasoner", {})
    sys_prompt = reasoner.get("stage2")
    if sys_prompt is None:
        logger.error("[predict_stage2] No system prompt found for stage2")
        raise ValueError("Missing system prompt for stage2")
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge};"'
    logger.debug(f"[predict_stage2] Payload: {payload}")
    return _predict_stage("stage2", sys_prompt, payload, cfg)

def predict_stage3(statement, unhelpful_thoughts, challenge, reframe):
    """
    Invoke agent's Stage3 reasoner to classify whether REFRAME is acceptable.
    """
    logger.info("[predict_stage3] Running Stage 3 prediction")
    cfg, prompts = _get_cfg_and_prompts()
    reasoner = prompts.get("reasoner", {})
    sys_prompt = reasoner.get("stage3")
    if sys_prompt is None:
        logger.error("[predict_stage3] No system prompt found for stage3")
        raise ValueError("Missing system prompt for stage3")
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge}; REFRAME: {reframe};"'
    logger.debug(f"[predict_stage3] Payload: {payload}")
    return _predict_stage("stage3", sys_prompt, payload, cfg)

def predict_stage_with_config(stage, payload, prompts_yaml_path=None, api_base=None, model=None, effort=None, timeout_seconds=None):
    """
    Generic entry that allows overriding the system prompt via YAML for maintenance only.
    If prompts_yaml_path is provided, we will load the exact copy and call agent's _chat_complete
    with that system prompt to keep behavior identical. Otherwise fallback to stageN_reasoner.
    This function keeps interfaces minimal for future RV/Analyzer reuse.
    """
    logger.info(f"[predict_stage_with_config] Predicting for stage={stage} using custom prompts at {prompts_yaml_path}")
    if not prompts_yaml_path:
        logger.error("[predict_stage_with_config] prompts_yaml_path must be provided for generic call")
        raise ValueError("prompts_yaml_path must be provided for generic call")
    prompts = load_yaml_file(prompts_yaml_path)
    reasoner = prompts.get("reasoner", {})
    sys_prompt = reasoner.get(stage)
    if sys_prompt is None:
        logger.error(f"[predict_stage_with_config] Unknown stage: {stage}")
        raise ValueError(f"Unknown stage: {stage}")
    logger.debug(f"[predict_stage_with_config] System prompt: {sys_prompt}")
    logger.debug(f"[predict_stage_with_config] Payload: {payload}")
    raw = chat_complete_jsonless(
        api_base, model, sys_prompt, payload, effort or "low", timeout_seconds
    )
    logger.debug(f"[predict_stage_with_config] Raw LLM output: {raw!r}")
    label = parse_decision(raw)
    logger.info(f"[predict_stage_with_config] Parsed label: {label}")
    return {"raw": raw, "label": label}