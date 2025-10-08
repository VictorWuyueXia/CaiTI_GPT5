from typing import Optional, Dict
from pathlib import Path
import logging

# Import utility functions for YAML loading and OpenAI API interaction
from utils.io import load_yaml_file
from LLM_bridge.openai_client import chat_complete_jsonless
# Import decision parsing function from CBT module (reused for RV)
from LLM_bridge.cbt import parse_decision

# Set up logger for this module
logger = logging.getLogger("LLM_bridge.rv")


def _get_cfg_and_prompts() -> (Dict, Dict):
    """
    Load and merge configuration and prompt files for RV and OpenAI.
    Returns a tuple of (merged_configuration, prompts_dictionary).
    """
    # Get the root directory (benchmarks/)
    root = Path(__file__).resolve().parent.parent
    logger.info("[_get_cfg_and_prompts] Loading RV and OpenAI configs")
    
    # Load RV-specific configuration
    rv_cfg = load_yaml_file(root / "configs" / "rv.yaml")
    # Load OpenAI API configuration
    openai_cfg = load_yaml_file(root / "configs" / "openai_config.yaml")
    # Merge configurations (OpenAI config takes precedence, then RV config)
    cfg = {**openai_cfg, **rv_cfg}
    
    # Get prompts path from RV config or use default
    prompts_path = rv_cfg.get("prompts_path", "configs/rv_prompts.yaml")
    logger.info(f"[_get_cfg_and_prompts] Loading prompts from {prompts_path}")
    
    # Load prompts YAML file
    prompts = load_yaml_file(root / prompts_path)
    return cfg, prompts


def predict_reasoner(
        topic: str,
        original_response: str,
        follow_up_response: str,
        *,
        prompts_yaml_path: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, object]:
    """
    Invoke RV reasoner to classify whether Follow-up is related (0) or not (1).
    Returns a dict with integer label and raw text for traceability.
    """
    logger.info("[predict_reasoner] Using default RV config and prompts")
    cfg, prompts = _get_cfg_and_prompts()
    sys_prompt = prompts.get("reasoner")
    if sys_prompt is None:
        logger.error("[predict_reasoner] No system prompt found for reasoner")
        raise ValueError("Missing system prompt for RV reasoner")
    
    # Format the input payload with proper string representation
    payload = f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow Up Response": {follow_up_response!r}}}'
    
    # Call OpenAI API with default configuration
    raw = chat_complete_jsonless(
        cfg.get("api_base"),
        cfg.get("model"),
        sys_prompt,
        payload,
        cfg.get("effort", "low"),
        cfg.get("timeout_seconds"),
    )
    # Parse the decision from the raw response
    label = parse_decision(raw)
    return {"label": label, "raw": raw}


# def predict_guide(topic: str, original_response: str, follow_up_response: str, *, prompts_yaml_path: Optional[str] = None, api_base: Optional[str] = None, model: Optional[str] = None, effort: Optional[str] = None, timeout_seconds: Optional[int] = None) -> str:
#     """
#     Optional helper to get RV guide text (not used for metrics).
#     Generates guidance text when follow-up response is unrelated to topic/original response.
#     """
#     # Handle custom prompts path if provided
#     if prompts_yaml_path:
#         root = Path(__file__).resolve().parent.parent
#         # Load prompts from absolute path or relative to root
#         prompts = load_yaml_file(root / prompts_yaml) if not Path(prompts_yaml_path).is_absolute() else load_yaml_file(prompts_yaml_path)
#         sys_prompt = prompts.get("guide")
#         # Call OpenAI API to generate guide text
#         return chat_complete_jsonless(api_base, model, sys_prompt, f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}', effort or "low", timeout_seconds)
    
#     # Use default configuration and prompts
#     cfg, prompts = _get_cfg_and_prompts()
#     sys极rompt = prompts.get("guide")
#     # Call OpenAI API with default configuration to generate guide text
#     return chat_complete_jsonless(cfg.get("api_base"), cfg.get("model"), sys_prompt, f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}', cfg.get("effort", "low"), cfg.get("timeout_seconds"))


# def predict_validation(topic: str, original_response: str, follow_up_response: str, *, prompts_yaml_path: Optional[str] =极, api_base: Optional[str] = None, model: Optional[str] = None, effort: Optional[str] = None, timeout_seconds: Optional[int] = None) -> str:
#     """
#     Optional helper to get RV validation text (not used for metrics).
#     Generates validation text to confirm the reasoner's decision.
#     """
#     # Handle custom prompts path if provided
#     if prompts_yaml_path:
#         root = Path(__file__).resolve().parent.parent
#         # Load prompts from absolute path or relative to root
#         prompts = load_yaml_file(root / prompts_yaml_path) if not Path(prompts_yaml_path).is_absolute() else load_yaml_file(prompts_yaml_path)
#         sys_prompt = prompts.get("validation")
#         # Call OpenAI API to generate validation text
#         return chat_complete_jsonless(api_base, model极 sys_prompt, f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}', effort or "low", timeout_seconds)
    
#     # Use default configuration and prompts
#     cfg, prompts = _get_cfg_and_prompts()
#     sys_prompt = prompts.get("validation")
#     # Call OpenAI API with default configuration to generate validation text
#     return chat_complete_jsonless(cfg.get("api_base"), cfg.get("model"), sys_prompt, f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}', cfg.get("effort", "low"), cfg.get("timeout_seconds"))

