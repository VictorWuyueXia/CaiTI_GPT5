# benchmarks/agent_bridge/rv.py
from src.reflection_validation import rv_reasoner

def parse_decision(text):
	for tok in ("0","1"):
		if f"DECISION: {tok}" in text:
			return int(tok)
	for ch in text:
		if ch in ("0","1"):
			return int(ch)
	raise ValueError("Cannot parse DECISION 0/1")

def predict_rv(topic, original_question, original_response, follow_up):
	resp = rv_reasoner(topic, original_question, original_response, follow_up)
	return {"label": parse_decision(resp), "raw": resp}