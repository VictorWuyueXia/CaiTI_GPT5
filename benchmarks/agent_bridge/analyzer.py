# benchmarks/agent_bridge/analyzer.py
from typing import Tuple
from src.response_analyzer import classify_dimension_and_score

def parse_dim_score(text):
	# Expect "dim, score" possibly with spaces/quotes
	s = text.strip().strip('"').strip("'")
	if "," in s:
		a, b = s.split(",", 1)
		dim = a.strip()
		score_str = b.strip()
		try:
			score = int(score_str)
		except:
			# Sometimes like "2." or stray chars
			score = int("".join(ch for ch in score_str if ch.isdigit())[0])
		return dim, score
	raise ValueError("Cannot parse 'DIMENSION, SCORE'")

def predict_analyzer(user_input):
	resp = classify_dimension_and_score(user_input)
	dim, score = parse_dim_score(resp)
	return {"dimension": dim, "score": score, "raw": resp}