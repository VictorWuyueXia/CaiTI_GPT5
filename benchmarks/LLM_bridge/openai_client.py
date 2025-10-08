import os
from openai import OpenAI

def chat_complete_jsonless(api_base, model, system_content, user_content, effort, timeout_seconds):
	"""
	Minimal OpenAI responses wrapper for standalone benchmarks using GPT-5.
	Relies on OPENAI_API_KEY in environment.
	Returns raw assistant text (no enforced JSON), so callers must parse themselves.
	"""
	client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=api_base)
	resp = client.responses.create(
		model=model,
		reasoning={"effort": effort},
		instructions=system_content,
		input=user_content,
		timeout=timeout_seconds,
	)
	return resp.output_text

