import os
from openai import OpenAI

def chat_complete_jsonless(api_base, model, system_content, user_content, temperature, max_tokens, timeout_seconds):
	"""
	Minimal OpenAI chat wrapper for standalone benchmarks.
	Relies on OPENAI_API_KEY in environment.
	Returns raw assistant text (no enforced JSON), so callers must parse themselves.
	"""
	client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=api_base)
	resp = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": system_content},
			{"role": "user", "content": user_content},
		],
		temperature=temperature,
		max_tokens=max_tokens,
		timeout=timeout_seconds,
	)
	return resp.choices[0].message.content

