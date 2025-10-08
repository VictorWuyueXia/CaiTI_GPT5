import os
from openai import OpenAI
from openai import AsyncOpenAI

_ASYNC_CLIENT = None
_ASYNC_CLIENT_BASE_URL = None

def get_async_openai_client(api_base):
    """
    Return a process-wide singleton AsyncOpenAI client. Initializes once with the given base URL.
    Raises if subsequent calls provide a different base URL to avoid silent mismatch.
    """
    global _ASYNC_CLIENT, _ASYNC_CLIENT_BASE_URL
    if _ASYNC_CLIENT is None:
        _ASYNC_CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=api_base)
        _ASYNC_CLIENT_BASE_URL = api_base
        return _ASYNC_CLIENT
    if api_base != _ASYNC_CLIENT_BASE_URL:
        raise ValueError("Async OpenAI client already initialized with a different base_url")
    return _ASYNC_CLIENT

async def _ask_async(client, model, system_content, user_content, effort, timeout_seconds):
    """
    Single async request returning assistant's raw text.
    """
    resp = await client.responses.create(
        model=model,
        reasoning={"effort": effort},
        instructions=system_content,
        input=user_content,
        timeout=timeout_seconds,
    )
    return resp.output_text

async def _gather_async(api_base, model, items, effort, timeout_seconds, max_batch):
    """
    Run async requests concurrently with a bounded semaphore equal to max_batch.
    Each item is a dict: {"system_content": str, "user_content": str}
    """
    import asyncio
    client = get_async_openai_client(api_base)
    sem = asyncio.Semaphore(max_batch)

    async def _runner(it):
        async with sem:
            return await _ask_async(
                client,
                model,
                it["system_content"],
                it["user_content"],
                effort,
                timeout_seconds,
            )

    tasks = [asyncio.create_task(_runner(it)) for it in items]
    return await asyncio.gather(*tasks)

def chat_complete_many(api_base, model, items, effort, timeout_seconds, parallel, max_batch):
    """
    Batch entry for callers. When parallel is False, it falls back to sequential sync calls.
    Each item is a dict: {"system_content": str, "user_content": str}
    Returns a list of raw texts in the same order as input.
    """
    if not parallel:
        # Sequential path, reuse existing sync function to minimize code duplication
        out = []
        for it in items:
            out.append(
                chat_complete_jsonless(
                    api_base,
                    model,
                    it["system_content"],
                    it["user_content"],
                    effort,
                    timeout_seconds,
                )
            )
        return out

    # Parallel path with asyncio
    import asyncio
    return asyncio.run(
        _gather_async(api_base, model, items, effort, timeout_seconds, max_batch)
    )

# Module-level singleton for OpenAI client
_CLIENT = None
_CLIENT_BASE_URL = None

def get_openai_client(api_base):
	"""
	Return a process-wide singleton OpenAI client. Initializes once with the given base URL.
	Raises if subsequent calls provide a different base URL to avoid silent mismatch.
	"""
	global _CLIENT, _CLIENT_BASE_URL
	if _CLIENT is None:
		_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=api_base)
		_CLIENT_BASE_URL = api_base
		return _CLIENT
	if api_base != _CLIENT_BASE_URL:
		raise ValueError("OpenAI client already initialized with a different base_url")
	return _CLIENT

def chat_complete_jsonless(api_base, model, system_content, user_content, effort, timeout_seconds):
	"""
	Minimal OpenAI responses wrapper for standalone benchmarks using GPT-5.
	Relies on OPENAI_API_KEY in environment.
	Returns raw assistant text (no enforced JSON), so callers must parse themselves.
	"""
	client = get_openai_client(api_base)
	resp = client.responses.create(
		model=model,
		reasoning={"effort": effort},
		instructions=system_content,
		input=user_content,
		timeout=timeout_seconds,
	)
	return resp.output_text

