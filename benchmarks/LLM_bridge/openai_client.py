import os
import logging
from openai import OpenAI
from openai import AsyncOpenAI

# Set up logger for this module
logger = logging.getLogger("LLM_bridge.openai_client")

# Module-level singleton for async OpenAI client and its base URL
_ASYNC_CLIENT = None
_ASYNC_CLIENT_BASE_URL = None

def get_async_openai_client(api_base):
    """
    Return a process-wide singleton AsyncOpenAI client. Initializes once with the given base URL.
    Raises if subsequent calls provide a different base URL to avoid silent mismatch.
    """
    global _ASYNC_CLIENT, _ASYNC_CLIENT_BASE_URL
    if _ASYNC_CLIENT is None:
        logger.info(f"[get_async_openai_client] Initializing AsyncOpenAI client with base_url={api_base}")
        _ASYNC_CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=api_base)
        _ASYNC_CLIENT_BASE_URL = api_base
        return _ASYNC_CLIENT
    if api_base != _ASYNC_CLIENT_BASE_URL:
        logger.error(f"[get_async_openai_client] Attempted to reinitialize AsyncOpenAI client with a different base_url: {api_base} (existing: {_ASYNC_CLIENT_BASE_URL})")
        raise ValueError("Async OpenAI client already initialized with a different base_url")
    logger.debug("[get_async_openai_client] Returning existing AsyncOpenAI client")
    return _ASYNC_CLIENT

async def _ask_async(client, model, system_content, user_content, effort, timeout_seconds):
    """
    Single async request returning assistant's raw text.
    """
    logger.debug(f"[_ask_async] Sending async request: model={model}, effort={effort}, timeout={timeout_seconds}")
    try:
        resp = await client.responses.create(
            model=model,
            reasoning={"effort": effort},
            instructions=system_content,
            input=user_content,
            timeout=timeout_seconds,
        )
        logger.debug(f"[_ask_async] Received response: {resp.output_text!r}")
        return resp.output_text
    except Exception as e:
        logger.error(f"[_ask_async] Exception during async OpenAI call: {e}")
        raise

async def _gather_async(api_base, model, items, effort, timeout_seconds, max_batch):
    """
    Run async requests concurrently with a bounded semaphore equal to max_batch.
    Each item is a dict: {"system_content": str, "user_content": str}
    """
    import asyncio
    logger.info(f"[_gather_async] Gathering {len(items)} async requests with max_batch={max_batch}")
    client = get_async_openai_client(api_base)
    sem = asyncio.Semaphore(max_batch)

    async def _runner(it):
        async with sem:
            logger.debug(f"[_gather_async:_runner] Processing item with system_content={it['system_content'][:30]!r}..., user_content={it['user_content'][:30]!r}...")
            return await _ask_async(
                client,
                model,
                it["system_content"],
                it["user_content"],
                effort,
                timeout_seconds,
            )

    tasks = [asyncio.create_task(_runner(it)) for it in items]
    try:
        results = await asyncio.gather(*tasks)
        logger.info(f"[_gather_async] Completed all async requests")
        return results
    except Exception as e:
        logger.error(f"[_gather_async] Exception during batch async requests: {e}")
        raise

def chat_complete_many(api_base, model, items, effort, timeout_seconds, parallel, max_batch):
    """
    Batch entry for callers. When parallel is False, it falls back to sequential sync calls.
    Each item is a dict: {"system_content": str, "user_content": str}
    Returns a list of raw texts in the same order as input.
    """
    logger.info(f"[chat_complete_many] Called with parallel={parallel}, num_items={len(items)}, max_batch={max_batch}")
    if not parallel:
        # Sequential path, reuse existing sync function to minimize code duplication
        logger.info("[chat_complete_many] Using sequential (sync) path")
        out = []
        for idx, it in enumerate(items):
            logger.debug(f"[chat_complete_many] Processing item {idx+1}/{len(items)} (sequential)")
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
        logger.info("[chat_complete_many] Completed all sequential requests")
        return out

    # Parallel path with asyncio
    logger.info("[chat_complete_many] Using parallel (async) path")
    import asyncio
    try:
        results = asyncio.run(
            _gather_async(api_base, model, items, effort, timeout_seconds, max_batch)
        )
        logger.info("[chat_complete_many] Completed all parallel requests")
        return results
    except Exception as e:
        logger.error(f"[chat_complete_many] Exception in parallel path: {e}")
        raise

# Module-level singleton for OpenAI client and its base URL
_CLIENT = None
_CLIENT_BASE_URL = None

def get_openai_client(api_base):
    """
    Return a process-wide singleton OpenAI client. Initializes once with the given base URL.
    Raises if subsequent calls provide a different base URL to avoid silent mismatch.
    """
    global _CLIENT, _CLIENT_BASE_URL
    if _CLIENT is None:
        logger.info(f"[get_openai_client] Initializing OpenAI client with base_url={api_base}")
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=api_base)
        _CLIENT_BASE_URL = api_base
        return _CLIENT
    if api_base != _CLIENT_BASE_URL:
        logger.error(f"[get_openai_client] Attempted to reinitialize OpenAI client with a different base_url: {api_base} (existing: {_CLIENT_BASE_URL})")
        raise ValueError("OpenAI client already initialized with a different base_url")
    logger.debug("[get_openai_client] Returning existing OpenAI client")
    return _CLIENT

def chat_complete_jsonless(api_base, model, system_content, user_content, effort, timeout_seconds):
    """
    Minimal OpenAI responses wrapper for standalone benchmarks using GPT-5.
    Relies on OPENAI_API_KEY in environment.
    Returns raw assistant text (no enforced JSON), so callers must parse themselves.
    """
    logger.debug(f"[chat_complete_jsonless] Sending sync request: model={model}, effort={effort}, timeout={timeout_seconds}")
    client = get_openai_client(api_base)
    try:
        resp = client.responses.create(
            model=model,
            reasoning={"effort": effort},
            instructions=system_content,
            input=user_content,
            timeout=timeout_seconds,
        )
        logger.debug(f"[chat_complete_jsonless] Received response: {resp.output_text!r}")
        return resp.output_text
    except Exception as e:
        logger.error(f"[chat_complete_jsonless] Exception during sync OpenAI call: {e}")
        raise
