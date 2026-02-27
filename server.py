"""HTTP API server for LLaMA inference with SSE streaming.

Loads a LLaMA model and serves inference requests over HTTP.
Supports both blocking JSON responses and token-by-token SSE streaming.

Usage:
    python server.py --model_path /path/to/model [--port 8080] [--tp 4]

Multi-host (run on each machine):
    python server.py --model_path /path/to/model --tp 4
"""

import argparse
import asyncio
import itertools
import json
import sys
import threading
import time
import traceback
from typing import AsyncGenerator, Optional

import jax
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from jax.sharding import Mesh
from pydantic import BaseModel

from inference import format_prompt, load_model
from models.engine import ServingConfig, ServingLoop, UserRequestPrompt
from models.sync_server import SyncServer

# ---------------------------------------------------------------------------
# Global state (set in main() before the server accepts requests)
# ---------------------------------------------------------------------------
serving_loop: Optional[ServingLoop] = None
tokenizer = None

_request_counter = itertools.count()
_counter_lock = threading.Lock()


def _next_id() -> int:
    with _counter_lock:
        return next(_request_counter)


# ---------------------------------------------------------------------------
# Background inference thread
# ---------------------------------------------------------------------------

def _inference_loop(shutdown: threading.Event) -> None:
    """Call serving_step() continuously.  Must run on exactly one thread."""
    while not shutdown.is_set():
        serving_loop.serving_step()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ReLax Inference Server")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None  # overrides server default when set (future use)


# -- Health ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# -- Blocking JSON response --------------------------------------------------

@app.post("/generate")
async def generate(req: GenerateRequest):
    req_id = _next_id()
    tokens = format_prompt(req.prompt, tokenizer)
    serving_loop.add_request(UserRequestPrompt(id=req_id, text=tokens))

    start = time.time()
    while True:
        await asyncio.sleep(0.01)
        result = serving_loop.results.get(req_id)
        if result is not None and result.done:
            text = tokenizer.decode(result.token_list)
            elapsed = round(time.time() - start, 3)
            # Clean up so the results dict doesn't grow unbounded
            del serving_loop.results[req_id]
            return {
                "id": req_id,
                "text": text,
                "tokens_generated": result.tokens_decoded,
                "elapsed_seconds": elapsed,
            }


# -- SSE streaming response --------------------------------------------------

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    req_id = _next_id()
    tokens = format_prompt(req.prompt, tokenizer)
    serving_loop.add_request(UserRequestPrompt(id=req_id, text=tokens))

    return StreamingResponse(
        _token_stream(req_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevent nginx from buffering SSE
        },
    )


async def _token_stream(req_id: int) -> AsyncGenerator[str, None]:
    """Yield SSE events as tokens arrive, one chunk per decode batch.

    The engine produces tokens in batches of `decode_steps` (default 10).
    We diff the decoded string each time the token list grows to handle
    multi-byte characters correctly.

    SSE event format:
        data: {"text": "<new text chunk>"}\n\n   — incremental text
        data: {"done": true, "tokens_generated": N, "elapsed_seconds": F}\n\n
    """
    last_text = ""
    start = time.time()
    try:
        while True:
            await asyncio.sleep(0.01)
            result = serving_loop.results.get(req_id)
            if result is None:
                continue

            # Take a GIL-safe snapshot so the background thread can keep appending
            token_list = list(result.token_list)
            done = result.done

            if token_list:
                full_text = tokenizer.decode(token_list)
                new_text = full_text[len(last_text):]
                if new_text:
                    yield f"data: {json.dumps({'text': new_text})}\n\n"
                    last_text = full_text

            if done:
                elapsed = round(time.time() - start, 3)
                yield (
                    f"data: {json.dumps({'done': True, 'tokens_generated': result.tokens_decoded, 'elapsed_seconds': elapsed})}\n\n"
                )
                break
    finally:
        # Clean up whether the client disconnected or generation finished
        serving_loop.results.pop(req_id, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global serving_loop, tokenizer

    jax.distributed.initialize()
    pid = jax.process_index()

    parser = argparse.ArgumentParser(description="ReLax inference API server")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="server_config.json",
                        help="Path to server config JSON (default: server_config.json)")
    args = parser.parse_args()

    # Load server config
    import json as _json
    with open(args.config) as f:
        cfg = _json.load(f)

    port               = cfg["port"]
    tp                 = cfg["tp"]
    max_decode_length  = cfg["max_decode_length"]
    decode_batch_size  = cfg["decode_batch_size"]
    prefill_batch_size = cfg["prefill_batch_size"]
    decode_steps       = cfg["decode_steps"]

    # Load model
    model, params, config, tokenizer = load_model(args.model_path)

    # Build device mesh
    devices = jax.devices()
    dp = len(devices) // tp
    mesh = Mesh(np.array(devices).reshape(dp, tp), ("dp", "tp"))
    print(f"[P{pid}] Mesh: dp={dp} tp={tp} ({len(devices)} devices total)")
    sys.stdout.flush()

    serve_cfg = ServingConfig(
        decode_steps=decode_steps,
        decode_batch_size=decode_batch_size,
        prefill_batch_size=prefill_batch_size,
        eos_tokens=(tokenizer.eot_id,),
        token_pad_idx=tokenizer.pad_id,
        max_decode_length=max_decode_length,
    )

    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        is_server=(pid == 0),
    )

    print(f"[P{pid}] Warming up (compiling JIT functions)...")
    sys.stdout.flush()
    serving_loop.warmup()
    print(f"[P{pid}] Warmup complete.")
    sys.stdout.flush()

    # Start background inference thread (all processes)
    shutdown = threading.Event()
    inference_thread = threading.Thread(
        target=_inference_loop, args=(shutdown,), daemon=True, name="inference"
    )
    inference_thread.start()

    if pid == 0:
        # Server process: run the HTTP server in the main thread
        import uvicorn
        print(f"[P{pid}] HTTP server listening on 0.0.0.0:{port}")
        sys.stdout.flush()
        try:
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        finally:
            shutdown.set()
    else:
        # Worker processes: just run the inference loop
        print(f"[P{pid}] Worker process running inference loop")
        sys.stdout.flush()
        try:
            inference_thread.join()
        except KeyboardInterrupt:
            shutdown.set()

    SyncServer.barrier("shutdown", 0)
    print(f"[P{pid}] Shutdown complete.")
    jax.distributed.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pid = jax.process_index()
        print(f"\n[P{pid}] FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
