"""OpenAI-compatible HTTP API server for LLaMA inference.

Loads a LLaMA model and serves inference requests over HTTP
via the standard /v1/chat/completions endpoint.

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
import uuid
from typing import List, Optional

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

import jax
import numpy as np
from fastapi import FastAPI, HTTPException
from jax.sharding import Mesh
from pydantic import BaseModel

from inference import load_model
from models.engine import ServingConfig, ServingLoop, UserRequestPrompt
from models.sync_server import SyncServer

# ---------------------------------------------------------------------------
# Global state (set in main() before the server accepts requests)
# ---------------------------------------------------------------------------
serving_loop: Optional[ServingLoop] = None
tokenizer = None
max_pending_requests: int = 500

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


def _wandb_system_loop(shutdown: threading.Event, interval: float = 5.0) -> None:
    """Periodically log system-level metrics to wandb."""
    while not shutdown.is_set():
        if serving_loop is not None:
            active = sum(1 for x in serving_loop.decode_work.active_results if x is not None)
            _wandb.log({
                "system/pending_requests": serving_loop.pending_prefill_count(),
                "system/active_slots": active,
                "system/decode_call_count": serving_loop._decode_call_count,
            })
        shutdown.wait(interval)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ReLax Inference Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 4000


def _format_chat_prompt(messages: List[ChatMessage], tok) -> List[int]:
    """Convert an OpenAI-style messages array into LLaMA chat-template tokens."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        parts.append(f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>")
    # Open the assistant turn so the model continues from here
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return tok.encode("".join(parts), bos=False, eos=False)


# -- Health ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# -- OpenAI-compatible chat completions --------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if serving_loop.pending_prefill_count() >= max_pending_requests:
        raise HTTPException(status_code=429, detail="Too many pending requests")
    req_id = _next_id()
    tokens = _format_chat_prompt(req.messages, tokenizer)
    prompt_tokens_count = len(tokens)
    if prompt_tokens_count >= serving_loop.serve_cfg.max_cache_seqlen:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long ({prompt_tokens_count} tokens), max is {serving_loop.serve_cfg.max_cache_seqlen - 1}",
        )
    serving_loop.add_request(UserRequestPrompt(id=req_id, text=tokens))

    start = time.time()
    while True:
        await asyncio.sleep(0.01)
        result = serving_loop.results.get(req_id)
        if result is not None and result.done:
            text = tokenizer.decode(result.token_list)
            completion_tokens_count = result.tokens_decoded
            del serving_loop.results[req_id]
            latency = time.time() - start
            if _wandb is not None and _wandb.run is not None:
                _wandb.log({
                    "request/prompt_tokens": prompt_tokens_count,
                    "request/completion_tokens": completion_tokens_count,
                    "request/total_tokens": prompt_tokens_count + completion_tokens_count,
                    "request/latency_s": latency,
                    "request/tokens_per_sec": completion_tokens_count / latency if latency > 0 else 0,
                })
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model or "relax",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens_count,
                    "completion_tokens": completion_tokens_count,
                    "total_tokens": prompt_tokens_count + completion_tokens_count,
                },
            }



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global serving_loop, tokenizer, max_pending_requests

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
    max_pending_requests = cfg.get("max_pending_requests", 500)


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
        max_cache_seqlen=max_decode_length,
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

    # Initialize wandb on P0 if available
    if pid == 0 and _wandb is not None:
        _wandb.init(project="relax", config={
            "tp": tp,
            "decode_batch_size": decode_batch_size,
            "prefill_batch_size": prefill_batch_size,
            "max_decode_length": max_decode_length,
            "decode_steps": decode_steps,
            "max_pending_requests": max_pending_requests,
        })
        print(f"[P{pid}] wandb initialized")
        sys.stdout.flush()

    # Start background inference thread (all processes)
    shutdown = threading.Event()
    inference_thread = threading.Thread(
        target=_inference_loop, args=(shutdown,), daemon=True, name="inference"
    )
    inference_thread.start()

    # Start wandb system metrics thread on P0
    if pid == 0 and _wandb is not None and _wandb.run is not None:
        wandb_thread = threading.Thread(
            target=_wandb_system_loop, args=(shutdown,), daemon=True, name="wandb-system"
        )
        wandb_thread.start()

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
