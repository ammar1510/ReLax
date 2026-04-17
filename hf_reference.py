"""Reference forward pass via HuggingFace transformers.

Loads the HF checkpoint, runs one forward pass on a fixed input, and saves
logits + per-layer hidden states for comparison against the JAX model.

Usage:
    python hf_reference.py --model_path ~/gemma-4-31B-it --out_dir ./ref
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = "The capital of France is"


def build_input_ids(tok, prompt: str, mode: str, enable_thinking: bool):
    """Produce the exact token IDs to feed to both HF and JAX.

    Modes (see gemma4_chat_template.md):
      raw  -> <bos> + tokenize(prompt)                           (math sanity)
      chat -> <bos> + Gemma 4 single-turn user prompt.

    Chat rendering, per §2/§3/§8:

      enable_thinking=False (default — matches HF chat_template default):
        <bos>
        <|turn>user\n{prompt}<turn|>\n
        <|turn>model\n
        <|channel>thought\n<channel|>     ← empty thought, suppresses reasoning

      enable_thinking=True:
        <bos>
        <|turn>system\n<|think|><turn|>\n  ← system turn with think marker
        <|turn>user\n{prompt}<turn|>\n
        <|turn>model\n
    """
    if mode == "raw":
        body = tok.encode(prompt, add_special_tokens=False)
    elif mode == "chat":
        parts = []
        if enable_thinking:
            parts.append("<|turn>system\n<|think|><turn|>\n")
        parts.append(f"<|turn>user\n{prompt}<turn|>\n")
        parts.append("<|turn>model\n")
        if not enable_thinking:
            parts.append("<|channel>thought\n<channel|>")
        body = tok.encode("".join(parts), add_special_tokens=False)
    else:
        raise ValueError(mode)

    bos = tok.bos_token_id
    assert bos is not None, "tokenizer has no bos_token_id"
    return [bos] + body


def build_input_ids_via_template(tok, prompt: str, enable_thinking: bool):
    """Alternative: let HF render the template. Use to cross-check the manual
    build against the tokenizer's own chat_template."""
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", default="./ref")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="auto",
                    help='"auto" (accelerate), "cpu", "cuda", or "cuda:0"')
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--mode", default="chat", choices=["raw", "chat"],
                    help="raw=<bos>+prompt (math check); chat=Gemma turn template")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Use thinking mode: inject system turn with <|think|> "
                         "instead of the thought-suppression trailer.")
    ap.add_argument("--verify_template", action="store_true",
                    help="Cross-check manual IDs against tok.apply_chat_template.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print(f"Loading tokenizer from {args.model_path}...")
    tok = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading model ({args.dtype}, device={args.device})...")
    # attn_implementation='eager' -> plain softmax attention, no flash-attn
    # kernel variance. Makes the reference numerically deterministic.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device if args.device != "cpu" else None,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    if args.device == "cpu":
        model = model.to("cpu")

    id_list = build_input_ids(tok, args.prompt, args.mode, args.enable_thinking)
    ids = torch.tensor([id_list], dtype=torch.long)
    print(f"\nMode: {args.mode}  enable_thinking={args.enable_thinking}")
    print(f"Token IDs ({len(id_list)}): {id_list}")
    print(f"Decoded: {tok.decode(id_list)!r}")

    if args.verify_template and args.mode == "chat":
        ref_ids = build_input_ids_via_template(tok, args.prompt, args.enable_thinking)
        match = list(ref_ids) == id_list
        print(f"apply_chat_template IDs ({len(ref_ids)}): {list(ref_ids)}")
        print(f"Manual build matches tokenizer template: {match}")
        if not match:
            print("WARNING: manual build diverges from tokenizer chat_template. "
                  "Trust apply_chat_template — update build_input_ids to match.")

    # Move ids to the model's first parameter device
    first_device = next(model.parameters()).device
    ids = ids.to(first_device)

    print("Running forward pass...")
    with torch.no_grad():
        out = model(
            ids,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    # logits: [1, seqlen, vocab]
    logits = out.logits.detach().to(torch.float32).cpu().numpy()
    # hidden_states: tuple of length n_layers + 1 (embeddings + after each layer)
    # Stack to [n_layers + 1, 1, seqlen, dim]
    hidden = np.stack(
        [h.detach().to(torch.float32).cpu().numpy() for h in out.hidden_states],
        axis=0,
    )

    np.save(os.path.join(args.out_dir, "input_ids.npy"), ids.cpu().numpy())
    np.save(os.path.join(args.out_dir, "logits.npy"), logits)
    np.save(os.path.join(args.out_dir, "hidden_states.npy"), hidden)

    print(f"\nSaved to {args.out_dir}/:")
    print(f"  input_ids.npy     {ids.shape}")
    print(f"  logits.npy        {logits.shape}  dtype=float32")
    print(f"  hidden_states.npy {hidden.shape}  (embed + per-layer residual stream)")
    print(f"\nTop-5 next-token predictions at final position:")
    last = logits[0, -1]
    top = np.argsort(-last)[:5]
    for t in top:
        print(f"  {t:6d}  {last[t]:+9.3f}  {tok.decode([int(t)])!r}")


if __name__ == "__main__":
    main()
