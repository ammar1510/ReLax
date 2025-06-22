"""
nano-Llama 3.1
Simpler version you can just forward on 1 GPU, without torchrun.
Changes:
- replace ColumnParallelLinear -> Linear
- replace RowParallelLinear -> Linear
- replace VocabParallelEmbedding -> Embedding

Run example:

python llama31.py \
    --ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
    --tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model
"""

import os
import glob
import fire
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.llama.tokenizer import Tokenizer

# -----------------------------------------------------------------------------
# ModelArgs

@dataclass
class ModelArgs:
    dim: int = 3072
    n_layers: int = 28
    n_heads: int = 24
    n_kv_heads: Optional[int] = None
    vocab_size: int = 128256
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_seqlen: int = 8192
    device: str = "cuda"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

# -----------------------------------------------------------------------------
# Transformer

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: str = "cuda"):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_scaling(freqs: torch.Tensor):
    # RoPE scaling (values obtained from grid search)
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32, device: str = "cuda"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].to(dtype) / dim))
    t = torch.arange(end, device=device, dtype=dtype)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real

def apply_rotary_emb(x, freqs_cis):
    # shape gymnastics let's go
    # x is (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    # freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(3)
    # x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    return x_out2.type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv) -> Tuple[torch.Tensor, torch.Tensor]:
        seqlen = xk.size(1)
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.flash = False # use flash attention?
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1 # AK: model parallel size is 1 for 1 GPU
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False )
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # will be KVCache object managed by inference context manager
        self.cache: Optional[KVCache] = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # rotate query, keys (RoPE)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            xk, xv = self.cache.update(start_pos, xk, xv)
        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # output projection
        proj = self.wo(output)
        return proj

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # hidden dim gymnastics that Meta simplified only later
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim, device=params.device)
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps, device=params.device)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False, device=params.device)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seqlen,
            params.rope_theta,
            params.use_scaled_rope,
            dtype=torch.float64,
            device=params.device
        ).to(torch.float32)

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        # for use during inference
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index=-100):
        # for use during training
        # ignore_index can be set to e.g. self.tokenizer.pad_id in the future
        # forward the model first
        _bsz, seqlen = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 # -1 disables KV caching logic
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        # and then loss
        loss = F.cross_entropy(
            input=logits.transpose(1, 2),
            target=targets,
            reduction="mean",
            ignore_index=ignore_index,
        )
        return loss

    def configure_optimizers(self, learning_rate, weight_decay=0.0, betas=(0.9, 0.97), device_type='cuda'):
        train_params = []

        finetune_type = "all"
        if finetune_type == "rmsnorm":
            # let's only train the RMSNorm parameters to start
            for name, param in self.named_parameters():
                if "norm" in name:
                    train_params.append(param)
        elif finetune_type == "all":
            # let's train all parameters
            for param in self.parameters():
                train_params.append(param)
        elif finetune_type == "all_no_pos":
            # let's train all parameters except the positional embeddings and lm_head
            n, m = 0, 0
            for name, param in self.named_parameters():
                if name == "output.weight":
                    # do not include
                    n += 1
                    continue
                elif name == "tok_embeddings.weight":
                    # do not include and also does not require grad
                    m += 1
                    param.requires_grad = False
                else:
                    # do include
                    train_params.append(param)
            assert n == 1, "did not find output.weight"
            assert m == 1, "did not find tok_embeddings.weight"

        print("number of parameters: ", sum(p.numel() for p in self.parameters()))
        print("number of trainable parameters: ", sum(p.numel() for p in train_params))
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = True #'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(train_params, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

# -----------------------------------------------------------------------------
# Llama wrapper

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seqlen: int,
        max_batch_size: int,
        device: str,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by loading the model weights and tokenizer.
        """
        start_time = time.time()
        
        # Seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Load model parameters from params.json
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args: ModelArgs = ModelArgs(
            max_seqlen=max_seqlen,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        # Initialize the tokenizer
        tokenizer = Tokenizer(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size
        
        # Determine the compute dtype
        compute_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16

        # Create the model
        model = Transformer(model_args).to(device, dtype=compute_dtype)

        # Load the checkpoint
        checkpoints = sorted(glob.glob(str(Path(ckpt_dir) / "consolidated.*.pth")))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

        state_dict = {}
        for ckpt_path in checkpoints:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict.update(checkpoint)
            del checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        del state_dict

        print(f"Loaded model in {time.time() - start_time:.2f} seconds")
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_seqlen = model.params.max_seqlen

    def setup_caches(self, batch_size: int, dtype: torch.dtype):
        # Initialize KV Caches for each layer
        for layer in self.model.layers:
            if hasattr(layer.attention, 'cache') and layer.attention.cache is None:  # type: ignore
                layer.attention.cache = KVCache(  # type: ignore
                    batch_size=batch_size,
                    seq_length=self.max_seqlen,
                    n_kv_heads=self.model.params.n_kv_heads,
                    head_dim=self.model.params.dim // self.model.params.n_heads,
                    dtype=dtype,
                    device=self.device
                )

    def cleanup_caches(self):
        # Free up memory by deleting caches
        for layer in self.model.layers:
            if hasattr(layer.attention, 'cache'):  # type: ignore
                layer.attention.cache = None  # type: ignore
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate a text completion for a single prompt.
        """
        self.setup_caches(batch_size=1, dtype=self.model.tok_embeddings.weight.dtype)

        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        
        # The tokens tensor starts with the prompt tokens
        tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Pre-fill the KV cache with the prompt tokens
        self.model.forward_inference(tokens, 0)
        
        # Generate tokens one by one
        generated_tokens = []
        logits = None
        for cur_pos in range(len(prompt_tokens), len(prompt_tokens) + max_gen_len):
            
            if cur_pos >= self.max_seqlen:
                break # sequence length limit

            # The input to the model is the last token generated
            # The start_pos is the position of that token in the sequence
            logits = self.model.forward_inference(tokens[:, cur_pos-1:cur_pos], cur_pos-1)
            
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            next_token_val = next_token.item()

            # Stop if the End of Sequence token is generated
            if next_token_val == self.tokenizer.eos_id:
                break
            
            # Add the generated token to our lists
            generated_tokens.append(next_token_val)
            tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)

        self.cleanup_caches()
        return self.tokenizer.decode(generated_tokens), logits

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompt: str = "The capital of France is",
    max_seqlen: int = 128,
    device: str = 'cuda'
):
    """
    Entry point for running the Llama 3.1 model for text generation.
    """
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seqlen=max_seqlen,
        max_batch_size=1,
        device=device,
    )

    result, logits = llama.generate(
        prompt,
        max_gen_len=1,
    )

    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    print("-" * 20)

    if logits is not None:
        np.save("logits_torch.npy", logits.cpu().numpy())
        print("Logits saved to logits_torch.npy")

if __name__ == "__main__":
    fire.Fire(main)
