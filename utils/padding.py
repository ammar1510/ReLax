"""Bucketing and padding utilities for efficient inference."""

import bisect
import jax
import jax.numpy as jnp


# Power-of-2 based buckets for efficient XLA compilation
DEFAULT_PREFILL_BUCKETS = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]


def take_nearest_bucket(buckets: list[int], length: int) -> int:
    """Find the nearest bucket size that is >= the given length.

    Uses binary search for efficient lookup.

    Args:
        buckets: List of bucket sizes (must be sorted ascending)
        length: Target sequence length

    Returns:
        Nearest bucket size >= length, or largest bucket if length exceeds all

    Examples:
        >>> take_nearest_bucket([16, 32, 64, 128], 10)
        16
        >>> take_nearest_bucket([16, 32, 64, 128], 32)
        32
        >>> take_nearest_bucket([16, 32, 64, 128], 100)
        128
        >>> take_nearest_bucket([16, 32, 64, 128], 200)
        128  # Largest bucket
    """
    pos = bisect.bisect_left(buckets, length)
    if pos == len(buckets):
        # Length exceeds all buckets, return largest
        return buckets[-1]
    return buckets[pos]


def pad_to_bucket(
    tokens: jax.Array,
    bucket_size: int,
    pad_id: int = 0,
) -> jax.Array:
    """Pad token array to the specified bucket size.

    If tokens are longer than bucket_size, truncates to bucket_size.

    Args:
        tokens: Token array of shape [bsz, seqlen]
        bucket_size: Target size to pad/truncate to
        pad_id: Token ID to use for padding (default: 0)

    Returns:
        Padded token array of shape [bsz, bucket_size]

    Examples:
        >>> tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> pad_to_bucket(tokens, 5, pad_id=0)
        Array([[1, 2, 3, 0, 0],
               [4, 5, 6, 0, 0]], dtype=int32)
    """
    bsz, seqlen = tokens.shape

    if seqlen >= bucket_size:
        # Truncate if too long (take last bucket_size tokens)
        return tokens[:, -bucket_size:]

    # Pad to bucket size
    padding = bucket_size - seqlen
    return jnp.pad(tokens, ((0, 0), (0, padding)), constant_values=pad_id)


def get_true_lengths(tokens: jax.Array, pad_id: int = 0) -> jax.Array:
    """Calculate true (non-padded) lengths for each sequence in batch.

    Args:
        tokens: Token array of shape [bsz, seqlen]
        pad_id: Token ID used for padding

    Returns:
        Array of shape [bsz] with true length of each sequence

    Examples:
        >>> tokens = jnp.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        >>> get_true_lengths(tokens, pad_id=0)
        Array([3, 2], dtype=int32)
    """
    # Count non-padding tokens in each sequence
    is_not_pad = tokens != pad_id
    lengths = jnp.sum(is_not_pad, axis=1, dtype=jnp.int32)
    return lengths


def pad_prompt(
    tokens: jax.Array,
    buckets: list[int] = None,
    pad_id: int = 0,
) -> tuple[jax.Array, int]:
    """Pad a single prompt to the nearest bucket size.

    Convenience function for padding individual prompts (without batch dimension).
    Adds batch dimension, finds appropriate bucket, and pads.

    Args:
        tokens: Token array of shape [seqlen] (single sequence, no batch dim)
        buckets: List of bucket sizes (default: DEFAULT_PREFILL_BUCKETS)
        pad_id: Token ID to use for padding (default: 0)

    Returns:
        Tuple of:
            - Padded tokens of shape [1, bucket_size]
            - True length (original sequence length)

    Examples:
        >>> prompt = jnp.array([1, 2, 3, 4, 5])
        >>> padded, true_len = pad_prompt(prompt)
        >>> padded.shape
        (1, 16)  # Padded to nearest bucket (16)
        >>> true_len
        5
    """
    if buckets is None:
        buckets = DEFAULT_PREFILL_BUCKETS

    # Get true length
    true_length = len(tokens)

    # Find appropriate bucket
    bucket_size = take_nearest_bucket(buckets, true_length)

    # Add batch dimension
    tokens_with_batch = tokens[None, :]  # [1, seqlen]

    # Pad to bucket
    padded_tokens = pad_to_bucket(tokens_with_batch, bucket_size, pad_id)

    return padded_tokens, true_length
