"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

Implements MSE-optimal (Algorithm 1) and inner-product-optimal (Algorithm 2)
variants from Zandieh et al., 2025 (arXiv:2504.19874).

Primary use case: KV cache compression for LLM inference.

Coordinate distribution insight: after multiplying a unit-sphere vector by a
random rotation matrix Π, each coordinate follows a Beta distribution that
converges to N(0, 1/d) in high dimensions (Lemma 1 of the paper). This lets us
apply an optimal *scalar* quantizer per coordinate independently.

Memory note: Centroid indices are stored as uint8 regardless of bit-width b,
giving ~2x compression over bfloat16.  Bit-packing (e.g. 4 indices per byte for
b=2) would achieve the full b-bit theoretical savings and is a future
optimization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import jit


# ---------------------------------------------------------------------------
# Pre-computed Max-Lloyd / Lloyd-Max centroids for N(0, 1).
#
# Scale by 1/sqrt(d) at runtime to get centroids for N(0, 1/d), which is
# the limiting coordinate distribution of a rotated unit-sphere vector of
# dimension d (Lemma 1, TurboQuant paper).
#
# Values for b=1,2,3 are from standard MMSE scalar quantizer tables for the
# Gaussian distribution.  The b=1 value sqrt(2/pi) ≈ 0.7979 is exact.
# For b=2 the paper cites ±0.453/sqrt(d) and ±1.51/sqrt(d) (before scaling).
# ---------------------------------------------------------------------------
_GAUSSIAN_CENTROIDS: dict[int, np.ndarray] = {
    1: np.array([-0.7979, 0.7979], dtype=np.float32),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32),
    3: np.array([
        -2.1520, -1.3439, -0.7560, -0.2451,
         0.2451,  0.7560,  1.3439,  2.1520,
    ], dtype=np.float32),
    4: np.array([
        -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9424,  1.2562,  1.6180,  2.0690,  2.7326,
    ], dtype=np.float32),
}


def compute_codebook(bit_width: int, dim: int) -> np.ndarray:
    """Return Max-Lloyd centroids for N(0, 1/d) — the TurboQuant scalar codebook.

    Args:
        bit_width: Bits per coordinate b in {1, 2, 3, 4}.
        dim: Vector dimension d (used to scale centroids from N(0,1) to N(0,1/d)).

    Returns:
        float32 array of shape [2^bit_width], sorted ascending.
    """
    if bit_width not in _GAUSSIAN_CENTROIDS:
        raise ValueError(
            f"bit_width must be in {sorted(_GAUSSIAN_CENTROIDS)}, got {bit_width}"
        )
    return (_GAUSSIAN_CENTROIDS[bit_width] / np.sqrt(dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Parameter structs (Flax pytree-compatible frozen dataclasses)
# ---------------------------------------------------------------------------

@struct.dataclass
class TurboQuantMSEParams:
    """Parameters for MSE-optimal TurboQuant (Algorithm 1).

    Attributes:
        rotation: Random orthogonal matrix Π ∈ R^{d×d}, float32, shape [dim, dim].
        codebook: Sorted optimal centroid values, float32, shape [2^bit_width].
    """
    rotation: jax.Array  # [dim, dim]
    codebook: jax.Array  # [2^bit_width]


@struct.dataclass
class TurboQuantProdParams:
    """Parameters for inner-product-optimal TurboQuant (Algorithm 2).

    Two-stage scheme: (b-1)-bit MSE quantization followed by 1-bit QJL on the
    residual, giving unbiased inner product estimates: E[<y, x̃>] = <y, x>.

    Attributes:
        mse: MSE-optimal params using (bit_width - 1) bits.
        projection: i.i.d. N(0,1) projection matrix S ∈ R^{d×d}, shape [dim, dim].
    """
    mse: TurboQuantMSEParams
    projection: jax.Array  # [dim, dim]


# ---------------------------------------------------------------------------
# Factory functions (call once at startup, outside JIT)
# ---------------------------------------------------------------------------

def create_mse_params(key: jax.Array, dim: int, bit_width: int) -> TurboQuantMSEParams:
    """Create TurboQuant MSE parameters with a fresh random rotation matrix.

    Args:
        key: JAX PRNG key.
        dim: Vector dimension (e.g. head_dim for KV cache compression).
        bit_width: Bits per coordinate b in {1, 2, 3, 4}.

    Returns:
        TurboQuantMSEParams with orthogonal rotation and pre-computed codebook.
    """
    raw = jax.random.normal(key, (dim, dim), dtype=jnp.float32)
    rotation, _ = jnp.linalg.qr(raw)
    codebook = jnp.array(compute_codebook(bit_width, dim))
    return TurboQuantMSEParams(rotation=rotation, codebook=codebook)


def create_prod_params(key: jax.Array, dim: int, bit_width: int) -> TurboQuantProdParams:
    """Create inner-product-optimal TurboQuant parameters (Algorithm 2).

    Splits the bit budget as (bit_width - 1) bits for MSE + 1 bit for QJL.

    Args:
        key: JAX PRNG key.
        dim: Vector dimension.
        bit_width: Total bits per coordinate (must be >= 2).

    Returns:
        TurboQuantProdParams.
    """
    if bit_width < 2:
        raise ValueError(
            f"bit_width must be >= 2 for inner-product TurboQuant (needs >= 1 MSE bit + 1 QJL bit), got {bit_width}"
        )
    k1, k2 = jax.random.split(key)
    mse = create_mse_params(k1, dim, bit_width - 1)
    projection = jax.random.normal(k2, (dim, dim), dtype=jnp.float32)
    return TurboQuantProdParams(mse=mse, projection=projection)


# ---------------------------------------------------------------------------
# MSE-optimal quantization (Algorithm 1)
# ---------------------------------------------------------------------------

@jit
def quantize_mse(
    params: TurboQuantMSEParams,
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Quantize vectors using MSE-optimal TurboQuant (QUANT_mse, Algorithm 1).

    Steps:
      1. Normalise x to the unit sphere; record L2 norm as scale.
      2. Rotate: y = Π · x_hat  (using row-vector convention: x_hat @ Π^T).
      3. Assign each coordinate to its nearest codebook centroid.

    Supports arbitrary leading batch dimensions via broadcasting.

    Args:
        params: TurboQuantMSEParams (rotation matrix + codebook).
        x: Input vectors, shape [..., dim].  Any floating dtype accepted.

    Returns:
        idx:   Centroid indices (uint8),    shape [..., dim].
        scale: L2 norms of x (bfloat16),   shape [...].
    """
    x = x.astype(jnp.float32)

    # Normalise to unit sphere; handle zero vectors gracefully
    scale = jnp.linalg.norm(x, axis=-1, keepdims=True)            # [..., 1]
    x_hat = x / jnp.where(scale > 0.0, scale, jnp.ones_like(scale))  # [..., dim]

    # Random rotation: y = Π · x_hat  (row-vector: x_hat @ Π^T)
    y = x_hat @ params.rotation.T                                  # [..., dim]

    # Nearest centroid: [..., dim, 1] vs [2^b] → [..., dim, 2^b]
    dists = jnp.abs(y[..., None] - params.codebook)
    idx = jnp.argmin(dists, axis=-1).astype(jnp.uint8)            # [..., dim]

    return idx, scale[..., 0].astype(jnp.bfloat16)                # drop keepdim


@jit
def dequantize_mse(
    params: TurboQuantMSEParams,
    idx: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Reconstruct vectors from MSE-quantized representation (DEQUANT_mse).

    Args:
        params: TurboQuantMSEParams.
        idx:   Centroid indices (uint8), shape [..., dim].
        scale: L2 norms,                shape [...].
               Pass jnp.ones(...) to reconstruct unit-sphere vectors.

    Returns:
        Reconstructed float32 vectors, shape [..., dim].
    """
    y_hat = params.codebook[idx.astype(jnp.int32)]      # [..., dim]
    # Inverse rotation: Π^T · ỹ  (row-vector: ỹ @ Π)
    x_hat = y_hat @ params.rotation                      # [..., dim]
    return x_hat * scale.astype(jnp.float32)[..., None]


# ---------------------------------------------------------------------------
# Inner-product-optimal quantization (Algorithm 2)
# ---------------------------------------------------------------------------

@jit
def quantize_prod(
    params: TurboQuantProdParams,
    x: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Quantize vectors for unbiased inner-product estimation (QUANT_prod, Algorithm 2).

    Two-stage:
      Stage 1 — (b-1)-bit MSE quantization of x.
      Stage 2 — 1-bit QJL (Quantized Johnson-Lindenstrauss) on the MSE residual.

    The reconstructed vector x̃ satisfies E[<y, x̃>] = <y, x> for any y.

    Args:
        params: TurboQuantProdParams.
        x: Input vectors, shape [..., dim].

    Returns:
        idx:           MSE centroid indices (uint8),  shape [..., dim].
        qjl:           QJL sign bits (int8, ±1),      shape [..., dim].
        scale:         L2 norms of x (bfloat16),      shape [...].
        residual_norm: ||r||_2 (bfloat16),             shape [...].
    """
    # Stage 1: MSE quantization
    idx, scale = quantize_mse(params.mse, x)

    # Reconstruct MSE approximation on the unit sphere (scale=1)
    x_mse_hat = dequantize_mse(
        params.mse, idx, jnp.ones(scale.shape, dtype=jnp.float32)
    )

    # Normalise input to unit sphere for residual computation
    x_f = x.astype(jnp.float32)
    scale_f = scale.astype(jnp.float32)
    x_hat = x_f / jnp.where(scale_f[..., None] > 0.0, scale_f[..., None], 1.0)

    # Residual on the unit sphere: r = x_hat - x̃_mse
    r = x_hat - x_mse_hat                                      # [..., dim]
    residual_norm = jnp.linalg.norm(r, axis=-1)                 # [...]

    # Stage 2: QJL — sign(S · r)  (row-vector: r @ S^T)
    r_proj = r @ params.projection.T                            # [..., dim]
    qjl = jnp.sign(r_proj).astype(jnp.int8)
    # Avoid zero signs (rare but possible at exact zero)
    qjl = jnp.where(qjl == 0, jnp.ones_like(qjl), qjl)

    return idx, qjl, scale, residual_norm.astype(jnp.bfloat16)


@jit
def dequantize_prod(
    params: TurboQuantProdParams,
    idx: jax.Array,
    qjl: jax.Array,
    scale: jax.Array,
    residual_norm: jax.Array,
) -> jax.Array:
    """Reconstruct vectors from inner-product-quantized representation (DEQUANT_prod).

    Provides unbiased inner product estimates: E[<y, x̃>] = <y, x>.

    The reconstruction is:
        x̃ = (x̃_mse + x̃_qjl) * scale
    where:
        x̃_mse = Π^T · c_{idx}
        x̃_qjl = (√(π/2) / d) · γ · S^T · qjl   (γ = residual_norm)

    Args:
        params:        TurboQuantProdParams.
        idx:           MSE centroid indices (uint8), shape [..., dim].
        qjl:           QJL sign bits (int8 ±1),      shape [..., dim].
        scale:         L2 norms (bfloat16),           shape [...].
        residual_norm: ||residual||_2 (bfloat16),     shape [...].

    Returns:
        Reconstructed float32 vectors, shape [..., dim].
    """
    dim = params.projection.shape[0]

    # MSE component (unit sphere, scale applied at end)
    x_mse = dequantize_mse(
        params.mse, idx, jnp.ones(scale.shape, dtype=jnp.float32)
    )

    # QJL component: (√(π/2) / d) · γ · S^T · qjl
    # Row-vector form: S^T · qjl_col  ≡  qjl_row @ S
    qjl_f = qjl.astype(jnp.float32)
    x_qjl = qjl_f @ params.projection                          # [..., dim]  = S^T · qjl
    gamma = residual_norm.astype(jnp.float32)
    x_qjl = x_qjl * (jnp.sqrt(jnp.pi / 2.0) / dim) * gamma[..., None]

    return (x_mse + x_qjl) * scale.astype(jnp.float32)[..., None]
