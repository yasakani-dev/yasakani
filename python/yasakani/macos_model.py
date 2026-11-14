# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Sequential Qwen3.5/Qwen3.8 inference over the published YSKN weights.

Compressed matrix operations are injected (Metal in the macOS runtime); NumPy
is the slow reference implementation. Activations, recurrent state and KV use
float32. This is not a reproduction of CUDA's integer activation arithmetic,
KV5 cache, speculative decoder or its published performance/quality results.
"""
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

from .yskn_format import Container, FormatError, NumpyOps


class _PrefillCancelled(Exception):
    """Internal control flow: partially advanced layers cannot be resumed."""


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    # exp(-abs(x)) is bounded even for large negative gates.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1, e) / (1 + e)


def silu(x):
    return x * sigmoid(x)


def rms_norm(x, weight, eps, offset=1):
    """Qwen RMSNorm uses 1 + weight; the GDN gated norm uses weight itself."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (weight + offset)


def rotary(x, position, rotary_dim, theta):
    """Partial split-half RoPE, leaving dimensions beyond rotary_dim unchanged."""
    out = x.copy()
    half = rotary_dim // 2
    freq = np.power(np.float32(theta), -np.arange(half, dtype=np.float32) * (2 / rotary_dim))
    angles = np.float32(position) * freq
    cosine, sine = np.cos(angles), np.sin(angles)
    left, right = x[..., :half], x[..., half:rotary_dim]
    out[..., :half] = left * cosine - right * sine
    out[..., half:rotary_dim] = right * cosine + left * sine
    return out


def delta_step(state, q, k, v, beta, log_decay):
    """One gated delta-rule step; state is [value heads, key dim, value dim].

    q/k have already been L2 normalised; q also includes 1/sqrt(key dim).
    Updates state in place and returns the output for this token.
    """
    repeat = state.shape[0] // k.shape[0]
    q, k = np.repeat(q, repeat, axis=0), np.repeat(k, repeat, axis=0)
    state *= np.exp(log_decay)[:, None, None]
    error = (v - np.einsum("hk,hkv->hv", k, state)) * beta[:, None]
    state += k[:, :, None] * error[:, None, :]
    return np.einsum("hk,hkv->hv", q, state)


def grouped_attention(q, keys, values):
    """Causal one-token GQA; keys/values contain only the committed prefix."""
    out = np.empty_like(q)
    group = q.shape[0] // keys.shape[1]
    scale = np.float32(1 / math.sqrt(q.shape[1]))
    for head in range(q.shape[0]):
        kv_head = head // group
        scores = keys[:, kv_head] @ q[head] * scale
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        out[head] = probabilities @ values[:, kv_head]
    return out


class Model:
    """One stateful, sequential model. Callers must serialize forward/reset.

    ``ops`` implements ``linear(name, x)``, ``embedding(token)`` and
    ``vector(name)`` in the original model's coordinates. Optional
    ``linear_batch(name, X)`` takes/returns row-major float32 matrices, while
    ``linear_multi(names, x)`` returns one float32 vector per named projection
    of a shared input. Linear outputs are unrounded float32. The default NumPy
    implementation is for checking correctness, not interactive use with a 27B model.
    """
    def __init__(self, path, max_ctx, ops=None):
        self.container = Container(path)
        self.config = self.container.model
        self.max_ctx = int(max_ctx)
        if self.max_ctx < 1 or self.max_ctx > 262144:
            raise ValueError("max_ctx must be between 1 and 262144")
        self._validate_config()
        self._validate_shapes()
        self.ops = ops if ops is not None else NumpyOps(self.container)
        self._vectors = {}
        for name, size in self._vector_sizes.items():
            value = np.asarray(self.ops.vector(name), dtype=np.float32)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise FormatError("%s: invalid vector values/shape" % name)
            self._vectors[name] = value
        self.reset()

    def _validate_config(self):
        c = self.config
        if c.get("arch") != "qwen3_5":
            raise FormatError("the macOS backend supports qwen3_5 YSKN models only")
        if c.get("tie_embeddings", False) or c.get("attn_output_gate", True) is not True or c.get("norm_offset", 1) != 1:
            raise FormatError("unsupported Qwen embedding, attention gate or norm configuration")
        for name in ("layers", "hidden", "ffn", "heads", "kv_heads", "head_dim", "vocab"):
            value = c.get(name)
            if not isinstance(value, int) or value < 1:
                raise FormatError("invalid model dimension: " + name)
            setattr(self, name, value)
        if (self.layers > 256 or self.hidden > 32768 or self.ffn > 262144 or self.heads > 256
                or self.head_dim > 1024 or self.vocab > 2000000 or self.heads % self.kv_heads):
            raise FormatError("unsupported model dimensions")
        self.layer_types = c.get("layer_types")
        if (not isinstance(self.layer_types, list) or len(self.layer_types) != self.layers
                or any(kind not in ("full_attention", "linear_attention") for kind in self.layer_types)):
            raise FormatError("invalid or unsupported layer_types")
        self.eps, self.rope_theta = float(c["eps"]), float(c["rope_theta"])
        self.rotary_dim = c.get("rotary_dim", 64)
        if (not np.isfinite(self.eps) or self.eps <= 0 or not np.isfinite(self.rope_theta) or self.rope_theta <= 0
                or not isinstance(self.rotary_dim, int) or self.rotary_dim < 2
                or self.rotary_dim % 2 or self.rotary_dim > self.head_dim):
            raise FormatError("invalid normalisation or rotary configuration")
        for attr, key, default in (("key_heads", "linear_num_key_heads", 16),
                                   ("value_heads", "linear_num_value_heads", 48),
                                   ("key_dim", "linear_key_head_dim", 128),
                                   ("value_dim", "linear_value_head_dim", 128),
                                   ("conv_size", "linear_conv_kernel_dim", 4)):
            value = c.get(key, default)
            if not isinstance(value, int) or not 1 <= value <= 1024:
                raise FormatError("invalid linear-attention dimension: " + key)
            setattr(self, attr, value)
        if self.value_heads % self.key_heads or self.conv_size != 4:
            raise FormatError("unsupported linear-attention grouping or convolution")
        self.conv_channels = 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    def _validate_shapes(self):
        self._vector_sizes = {}
        tensors, vectors = self.container.tensors, self.container.vectors

        def tensor(name, rows, cols):
            desc = tensors.get(name, {})
            if (desc.get("rows"), desc.get("cols")) != (rows, cols):
                raise FormatError("%s: missing tensor or shape mismatch (expected %d x %d)" % (name, rows, cols))

        def vector(name, size):
            desc = vectors.get(name, {})
            if desc.get("n") != size or desc.get("dtype", "f16") not in ("f16", "bf16"):
                raise FormatError("%s: missing vector or shape/dtype mismatch" % name)
            self.container.array(desc["off"], "<u2", size)
            self._vector_sizes[name] = size

        if "embed_tokens" in tensors:
            tensor("embed_tokens", self.vocab, self.hidden)
            for name in ("embed_tokens", "embed_tokens.res", "embed_tokens.res2"):
                if name in tensors:
                    tensor(name, self.vocab, self.hidden)
                    desc = tensors[name]
                    rotation = desc.get("rot_out")
                    if (desc.get("role") != "embedding" or (rotation is not None
                            and (not isinstance(rotation, dict) or rotation.get("kind") != "none"))):
                        raise FormatError("unsupported embedding role or output rotation")
            if "embed_tokens.res2" in tensors and "embed_tokens.res" not in tensors:
                raise FormatError("embed_tokens.res2 requires embed_tokens.res")
        else:
            # Do not eagerly materialise the multi-gigabyte uncompressed table.
            desc = vectors.get("embed_tokens", {})
            if desc.get("n") != self.vocab * self.hidden or desc.get("dtype", "f16") not in ("f16", "bf16"):
                raise FormatError("missing or invalid embedding table")
            self.container.array(desc["off"], "<u2", desc["n"])
        tensor("lm_head", self.vocab, self.hidden)
        vector("norm", self.hidden)
        for layer, kind in enumerate(self.layer_types):
            p = "layers.%d." % layer
            vector(p + "input_layernorm", self.hidden)
            vector(p + "post_attention_layernorm", self.hidden)
            tensor(p + "mlp.gate_proj", self.ffn, self.hidden)
            tensor(p + "mlp.up_proj", self.ffn, self.hidden)
            tensor(p + "mlp.down_proj", self.hidden, self.ffn)
            if kind == "full_attention":
                tensor(p + "self_attn.q_proj", 2 * self.heads * self.head_dim, self.hidden)
                tensor(p + "self_attn.k_proj", self.kv_heads * self.head_dim, self.hidden)
                tensor(p + "self_attn.v_proj", self.kv_heads * self.head_dim, self.hidden)
                tensor(p + "self_attn.o_proj", self.hidden, self.heads * self.head_dim)
                vector(p + "self_attn.q_norm", self.head_dim)
                vector(p + "self_attn.k_norm", self.head_dim)
            else:
                tensor(p + "linear_attn.in_proj_qkv", self.conv_channels, self.hidden)
                tensor(p + "linear_attn.in_proj_z", self.value_heads * self.value_dim, self.hidden)
                tensor(p + "linear_attn.out_proj", self.hidden, self.value_heads * self.value_dim)
                for suffix in ("in_proj_a", "in_proj_b"):
                    vector(p + "linear_attn." + suffix, self.value_heads * self.hidden)
                vector(p + "linear_attn.conv1d", self.conv_channels * self.conv_size)
                vector(p + "linear_attn.A_log", self.value_heads)
                vector(p + "linear_attn.dt_bias", self.value_heads)
                vector(p + "linear_attn.norm", self.value_dim)

    def reset(self):
        self.position = 0
        self._failed = False
        self._linear_state = {}
        self._kv = {}
        for layer, kind in enumerate(self.layer_types):
            if kind == "linear_attention":
                self._linear_state[layer] = (
                    np.zeros((self.conv_size - 1, self.conv_channels), np.float32),
                    np.zeros((self.value_heads, self.key_dim, self.value_dim), np.float32))

    def _vector(self, name):
        return self._vectors[name]

    def _linear(self, name, x):
        out = np.asarray(self.ops.linear(name, np.ascontiguousarray(x, dtype=np.float32)), dtype=np.float32)
        if out.shape != (self.container.tensors[name]["rows"],) or not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite result or incorrect shape from " + name)
        return out

    def _linears(self, names, x):
        """Project one input several ways, allowing a single native submission."""
        multi = getattr(self.ops, "linear_multi", None)
        if not callable(multi):
            return [self._linear(name, x) for name in names]
        outputs = multi(names, np.ascontiguousarray(x, dtype=np.float32))
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(names):
            raise FloatingPointError("incorrect number of results from grouped projections")
        checked = []
        for name, output in zip(names, outputs):
            out = np.asarray(output, dtype=np.float32)
            if out.shape != (self.container.tensors[name]["rows"],) or not np.all(np.isfinite(out)):
                raise FloatingPointError("non-finite result or incorrect shape from " + name)
            checked.append(out)
        return checked

    def _linear_batch(self, name, x, check_cancel):
        check_cancel()
        x = np.ascontiguousarray(x, dtype=np.float32)
        batch = getattr(self.ops, "linear_batch", None)
        if callable(batch):
            out = np.asarray(batch(name, x), dtype=np.float32)
        else:
            # A reference or older ops implementation keeps the identical scalar
            # projection arithmetic, while the model still executes layer-major.
            rows = []
            for row in x:
                check_cancel()
                rows.append(self._linear(name, row))
            out = np.stack(rows)
        check_cancel()
        if out.shape != (len(x), self.container.tensors[name]["rows"]) or not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite result or incorrect batch shape from " + name)
        return out

    def _attention(self, layer, p, x):
        p += "self_attn."
        q_gate, k, v = self._linears((p + "q_proj", p + "k_proj", p + "v_proj"), x)
        out = self._attention_step(layer, p, q_gate, k, v, self.position)
        return self._linear(p + "o_proj", out)

    def _attention_step(self, layer, p, q_gate, k, v, position):
        """Commit one projected token; shared by scalar decode and chunk prefill."""
        # Q and the output gate are interleaved per attention head, not two global halves.
        q_gate = q_gate.reshape(self.heads, 2, self.head_dim)
        q, gate = q_gate[:, 0], q_gate[:, 1]
        k = k.reshape(self.kv_heads, self.head_dim)
        v = v.reshape(self.kv_heads, self.head_dim)
        q = rotary(rms_norm(q, self._vector(p + "q_norm"), self.eps), position, self.rotary_dim, self.rope_theta)
        k = rotary(rms_norm(k, self._vector(p + "k_norm"), self.eps), position, self.rotary_dim, self.rope_theta)
        if layer not in self._kv:
            capacity = min(64, self.max_ctx)
            self._kv[layer] = (np.empty((capacity, self.kv_heads, self.head_dim), np.float32),
                               np.empty((capacity, self.kv_heads, self.head_dim), np.float32))
        keys, values = self._kv[layer]
        if position == len(keys):
            capacity = min(self.max_ctx, 2 * len(keys))
            new_keys = np.empty((capacity, self.kv_heads, self.head_dim), np.float32)
            new_values = np.empty_like(new_keys)
            new_keys[:position], new_values[:position] = keys, values
            keys, values = self._kv[layer] = new_keys, new_values
        keys[position], values[position] = k, v
        out = grouped_attention(q, keys[:position + 1], values[:position + 1])
        return (out * sigmoid(gate)).reshape(-1)

    def _gdn(self, layer, p, x):
        p += "linear_attn."
        mixed, z = self._linears((p + "in_proj_qkv", p + "in_proj_z"), x)
        out = self._gdn_step(layer, p, x, mixed, z)
        return self._linear(p + "out_proj", out)

    def _gdn_step(self, layer, p, x, mixed, z):
        """Commit one convolution/delta-rule step using already projected values."""
        history, state = self._linear_state[layer]
        z = z.reshape(self.value_heads, self.value_dim)
        weights = self._vector(p + "conv1d").reshape(self.conv_channels, self.conv_size)
        convolved = np.sum(history.T * weights[:, :-1], axis=1) + mixed * weights[:, -1]
        history[:-1] = history[1:]
        history[-1] = mixed
        convolved = silu(convolved)
        split = self.key_heads * self.key_dim
        q = convolved[:split].reshape(self.key_heads, self.key_dim)
        k = convolved[split:2 * split].reshape(self.key_heads, self.key_dim)
        v = convolved[2 * split:].reshape(self.value_heads, self.value_dim)
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) / np.float32(math.sqrt(self.key_dim))
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        a = self._vector(p + "in_proj_a").reshape(self.value_heads, self.hidden) @ x
        b = self._vector(p + "in_proj_b").reshape(self.value_heads, self.hidden) @ x
        beta = sigmoid(b)
        decay = -np.exp(self._vector(p + "A_log")) * np.logaddexp(0, a + self._vector(p + "dt_bias"))
        core = delta_step(state, q, k, v, beta, decay)
        out = rms_norm(core, self._vector(p + "norm"), self.eps, offset=0) * silu(z)
        return out.reshape(-1)

    def prefill(self, tokens, *, cancel=None, logits=True):
        """Commit up to 32 prompt tokens, batching projections by layer.

        Chunks may continue an existing prefix. Attention sees only preceding
        tokens, and each layer's convolution/recurrent state advances in token
        order. Only the final token needs a vocabulary projection. A cancelled
        or failed chunk invalidates the state: ``reset()`` is required before
        another operation. Cancellation returns None; successful logits=False
        also returns None, with position advanced by the complete chunk.
        """
        if self._failed:
            raise RuntimeError("model state is invalid after a failed prefill; call reset()")
        ids = np.asarray(tokens)
        if (ids.ndim != 1 or not 1 <= len(ids) <= 32
                or not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("prefill expects an integer vector of 1 to 32 tokens")
        if np.any(ids < 0) or np.any(ids >= self.vocab):
            raise ValueError("token id is outside the model vocabulary")
        if self.position + len(ids) > self.max_ctx:
            raise ValueError("prefill exceeds the model context")
        if cancel is not None and not callable(cancel):
            raise TypeError("cancel must be callable")

        def check_cancel():
            if cancel is not None and cancel():
                raise _PrefillCancelled()

        def project(name, x):
            return self._linear_batch(name, x, check_cancel)

        start = self.position
        try:
            x = np.empty((len(ids), self.hidden), dtype=np.float32)
            for index, token in enumerate(ids):
                check_cancel()
                embedding = np.asarray(self.ops.embedding(int(token)), dtype=np.float32)
                if embedding.shape != (self.hidden,) or not np.all(np.isfinite(embedding)):
                    raise FloatingPointError("invalid embedding")
                x[index] = embedding
            for layer, kind in enumerate(self.layer_types):
                check_cancel()
                p = "layers.%d." % layer
                norm = rms_norm(x, self._vector(p + "input_layernorm"), self.eps)
                if kind == "linear_attention":
                    a = p + "linear_attn."
                    mixed = project(a + "in_proj_qkv", norm)
                    z = project(a + "in_proj_z", norm)
                    attended = np.empty((len(ids), self.value_heads * self.value_dim), np.float32)
                    for index in range(len(ids)):
                        check_cancel()
                        attended[index] = self._gdn_step(layer, a, norm[index], mixed[index], z[index])
                    x += project(a + "out_proj", attended)
                else:
                    a = p + "self_attn."
                    q = project(a + "q_proj", norm)
                    k = project(a + "k_proj", norm)
                    v = project(a + "v_proj", norm)
                    attended = np.empty((len(ids), self.heads * self.head_dim), np.float32)
                    for index in range(len(ids)):
                        check_cancel()
                        attended[index] = self._attention_step(layer, a, q[index], k[index], v[index], start + index)
                    x += project(a + "o_proj", attended)
                norm = rms_norm(x, self._vector(p + "post_attention_layernorm"), self.eps)
                gate = project(p + "mlp.gate_proj", norm)
                up = project(p + "mlp.up_proj", norm)
                x += project(p + "mlp.down_proj", silu(gate) * up)
            check_cancel()
            result = self._linear("lm_head", rms_norm(x[-1], self._vector("norm"), self.eps)) if logits else None
            check_cancel()
            self.position = start + len(ids)
            return result
        except _PrefillCancelled:
            self._failed = True
            return None
        except BaseException:
            self._failed = True
            raise

    def forward(self, token, logits=True):
        """Commit one token; omit the expensive vocabulary projection during prefill."""
        if self._failed:
            raise RuntimeError("model state is invalid after a failed forward; call reset()")
        if not isinstance(token, (int, np.integer)) or not 0 <= token < self.vocab:
            raise ValueError("token id is outside the model vocabulary")
        if self.position >= self.max_ctx:
            raise ValueError("model context is full")
        try:
            x = np.asarray(self.ops.embedding(int(token)), dtype=np.float32).copy()
            if x.shape != (self.hidden,) or not np.all(np.isfinite(x)):
                raise FloatingPointError("invalid embedding")
            for layer, kind in enumerate(self.layer_types):
                p = "layers.%d." % layer
                norm = rms_norm(x, self._vector(p + "input_layernorm"), self.eps)
                x += self._attention(layer, p, norm) if kind == "full_attention" else self._gdn(layer, p, norm)
                norm = rms_norm(x, self._vector(p + "post_attention_layernorm"), self.eps)
                gate, up = self._linears((p + "mlp.gate_proj", p + "mlp.up_proj"), norm)
                x += self._linear(p + "mlp.down_proj", silu(gate) * up)
            result = self._linear("lm_head", rms_norm(x, self._vector("norm"), self.eps)) if logits else None
            self.position += 1
            return result
        except BaseException:
            # Earlier layers may already have committed recurrent state; never reuse it silently.
            self._failed = True
            raise

    def state_hash(self):
        """Digest only committed state, excluding spare/uninitialised KV capacity."""
        digest = hashlib.sha256(struct.pack("<Q?", self.position, self._failed))
        for layer in range(self.layers):
            if layer in self._linear_state:
                for value in self._linear_state[layer]:
                    digest.update(np.asarray(value, dtype="<f4").tobytes())
            elif layer in self._kv:
                for value in self._kv[layer]:
                    digest.update(np.asarray(value[:self.position], dtype="<f4").tobytes())
        return np.frombuffer(digest.digest(), dtype="<u8").copy()
