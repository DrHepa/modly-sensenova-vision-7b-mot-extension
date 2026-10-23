# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file is an extension-side adaptation for the vendored
# OpenSenseNova/SenseNova-Vision runtime.  It keeps FlashAttention optional and
# provides a PyTorch SDPA implementation for packed variable-length attention.

"""Optional FlashAttention dispatch with a packed-sequence SDPA fallback.

The Bagel Qwen and SigLIP paths operate on a single packed token tensor and a
pair of cumulative sequence-length tensors.  ``torch.nn.functional``'s SDPA
API operates on one batch at a time, so the fallback unpacks each sequence,
performs attention, and packs the outputs again.  This is intentionally a
correctness fallback rather than a performance replacement for FlashAttention.
"""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import torch
from torch.nn.functional import scaled_dot_product_attention


class AttentionBackendError(RuntimeError):
    """Raised when a packed attention request cannot be represented safely."""


_FLASH_ATTN_FUNCTION: Any | None = None
_FLASH_ATTN_ATTEMPTED = False
_FLASH_ATTN_ERROR: BaseException | None = None


def _load_flash_attn() -> Any | None:
    """Load the optional FlashAttention entry point exactly once."""

    global _FLASH_ATTN_ATTEMPTED, _FLASH_ATTN_FUNCTION, _FLASH_ATTN_ERROR
    if _FLASH_ATTN_ATTEMPTED:
        return _FLASH_ATTN_FUNCTION

    _FLASH_ATTN_ATTEMPTED = True
    try:
        module = importlib.import_module("flash_attn")
        function = getattr(module, "flash_attn_varlen_func")
        if not callable(function):
            raise TypeError("flash_attn_varlen_func is not callable")
    except Exception as exc:  # native imports can raise OSError as well as ImportError
        _FLASH_ATTN_ERROR = exc
        _FLASH_ATTN_FUNCTION = None
    else:
        _FLASH_ATTN_FUNCTION = function
    return _FLASH_ATTN_FUNCTION


def reset_attention_backend_cache() -> None:
    """Reset optional-import state for tests and interpreter reconfiguration."""

    global _FLASH_ATTN_ATTEMPTED, _FLASH_ATTN_FUNCTION, _FLASH_ATTN_ERROR
    _FLASH_ATTN_ATTEMPTED = False
    _FLASH_ATTN_FUNCTION = None
    _FLASH_ATTN_ERROR = None


def get_attention_backend() -> str:
    """Return the backend available to the current interpreter.

    The result is deliberately based on an importable callable rather than a
    package name alone: a partially installed native module must not block the
    SDPA path.
    """

    return "flash-attn" if _load_flash_attn() is not None else "sdpa"


def _lengths_from_cumulative(
    cumulative: torch.Tensor,
    *,
    total: int,
    name: str,
) -> list[int]:
    if not isinstance(cumulative, torch.Tensor) or cumulative.ndim != 1:
        raise AttentionBackendError(f"{name} must be a one-dimensional tensor")
    if cumulative.numel() < 2:
        raise AttentionBackendError(f"{name} must contain at least one sequence")
    try:
        values = [
            int(value)
            for value in cumulative.detach().to(device="cpu", dtype=torch.int64).tolist()
        ]
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AttentionBackendError(f"{name} must contain integer cumulative lengths") from exc
    if values[0] != 0 or values[-1] != total:
        raise AttentionBackendError(
            f"{name} must start at 0 and end at the packed length {total}; got {values!r}"
        )
    if any(left > right for left, right in zip(values, values[1:])):
        raise AttentionBackendError(f"{name} must be monotonically non-decreasing")
    return [right - left for left, right in zip(values, values[1:])]


def _validate_packed_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[list[int], list[int]]:
    for name, tensor in (("q", query), ("k", key), ("v", value)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise AttentionBackendError(f"{name} must have shape [packed_length, heads, head_dim]")
        if tensor.numel() == 0:
            raise AttentionBackendError(f"{name} must not be empty")
    if query.device != key.device or query.device != value.device:
        raise AttentionBackendError("q, k, and v must be on the same device")
    if key.shape != value.shape:
        raise AttentionBackendError(
            f"k and v must have the same shape; got {tuple(key.shape)} and {tuple(value.shape)}"
        )
    if query.shape[-1] != key.shape[-1]:
        raise AttentionBackendError("q, k, and v must use the same head dimension")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise AttentionBackendError("q, k, and v must use the same dtype")
    q_lengths = _lengths_from_cumulative(
        cu_seqlens_q, total=query.shape[0], name="cu_seqlens_q"
    )
    k_lengths = _lengths_from_cumulative(
        cu_seqlens_k, total=key.shape[0], name="cu_seqlens_k"
    )
    if len(q_lengths) != len(k_lengths):
        raise AttentionBackendError(
            "cu_seqlens_q and cu_seqlens_k must describe the same number of sequences"
        )
    if any(length <= 0 for length in q_lengths + k_lengths):
        raise AttentionBackendError("packed attention does not support empty sequences")
    if query.shape[1] == 0 or key.shape[1] == 0:
        raise AttentionBackendError("packed attention requires at least one head")
    if query.shape[1] % key.shape[1] != 0:
        raise AttentionBackendError(
            "query head count must be divisible by key/value head count for GQA"
        )
    return q_lengths, k_lengths


def _causal_mask(
    query_length: int,
    key_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build FlashAttention's bottom-right causal mask for one sequence."""

    query_positions = torch.arange(query_length, device=device).unsqueeze(1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    # For q_len != k_len this aligns the diagonal at the bottom-right, which
    # is the semantics used by modern FlashAttention for cached decoding.
    return key_positions <= query_positions + (key_length - query_length)


def _normalise_mask(mask: torch.Tensor, *, query: torch.Tensor) -> torch.Tensor:
    """Convert a per-sequence mask to SDPA's [batch, heads, q, k] shape."""

    if not isinstance(mask, torch.Tensor):
        raise AttentionBackendError("attention masks must be tensors")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    elif mask.ndim != 4:
        raise AttentionBackendError(
            "attention masks must have shape [q, k], [heads, q, k], or [batch, heads, q, k]"
        )
    if mask.shape[0] not in (1, query.shape[0]):
        raise AttentionBackendError("attention mask batch dimension is not broadcastable")
    if mask.shape[1] not in (1, query.shape[1]):
        raise AttentionBackendError("attention mask head dimension is not broadcastable")
    if mask.device != query.device:
        mask = mask.to(device=query.device)
    if mask.dtype != torch.bool and not torch.is_floating_point(mask):
        raise AttentionBackendError("attention masks must be boolean or floating-point tensors")
    return mask


def _sample_mask(
    attention_mask: torch.Tensor | Sequence[torch.Tensor] | None,
    *,
    index: int,
    query_start: int,
    query_length: int,
    key_start: int,
    key_length: int,
    total_query: int,
    total_key: int,
    sequence_count: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if isinstance(attention_mask, (list, tuple)):
        if index >= len(attention_mask):
            raise AttentionBackendError("attention mask list is shorter than the packed batch")
        mask = attention_mask[index]
    elif isinstance(attention_mask, torch.Tensor):
        mask = attention_mask
        if mask.ndim >= 2 and tuple(mask.shape[-2:]) == (total_query, total_key):
            mask = mask[..., query_start : query_start + query_length, key_start : key_start + key_length]
            if mask.ndim >= 3 and mask.shape[0] == sequence_count:
                mask = mask[index]
        elif mask.ndim >= 3 and mask.shape[0] == sequence_count:
            mask = mask[index]
        elif mask.ndim >= 3 and mask.shape[0] > 1 and mask.shape[0] == total_query:
            raise AttentionBackendError(
                "packed attention masks must use a leading sequence dimension, not query length"
            )
    else:
        raise AttentionBackendError("attention_mask must be a tensor, sequence of tensors, or None")

    if mask is None:
        return None
    if tuple(mask.shape[-2:]) != (query_length, key_length):
        raise AttentionBackendError(
            "attention mask does not match the current packed query/key sequence lengths"
        )
    return mask


def _sdpa_one_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    attention_mask: torch.Tensor | None,
    dropout_p: float,
    scale: float | None,
) -> torch.Tensor:
    if query.shape[1] != key.shape[1]:
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    q_len, k_len = query.shape[0], key.shape[0]
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    mask = _normalise_mask(attention_mask, query=q) if attention_mask is not None else None
    if is_causal:
        causal = _causal_mask(q_len, k_len, device=query.device)
        causal = causal.unsqueeze(0).unsqueeze(0)
        if mask is None:
            mask = causal
        elif mask.dtype == torch.bool:
            mask = mask & causal
        else:
            # Floating masks are additive biases.  Apply the causal exclusion
            # without discarding caller-provided logits bias values.
            causal_bias = torch.zeros_like(causal, dtype=q.dtype).masked_fill(
                ~causal, torch.finfo(q.dtype).min
            )
            mask = mask.to(dtype=q.dtype) + causal_bias

    kwargs: dict[str, Any] = {
        "attn_mask": mask,
        "dropout_p": dropout_p,
        "is_causal": False,
    }
    if scale is not None:
        kwargs["scale"] = scale
    try:
        output = scaled_dot_product_attention(q, k, v, **kwargs)
    except (RuntimeError, TypeError) as exc:
        raise AttentionBackendError(
            "PyTorch SDPA could not represent the packed attention request; "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, causal={is_causal}: {exc}"
        ) from exc
    return output.squeeze(0).transpose(0, 1).contiguous()


def sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    causal: bool = False,
    attention_mask: torch.Tensor | Sequence[torch.Tensor] | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Evaluate packed variable-length attention with PyTorch SDPA.

    ``attention_mask`` can be a list/tuple containing one [q, k] (or
    broadcastable) mask per sequence, or one global mask whose final two
    dimensions cover the full packed q/k tensors.  Arbitrary masks are not
    representable by FlashAttention's varlen API and therefore deliberately
    select this fallback.
    """

    q_lengths, k_lengths = _validate_packed_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if not isinstance(causal, bool):
        raise AttentionBackendError("causal must be a boolean")
    if dropout_p < 0 or dropout_p >= 1:
        raise AttentionBackendError("dropout_p must be in the interval [0, 1)")
    if max_seqlen_q is not None and int(max_seqlen_q) < max(q_lengths):
        raise AttentionBackendError("max_seqlen_q is smaller than a packed query sequence")
    if max_seqlen_k is not None and int(max_seqlen_k) < max(k_lengths):
        raise AttentionBackendError("max_seqlen_k is smaller than a packed key sequence")

    outputs: list[torch.Tensor] = []
    query_start = key_start = 0
    for index, (query_length, key_length) in enumerate(zip(q_lengths, k_lengths)):
        query_end = query_start + query_length
        key_end = key_start + key_length
        mask = _sample_mask(
            attention_mask,
            index=index,
            query_start=query_start,
            query_length=query_length,
            key_start=key_start,
            key_length=key_length,
            total_query=q.shape[0],
            total_key=k.shape[0],
            sequence_count=len(q_lengths),
        )
        outputs.append(
            _sdpa_one_sequence(
                q[query_start:query_end],
                k[key_start:key_end],
                v[key_start:key_end],
                is_causal=causal,
                attention_mask=mask,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )
        )
        query_start = query_end
        key_start = key_end
    return torch.cat(outputs, dim=0)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    attention_mask: torch.Tensor | Sequence[torch.Tensor] | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch packed attention to verified FlashAttention or SDPA.

    ``backend`` is primarily useful for deterministic setup/tests.  ``None``
    prefers FlashAttention when its varlen callable imports successfully and
    otherwise selects SDPA.  Explicit ``flash-attn`` raises a clear error when
    the operation needs an arbitrary mask, because silently changing masking
    semantics would be worse than using the slower fallback.
    """

    requested = backend.casefold() if isinstance(backend, str) else None
    if requested not in (None, "flash-attn", "flash_attn", "sdpa"):
        raise AttentionBackendError(f"Unsupported attention backend: {backend}")

    flash = _load_flash_attn() if requested != "sdpa" else None
    use_flash = flash is not None and requested != "sdpa"
    if requested in {"flash-attn", "flash_attn"} and flash is None:
        detail = f": {_FLASH_ATTN_ERROR}" if _FLASH_ATTN_ERROR is not None else ""
        raise AttentionBackendError(f"FlashAttention varlen backend is unavailable{detail}")
    if requested in {"flash-attn", "flash_attn"} and attention_mask is not None:
        raise AttentionBackendError(
            "Explicit FlashAttention dispatch cannot represent arbitrary attention masks; "
            "request the SDPA backend instead"
        )
    if use_flash and attention_mask is None:
        try:
            return flash(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                cu_seqlens_k=cu_seqlens_k.to(torch.int32),
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=int(max_seqlen_k),
                causal=causal,
                **kwargs,
            )
        except TypeError:
            # Some audited FlashAttention releases use positional q/k/v.  A
            # type mismatch here is not evidence that the native operation is
            # broken; retry with the public positional form.
            try:
                return flash(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                    cu_seqlens_k=cu_seqlens_k.to(torch.int32),
                    max_seqlen_q=int(max_seqlen_q),
                    max_seqlen_k=int(max_seqlen_k),
                    causal=causal,
                    **kwargs,
                )
            except Exception as exc:
                if requested in {"flash-attn", "flash_attn"}:
                    raise AttentionBackendError(
                        f"FlashAttention varlen call failed: {exc}"
                    ) from exc
        except Exception as exc:
            if requested in {"flash-attn", "flash_attn"}:
                raise AttentionBackendError(f"FlashAttention varlen call failed: {exc}") from exc
            # Automatic dispatch may fall back when an optional native module
            # imports but cannot execute on this particular device/dtype.

    dropout_p = float(kwargs.pop("dropout_p", 0.0))
    softmax_scale = kwargs.pop("softmax_scale", None)
    unsupported = sorted(kwargs)
    if unsupported:
        raise AttentionBackendError(
            "SDPA fallback cannot represent FlashAttention options: "
            + ", ".join(unsupported)
        )
    return sdpa_varlen_attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        attention_mask=attention_mask,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
    )


__all__ = [
    "AttentionBackendError",
    "get_attention_backend",
    "reset_attention_backend_cache",
    "sdpa_varlen_attention",
    "varlen_attention",
]
