# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FP8_PB_WO dispatch in ModelOptMixedPrecisionConfig.

Hermetic: no network, no GPU, no real model load.
"""

from typing import Any
from unittest.mock import patch

import torch

from vllm.config import VllmConfig
from vllm.config.model import ModelConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import modelopt as modelopt_module
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8LinearMethod,
    ModelOptFp8PbWoLinearMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4LinearMethod,
    UnquantizedLinearMethod,
)


class _StubLinear(LinearBase):
    """Minimal LinearBase for isinstance checks without GPU or distributed setup."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


def _make_mixed_config(
    quantized_layers: dict[str, dict[str, Any]],
    kv_cache_quant_method: str | None = None,
    exclude_modules: list[str] | None = None,
) -> ModelOptMixedPrecisionConfig:
    """Build a ModelOptMixedPrecisionConfig from a synthetic layer dict."""
    original_config: dict[str, Any] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": quantized_layers,
        }
    }
    return ModelOptMixedPrecisionConfig._from_config(
        quant_method="MIXED_PRECISION",
        kv_cache_quant_method=kv_cache_quant_method,
        exclude_modules=exclude_modules or [],
        original_config=original_config,
        group_size=None,
    )


def test_mixed_pbwo_dispatch_returns_fp8_pbwo_linear_method(default_vllm_config):
    """FP8_PB_WO prefix routes to ModelOptFp8PbWoLinearMethod."""
    default_vllm_config.model_config = ModelConfig()
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    method = config.get_quant_method(
        layer=_StubLinear(), prefix="model.layers.0.self_attn.q_proj"
    )
    assert isinstance(method, ModelOptFp8PbWoLinearMethod)


def test_mixed_nvfp4_dispatch_still_works():
    """NVFP4 prefix in a mixed config routes to ModelOptNvFp4LinearMethod."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    method = config.get_quant_method(
        layer=_StubLinear(), prefix="model.layers.0.mlp.experts.0.w13"
    )
    assert isinstance(method, ModelOptNvFp4LinearMethod)


def test_mixed_no_pbwo_entries_fp8_pbwo_config_is_none():
    """fp8_pbwo_config stays None when no FP8_PB_WO entries are present."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    assert config.fp8_pbwo_config is None


def test_mixed_no_pbwo_entries_fp8_dispatch_preserved(default_vllm_config):
    """Plain FP8 entry in a mixed config routes to ModelOptFp8LinearMethod."""
    default_vllm_config.model_config = ModelConfig()
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    method = config.get_quant_method(
        layer=_StubLinear(), prefix="model.layers.0.self_attn.q_proj"
    )
    assert isinstance(method, ModelOptFp8LinearMethod)


def test_mixed_unknown_quant_algo_returns_unquantized():
    """Unknown quant_algo falls back to UnquantizedLinearMethod."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "BOGUS_UNKNOWN_ALGO"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    method = config.get_quant_method(
        layer=_StubLinear(), prefix="model.layers.0.self_attn.q_proj"
    )
    assert isinstance(method, UnquantizedLinearMethod)


def test_mixed_pbwo_has_blocked_weights_true_when_pbwo_present():
    """has_blocked_weights() returns True when FP8_PB_WO entries exist."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    assert config.has_blocked_weights() is True


def test_mixed_pbwo_has_blocked_weights_false_when_no_pbwo():
    """has_blocked_weights() returns False when no FP8_PB_WO entries exist."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    assert config.has_blocked_weights() is False


def test_mixed_pbwo_vllmconfig_gate_enables_quant_fp8():
    """VllmConfig enables +quant_fp8 custom op when quant_config carries FP8_PB_WO."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    vllm_cfg = VllmConfig(quant_config=config)
    assert "+quant_fp8" in vllm_cfg.compilation_config.custom_ops


def test_mixed_no_pbwo_vllmconfig_gate_does_not_enable_quant_fp8():
    """VllmConfig does NOT enable +quant_fp8 when quant_config carries no FP8_PB_WO."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    config = _make_mixed_config(quantized_layers)
    vllm_cfg = VllmConfig(quant_config=config)
    assert "+quant_fp8" not in vllm_cfg.compilation_config.custom_ops


def test_mixed_pbwo_log_line_contains_exact_phrase():
    """Detection log line names the FP8_PB_WO entry count."""
    quantized_layers: dict[str, dict[str, Any]] = {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
        "model.layers.0.self_attn.k_proj": {"quant_algo": "FP8_PB_WO"},
        "model.layers.0.mlp.experts.0.w13": {"quant_algo": "NVFP4", "group_size": 16},
    }
    with patch.object(modelopt_module.logger, "info") as mock_info:
        _make_mixed_config(quantized_layers)

    expected = "Detected ModelOpt mixed-precision checkpoint with 2 FP8_PB_WO entries"
    formatted_msgs = [call.args[0] % call.args[1:] for call in mock_info.call_args_list]
    assert any(expected in msg for msg in formatted_msgs)
