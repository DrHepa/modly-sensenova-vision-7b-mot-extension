from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_setup():
    spec = importlib.util.spec_from_file_location("sensenova_windows_setup", ROOT / "setup.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_required_source(extension: Path) -> None:
    (extension / "vendor/opensensenova_sensenova_vision").mkdir(parents=True)
    for name in ("manifest.json", "generator.py"):
        shutil.copy2(ROOT / name, extension / name)
    shutil.copy2(
        ROOT / "vendor/opensensenova_sensenova_vision/PROVENANCE.json",
        extension / "vendor/opensensenova_sensenova_vision/PROVENANCE.json",
    )


def test_windows_setup_skips_flashattention_source_install(tmp_path: Path, monkeypatch) -> None:
    setup = _load_setup()
    extension = tmp_path / "extension"
    _copy_required_source(extension)
    monkeypatch.setattr(setup, "SCRIPT_DIR", extension.resolve())

    venv_python = extension / "venv/Scripts/python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"windows-python")
    monkeypatch.setattr(setup, "_venv_python", lambda _ext, _system=None: venv_python)
    monkeypatch.setattr(setup, "_venv_matches_profile", lambda *_args, **_kwargs: True)

    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    logger = setup.SetupLogger(tmp_path / "setup.log")
    try:
        result = setup.run_setup(
            setup.SetupConfig(
                python_exe=Path(sys.executable).resolve(),
                ext_dir=extension,
                gpu_sm=120,
                cuda_version=134,
                platform_name="windows",
                arch="x86_64",
            ),
            logger,
            runner=runner,
            system="Windows",
            machine="x86_64",
            python_version=(3, 12),
        )
    finally:
        logger.close()

    install_commands = [" ".join(command) for command in calls if "-m" in command and "pip" in command]
    assert result["lane"]["cuda"] == 134
    assert result["lane"]["torch_build"] == "cu128"
    assert any("--index-url https://download.pytorch.org/whl/cu128" in command for command in install_commands)
    assert not any("flash-attn" in command or "flash_attn" in command for command in install_commands)
    assert result["attention_backend"] == {
        "selected": "sdpa",
        "fallback": "torch.nn.functional.scaled_dot_product_attention",
        "flash_attn_available": False,
    }
    assert result["model_weights_downloaded"] is False


def test_cuda_13_4_encoding_maps_to_last_selected_cu128_lane() -> None:
    setup = _load_setup()
    assert setup._torch_build(134) == "cu128"
    lane = setup.validate_lane(
        setup.SetupConfig(
            python_exe=Path(sys.executable),
            ext_dir=ROOT,
            gpu_sm=120,
            cuda_version=134,
            platform_name="windows",
            arch="x86_64",
        ),
        system="Windows",
        machine="x86_64",
        python_version=(3, 12),
    )
    assert lane["torch_build"] == "cu128"


def test_windows_smoke_requires_matching_cuda_torch_build() -> None:
    setup = _load_setup()
    smoke = setup._smoke_script(
        ROOT,
        attention_backend="sdpa",
        torch_build="cu128",
        cuda_version=134,
        accelerator="cuda",
    )
    compile(smoke, "<sensenova-windows-smoke>", "exec")
    assert "expected_torch_build = 'cu128'" in smoke
    assert "expected_cuda_version = 134" in smoke
    assert "expected_cuda_runtime = '12.8'" in smoke
    assert "torch.cuda.is_available()" in smoke
    assert '"+{expected_torch_build}" in torch_version' in smoke


def _load_attention_backend():
    torch = pytest.importorskip("torch")
    path = ROOT / "vendor/opensensenova_sensenova_vision/modeling/bagel/attention_backend.py"
    module_name = "_sensenova_attention_backend_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return torch, module


def test_sdpa_fallback_handles_packed_gqa_masks_and_noncausal() -> None:
    torch, attention = _load_attention_backend()
    attention.reset_attention_backend_cache()
    q = torch.randn((5, 4, 8), dtype=torch.float32)
    k = torch.randn((7, 2, 8), dtype=torch.float32)
    v = torch.randn((7, 2, 8), dtype=torch.float32)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)
    masks = [
        torch.tensor([[True, True, False], [True, False, True]]),
        torch.ones((3, 4), dtype=torch.bool),
    ]

    output = attention.varlen_attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=3,
        max_seqlen_k=4,
        causal=False,
        attention_mask=masks,
        backend="sdpa",
    )

    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert bool(torch.isfinite(output).all().item())


def test_sdpa_fallback_uses_bottom_right_causal_alignment() -> None:
    torch, attention = _load_attention_backend()
    attention.reset_attention_backend_cache()
    q = torch.tensor([[[1.0]], [[1.0]]])
    k = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    v = torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 4], dtype=torch.int32)

    output = attention.sdpa_varlen_attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=2,
        max_seqlen_k=4,
        causal=True,
    )

    # Bottom-right causal alignment permits the first query to see the first
    # three cached keys and the final query to see all four.
    allowed = torch.tensor([[True, True, True, False], [True, True, True, True]])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=allowed.unsqueeze(0).unsqueeze(0),
        is_causal=False,
    ).squeeze(0).transpose(0, 1)
    assert torch.allclose(output, expected)


def test_flashattention_dispatch_is_optional_and_prefers_callable(monkeypatch) -> None:
    torch, attention = _load_attention_backend()
    calls = []

    fake = types.ModuleType("flash_attn")

    def fake_varlen(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs["q"]

    fake.flash_attn_varlen_func = fake_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", fake)
    attention.reset_attention_backend_cache()
    q = torch.randn((2, 1, 4))
    cu = torch.tensor([0, 2], dtype=torch.int32)
    output = attention.varlen_attention(
        q,
        q,
        q,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=2,
        max_seqlen_k=2,
    )
    assert attention.get_attention_backend() == "flash-attn"
    assert calls and output is q
