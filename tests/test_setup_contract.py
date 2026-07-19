from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FLASH_WHEEL = (
    ROOT
    / "wheels/flash_attn/flash_attn-2.7.3-cp312-cp312-linux_aarch64.whl"
)
FLASH_SHA256 = "8b7c2138cce6ff36f13051d3d83ac688ed0f997c629e5c1e140c120b552a9fc6"


def load_setup_module():
    spec = importlib.util.spec_from_file_location("sensenova_extension_setup", ROOT / "setup.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_required_source(extension: Path) -> None:
    (extension / "vendor/opensensenova_sensenova_vision").mkdir(parents=True)
    for name in ("manifest.json", "generator.py", "flash-attn-wheelhouse.manifest.json"):
        shutil.copy2(ROOT / name, extension / name)
    shutil.copy2(
        ROOT / "vendor/opensensenova_sensenova_vision/PROVENANCE.json",
        extension / "vendor/opensensenova_sensenova_vision/PROVENANCE.json",
    )


def _real_open3d_asset(module):
    return module.WheelAsset(
        package="open3d",
        version="0.19.0",
        path=next((ROOT / "wheels/open3d").glob("*.whl")),
        sha256="6dede879da0848d1dacc62bf2f2015ab3623f867630f099b52c2d21aca88cca8",
        size_bytes=17_360_015,
    )


def _real_flash_asset(module):
    return module.WheelAsset(
        package="flash-attn",
        version="2.7.3",
        path=FLASH_WHEEL,
        sha256=FLASH_SHA256,
        size_bytes=406_023_844,
    )


def test_setup_parses_electron_json_and_legacy_argv() -> None:
    module = load_setup_module()
    payload = json.dumps(
        {
            "python_exe": sys.executable,
            "ext_dir": str(ROOT),
            "gpu_sm": 121,
            "cuda_version": 128,
        }
    )
    electron = module.parse_setup_args([payload])
    legacy = module.parse_setup_args([sys.executable, str(ROOT), "121", "128"])
    assert electron == legacy


def test_setup_contains_no_weight_download_or_host_package_shortcut() -> None:
    source = (ROOT / "setup.py").read_text(encoding="utf-8").casefold()
    forbidden = (
        "snapshot_download",
        "hf_hub_download",
        "sensenova/sensenova-vision-7b-mot",
        "apt-get",
        "sudo ",
        "git clone",
        "--no-build-isolation",
    )
    for marker in forbidden:
        assert marker not in source
    assert "flash-attn-wheelhouse-missing" in source
    assert "will not build flashattention from source" in source
    assert 'upstream_flash_attn_version = "2.6.3"' in source
    assert 'flash_attn_version = "2.7.3"' in source


def test_real_local_wheels_match_the_verified_lane() -> None:
    module = load_setup_module()
    open3d = module.resolve_open3d_asset(ROOT)
    flash = module.resolve_flash_attn_asset(ROOT)

    assert open3d.version == "0.19.0"
    assert open3d.size_bytes == 17_360_015
    assert open3d.sha256 == "6dede879da0848d1dacc62bf2f2015ab3623f867630f099b52c2d21aca88cca8"
    assert flash.path == FLASH_WHEEL
    assert flash.version == "2.7.3"
    assert flash.size_bytes == 406_023_844
    assert flash.sha256 == FLASH_SHA256


def test_flash_wheel_missing_and_wrong_bytes_fail_closed(tmp_path: Path) -> None:
    module = load_setup_module()
    extension = tmp_path / "extension"
    extension.mkdir()
    shutil.copy2(
        ROOT / "flash-attn-wheelhouse.manifest.json",
        extension / "flash-attn-wheelhouse.manifest.json",
    )

    with pytest.raises(module.SetupError) as missing:
        module.resolve_flash_attn_asset(extension)
    assert missing.value.code == "flash-attn-wheelhouse-missing"
    assert "FlashAttention wheel file is missing" in str(missing.value)

    wrong = extension / "wheels/flash_attn" / FLASH_WHEEL.name
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(b"not the verified wheel")
    with pytest.raises(module.SetupError) as invalid:
        module.resolve_flash_attn_asset(extension)
    assert invalid.value.code == "flash-attn-wheelhouse-invalid"
    assert "size does not match" in str(invalid.value)


@pytest.mark.parametrize("field", ["selectors", "verification_evidence"])
def test_flash_wheel_wrong_lane_or_evidence_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    module = load_setup_module()
    manifest = json.loads(
        (ROOT / "flash-attn-wheelhouse.manifest.json").read_text(encoding="utf-8")
    )
    manifest["assets"][0][field]["gpu_sm"] = 120
    (tmp_path / "flash-attn-wheelhouse.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(module.SetupError) as caught:
        module.resolve_flash_attn_asset(tmp_path)
    assert caught.value.code == "flash-attn-wheelhouse-missing"


def test_setup_stops_before_installing_when_flash_wheel_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_setup_module()
    extension = tmp_path / "extension"
    _copy_required_source(extension)
    monkeypatch.setattr(module, "resolve_open3d_asset", lambda _ext: _real_open3d_asset(module))
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(list(command))
        raise AssertionError("runner must not execute before wheel preflight succeeds")

    logger = module.SetupLogger(tmp_path / "setup.log")
    config = module.SetupConfig(
        python_exe=Path(sys.executable).resolve(),
        ext_dir=extension,
        gpu_sm=121,
        cuda_version=128,
    )
    try:
        with pytest.raises(module.SetupError) as caught:
            module.run_setup(
                config,
                logger,
                runner=runner,
                system="Linux",
                machine="aarch64",
                python_version=(3, 12),
            )
    finally:
        logger.close()
    assert caught.value.code == "flash-attn-wheelhouse-missing"
    assert calls == []


def test_python_launcher_replaces_symlink_without_overwriting_target_and_is_idempotent(
    tmp_path: Path,
) -> None:
    module = load_setup_module()
    bin_dir = tmp_path / "venv/bin"
    bin_dir.mkdir(parents=True)
    system_python = tmp_path / "system-python3.12"
    system_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$GLIBC_TUNABLES\"\n"
        "printf '%s\\n' \"$1\"\n",
        encoding="utf-8",
    )
    system_python.chmod(0o755)
    venv_python312 = bin_dir / "python3.12"
    venv_python312.symlink_to(system_python)
    venv_python = bin_dir / "python"
    venv_python.symlink_to("python3.12")
    original_system_python = system_python.read_bytes()

    module._install_python_launcher(venv_python)

    assert not venv_python.is_symlink()
    assert os.access(venv_python, os.X_OK)
    assert system_python.read_bytes() == original_system_python
    first_launcher = venv_python.read_bytes()
    env = {
        "GLIBC_TUNABLES": (
            "glibc.malloc.trim_threshold=128:"
            "glibc.rtld.optional_static_tls=1:"
            "glibc.cpu.hwcaps=-AVX2"
        )
    }
    completed = subprocess.run(
        [str(venv_python), "argument preserved"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout.splitlines() == [
        "glibc.malloc.trim_threshold=128:glibc.cpu.hwcaps=-AVX2:"
        "glibc.rtld.optional_static_tls=262144",
        "argument preserved",
    ]

    module._install_python_launcher(venv_python)

    assert venv_python.read_bytes() == first_launcher
    assert system_python.read_bytes() == original_system_python


def test_verified_assets_produce_only_pinned_dependency_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_setup_module()
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    system_python = tmp_path / "system-python3.12"
    system_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    system_python.chmod(0o755)
    (venv_python.parent / "python3.12").symlink_to(system_python)
    venv_python.symlink_to("python3.12")
    monkeypatch.setattr(module, "_venv_python", lambda _ext: venv_python)
    monkeypatch.setattr(module, "resolve_open3d_asset", lambda _ext: _real_open3d_asset(module))
    monkeypatch.setattr(module, "resolve_flash_attn_asset", lambda _ext: _real_flash_asset(module))
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    calls: list[tuple[list[str], dict]] = []

    def runner(command, **kwargs):
        assert not venv_python.is_symlink()
        assert "glibc.rtld.optional_static_tls=262144" in venv_python.read_text(
            encoding="utf-8"
        )
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    logger = module.SetupLogger(tmp_path / "setup-success.log")
    try:
        result = module.run_setup(
            module.SetupConfig(
                python_exe=Path(sys.executable).resolve(),
                ext_dir=ROOT,
                gpu_sm=121,
                cuda_version=128,
            ),
            logger,
            runner=runner,
            system="Linux",
            machine="aarch64",
            python_version=(3, 12),
        )
    finally:
        logger.close()

    joined = [" ".join(command) for command, _kwargs in calls]
    assert any("torch==2.7.0" in command and "torchvision==0.22.0" in command for command in joined)
    assert any("cu128" in command for command in joined)
    assert any(_real_open3d_asset(module).path.name in command for command in joined)
    assert any("--no-deps" in command and FLASH_WHEEL.name in command for command in joined)
    assert all("sensenova/SenseNova-Vision-7B-MoT" not in command for command in joined)
    assert all("--no-build-isolation" not in command for command in joined)
    smoke_command, smoke_kwargs = calls[-1]
    assert smoke_command[1:3] == ["-B", "-c"]
    assert "GLIBC_TUNABLES" not in smoke_kwargs["env"]
    assert result["flash_attn"] == {
        "path": str(FLASH_WHEEL),
        "sha256": FLASH_SHA256,
        "version": "2.7.3",
    }


def test_flash_attention_smoke_executes_bf16_cuda_varlen_kernel_contract() -> None:
    module = load_setup_module()
    smoke = module._smoke_script(ROOT)
    compile(smoke, "<sensenova-setup-smoke>", "exec")

    assert "from flash_attn import flash_attn_varlen_func" in smoke
    assert "str(torch.__version__) == '2.7.0+cu128'" in smoke
    assert "torch.version.cuda == '12.8'" in smoke
    assert "torch.cuda.get_device_capability(0) == (12, 1)" in smoke
    assert "flash_attn.__version__ == '2.7.3'" in smoke
    assert "dtype=torch.bfloat16" in smoke
    assert "for causal in (False, True)" in smoke
    assert "output.shape == q.shape" in smoke
    assert "torch.isfinite(output)" in smoke
    assert "glibc.rtld.optional_static_tls=262144" in smoke
    assert "os.environ.get('GLIBC_TUNABLES', '')" in smoke


def test_verified_lane_rejects_unverified_gpu_sm() -> None:
    module = load_setup_module()
    config = module.SetupConfig(
        python_exe=Path(sys.executable),
        ext_dir=ROOT,
        gpu_sm=120,
        cuda_version=128,
    )
    with pytest.raises(module.SetupError) as caught:
        module.validate_lane(
            config,
            system="Linux",
            machine="aarch64",
            python_version=(3, 12),
        )
    assert caught.value.code == "unsupported-platform"
    assert "requires SM 121" in str(caught.value)


def test_main_keeps_stdout_empty_and_persists_actionable_missing_wheel_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_setup_module()
    extension = tmp_path / "extension"
    _copy_required_source(extension)
    monkeypatch.setattr(module, "resolve_open3d_asset", lambda _ext: _real_open3d_asset(module))
    original_run_setup = module.run_setup

    def target_lane(config, logger):
        return original_run_setup(
            config,
            logger,
            system="Linux",
            machine="aarch64",
            python_version=(3, 12),
        )

    monkeypatch.setattr(module, "run_setup", target_lane)
    payload = json.dumps(
        {
            "python_exe": sys.executable,
            "ext_dir": str(extension),
            "gpu_sm": 121,
            "cuda_version": 128,
        }
    )
    assert module.main([payload]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "flash-attn-wheelhouse-missing" in captured.err
    status = json.loads(
        (extension / ".modly/setup/setup-status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "blocked"
    assert status["error"]["code"] == "flash-attn-wheelhouse-missing"
    assert status["model_weights"]["downloaded_by_setup"] is False


def test_docs_have_no_stale_flash_attention_blocker_language() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").casefold()
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8").casefold()

    for stale in (
        "current verified blocker",
        "there is currently no verified flashattention",
        "after setup is unblocked",
    ):
        assert stale not in readme
    assert "**unblocked**" in readme
    assert "406,023,844" in readme or "406023844" in readme
    assert FLASH_WHEEL.name.casefold() in readme
    assert "source build" in readme
    assert "redistributes" in notices
    assert "bsd-3-clause" in notices
