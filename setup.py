#!/usr/bin/env python3
"""Prepare the isolated SenseNova runtime without downloading model weights."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


EXTENSION_ID = "sensenova-vision-7b-mot"
STATUS_SCHEMA = "modly.setup-status.v1"
SCRIPT_DIR = Path(__file__).resolve().parent

PYTHON_VERSION = (3, 12)
TORCH_VERSION = "2.7.0"
TORCHVISION_VERSION = "0.22.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CUDA_VERSION = 128
UPSTREAM_FLASH_ATTN_VERSION = "2.6.3"
FLASH_ATTN_VERSION = "2.7.3"
OPEN3D_VERSION = "0.19.0"
OPTIONAL_STATIC_TLS_TUNABLE = "glibc.rtld.optional_static_tls=262144"
PYTHON_LAUNCHER = f"""#!/bin/sh
_required_tunable={OPTIONAL_STATIC_TLS_TUNABLE!r}
_remaining=${{GLIBC_TUNABLES-}}
_preserved=
while [ -n "$_remaining" ]; do
    case "$_remaining" in
        *:*)
            _entry=${{_remaining%%:*}}
            _remaining=${{_remaining#*:}}
            ;;
        *)
            _entry=$_remaining
            _remaining=
            ;;
    esac
    case "$_entry" in
        glibc.rtld.optional_static_tls=*)
            ;;
        "")
            ;;
        *)
            if [ -n "$_preserved" ]; then
                _preserved="${{_preserved}}:${{_entry}}"
            else
                _preserved=$_entry
            fi
            ;;
    esac
done
if [ -n "$_preserved" ]; then
    GLIBC_TUNABLES="${{_preserved}}:${{_required_tunable}}"
else
    GLIBC_TUNABLES=$_required_tunable
fi
export GLIBC_TUNABLES
case "$0" in
    */*) _launcher_dir=${{0%/*}} ;;
    *) _launcher_dir=. ;;
esac
exec "$_launcher_dir/python3.12" "$@"
"""

OPEN3D_MANIFEST_NAME = "open3d-wheelhouse.manifest.json"
OPEN3D_MANIFEST_SCHEMA = "sensenova-open3d-wheelhouse/v1"
FLASH_MANIFEST_NAME = "flash-attn-wheelhouse.manifest.json"
FLASH_MANIFEST_SCHEMA = "sensenova-flash-attn-wheelhouse/v1"
FLASH_ATTN_SELECTORS = {
    "os": "linux",
    "arch": "aarch64",
    "python_tag": "cp312",
    "torch": "2.7.0+cu128",
    "cuda": "12.8",
    "gpu_sm": 121,
}
FLASH_ATTN_VERIFICATION_EVIDENCE = {
    "verified_at": "2026-07-16",
    "python": "3.12.3",
    "torch": "2.7.0+cu128",
    "cuda_runtime": "12.8",
    "gpu_sm": 121,
    "nvidia_driver": "580.159.03",
    "import_source": "extracted wheel contents",
    "excluded_installed_version": "flash-attn 2.8.3",
    "kernel": "flash_attn_varlen_func",
    "dtype": "bfloat16",
    "modes": ["causal", "noncausal"],
    "checks": ["output shape", "finite output"],
}

STATUS_RELATIVE_PATH = Path(".modly/setup/setup-status.json")
LOG_RELATIVE_PATH = Path(".modly/setup/logs/setup.log")

BOOTSTRAP_REQUIREMENTS = (
    "pip==25.1.1",
    "setuptools==80.9.0",
    "wheel==0.45.1",
)
RUNTIME_REQUIREMENTS = (
    "accelerate==1.7.0",
    "einops==0.8.1",
    "httpx==0.28.1",
    "huggingface_hub==0.29.1",
    "numpy==1.26.4",
    "Pillow==11.2.1",
    "regex==2024.11.6",
    "safetensors==0.4.5",
    "scipy==1.14.1",
    "tqdm==4.67.1",
    "transformers==4.49.0",
    "trimesh==4.12.2",
)
REQUIRED_EXTENSION_FILES = (
    Path("manifest.json"),
    Path("generator.py"),
    Path("vendor/opensensenova_sensenova_vision/PROVENANCE.json"),
)


class SetupError(RuntimeError):
    """Expected, actionable setup failure shown by Modly."""

    def __init__(self, message: str, code: str = "setup-error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SetupConfig:
    python_exe: Path
    ext_dir: Path
    gpu_sm: int
    cuda_version: int


@dataclass(frozen=True)
class WheelAsset:
    package: str
    version: str
    path: Path
    sha256: str
    size_bytes: int


class SetupLogger:
    """Mirror human-readable setup output to stderr and a durable log file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def write(self, message: str) -> None:
        line = _redact(str(message).rstrip())
        if not line:
            return
        print(line, file=sys.stderr, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()

    def info(self, message: str) -> None:
        self.write(f"[setup:{EXTENSION_ID}] {message}")

    def error(self, message: str) -> None:
        self.write(f"[setup:{EXTENSION_ID}][error] {message}")


def _redact(value: str) -> str:
    value = re.sub(r"hf_[A-Za-z0-9]{8,}", "hf_[REDACTED]", value)
    value = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1[REDACTED]",
        value,
    )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SetupError(f"{field} must be an integer", "invalid-arguments")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SetupError(f"{field} must be an integer", "invalid-arguments") from exc


def parse_setup_args(argv: Sequence[str]) -> SetupConfig:
    """Accept Electron's single JSON argv and the established legacy form."""
    if len(argv) == 1:
        try:
            payload = json.loads(argv[0])
        except json.JSONDecodeError as exc:
            raise SetupError(
                "The single setup argument must be a JSON object",
                "invalid-arguments",
            ) from exc
        if not isinstance(payload, Mapping):
            raise SetupError("Setup JSON must be an object", "invalid-arguments")
        required = {"python_exe", "ext_dir", "gpu_sm", "cuda_version"}
        missing = sorted(required - set(payload))
        if missing:
            raise SetupError(
                "Setup JSON is missing: " + ", ".join(missing),
                "invalid-arguments",
            )
        python_exe = Path(str(payload["python_exe"])).expanduser()
        ext_dir = Path(str(payload["ext_dir"])).expanduser()
        gpu_sm = _require_int(payload["gpu_sm"], "gpu_sm")
        cuda_version = _require_int(payload["cuda_version"], "cuda_version")
    elif len(argv) in {3, 4}:
        python_exe = Path(argv[0]).expanduser()
        ext_dir = Path(argv[1]).expanduser()
        gpu_sm = _require_int(argv[2], "gpu_sm")
        cuda_version = _require_int(argv[3], "cuda_version") if len(argv) == 4 else CUDA_VERSION
    else:
        raise SetupError(
            "Usage: setup.py '{\"python_exe\":\"...\",\"ext_dir\":\"...\","
            "\"gpu_sm\":121,\"cuda_version\":128}' or "
            "setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]",
            "invalid-arguments",
        )

    return SetupConfig(
        python_exe=python_exe.resolve(),
        ext_dir=ext_dir.resolve(),
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
    )


def validate_lane(
    config: SetupConfig,
    *,
    system: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> dict[str, Any]:
    detected_system = (system or platform.system()).casefold()
    detected_machine = (machine or platform.machine()).casefold()
    aliases = {"arm64": "aarch64"}
    detected_machine = aliases.get(detected_machine, detected_machine)
    detected_python = python_version or (sys.version_info.major, sys.version_info.minor)

    problems: list[str] = []
    if detected_system != "linux":
        problems.append(f"operating system is {detected_system or 'unknown'}, expected linux")
    if detected_machine != "aarch64":
        problems.append(f"architecture is {detected_machine or 'unknown'}, expected aarch64")
    if tuple(detected_python) != PYTHON_VERSION:
        problems.append(
            f"Python is {detected_python[0]}.{detected_python[1]}, expected 3.12"
        )
    if config.cuda_version != CUDA_VERSION:
        problems.append(
            f"CUDA lane is {config.cuda_version}, expected {CUDA_VERSION} (12.8)"
        )
    if config.gpu_sm != 121:
        problems.append(
            f"GPU SM is {config.gpu_sm}; the verified FlashAttention wheel requires SM 121"
        )
    if problems:
        raise SetupError(
            "Unsupported SenseNova setup lane: " + "; ".join(problems),
            "unsupported-platform",
        )
    return {
        "id": "linux-aarch64/cp312/cu128/torch2.7.0/sm121",
        "os": "linux",
        "arch": "aarch64",
        "python": "3.12",
        "python_tag": "cp312",
        "cuda": "12.8",
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "gpu_sm": config.gpu_sm,
    }


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SetupError(f"Required wheel manifest is missing: {path.name}", code) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"Wheel manifest is invalid: {path.name}", code) from exc
    if not isinstance(value, dict):
        raise SetupError(f"Wheel manifest must be an object: {path.name}", code)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_asset_path(ext_dir: Path, relative: Any, code: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise SetupError("Wheel filename must be a safe relative path", code)
    raw = Path(relative)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise SetupError("Wheel filename must be a safe relative path", code)
    candidate = (ext_dir / raw).resolve()
    try:
        candidate.relative_to(ext_dir.resolve())
    except ValueError as exc:
        raise SetupError("Wheel path escapes the extension directory", code) from exc
    return candidate


def _wheel_metadata(path: Path, code: str) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise SetupError("Wheel must contain exactly one METADATA file", code)
            text = archive.read(metadata_names[0]).decode("utf-8")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise SetupError(f"Wheel is not a valid Python wheel: {path.name}", code) from exc
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in {"Name", "Version"} and key not in fields:
                fields[key] = value.strip()
    if "Name" not in fields or "Version" not in fields:
        raise SetupError(f"Wheel metadata is incomplete: {path.name}", code)
    return fields["Name"].casefold().replace("_", "-"), fields["Version"]


def _validated_asset(
    ext_dir: Path,
    raw: Mapping[str, Any],
    *,
    package: str,
    version: str,
    code: str,
) -> WheelAsset:
    path = _safe_asset_path(ext_dir, raw.get("filename"), code)
    if path.is_symlink() or not path.is_file():
        raise SetupError(f"Verified {package} wheel file is missing: {path}", code)
    expected_size = raw.get("size_bytes")
    expected_hash = raw.get("sha256")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise SetupError(f"{package} wheel size is missing from its manifest", code)
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise SetupError(f"{package} wheel SHA-256 is missing from its manifest", code)
    if path.stat().st_size != expected_size:
        raise SetupError(f"{package} wheel size does not match its manifest", code)
    actual_hash = _sha256(path)
    if actual_hash != expected_hash.casefold():
        raise SetupError(f"{package} wheel SHA-256 does not match its manifest", code)
    metadata_name, metadata_version = _wheel_metadata(path, code)
    if metadata_name != package.casefold().replace("_", "-") or metadata_version != version:
        raise SetupError(
            f"Wheel metadata is {metadata_name}=={metadata_version}; expected "
            f"{package}=={version}",
            code,
        )
    return WheelAsset(
        package=package,
        version=version,
        path=path,
        sha256=actual_hash,
        size_bytes=expected_size,
    )


def resolve_open3d_asset(ext_dir: Path) -> WheelAsset:
    code = "open3d-wheelhouse-invalid"
    payload = _load_json(ext_dir / OPEN3D_MANIFEST_NAME, code)
    if (
        payload.get("schema_version") != OPEN3D_MANIFEST_SCHEMA
        or payload.get("extension_id") != EXTENSION_ID
    ):
        raise SetupError("Open3D wheel manifest identity is invalid", code)
    selectors = {
        "os": "linux",
        "arch": "aarch64",
        "python_tag": "cp312",
        "accelerator_lane": "openblas-cpu",
    }
    for raw in payload.get("assets", []):
        if not isinstance(raw, Mapping):
            continue
        if (
            raw.get("package") == "open3d"
            and raw.get("version") == OPEN3D_VERSION
            and raw.get("selectors") == selectors
        ):
            return _validated_asset(
                ext_dir,
                raw,
                package="open3d",
                version=OPEN3D_VERSION,
                code=code,
            )
    raise SetupError(
        "The Open3D wheel manifest has no audited CPython 3.12 Linux aarch64 asset",
        "open3d-wheelhouse-missing",
    )


def resolve_flash_attn_asset(ext_dir: Path) -> WheelAsset:
    invalid_code = "flash-attn-wheelhouse-invalid"
    payload = _load_json(ext_dir / FLASH_MANIFEST_NAME, invalid_code)
    if (
        payload.get("schema_version") != FLASH_MANIFEST_SCHEMA
        or payload.get("extension_id") != EXTENSION_ID
    ):
        raise SetupError("FlashAttention wheel manifest identity is invalid", invalid_code)
    for raw in payload.get("assets", []):
        if not isinstance(raw, Mapping):
            continue
        if (
            raw.get("package") in {"flash-attn", "flash_attn"}
            and raw.get("version") == FLASH_ATTN_VERSION
            and raw.get("selectors") == FLASH_ATTN_SELECTORS
            and raw.get("verification_evidence") == FLASH_ATTN_VERIFICATION_EVIDENCE
        ):
            candidate = _safe_asset_path(
                ext_dir,
                raw.get("filename"),
                invalid_code,
            )
            if candidate.is_symlink() or not candidate.is_file():
                raise SetupError(
                    f"Verified FlashAttention wheel file is missing: {candidate}",
                    "flash-attn-wheelhouse-missing",
                )
            return _validated_asset(
                ext_dir,
                raw,
                package="flash-attn",
                version=FLASH_ATTN_VERSION,
                code=invalid_code,
            )
    raise SetupError(
        "No verified FlashAttention 2.7.3 compatibility-adaptation wheel is "
        "available for CPython 3.12, Linux aarch64, Torch 2.7.0+cu128, "
        "CUDA 12.8, and SM 121. The official SenseNova setup.sh reference to "
        f"flash-attn=={UPSTREAM_FLASH_ATTN_VERSION} remains recorded separately. "
        "Restore the audited local wheel and its exact manifest evidence. Setup "
        "will not build FlashAttention from source or download a replacement.",
        "flash-attn-wheelhouse-missing",
    )


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run_command(
    command: Sequence[str],
    logger: SetupLogger,
    *,
    runner: Runner,
    env: dict[str, str] | None = None,
) -> None:
    logger.info("Running: " + " ".join(_redact(value) for value in command))
    result = runner(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    output = result.stdout or ""
    for line in output.splitlines():
        logger.write(line)
    if result.returncode != 0:
        raise SetupError(
            f"Command failed with exit code {result.returncode}: {command[0]}",
            "dependency-install-failed",
        )


def _venv_python(ext_dir: Path) -> Path:
    return ext_dir / "venv" / "bin" / "python"


def _install_python_launcher(venv_python: Path) -> None:
    """Atomically replace the venv's unversioned Python symlink with our launcher."""
    versioned_python = venv_python.with_name("python3.12")
    if not versioned_python.is_file():
        raise SetupError(
            "Virtual environment creation did not produce venv/bin/python3.12",
            "python-launcher-failed",
        )

    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".modly-python-launcher-",
            dir=venv_python.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(PYTHON_LAUNCHER)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o755)
        os.replace(temporary, venv_python)
        temporary = None
    except OSError as exc:
        raise SetupError(
            f"Could not install the virtual-environment Python launcher: {exc}",
            "python-launcher-failed",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _smoke_script(ext_dir: Path) -> str:
    vendor = ext_dir / "vendor" / "opensensenova_sensenova_vision"
    return f"""
import importlib
import json
import os
import sys
from pathlib import Path

os.environ.update({{
    'HF_HUB_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
}})
glibc_tunables = os.environ.get('GLIBC_TUNABLES', '').split(':')
assert glibc_tunables.count({OPTIONAL_STATIC_TLS_TUNABLE!r}) == 1
sys.path.insert(0, {str(vendor)!r})
open3d = importlib.import_module('open3d')
torch = importlib.import_module('torch')
flash_attn = importlib.import_module('flash_attn')
from flash_attn import flash_attn_varlen_func
runtime = importlib.import_module('inference.sensenova_vision')
assert Path(runtime.__file__).resolve().is_relative_to(Path({str(vendor)!r}).resolve())
assert open3d.__version__ == {OPEN3D_VERSION!r}
assert str(torch.__version__) == {f"{TORCH_VERSION}+cu128"!r}
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 1)
assert flash_attn.__version__ == {FLASH_ATTN_VERSION!r}

torch.manual_seed(0)
device = torch.device('cuda')
cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device=device)
q = torch.randn((4, 2, 64), dtype=torch.bfloat16, device=device)
k = torch.randn((4, 2, 64), dtype=torch.bfloat16, device=device)
v = torch.randn((4, 2, 64), dtype=torch.bfloat16, device=device)
kernel_checks = {{}}
for causal in (False, True):
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        4,
        4,
        dropout_p=0.0,
        causal=causal,
    )
    assert output.shape == q.shape
    assert output.dtype == torch.bfloat16
    assert bool(torch.isfinite(output).all().item())
    kernel_checks['causal' if causal else 'noncausal'] = {{
        'shape': list(output.shape),
        'finite': True,
        'dtype': str(output.dtype),
    }}
torch.cuda.synchronize()

print(json.dumps({{
    'open3d': open3d.__version__,
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gpu_sm': 121,
    'flash_attn': flash_attn.__version__,
    'flash_attn_varlen_func': kernel_checks,
    'runtime': str(Path(runtime.__file__).resolve()),
}}, sort_keys=True))
"""


def run_setup(
    config: SetupConfig,
    logger: SetupLogger,
    *,
    runner: Runner = subprocess.run,
    system: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> dict[str, Any]:
    lane = validate_lane(
        config,
        system=system,
        machine=machine,
        python_version=python_version,
    )
    if not config.python_exe.is_file():
        raise SetupError(
            f"Python executable does not exist: {config.python_exe}",
            "invalid-python",
        )
    missing = [
        str(relative) for relative in REQUIRED_EXTENSION_FILES
        if not (config.ext_dir / relative).is_file()
    ]
    if missing:
        raise SetupError(
            "Extension source is incomplete: " + ", ".join(missing),
            "extension-source-missing",
        )

    logger.info(f"Selected audited lane: {lane['id']}")
    open3d = resolve_open3d_asset(config.ext_dir)
    logger.info(f"Verified Open3D wheel: {open3d.path.name} ({open3d.sha256})")
    flash_attn = resolve_flash_attn_asset(config.ext_dir)
    logger.info(
        f"Verified FlashAttention wheel: {flash_attn.path.name} ({flash_attn.sha256})"
    )

    venv_python = _venv_python(config.ext_dir)
    if not venv_python.is_file():
        _run_command(
            [str(config.python_exe), "-m", "venv", str(config.ext_dir / "venv")],
            logger,
            runner=runner,
        )
    if not venv_python.is_file():
        raise SetupError("Virtual environment creation did not produce venv/bin/python")

    _install_python_launcher(venv_python)
    logger.info("Installed Open3D static-TLS Python launcher")

    pip = [str(venv_python), "-m", "pip", "install", "--no-input"]
    _run_command([*pip, "--upgrade", *BOOTSTRAP_REQUIREMENTS], logger, runner=runner)
    _run_command(
        [
            *pip,
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url",
            TORCH_INDEX_URL,
        ],
        logger,
        runner=runner,
    )
    _run_command([*pip, *RUNTIME_REQUIREMENTS], logger, runner=runner)
    _run_command([*pip, str(open3d.path)], logger, runner=runner)
    _run_command([*pip, "--no-deps", str(flash_attn.path)], logger, runner=runner)
    _run_command(
        [str(venv_python), "-m", "pip", "check"],
        logger,
        runner=runner,
    )

    smoke_env = dict(os.environ)
    smoke_env.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    })
    _run_command(
        [str(venv_python), "-B", "-c", _smoke_script(config.ext_dir)],
        logger,
        runner=runner,
        env=smoke_env,
    )
    return {
        "lane": lane,
        "venv_python": str(venv_python),
        "open3d": {
            "path": str(open3d.path),
            "sha256": open3d.sha256,
            "version": open3d.version,
        },
        "flash_attn": {
            "path": str(flash_attn.path),
            "sha256": flash_attn.sha256,
            "version": flash_attn.version,
        },
    }


def _write_status(
    config: SetupConfig,
    *,
    status: str,
    started_at: str,
    result: dict[str, Any] | None = None,
    error: SetupError | None = None,
) -> None:
    path = config.ext_dir / STATUS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "extension_id": EXTENSION_ID,
        "status": status,
        "started_at": started_at,
        "completed_at": _utc_now() if status != "running" else None,
        "log_path": str(config.ext_dir / LOG_RELATIVE_PATH),
        "venv_dir": str(config.ext_dir / "venv"),
        "model_weights": {
            "managed_by": "modly-ui/downloader",
            "downloaded_by_setup": False,
            "target": "checkpoint",
        },
        "result": result,
        "error": (
            {"code": error.code, "message": str(error)} if error is not None else None
        ),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        config = parse_setup_args(arguments)
    except SetupError as exc:
        print(f"[setup:{EXTENSION_ID}][error] {exc}", file=sys.stderr, flush=True)
        return 2

    started_at = _utc_now()
    logger = SetupLogger(config.ext_dir / LOG_RELATIVE_PATH)
    _write_status(config, status="running", started_at=started_at)
    try:
        logger.info("Preparing dependencies only; model weights remain UI-managed")
        result = run_setup(config, logger)
        _write_status(
            config,
            status="ready",
            started_at=started_at,
            result=result,
        )
        logger.info("SenseNova extension runtime is ready")
        return 0
    except SetupError as exc:
        _write_status(
            config,
            status="blocked",
            started_at=started_at,
            error=exc,
        )
        logger.error(f"{exc.code}: {exc}")
        return 1
    except Exception as exc:
        wrapped = SetupError(
            f"Unexpected setup failure: {exc.__class__.__name__}: {exc}",
            "unexpected-setup-error",
        )
        _write_status(
            config,
            status="failed",
            started_at=started_at,
            error=wrapped,
        )
        logger.error(str(wrapped))
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

