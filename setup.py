#!/usr/bin/env python3
"""Prepare the isolated SenseNova runtime without downloading model weights."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
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

SUPPORTED_PYTHON_VERSIONS = {(3, 11), (3, 12)}
TORCH_VERSION = "2.7.0"
TORCHVISION_VERSION = "0.22.0"
TORCH_INDEX_URLS = {
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}
TORCH_BUILD_CUDA_VERSIONS = {
    "cu118": "11.8",
    "cu126": "12.6",
    "cu128": "12.8",
}
UPSTREAM_FLASH_ATTN_VERSION = "2.6.3"
FLASH_ATTN_VERSION = "2.7.3"
OPEN3D_VERSION = "0.19.0"
FLASH_ATTN_ARM64_RELEASE_URL = (
    "https://github.com/DrHepa/modly-sensenova-vision-7b-mot-extension/"
    "releases/download/v0.2.2/"
    "flash_attn-2.7.3-cp312-cp312-linux_aarch64.whl#sha256=8b7c2138cce6ff36f13051d3d83ac688ed0f997c629e5c1e140c120b552a9fc6"
)
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
VENV_TRANSACTION_MARKER_RELATIVE_PATH = Path(".modly/setup/venv-transaction.json")
VENV_TRANSACTION_BACKUP_RELATIVE_PATH = Path(".modly/setup/venv-backup")

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
    "packaging==24.2",
    "ninja==1.11.1.3",
    "psutil==6.1.1",
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
    accelerator: str = ""
    platform_name: str = ""
    arch: str = ""


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


class VenvTransaction:
    """Replace the shared extension venv without losing the last working copy."""

    def __init__(self, ext_dir: Path, logger: SetupLogger) -> None:
        self.venv_dir = ext_dir / "venv"
        self.marker_path = ext_dir / VENV_TRANSACTION_MARKER_RELATIVE_PATH
        self.backup_dir = ext_dir / VENV_TRANSACTION_BACKUP_RELATIVE_PATH
        self.logger = logger
        self._active = False

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path)

    def _read_had_existing(self) -> bool:
        try:
            payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.backup_dir.exists()
        return bool(payload.get("had_existing", self.backup_dir.exists()))

    def _write_marker(self, had_existing: bool) -> None:
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.marker_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "modly.venv-transaction.v1",
                    "had_existing": had_existing,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.marker_path)

    def rollback(self) -> str | None:
        """Restore the previous venv, or remove a failed fresh installation."""

        if not self._active and not self.marker_path.exists():
            return None
        had_existing = self._read_had_existing()
        try:
            if self.backup_dir.exists():
                self._remove_path(self.venv_dir)
                os.replace(self.backup_dir, self.venv_dir)
                self.logger.info("Restored the previous extension venv")
            elif not had_existing:
                self._remove_path(self.venv_dir)
            self.marker_path.unlink(missing_ok=True)
            self._active = False
            return None
        except OSError as exc:
            message = f"Could not restore the previous extension venv: {exc}"
            self.logger.error(message)
            return message

    def begin(self) -> None:
        if self.marker_path.exists():
            self.logger.info("Recovering an interrupted venv replacement")
            error = self.rollback()
            if error is not None:
                raise SetupError(error, "venv-rollback-failed")
        if self.backup_dir.exists():
            self.logger.info("Removing an orphaned extension venv backup")
            self._remove_path(self.backup_dir)

        had_existing = self.venv_dir.exists() or self.venv_dir.is_symlink()
        self._write_marker(had_existing)
        self._active = True
        if had_existing:
            try:
                os.replace(self.venv_dir, self.backup_dir)
            except OSError as exc:
                error = self.rollback()
                detail = f"; rollback failed: {error}" if error else ""
                raise SetupError(
                    f"Could not preserve the existing extension venv: {exc}{detail}",
                    "venv-backup-failed",
                ) from exc
            self.logger.info("Preserved the previous extension venv until setup succeeds")

    def commit(self) -> None:
        if not self._active:
            return
        try:
            self.marker_path.unlink()
        except OSError as exc:
            raise SetupError(
                f"Could not commit the extension venv replacement: {exc}",
                "venv-commit-failed",
            ) from exc
        self._active = False
        try:
            self._remove_path(self.backup_dir)
        except OSError as exc:
            self.logger.info(f"Could not remove the old venv backup yet: {exc}")


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


def _same_directory(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return os.path.normcase(str(left.resolve(strict=False))) == os.path.normcase(
            str(right.resolve(strict=False))
        )



def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SetupError(f"{field} must be an integer", "invalid-arguments")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SetupError(f"{field} must be an integer", "invalid-arguments") from exc


def parse_setup_args(argv: Sequence[str]) -> SetupConfig:
    """Accept Modly's single JSON argv and the established legacy form."""

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
        required = {"python_exe", "ext_dir"}
        missing = sorted(required - set(payload))
        if missing:
            raise SetupError(
                "Setup JSON is missing: " + ", ".join(missing),
                "invalid-arguments",
            )
        python_exe = Path(str(payload["python_exe"])).expanduser()
        ext_dir = Path(str(payload["ext_dir"])).expanduser()
        gpu_sm = _require_int(payload.get("gpu_sm", 0), "gpu_sm")
        cuda_version = _require_int(payload.get("cuda_version", 0), "cuda_version")
        accelerator = str(payload.get("accelerator") or "")
        platform_name = str(payload.get("platform") or "")
        arch = str(payload.get("arch") or "")
    elif len(argv) in {3, 4}:
        python_exe = Path(argv[0]).expanduser()
        ext_dir = Path(argv[1]).expanduser()
        gpu_sm = _require_int(argv[2], "gpu_sm")
        cuda_version = _require_int(argv[3], "cuda_version") if len(argv) == 4 else 0
        accelerator = ""
        platform_name = ""
        arch = ""
    else:
        raise SetupError(
            "Usage: setup.py with one JSON object containing python_exe, ext_dir, "
            "and optional gpu_sm, cuda_version, platform, and arch; or "
            "setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]",
            "invalid-arguments",
        )

    return SetupConfig(
        python_exe=python_exe.resolve(),
        ext_dir=ext_dir.resolve(),
        gpu_sm=max(0, gpu_sm),
        cuda_version=max(0, cuda_version),
        accelerator=accelerator,
        platform_name=platform_name,
        arch=arch,
    )


def _normalize_system(value: str) -> str:
    aliases = {
        "linux": "linux",
        "win32": "windows",
        "windows": "windows",
    }
    normalized = aliases.get(str(value or "").strip().casefold())
    if normalized is None:
        raise SetupError(
            f"Unsupported operating system: {value or 'unknown'}; expected Windows or Linux",
            "unsupported-platform",
        )
    return normalized


def _normalize_machine(value: str) -> str:
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }
    normalized = aliases.get(str(value or "").strip().casefold())
    if normalized is None:
        raise SetupError(
            f"Unsupported architecture: {value or 'unknown'}; expected x86_64 or aarch64",
            "unsupported-platform",
        )
    return normalized


def _torch_build(cuda_version: int) -> str:
    # Modly encodes CUDA 13.4 as 134 on Windows.  There is no independently
    # validated cu134 dependency lane here, so 134 (and newer reports) use the
    # highest selected Torch wheel, cu128.  This is a packaging mapping, not a
    # claim that a CUDA 13.x toolkit or native GPU path was validated.
    if cuda_version == 134:
        return "cu128"
    if cuda_version >= 128:
        return "cu128"
    if cuda_version >= 126:
        return "cu126"
    if cuda_version >= 118:
        return "cu118"
    return "pypi"


_PYTHON_PROBE_SCRIPT = """
import json
import platform
import sys

print(json.dumps({
    "implementation": platform.python_implementation().lower(),
    "version": list(sys.version_info[:3]),
    "platform": sys.platform,
    "machine": platform.machine(),
}, sort_keys=True))
"""


def _probe_python_environment(
    python_exe: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    try:
        result = runner(
            [str(python_exe), "-I", "-B", "-c", _PYTHON_PROBE_SCRIPT],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupError(
            f"Could not execute the supplied Python interpreter: {python_exe}: {exc}",
            "python-probe-failed",
        ) from exc
    if result.returncode != 0:
        detail = _redact((result.stderr or result.stdout or "").strip())
        raise SetupError(
            f"Supplied Python probe failed with exit code {result.returncode}: "
            f"{detail or python_exe}",
            "python-probe-failed",
        )
    try:
        payload = json.loads(result.stdout)
        version = tuple(int(value) for value in payload["version"][:2])
        implementation = str(payload["implementation"]).casefold()
        detected_system = _normalize_system(str(payload["platform"]))
        detected_machine = _normalize_machine(str(payload["machine"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SetupError(
            "Supplied Python returned invalid environment information",
            "python-probe-failed",
        ) from exc
    return {
        "implementation": implementation,
        "version": version,
        "system": detected_system,
        "machine": detected_machine,
    }



def validate_lane(
    config: SetupConfig,
    *,
    system: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int] | None = None,
    python_implementation: str = "cpython",
) -> dict[str, Any]:
    detected_system = _normalize_system(system or platform.system())
    detected_machine = _normalize_machine(machine or platform.machine())
    detected_python = python_version or (sys.version_info.major, sys.version_info.minor)
    if python_implementation.strip().casefold() != "cpython":
        raise SetupError(
            f"Unsupported Python implementation: {python_implementation}; expected CPython",
            "unsupported-python",
        )
    if tuple(detected_python) not in SUPPORTED_PYTHON_VERSIONS:
        raise SetupError(
            f"Unsupported Python {detected_python[0]}.{detected_python[1]}; "
            "expected CPython 3.11 or 3.12",
            "unsupported-python",
        )
    if config.platform_name and _normalize_system(config.platform_name) != detected_system:
        raise SetupError(
            "Modly platform metadata does not match the supplied Python host",
            "environment-mismatch",
        )
    if config.arch and _normalize_machine(config.arch) != detected_machine:
        raise SetupError(
            "Modly architecture metadata does not match the supplied Python host",
            "environment-mismatch",
        )
    raw_accelerator = (
        config.accelerator or ("cuda" if config.cuda_version > 0 else "unknown")
    ).strip().casefold()
    accelerator_aliases = {
        "cuda": "cuda",
        "nvidia": "cuda",
        "unknown": "unknown",
        "": "unknown",
        "cpu": "cpu",
        "none": "cpu",
    }
    accelerator = accelerator_aliases.get(raw_accelerator)
    if accelerator is None:
        raise SetupError(
            f"Unsupported accelerator metadata: {config.accelerator}",
            "unsupported-accelerator",
        )
    build = _torch_build(config.cuda_version)
    return {
        "id": (
            f"{detected_system}-{detected_machine}/"
            f"cp{detected_python[0]}{detected_python[1]}/{build}"
        ),
        "os": detected_system,
        "arch": detected_machine,
        "python": f"{detected_python[0]}.{detected_python[1]}",
        "python_tag": f"cp{detected_python[0]}{detected_python[1]}",
        "cuda": config.cuda_version,
        "torch_build": build,
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "gpu_sm": config.gpu_sm,
        "accelerator": accelerator,
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
        "Setup will use the checksum-pinned release asset when the audited "
        "local wheel is absent; it will not build FlashAttention from source.",
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
        command_text = " ".join(_redact(str(value)) for value in command)
        output_lines = [line.strip() for line in output.splitlines() if line.strip()]
        detail = _redact(output_lines[-1]) if output_lines else "no command output"
        raise SetupError(
            f"Command failed with exit code {result.returncode}: {command_text}. "
            f"Last output: {detail}. Full log: {logger.path}",
            "dependency-install-failed",
        )


def _venv_python(ext_dir: Path, system: str | None = None) -> Path:
    detected_system = _normalize_system(system or platform.system())
    relative = (
        Path("Scripts/python.exe")
        if detected_system == "windows"
        else Path("bin/python")
    )
    return ext_dir / "venv" / relative


def _venv_matches_profile(
    venv_python: Path,
    lane: Mapping[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> bool:
    try:
        probe = _probe_python_environment(venv_python, runner=runner)
    except SetupError:
        return False
    expected_python = tuple(int(part) for part in str(lane["python"]).split("."))
    return (
        probe["implementation"] == "cpython"
        and tuple(probe["version"]) == expected_python
        and probe["system"] == lane["os"]
        and probe["machine"] == lane["arch"]
    )


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


def _smoke_script(
    ext_dir: Path,
    *,
    require_tls_launcher: bool = False,
    flash_version: str | None = None,
    attention_backend: str = "sdpa",
    torch_build: str = "pypi",
    cuda_version: int = 0,
    accelerator: str = "unknown",
) -> str:
    vendor = ext_dir / "vendor" / "opensensenova_sensenova_vision"
    return f"""
import importlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

os.environ.update({{
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}})
if {require_tls_launcher!r}:
    glibc_tunables = os.environ.get("GLIBC_TUNABLES", "").split(":")
    assert glibc_tunables.count({OPTIONAL_STATIC_TLS_TUNABLE!r}) == 1

sys.path.insert(0, {str(vendor)!r})
open3d = importlib.import_module("open3d")
torch = importlib.import_module("torch")
torchvision = importlib.import_module("torchvision")
runtime = importlib.import_module("inference.sensenova_vision")
attention = importlib.import_module("modeling.bagel.attention_backend")

assert Path(runtime.__file__).resolve().is_relative_to(Path({str(vendor)!r}).resolve())
assert open3d.__version__ == {OPEN3D_VERSION!r}
torch_version = str(torch.__version__)
assert torch_version.split("+", 1)[0] == {TORCH_VERSION!r}
assert metadata.version("torchvision").split("+", 1)[0] == {TORCHVISION_VERSION!r}
assert torchvision.extension._has_ops(), "torchvision compiled operators are unavailable"
expected_attention_backend = {attention_backend!r}
expected_flash_version = {flash_version!r}
expected_torch_build = {torch_build!r}
expected_cuda_version = {cuda_version!r}
expected_accelerator = {accelerator!r}
expected_cuda_runtime = {TORCH_BUILD_CUDA_VERSIONS.get(torch_build)!r}
requires_cuda = expected_accelerator == "cuda" or expected_cuda_version > 0
cuda_available = bool(torch.cuda.is_available())
if expected_cuda_runtime is not None:
    assert torch.version.cuda == expected_cuda_runtime, (
        "Installed Torch CUDA runtime does not match the selected "
        f"{{expected_torch_build}} lane: {{torch.version.cuda!r}}"
    )
    assert f"+{{expected_torch_build}}" in torch_version, (
        "Installed Torch version does not carry the selected CUDA wheel suffix: "
        f"{{torch_version}}"
    )
if requires_cuda:
    assert torch.version.cuda is not None, "CUDA lane requires a CUDA-enabled Torch build"
    assert cuda_available, "CUDA lane requires torch.cuda.is_available()"
assert attention.get_attention_backend() == expected_attention_backend
if expected_flash_version is not None:
    assert metadata.version("flash-attn") == expected_flash_version

# Exercise the extension-side packed SDPA path even on a host without CUDA.
if expected_attention_backend == "sdpa":
    torch.manual_seed(0)
    sdpa_device = "cuda" if cuda_available else "cpu"
    sdpa_dtype = torch.bfloat16 if sdpa_device == "cuda" else torch.float32
    sdpa_q = torch.randn((5, 2, 8), dtype=sdpa_dtype, device=sdpa_device)
    sdpa_k = torch.randn((7, 1, 8), dtype=sdpa_dtype, device=sdpa_device)
    sdpa_v = torch.randn((7, 1, 8), dtype=sdpa_dtype, device=sdpa_device)
    sdpa_cu_q = torch.tensor([0, 2, 5], dtype=torch.int32, device=sdpa_device)
    sdpa_cu_k = torch.tensor([0, 3, 7], dtype=torch.int32, device=sdpa_device)
    sdpa_output = attention.varlen_attention(
        q=sdpa_q,
        k=sdpa_k,
        v=sdpa_v,
        cu_seqlens_q=sdpa_cu_q,
        cu_seqlens_k=sdpa_cu_k,
        max_seqlen_q=3,
        max_seqlen_k=4,
        causal=False,
        backend="sdpa",
    )
    assert sdpa_output.shape == sdpa_q.shape
    assert bool(torch.isfinite(sdpa_output).all().item())

    # Exercise FlashAttention's bottom-right causal alignment through the
    # extension fallback with unequal packed query/key lengths.
    causal_q = torch.ones((2, 1, 1), dtype=sdpa_dtype, device=sdpa_device)
    causal_k = torch.arange(1, 5, dtype=sdpa_dtype, device=sdpa_device).reshape(4, 1, 1)
    causal_v = (10 * torch.arange(1, 5, dtype=sdpa_dtype, device=sdpa_device)).reshape(4, 1, 1)
    causal_cu_q = torch.tensor([0, 2], dtype=torch.int32, device=sdpa_device)
    causal_cu_k = torch.tensor([0, 4], dtype=torch.int32, device=sdpa_device)
    causal_output = attention.varlen_attention(
        q=causal_q,
        k=causal_k,
        v=causal_v,
        cu_seqlens_q=causal_cu_q,
        cu_seqlens_k=causal_cu_k,
        max_seqlen_q=2,
        max_seqlen_k=4,
        causal=True,
        backend="sdpa",
    )
    causal_allowed = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
        dtype=torch.bool,
        device=sdpa_device,
    )
    causal_reference = torch.nn.functional.scaled_dot_product_attention(
        causal_q.transpose(0, 1).unsqueeze(0),
        causal_k.transpose(0, 1).unsqueeze(0),
        causal_v.transpose(0, 1).unsqueeze(0),
        attn_mask=causal_allowed.unsqueeze(0).unsqueeze(0),
        is_causal=False,
    ).squeeze(0).transpose(0, 1)
    assert causal_output.shape == causal_q.shape
    assert torch.allclose(causal_output, causal_reference, rtol=1e-2, atol=1e-2)

capability = None
dtype = None
kernel_checks = {{"status": "not-run"}}
if cuda_available and expected_attention_backend == "flash-attn":
    capability = list(torch.cuda.get_device_capability(0))
    dtype = (
        torch.bfloat16
        if capability[0] >= 8 and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    from flash_attn import flash_attn_varlen_func
    torch.manual_seed(0)
    q = torch.randn((4, 2, 64), dtype=dtype, device="cuda")
    k = torch.randn((4, 2, 64), dtype=dtype, device="cuda")
    v = torch.randn((4, 2, 64), dtype=dtype, device="cuda")
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
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
        assert output.dtype == dtype
        assert bool(torch.isfinite(output).all().item())
        kernel_checks["causal" if causal else "noncausal"] = True
    torch.cuda.synchronize()
print(json.dumps({{
    "open3d": open3d.__version__,
    "torch": str(torch.__version__),
    "torchvision": metadata.version("torchvision"),
    "cuda": torch.version.cuda,
    "cuda_available": cuda_available,
    "gpu_capability": capability,
    "flash_dtype": str(dtype) if dtype is not None else None,
    "flash_kernel": kernel_checks,
    "flash_attn": (
        metadata.version("flash-attn") if expected_attention_backend == "flash-attn" else None
    ),
    "attention_backend": expected_attention_backend,
    "runtime": str(Path(runtime.__file__).resolve()),
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
    python_implementation: str = "cpython",
) -> dict[str, Any]:
    if not _same_directory(config.ext_dir, SCRIPT_DIR):
        raise SetupError(
            f"Setup ext_dir must be this extension directory: {SCRIPT_DIR}",
            "invalid-extension-directory",
        )
    if not config.python_exe.is_file():
        raise SetupError(
            f"Python executable does not exist: {config.python_exe}",
            "invalid-python",
        )
    if system is None and machine is None and python_version is None:
        probe = _probe_python_environment(config.python_exe, runner=runner)
        system = str(probe["system"])
        machine = str(probe["machine"])
        python_version = tuple(probe["version"])
        python_implementation = str(probe["implementation"])
    lane = validate_lane(
        config,
        system=system,
        machine=machine,
        python_version=python_version,
        python_implementation=python_implementation,
    )
    missing = [
        str(relative)
        for relative in REQUIRED_EXTENSION_FILES
        if not (config.ext_dir / relative).is_file()
    ]
    if missing:
        raise SetupError(
            "Extension source is incomplete: " + ", ".join(missing),
            "extension-source-missing",
        )

    logger.info(f"Detected setup profile: {lane['id']}")
    exact_arm_lane = (
        lane["os"] == "linux"
        and lane["arch"] == "aarch64"
        and lane["python_tag"] == "cp312"
    )
    exact_flash_lane = (
        exact_arm_lane
        and lane["torch_build"] == "cu128"
        and config.gpu_sm == 121
    )

    open3d_asset: WheelAsset | None = None
    if exact_arm_lane:
        try:
            open3d_asset = resolve_open3d_asset(config.ext_dir)
        except SetupError as exc:
            logger.info(
                "Optional verified Open3D wheel is unavailable; "
                f"falling back to pip: {exc}"
            )
    open3d_requirement = (
        str(open3d_asset.path)
        if open3d_asset is not None
        else f"open3d=={OPEN3D_VERSION}"
    )

    flash_asset: WheelAsset | None = None
    flash_requirement: str | None = None
    flash_source = "not-installed"
    flash_version: str | None = None
    attention_backend = "sdpa"
    if exact_flash_lane:
        try:
            flash_asset = resolve_flash_attn_asset(config.ext_dir)
        except SetupError as exc:
            logger.info(
                "Optional local FlashAttention wheel is unavailable; "
                f"using the checksum-pinned release asset: {exc}"
            )
        if flash_asset is not None:
            flash_requirement = str(flash_asset.path)
            flash_source = "verified-local-wheel"
        else:
            flash_requirement = FLASH_ATTN_ARM64_RELEASE_URL
            flash_source = "verified-release-wheel"
        flash_version = FLASH_ATTN_VERSION
        attention_backend = "flash-attn"
    else:
        logger.info(
            "No verified FlashAttention wheel matches this lane; "
            "using the extension-side PyTorch SDPA backend"
        )

    venv_dir = config.ext_dir / "venv"
    venv_python = _venv_python(config.ext_dir, lane["os"])
    if venv_dir.exists() and (
        not venv_python.is_file()
        or not _venv_matches_profile(venv_python, lane, runner=runner)
    ):
        raise SetupError(
            "Existing extension venv does not match the requested profile; "
            "run setup through its main entry point for transactional replacement",
            "venv-profile-mismatch",
        )
    if not venv_python.is_file():
        _run_command(
            [str(config.python_exe), "-m", "venv", str(venv_dir)],
            logger,
            runner=runner,
        )
    if not venv_python.is_file():
        raise SetupError(
            f"Virtual environment creation did not produce {venv_python}",
            "venv-creation-failed",
        )

    require_tls_launcher = open3d_asset is not None and lane["os"] == "linux"
    if require_tls_launcher:
        _install_python_launcher(venv_python)
        logger.info("Enabled the verified ARM Open3D static-TLS launcher")

    pip = [str(venv_python), "-m", "pip", "install", "--no-input"]
    _run_command([*pip, "--upgrade", *BOOTSTRAP_REQUIREMENTS], logger, runner=runner)

    torch_command = [
        *pip,
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
    ]
    torch_index = TORCH_INDEX_URLS.get(lane["torch_build"])
    if torch_index is not None:
        torch_command.extend(["--index-url", torch_index])
    _run_command(torch_command, logger, runner=runner)

    _run_command([*pip, *RUNTIME_REQUIREMENTS], logger, runner=runner)
    _run_command([*pip, open3d_requirement], logger, runner=runner)

    if flash_requirement is not None:
        _run_command(
            [
                *pip,
                "--no-deps",
                flash_requirement,
            ],
            logger,
            runner=runner,
        )
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
    def run_smoke() -> None:
        _run_command(
            [
                str(venv_python),
                "-B",
                "-c",
                _smoke_script(
                    config.ext_dir,
                    require_tls_launcher=require_tls_launcher,
                    flash_version=flash_version,
                    attention_backend=attention_backend,
                    torch_build=lane["torch_build"],
                    cuda_version=config.cuda_version,
                    accelerator=lane["accelerator"],
                ),
            ],
            logger,
            runner=runner,
            env=smoke_env,
        )

    run_smoke()
    return {
        "lane": lane,
        "venv_python": str(venv_python),
        "open3d": {
            "source": (
                "verified-local-wheel"
                if open3d_asset is not None
                else "pip"
            ),
            "requirement": open3d_requirement,
            "sha256": open3d_asset.sha256 if open3d_asset is not None else None,
            "version": OPEN3D_VERSION,
        },
        "flash_attn": {
            "source": flash_source,
            "requirement": flash_requirement,
            "sha256": (
                flash_asset.sha256
                if flash_source == "verified-local-wheel" and flash_asset is not None
                else None
            ),
            "version": flash_version,
        },
        "attention_backend": {
            "selected": attention_backend,
            "fallback": "torch.nn.functional.scaled_dot_product_attention",
            "flash_attn_available": flash_requirement is not None,
        },
        "model_weights_downloaded": False,
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
    if not _same_directory(config.ext_dir, SCRIPT_DIR):
        print(
            f"[setup:{EXTENSION_ID}][error] invalid-extension-directory: "
            f"ext_dir must be {SCRIPT_DIR}",
            file=sys.stderr,
            flush=True,
        )
        return 2


    started_at = _utc_now()
    logger = SetupLogger(config.ext_dir / LOG_RELATIVE_PATH)
    _write_status(config, status="running", started_at=started_at)
    transaction = VenvTransaction(config.ext_dir, logger)
    try:
        logger.info("Preparing dependencies only; model weights remain UI-managed")
        transaction.begin()
        result = run_setup(config, logger)
        _write_status(
            config,
            status="ready",
            started_at=started_at,
            result=result,
        )
        logger.info("SenseNova extension runtime is ready")
        transaction.commit()
        return 0
    except SetupError as exc:
        rollback_error = transaction.rollback()
        if rollback_error is not None:
            exc = SetupError(
                f"{exc}; rollback failed: {rollback_error}",
                "venv-rollback-failed",
            )
        _write_status(
            config,
            status="blocked",
            started_at=started_at,
            error=exc,
        )
        logger.error(f"{exc.code}: {exc}")
        return 1
    except Exception as exc:
        rollback_error = transaction.rollback()
        rollback_detail = (
            f"; rollback failed: {rollback_error}" if rollback_error else ""
        )
        wrapped = SetupError(
            f"Unexpected setup failure: {exc.__class__.__name__}: {exc}{rollback_detail}",
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
