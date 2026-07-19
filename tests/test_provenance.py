from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/opensensenova_sensenova_vision"
FLASH_SHA256 = "8b7c2138cce6ff36f13051d3d83ac688ed0f997c629e5c1e140c120b552a9fc6"
NON_DISTRIBUTED_DIRS = {".git", ".modly", ".pytest_cache", "__pycache__", "venv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_vendored_runtime_closure_matches_every_recorded_hash() -> None:
    provenance = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["upstream"]["commit"] == (
        "beea1f771b8192c597085c0e668f4430d2089d8d"
    )
    assert provenance["upstream"]["license"] == "Apache-2.0"
    records = provenance["files"]
    assert len(records) == 27
    for record in records:
        path = VENDOR / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["size_bytes"]
        assert sha256(path) == record["sha256"]


def test_vendored_tree_contains_only_the_declared_closure_and_provenance() -> None:
    provenance = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))
    declared = {record["path"] for record in provenance["files"]}
    actual = {
        path.relative_to(VENDOR).as_posix()
        for path in VENDOR.rglob("*")
        if (
            path.is_file()
            and path.name != "PROVENANCE.json"
            and "__pycache__" not in path.parts
        )
    }
    assert actual == declared


def test_open3d_wheel_hash_matches_reuse_manifest() -> None:
    manifest = json.loads(
        (ROOT / "open3d-wheelhouse.manifest.json").read_text(encoding="utf-8")
    )
    asset = manifest["assets"][0]
    wheel = ROOT / asset["filename"]
    assert wheel.stat().st_size == asset["size_bytes"]
    assert sha256(wheel) == asset["sha256"]
    assert asset["sha256"] == manifest["reused_from"]["asset_sha256"]


def test_flash_attention_wheel_and_compatibility_evidence_match_manifest() -> None:
    manifest = json.loads(
        (ROOT / "flash-attn-wheelhouse.manifest.json").read_text(encoding="utf-8")
    )
    asset = manifest["assets"][0]
    wheel = ROOT / asset["filename"]

    assert manifest["status"] == "verified-local-compatibility-adaptation"
    assert manifest["official_upstream_reference"]["requirement"] == "flash-attn==2.6.3"
    assert manifest["selected_runtime_adaptation"]["requirement"] == "flash-attn==2.7.3"
    assert asset["version"] == "2.7.3"
    assert asset["size_bytes"] == 406_023_844
    assert asset["sha256"] == FLASH_SHA256
    assert wheel.stat().st_size == asset["size_bytes"]
    assert sha256(wheel) == asset["sha256"]
    assert asset["selectors"] == {
        "os": "linux",
        "arch": "aarch64",
        "python_tag": "cp312",
        "torch": "2.7.0+cu128",
        "cuda": "12.8",
        "gpu_sm": 121,
    }
    assert asset["verification_evidence"]["import_source"] == "extracted wheel contents"
    assert asset["verification_evidence"]["excluded_installed_version"] == "flash-attn 2.8.3"
    assert asset["verification_evidence"]["modes"] == ["causal", "noncausal"]


def test_flash_attention_licenses_match_embedded_wheel_bytes() -> None:
    manifest = json.loads(
        (ROOT / "flash-attn-wheelhouse.manifest.json").read_text(encoding="utf-8")
    )
    asset = manifest["assets"][0]
    wheel = ROOT / asset["filename"]
    embedded = {
        "LICENSE": "flash_attn-2.7.3.dist-info/licenses/LICENSE",
        "AUTHORS": "flash_attn-2.7.3.dist-info/licenses/AUTHORS",
    }

    with zipfile.ZipFile(wheel) as archive:
        for record in asset["license_files"]:
            local = ROOT / record["path"]
            name = local.name
            assert local.is_file()
            assert sha256(local) == record["sha256"]
            assert local.read_bytes() == archive.read(embedded[name])


def redistributed_model_weight_paths(root: Path) -> list[str]:
    forbidden_suffixes = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
    offenders = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if NON_DISTRIBUTED_DIRS.intersection(relative.parts):
            continue
        if path.is_file() and path.suffix.casefold() in forbidden_suffixes:
            offenders.append(relative.as_posix())
    return offenders


def test_repository_does_not_redistribute_model_weights() -> None:
    assert redistributed_model_weight_paths(ROOT) == []


def test_weight_scan_ignores_installed_runtime_state_but_not_payload(
    tmp_path: Path,
) -> None:
    runtime_metadata = tmp_path / "venv/lib/python3.12/site-packages"
    runtime_metadata.mkdir(parents=True)
    (runtime_metadata / "distutils-precedence.pth").write_text(
        "import _distutils_hack\n",
        encoding="utf-8",
    )
    assert redistributed_model_weight_paths(tmp_path) == []

    (tmp_path / "accidental-model.safetensors").write_bytes(b"not-a-real-weight")
    assert redistributed_model_weight_paths(tmp_path) == [
        "accidental-model.safetensors"
    ]
