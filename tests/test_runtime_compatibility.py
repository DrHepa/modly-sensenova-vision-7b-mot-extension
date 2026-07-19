from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import types
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "generator.py"
SETUP_PATH = ROOT / "setup.py"
FORBIDDEN_HOST_DTOS = {
    "GenerationRequest",
    "GenerationOutput",
    "GenerationResource",
    "GenerationResult",
}


def _load_generator(monkeypatch: pytest.MonkeyPatch):
    services = types.ModuleType("services")
    services.__path__ = []
    generators = types.ModuleType("services.generators")
    generators.__path__ = []
    base = types.ModuleType("services.generators.base")

    class BaseGenerator:
        pass

    class GenerationCancelled(Exception):
        pass

    base.BaseGenerator = BaseGenerator
    base.GenerationCancelled = GenerationCancelled
    monkeypatch.setitem(sys.modules, "services", services)
    monkeypatch.setitem(sys.modules, "services.generators", generators)
    monkeypatch.setitem(sys.modules, "services.generators.base", base)

    module_name = f"_sensenova_generator_contract_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, BaseGenerator, GenerationCancelled


def test_setup_pins_httpx_for_the_extension_runtime() -> None:
    tree = ast.parse(SETUP_PATH.read_text(encoding="utf-8"))
    requirements = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "RUNTIME_REQUIREMENTS"
            for target in node.targets
        ):
            requirements = ast.literal_eval(node.value)
            break

    assert requirements is not None
    assert [item for item in requirements if item.startswith("httpx")] == [
        "httpx==0.28.1"
    ]


def test_generator_imports_only_the_established_host_base_contract(monkeypatch) -> None:
    module, base_class, cancelled_class = _load_generator(monkeypatch)

    assert issubclass(module.SenseNovaVisionGenerator, base_class)
    assert module.GenerationCancelled is cancelled_class
    assert FORBIDDEN_HOST_DTOS.isdisjoint(vars(module))

    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "services.generators.base"
    ]
    assert len(imports) == 1
    assert {alias.name for alias in imports[0].names} == {
        "BaseGenerator",
        "GenerationCancelled",
    }


def test_generator_exposes_only_direct_model_generation_entrypoint(monkeypatch) -> None:
    module, _base_class, _cancelled_class = _load_generator(monkeypatch)
    generator_class = module.SenseNovaVisionGenerator

    assert "generate_v2" not in vars(generator_class)
    signature = inspect.signature(generator_class.generate)
    assert list(signature.parameters) == [
        "self",
        "image_bytes",
        "params",
        "progress_cb",
        "cancel_event",
    ]
    assert signature.parameters["progress_cb"].default is None
    assert signature.parameters["cancel_event"].default is None


def test_no_custom_io_contract_or_plural_route_seam_remains() -> None:
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    lowered = source.casefold()
    assert "named-v1" not in lowered
    assert "io_contract" not in lowered
    assert "generate_v2" not in lowered
    assert "from-images" not in lowered
    assert "/generate/from-images" not in lowered
    for name in FORBIDDEN_HOST_DTOS:
        assert name not in source
