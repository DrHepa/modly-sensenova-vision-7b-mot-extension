from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

# Keep the repository test suite self-contained. The extension's only host-side
# imports are the established model base class and cancellation exception.
services = types.ModuleType("services")
services.__path__ = []
generators = types.ModuleType("services.generators")
generators.__path__ = []
base = types.ModuleType("services.generators.base")


class GenerationCancelled(Exception):
    pass


class BaseGenerator:
    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.outputs_dir = Path(outputs_dir)
        self._model = None
        self.model_id = ""
        self.node_id = ""

    def _check_cancelled(self, cancel_event: Optional[threading.Event]) -> None:
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()


base.BaseGenerator = BaseGenerator
base.GenerationCancelled = GenerationCancelled
sys.modules["services"] = services
sys.modules["services.generators"] = generators
sys.modules["services.generators.base"] = base
