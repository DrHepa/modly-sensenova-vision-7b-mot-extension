"""Modly adapter for the pinned SenseNova-Vision-7B-MoT runtime."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import struct
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from services.generators.base import BaseGenerator, GenerationCancelled


@dataclass(frozen=True)
class _GenerationTask:
    node_id: str
    images: tuple[bytes, ...] = ()
    image_names: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None
    outputs_dir: Optional[Path] = None


@dataclass(frozen=True)
class _GeneratedArtifact:
    source_handle: str
    kind: str
    path: Optional[Path] = None
    text: Optional[str] = None
    display_name: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SidecarResource:
    name: str
    path: Path
    media_type: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _DispatchResult:
    outputs: Sequence[_GeneratedArtifact]
    resources: Sequence[_SidecarResource] = ()


HF_REPOSITORY = "sensenova/SenseNova-Vision-7B-MoT"
HF_REVISION = "79548fcc5b954598799b9317f8d3ec5e347d5c0e"
UPSTREAM_COMMIT = "beea1f771b8192c597085c0e668f4430d2089d8d"

EXTENSION_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = EXTENSION_ROOT / "vendor" / "opensensenova_sensenova_vision"
VENDOR_PROVENANCE = VENDOR_ROOT / "PROVENANCE.json"

REQUIRED_CHECKPOINT_FILES = (
    "ae.safetensors",
    "ema.safetensors",
    "config.json",
    "generation_config.json",
    "llm_config.json",
    "vit_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
)

VISUAL_PROMPT_SEGMENTATION_PROMPT = (
    "<image>Can you provide segmentation masks for this image based on these "
    "regions: <box><image>? Please provide the segmentation masks."
)
CAPTIONED_SEGMENTATION_PROMPT = (
    "Please briefly describe the contents of the image. Please respond with "
    "interleaved segmentation masks for the corresponding parts of the answer."
)
DEPTH_PROMPT = (
    "Estimate relative depth for each pixel in the image, with closer objects "
    "appearing brighter and distant objects appearing darker. Output is a "
    "grayscale image with pixel values ranging from 0-255."
)
NORMAL_PROMPT = (
    "Estimate surface normals and encode as an RGB image. Each channel "
    "corresponds to a direction component (X, Y, Z) with continuous value "
    "variations, creating smooth color gradients distinct from other task outputs."
)
OCR_PROMPT = (
    "Perform word-level text detection and recognition on the entire image. "
    "Output a structured text list containing every detected word, its bounding "
    "box coordinates with <bbox> format, and the recognized text content."
)
BBOX_DETECTION_PROMPT_TEMPLATE = (
    "Detect all instances of {categories} in the image. Output the results "
    "as a structured text list with each detection including category and "
    "bounding box coordinates in <bbox> format."
)
POINT_DETECTION_PROMPT_TEMPLATE = (
    "Locate and identify {categories} within the scene. Output detection "
    "results as text entries, each containing the object class and pixel "
    "coordinates defining the object point location."
)
KEYPOINT_PROMPT_TEMPLATE = {
    "human": (
        "Detect all instances of {categories} in the image. For each instance, "
        "output a bounding box in <bbox> format and the coordinates of its "
        "nose, left eye, right eye, left ear, right ear, left shoulder, right "
        "shoulder, left elbow, right elbow, left wrist, right wrist, left hip, "
        "right hip, left knee, right knee, left ankle, right ankle in "
        "<kpt>[x,y]</kpt> format. Return results as a structured list."
    ),
    "animal": (
        "Detect all instances of {categories} in the image. For each instance, "
        "output a bounding box in <bbox> format and the coordinates of its "
        "left eye, right eye, nose, neck, root of tail, left shoulder, left "
        "elbow, left front paw, right shoulder, right elbow, right front paw, "
        "left hip, left knee, left back paw, right hip, right knee, right back "
        "paw in <kpt>[x,y]</kpt> format. Return results as a structured list."
    ),
    "mixed": (
        "Detect all instances of {categories} in the image. For human "
        "instances, output a bounding box in <bbox> format and the COCO human "
        "keypoints: nose, left eye, right eye, left ear, right ear, left "
        "shoulder, right shoulder, left elbow, right elbow, left wrist, right "
        "wrist, left hip, right hip, left knee, right knee, left ankle, right "
        "ankle. For animal instances, output a bounding box in <bbox> format "
        "and the AP-10K animal keypoints: left eye, right eye, nose, neck, "
        "root of tail, left shoulder, left elbow, left front paw, right "
        "shoulder, right elbow, right front paw, left hip, left knee, left back "
        "paw, right hip, right knee, right back paw. Use <kpt>[x,y]</kpt> "
        "format for every keypoint and return results as a structured list."
    ),
}
HUMAN_KEYPOINT_CATEGORIES = frozenset({
    "human",
    "person",
    "people",
    "pedestrian",
})
MASK_QUESTION = (
    "Can you segment the image based on the following categories: {categories}? "
    "Please output the {task_type} segmentation masks."
)
PANOPTIC_CAPTION_QUESTION = (
    "Please find all instances in the image and assign color to each instance "
    "in the EXACT format: <p>instance-no<color>(R,G,B)</color></p>, then "
    "respond with {task_type} segmentation masks."
)

# Exact category-name order from upstream data/prompts.py::COCO_CATEGORIES at
# UPSTREAM_COMMIT. Panoptic prompts normalize the three official suffix forms.
COCO_CATEGORY_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush", "banner",
    "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff",
    "floor-wood", "flower", "fruit", "gravel", "house", "light",
    "mirror-stuff", "net", "pillow", "platform", "playingfield",
    "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow",
    "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile",
    "wall-wood", "water-other", "window-blind", "window-other",
    "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged",
    "cabinet-merged", "table-merged", "floor-other-merged",
    "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged",
    "paper-merged", "food-other-merged", "building-other-merged",
    "rock-merged", "wall-other-merged", "rug-merged",
)
RECON3D_PROMPT = (
    "Reconstruct a scene from multiple input images and output one dense 3D "
    "coordinate map per view, all aligned to the first camera's perspective."
)
CAMERA_POSE_PROMPT = (
    "With the first frame as the reference frame, output the relative pose of"
    " all subsequent frames (excluding the first frame) with respect to the"
    " first frame, following the input order and adhering to the strict format"
    " below:Rotation: Represented by a quaternion in the format"
    " <quat>[x,y,z,w], enclosed in <quat> tags;Translation: Represented by a"
    " unit vector (direction) in the format <offset>[x,y,z], enclosed in"
    " <offset> tags (the vector has no absolute physical meaning, only"
    " directional information);Scale: Represented by a numerical value in the"
    " format <scale>value</scale> tags, where the value denotes the magnitude"
    " of translation (corresponding to the length of the translation unit"
    " vector);Enclose the result of each frame in <frame> tags, with no extra"
    " characters, spaces, or line breaks outside the tags."
)

NODE_IDS = frozenset({
    "open-image-generation",
    "image-to-image-vision",
    "visual-prompt-segmentation",
    "captioned-segmentation",
    "structured-vision",
    "multi-view-reconstruction",
    "camera-pose",
    "relative-depth-estimation",
    "surface-normal-estimation",
    "binary-segmentation",
    "panoptic-segmentation",
    "bounding-box-detection",
    "point-detection",
    "keypoint-detection",
    "ocr",
})


class _ProgressReporter:
    def __init__(self, callback: Optional[Callable[[int, str], None]]) -> None:
        self._callback = callback
        self._last = -1

    def emit(self, percent: int, label: str) -> None:
        value = max(0, min(100, int(percent)))
        if value < self._last:
            raise RuntimeError("Internal progress must be monotonic")
        self._last = value
        if self._callback is not None:
            self._callback(value, label)


def _model_question(prompt: str, image_count: int) -> str:
    """Match the official inference_demo.model_question implementation."""
    if image_count <= 0:
        return prompt
    return f'{("<image>" * image_count)} {prompt}'.strip()


def _format_categories(query: str) -> str:
    """Match inference_demo._format_categories at the pinned upstream commit."""
    text = query.strip()
    if not text:
        raise ValueError("Category tasks require explicit categories")
    if "<p>" in text and "</p>" in text:
        return text
    categories = [item.strip() for item in text.split(",") if item.strip()]
    if not categories:
        raise ValueError("Category tasks require explicit categories")
    return ", ".join(f"<p>{category}</p>" for category in categories)


def _extract_category_names(categories: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(
        r"<p>(.*?)</p>",
        categories,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        name = " ".join(match.group(1).split()).lower()
        if name:
            names.append(name)
    return names


def _prompt_category_name(name: str) -> str:
    for suffix in ("-stuff", "-other", "-merged"):
        name = name.replace(suffix, "")
    return name


def _format_coco_categories() -> str:
    return ", ".join(
        f"<p>{_prompt_category_name(category)}</p>"
        for category in COCO_CATEGORY_NAMES
    )


def _binary_seg_prompt(query: str) -> str:
    return MASK_QUESTION.format(
        categories=_format_categories(query),
        task_type="binary",
    )


def _panoptic_prompt(query: str) -> str:
    categories = _format_categories(query) if query.strip() else _format_coco_categories()
    category_prompt = MASK_QUESTION.format(
        categories=categories,
        task_type="panoptic",
    )
    caption_prompt = PANOPTIC_CAPTION_QUESTION.format(task_type="panoptic")
    return f"{category_prompt} {caption_prompt}"


def _category_prompt(template: str, query: str) -> str:
    return template.format(categories=_format_categories(query))


def _keypoint_prompt(query: str) -> str:
    categories = _format_categories(query)
    category_names = _extract_category_names(categories)
    has_human = any(name in HUMAN_KEYPOINT_CATEGORIES for name in category_names)
    has_animal = any(name not in HUMAN_KEYPOINT_CATEGORIES for name in category_names)
    if has_human and has_animal:
        keypoint_type = "mixed"
    elif has_human:
        keypoint_type = "human"
    else:
        keypoint_type = "animal"
    return KEYPOINT_PROMPT_TEMPLATE[keypoint_type].format(categories=categories)


def _safe_bool(params: dict[str, Any], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"Parameter '{key}' must be a boolean")
    return value


def _safe_int(
    params: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = params.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Parameter '{key}' must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(
            f"Parameter '{key}' must be between {minimum} and {maximum}"
        )
    return value


def _safe_float(
    params: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = params.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Parameter '{key}' must be numeric")
    number = float(value)
    if number < minimum or number > maximum:
        raise ValueError(
            f"Parameter '{key}' must be between {minimum} and {maximum}"
        )
    return number


class SenseNovaVisionGenerator(BaseGenerator):
    """One shared, offline-only SenseNova runtime serving fifteen Modly nodes."""

    MODEL_ID = "sensenova-vision-7b-mot"
    DISPLAY_NAME = "SenseNova Vision 7B MoT"
    VRAM_GB = 40

    def load(self) -> None:
        if self._model is not None:
            return

        checkpoint_dir = self._validate_checkpoint_before_imports()
        self._verify_vendored_source()
        self._configure_offline_environment(checkpoint_dir)

        vendor_path = str(VENDOR_ROOT)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

        try:
            # This audited Open3D wheel is built without OpenMP. Importing it
            # before Torch also avoids exhausting aarch64 static TLS slots.
            with contextlib.redirect_stdout(sys.stderr):
                open3d = importlib.import_module("open3d")
                torch = importlib.import_module("torch")
                runtime = importlib.import_module("inference.sensenova_vision")
                pose_parser = importlib.import_module(
                    "inference.utils_3d.camera_pose_parser"
                )
        except Exception as exc:
            raise RuntimeError(
                "SenseNova runtime imports failed. Run this extension's setup.py "
                "and inspect .modly/setup/logs/setup.log."
            ) from exc

        runtime_path = Path(runtime.__file__).resolve()
        if VENDOR_ROOT.resolve() not in runtime_path.parents:
            raise RuntimeError(
                "SenseNova runtime resolved outside the pinned vendored source tree"
            )
        if getattr(open3d, "__version__", None) != "0.19.0":
            raise RuntimeError("SenseNova requires the audited Open3D 0.19.0 wheel")
        if not torch.cuda.is_available():
            raise RuntimeError("SenseNova-Vision-7B-MoT requires a CUDA GPU")
        if not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            raise RuntimeError("SenseNova-Vision-7B-MoT requires CUDA BF16 support")

        offload_dir = self.model_dir / ".modly-runtime-offload"
        offload_dir.mkdir(parents=True, exist_ok=True)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                model = runtime.SenseNovaVisionModel(
                    model_path=str(checkpoint_dir),
                    checkpoint_name="ema.safetensors",
                    dtype="bf16",
                    device="cuda",
                    max_mem_per_gpu="80GiB",
                    offload_folder=str(offload_dir),
                    download_revision=HF_REVISION,
                    download_local_files_only=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the local SenseNova checkpoint in CUDA BF16 mode. "
                "No network fallback was attempted."
            ) from exc

        self._runtime_module = runtime
        self._pose_parser = pose_parser.resolve_pose_string
        self._model = model

    def unload(self) -> None:
        self._model = None
        self._runtime_module = None
        self._pose_parser = None
        import gc

        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Any = None,
    ) -> Path:
        """Run one established Modly model-node request and return its primary artifact."""
        node_id = self._normalized_node_id(
            getattr(self, "node_id", "") or getattr(self, "model_id", "")
        )
        if node_id not in NODE_IDS:
            raise ValueError(f"Unsupported SenseNova node: {node_id or '<unset>'}")
        if not isinstance(params, dict):
            raise TypeError("SenseNova generation parameters must be a dictionary")

        images, image_names = self._legacy_inputs(node_id, image_bytes, params)
        task = _GenerationTask(
            node_id=node_id,
            images=images,
            image_names=image_names,
            params=dict(params),
            prompt=params.get("prompt"),
            outputs_dir=self.outputs_dir,
        )
        result, run_dir = self._execute_task(task, progress_cb, cancel_event)
        return self._finalize_primary_output(result, run_dir)

    def _legacy_inputs(
        self,
        node_id: str,
        image_bytes: bytes,
        params: dict[str, Any],
    ) -> tuple[tuple[bytes, ...], tuple[str, ...]]:
        if node_id == "open-image-generation":
            if image_bytes:
                raise ValueError("open-image-generation does not accept an image input")
            return (), ()

        if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or not image_bytes:
            raise ValueError(f"{node_id} requires a primary image input")
        primary = bytes(image_bytes)

        if node_id == "visual-prompt-segmentation":
            prompt_image = self._read_supplemental_image(
                params,
                "prompt_image_path",
                required=True,
            )
            return (primary, prompt_image), ("scene", "prompt")

        if node_id in {"multi-view-reconstruction", "camera-pose"}:
            supplemental = self._ordered_view_images(
                params,
                require_view_2=node_id == "camera-pose",
            )
            return (primary, *supplemental), ("views",) * (1 + len(supplemental))

        return (primary,), ("image",)

    @classmethod
    def _ordered_view_images(
        cls,
        params: dict[str, Any],
        *,
        require_view_2: bool,
    ) -> tuple[bytes, ...]:
        values: list[bytes] = []
        first_gap: Optional[int] = None
        for index in range(2, 11):
            key = f"view_{index}_image_path"
            value = params.get(key, "")
            if not isinstance(value, str):
                raise TypeError(f"Parameter '{key}' must be an image path string")
            if not value.strip():
                if index == 2 and require_view_2:
                    raise ValueError("camera-pose requires view_2_image_path")
                first_gap = first_gap or index
                continue
            if first_gap is not None:
                raise ValueError(
                    "Supplemental view paths must be sequential without gaps: "
                    f"view_{index}_image_path is set after missing "
                    f"view_{first_gap}_image_path"
                )
            values.append(cls._read_image_path(value, key))
        return tuple(values)

    @classmethod
    def _read_supplemental_image(
        cls,
        params: dict[str, Any],
        key: str,
        *,
        required: bool,
    ) -> bytes:
        value = params.get(key, "")
        if not isinstance(value, str):
            raise TypeError(f"Parameter '{key}' must be an image path string")
        if not value.strip():
            if required:
                raise ValueError(f"Parameter '{key}' requires an image selected in Modly")
            return b""
        return cls._read_image_path(value, key)

    @staticmethod
    def _read_image_path(value: str, key: str) -> bytes:
        if "\x00" in value:
            raise ValueError(f"Parameter '{key}' contains an invalid image path")
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Parameter '{key}' does not point to an existing image") from exc
        if not path.is_file():
            raise ValueError(f"Parameter '{key}' must point to an image file")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Parameter '{key}' image could not be read") from exc
        if not payload:
            raise ValueError(f"Parameter '{key}' image is empty")
        return payload

    def _execute_task(
        self,
        task: _GenerationTask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Any = None,
    ) -> tuple[_DispatchResult, Path]:
        node_id = self._normalized_node_id(task.node_id)
        if node_id not in NODE_IDS:
            raise ValueError(f"Unsupported SenseNova node: {task.node_id}")
        if self._model is None:
            self.load()

        progress = _ProgressReporter(progress_cb)
        self._check_cancelled(cancel_event)
        progress.emit(0, "Preparing SenseNova request")

        run_dir = self._create_request_dir(task, node_id)
        try:
            records = self._materialize_images(task, run_dir)
            self._check_cancelled(cancel_event)
            progress.emit(10, "Validated ordered inputs")
            result = self._dispatch(
                node_id,
                task,
                records,
                run_dir,
                progress,
                cancel_event,
            )
            self._check_cancelled(cancel_event)
            progress.emit(100, "SenseNova generation complete")
            return result, run_dir
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(run_dir / ".inputs", ignore_errors=True)

    @staticmethod
    def _sidecar_path(run_dir: Path, source_handle: str, suffix: str) -> Path:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source_handle).strip(".-") or "output"
        candidate = run_dir / f"{stem}{suffix}"
        counter = 2
        while candidate.exists():
            candidate = run_dir / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    @classmethod
    def _finalize_primary_output(
        cls,
        result: _DispatchResult,
        run_dir: Path,
    ) -> Path:
        if not result.outputs:
            raise RuntimeError("SenseNova dispatch returned no output")
        resolved_run_dir = run_dir.resolve()

        def validate_run_file(path_value: str | Path, label: str) -> Path:
            path = Path(path_value).resolve()
            try:
                path.relative_to(resolved_run_dir)
            except ValueError as exc:
                raise RuntimeError(
                    f"{label} is outside the SenseNova run directory"
                ) from exc
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"{label} is missing or empty")
            return path

        for resource in result.resources:
            label = f"SenseNova sidecar resource {resource.name}"
            path = validate_run_file(resource.path, label)
            if path.parent != resolved_run_dir:
                target = cls._sidecar_path(run_dir, path.stem, path.suffix)
                shutil.copy2(path, target)
                validate_run_file(target, f"{label} copy")

        for output in result.outputs[1:]:
            if output.text is not None:
                target = cls._sidecar_path(run_dir, output.source_handle, ".txt")
                target.write_text(
                    output.text,
                    encoding="utf-8",
                )
                validate_run_file(
                    target,
                    f"SenseNova secondary text output {output.source_handle}",
                )
            elif output.path is not None:
                label = f"SenseNova secondary output {output.source_handle}"
                path = validate_run_file(output.path, label)
                if path.parent != resolved_run_dir:
                    target = cls._sidecar_path(run_dir, path.stem, path.suffix)
                    shutil.copy2(path, target)
                    validate_run_file(target, f"{label} copy")
            else:
                raise RuntimeError(
                    f"SenseNova secondary output has no payload: {output.source_handle}"
                )

        primary = result.outputs[0]
        if primary.path is not None:
            return validate_run_file(
                primary.path,
                "SenseNova primary file output",
            )
        if primary.text is not None:
            if not isinstance(primary.text, str) or not primary.text.strip():
                raise RuntimeError("SenseNova primary text output is empty")
            target = cls._sidecar_path(run_dir, primary.source_handle, ".txt")
            target.write_text(primary.text, encoding="utf-8")
            return validate_run_file(
                target,
                f"SenseNova primary text output {primary.source_handle}",
            )
        raise RuntimeError("SenseNova primary output has no payload")

    def _dispatch(
        self,
        node_id: str,
        request: _GenerationTask,
        records: list[tuple[str, Path]],
        run_dir: Path,
        progress: _ProgressReporter,
        cancel_event: Any,
    ) -> _DispatchResult:
        params = dict(request.params)
        seed = _safe_int(params, "seed", 42, -1, 2_147_483_647)
        noise_seed: Optional[int] = None if seed == -1 else seed

        if node_id == "open-image-generation":
            self._require_no_images(records)
            prompt = self._required_prompt(request, params)
            reasoning_mode = _safe_bool(params, "reasoning_mode", False)
            mode = "think_generate" if reasoning_mode else "generate"
            progress.emit(20, "Generating image")
            call_params = {
                "question": prompt,
                "images": [],
                "mode": mode,
                "noise_seed": noise_seed,
                **self._image_generation_overrides(params, 4.0, 1.0),
            }
            if reasoning_mode:
                call_params["return_intermediate_outputs"] = True
            value = self._call_generate(**call_params)
            if reasoning_mode:
                image, reasoning = self._require_image_optional_text(value)
            else:
                image, reasoning = value, None
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing PNG")
            path = self._save_image(image, run_dir / "image.png", color_mode="RGB")
            outputs = [_GeneratedArtifact(
                source_handle="image",
                kind="image",
                path=path,
                display_name="SenseNova generated image",
                metadata={"mode": mode, "seed": noise_seed},
            )]
            if reasoning:
                outputs.append(_GeneratedArtifact(
                    source_handle="reasoning",
                    kind="text",
                    text=reasoning,
                    display_name="SenseNova generation reasoning",
                    metadata={
                        "mode": mode,
                        "format": "upstream-raw-reasoning-text",
                    },
                ))
            return _DispatchResult(outputs=tuple(outputs))

        if node_id == "image-to-image-vision":
            image_paths = self._require_unique_ports(records, ("image",))
            prompt = self._required_prompt(request, params)
            reasoning_mode = _safe_bool(params, "reasoning_mode", False)
            mode = "think_edit" if reasoning_mode else "edit"
            progress.emit(20, "Running image edit")
            call_params = {
                "question": _model_question(prompt, 1),
                "images": image_paths,
                "mode": mode,
                "noise_seed": noise_seed,
                **self._image_generation_overrides(params, 4.0, 2.0),
            }
            if reasoning_mode:
                call_params["return_intermediate_outputs"] = True
            value = self._call_generate(**call_params)
            if reasoning_mode:
                image, reasoning = self._require_image_optional_text(value)
            else:
                image, reasoning = value, None
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing edited PNG")
            path = self._save_image(image, run_dir / "image.png", color_mode="RGB")
            outputs = [_GeneratedArtifact(
                source_handle="image",
                kind="image",
                path=path,
                display_name="SenseNova edited image",
                metadata={"mode": mode, "seed": noise_seed},
            )]
            if reasoning:
                outputs.append(_GeneratedArtifact(
                    source_handle="reasoning",
                    kind="text",
                    text=reasoning,
                    display_name="SenseNova editing reasoning",
                    metadata={
                        "mode": mode,
                        "format": "upstream-raw-reasoning-text",
                    },
                ))
            return _DispatchResult(outputs=tuple(outputs))

        if node_id == "visual-prompt-segmentation":
            image_paths = self._require_unique_ports(records, ("scene", "prompt"))
            progress.emit(20, "Running visual-prompt segmentation")
            value = self._call_generate(
                question=VISUAL_PROMPT_SEGMENTATION_PROMPT,
                images=image_paths,
                mode="dense_perception",
                noise_seed=noise_seed,
                **self._image_generation_overrides(params, 4.0, 1.0),
            )
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing segmentation PNG")
            path = self._save_image(value, run_dir / "segmentation.png")
            return _DispatchResult(outputs=(_GeneratedArtifact(
                source_handle="segmentation",
                kind="image",
                path=path,
                display_name="Visual-prompt segmentation",
                metadata={
                    "mode": "dense_perception",
                    "input_order": ["scene", "prompt"],
                    "seed": noise_seed,
                },
            ),))

        if node_id == "captioned-segmentation":
            image_paths = self._require_unique_ports(records, ("image",))
            progress.emit(20, "Running grounded caption segmentation")
            value = self._call_generate(
                question=_model_question(CAPTIONED_SEGMENTATION_PROMPT, 1),
                images=image_paths,
                mode="caption_generate",
                noise_seed=noise_seed,
                return_intermediate_outputs=True,
                **self._image_generation_overrides(params, 4.0, 1.0),
            )
            image, text = self._require_image_and_text(value)
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing segmentation and caption")
            path = self._save_image(image, run_dir / "segmentation.png")
            return _DispatchResult(outputs=(
                _GeneratedArtifact(
                    source_handle="segmentation",
                    kind="image",
                    path=path,
                    display_name="Captioned segmentation",
                    metadata={"mode": "caption_generate", "seed": noise_seed},
                ),
                _GeneratedArtifact(
                    source_handle="caption",
                    kind="text",
                    text=text,
                    display_name="Grounded caption",
                    metadata={"format": "upstream-raw-tagged-text"},
                ),
            ))

        if node_id == "structured-vision":
            image_paths = self._require_unique_ports(records, ("image",))
            prompt = self._required_prompt(request, params)
            reasoning_mode = _safe_bool(params, "reasoning_mode", False)
            mode = "think_understanding" if reasoning_mode else "understanding"
            progress.emit(20, "Running structured visual understanding")
            value = self._call_generate(
                question=_model_question(prompt, 1),
                images=image_paths,
                mode=mode,
                noise_seed=noise_seed,
            )
            raw_text = self._require_text(value)
            self._check_cancelled(cancel_event)
            progress.emit(90, "Preparing raw structured text")
            return self._tagged_text_result(
                raw_text=raw_text,
                run_dir=run_dir,
                source_handle="raw_text",
                display_name="SenseNova raw structured response",
                schema="sensenova.structured-vision.raw.v1",
                mode=mode,
                prompt=prompt,
                include_json=_safe_bool(params, "include_json_resource", True),
            )

        if node_id == "relative-depth-estimation":
            image_paths = self._require_unique_ports(records, ("image",))
            progress.emit(20, "Estimating relative depth")
            value = self._call_generate(
                question=_model_question(DEPTH_PROMPT, 1),
                images=image_paths,
                mode="dense_perception",
                noise_seed=noise_seed,
                **self._image_generation_overrides(params, 4.0, 1.0),
            )
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing 8-bit grayscale depth map")
            path = self._save_image(
                value,
                run_dir / "depth.png",
                color_mode="L",
            )
            return _DispatchResult(outputs=(_GeneratedArtifact(
                source_handle="depth",
                kind="image",
                path=path,
                display_name="SenseNova relative depth map",
                metadata={
                    "mode": "dense_perception",
                    "encoding": "grayscale-uint8",
                    "range": [0, 255],
                    "near": "brighter",
                    "far": "darker",
                    "metric_depth": False,
                    "seed": noise_seed,
                },
            ),))

        if node_id == "surface-normal-estimation":
            image_paths = self._require_unique_ports(records, ("image",))
            progress.emit(20, "Estimating surface normals")
            value = self._call_generate(
                question=_model_question(NORMAL_PROMPT, 1),
                images=image_paths,
                mode="dense_perception",
                noise_seed=noise_seed,
                **self._image_generation_overrides(params, 4.0, 1.0),
            )
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing RGB normal map")
            path = self._save_image(
                value,
                run_dir / "normals.png",
                color_mode="RGB",
            )
            return _DispatchResult(outputs=(_GeneratedArtifact(
                source_handle="normals",
                kind="image",
                path=path,
                display_name="SenseNova surface normal map",
                metadata={
                    "mode": "dense_perception",
                    "encoding": "rgb-xyz-directions",
                    "seed": noise_seed,
                },
            ),))

        if node_id == "binary-segmentation":
            image_paths = self._require_unique_ports(records, ("image",))
            query = self._required_string_param(
                params,
                "query",
                "Binary segmentation query",
            )
            prompt = _binary_seg_prompt(query)
            progress.emit(20, "Running binary segmentation")
            value = self._call_generate(
                question=_model_question(prompt, 1),
                images=image_paths,
                mode="dense_perception",
                noise_seed=noise_seed,
                **self._image_generation_overrides(params, 4.0, 1.0),
            )
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing binary segmentation PNG")
            path = self._save_image(
                value,
                run_dir / "segmentation.png",
                color_mode="RGB",
            )
            return _DispatchResult(outputs=(_GeneratedArtifact(
                source_handle="segmentation",
                kind="image",
                path=path,
                display_name="SenseNova binary segmentation",
                metadata={
                    "mode": "dense_perception",
                    "query": query,
                    "seed": noise_seed,
                },
            ),))

        if node_id == "panoptic-segmentation":
            image_paths = self._require_unique_ports(records, ("image",))
            categories = self._optional_string_param(params, "categories", "")
            prompt = _panoptic_prompt(categories)
            progress.emit(20, "Running panoptic segmentation")
            value = self._call_generate(
                question=_model_question(prompt, 1),
                images=image_paths,
                mode="caption_generate",
                noise_seed=noise_seed,
                return_intermediate_outputs=True,
                **self._image_generation_overrides(params, 4.0, 1.0),
            )
            image, caption = self._require_image_and_text(value)
            self._check_cancelled(cancel_event)
            progress.emit(90, "Serializing panoptic segmentation and caption")
            path = self._save_image(
                image,
                run_dir / "segmentation.png",
                color_mode="RGB",
            )
            return _DispatchResult(outputs=(
                _GeneratedArtifact(
                    source_handle="segmentation",
                    kind="image",
                    path=path,
                    display_name="SenseNova panoptic segmentation",
                    metadata={
                        "mode": "caption_generate",
                        "categories": categories or "official-coco-default",
                        "seed": noise_seed,
                    },
                ),
                _GeneratedArtifact(
                    source_handle="caption",
                    kind="text",
                    text=caption,
                    display_name="Panoptic instance caption",
                    metadata={"format": "upstream-raw-tagged-text"},
                ),
            ))

        if node_id in {
            "bounding-box-detection",
            "point-detection",
            "keypoint-detection",
            "ocr",
        }:
            image_paths = self._require_unique_ports(records, ("image",))
            if node_id == "ocr":
                prompt = OCR_PROMPT
                mode = "dense_OCR"
                source_handle = "ocr_text"
                display_name = "SenseNova word-level OCR"
            else:
                categories = self._required_string_param(
                    params,
                    "categories",
                    "Detection categories",
                )
                mode = "dense_detection"
                if node_id == "bounding-box-detection":
                    prompt = _category_prompt(BBOX_DETECTION_PROMPT_TEMPLATE, categories)
                    source_handle = "detections"
                    display_name = "SenseNova bounding-box detections"
                elif node_id == "point-detection":
                    prompt = _category_prompt(POINT_DETECTION_PROMPT_TEMPLATE, categories)
                    source_handle = "detections"
                    display_name = "SenseNova point detections"
                else:
                    prompt = _keypoint_prompt(categories)
                    source_handle = "keypoints"
                    display_name = "SenseNova keypoint detections"
            progress.emit(20, f"Running {display_name}")
            value = self._call_generate(
                question=_model_question(prompt, 1),
                images=image_paths,
                mode=mode,
                noise_seed=noise_seed,
            )
            raw_text = self._require_text(value)
            self._check_cancelled(cancel_event)
            progress.emit(90, "Preparing upstream tagged text")
            return self._tagged_text_result(
                raw_text=raw_text,
                run_dir=run_dir,
                source_handle=source_handle,
                display_name=display_name,
                schema=f"sensenova.{node_id}.raw.v1",
                mode=mode,
                prompt=prompt,
                include_json=_safe_bool(params, "include_json_resource", True),
            )

        if node_id == "multi-view-reconstruction":
            image_paths = self._require_repeatable_port(records, "views", 1, 10)
            raw_path = run_dir / "pointmaps.npy"
            glb_path = run_dir / "pointcloud.glb"
            progress.emit(20, "Reconstructing aligned point maps")
            with contextlib.redirect_stdout(sys.stderr):
                value = self._model.reconstruct_3d(
                    images=[str(path) for path in image_paths],
                    prompt=RECON3D_PROMPT,
                    noise_seed=noise_seed,
                    postprocess_predictions=True,
                    max_images=10,
                    raw_output=str(raw_path),
                    glb_output=str(glb_path),
                    mask_edge=_safe_bool(params, "mask_edge", True),
                    mask_black_bg=_safe_bool(params, "mask_black_bg", False),
                    mask_white_bg=_safe_bool(params, "mask_white_bg", False),
                    num_timesteps=_safe_int(params, "num_timesteps", 50, 1, 100),
                )
            del value
            self._check_cancelled(cancel_event)
            progress.emit(90, "Validating POINTS GLB and NPY point maps")
            shape, dtype = self._validate_pointmaps(raw_path, len(image_paths))
            primitive_count = self._verify_glb_points(glb_path)
            metadata = {
                "geometry_kind": "pointcloud",
                "gltf_primitive_mode": "POINTS",
                "primitive_mode": 0,
                "view_count": len(image_paths),
                "coordinate_space": "first-camera",
                "pointmap_shape": shape,
                "pointmap_dtype": dtype,
                "glb_point_primitives": primitive_count,
                "upstream_method": "SenseNovaVisionModel.reconstruct_3d",
            }
            return _DispatchResult(
                outputs=(_GeneratedArtifact(
                    source_handle="pointcloud",
                    kind="mesh",
                    path=glb_path,
                    display_name="SenseNova aligned point cloud",
                    metadata=metadata,
                ),),
                resources=(_SidecarResource(
                    name="pointmaps",
                    path=raw_path,
                    media_type="application/x-npy",
                    metadata={
                        "geometry_kind": "pointcloud",
                        "shape": shape,
                        "dtype": dtype,
                        "coordinate_space": "first-camera",
                    },
                ),),
            )

        if node_id == "camera-pose":
            image_paths = self._require_repeatable_port(records, "views", 2, 10)
            progress.emit(20, "Estimating ordered camera poses")
            raw_value = self._call_generate(
                question=_model_question(CAMERA_POSE_PROMPT, len(image_paths)),
                images=image_paths,
                mode="understanding",
                noise_seed=noise_seed,
                vit_transform=self._model.camera_vit_transform,
            )
            raw_text = self._require_text(raw_value)
            self._check_cancelled(cancel_event)
            with contextlib.redirect_stdout(sys.stderr):
                parser = getattr(self, "_pose_parser", None)
                if parser is None:
                    parser_module = importlib.import_module(
                        "inference.utils_3d.camera_pose_parser"
                    )
                    parser = parser_module.resolve_pose_string
                parsed = parser(raw_text)
            parsed = self._validate_pose_result(parsed, len(image_paths) - 1)
            deterministic = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            raw_path = run_dir / "raw-pose.txt"
            json_path = run_dir / "camera-poses.json"
            raw_path.write_text(raw_text, encoding="utf-8")
            json_path.write_text(deterministic + "\n", encoding="utf-8")
            progress.emit(90, "Serializing validated camera poses")
            return _DispatchResult(
                outputs=(_GeneratedArtifact(
                    source_handle="poses",
                    kind="text",
                    text=deterministic,
                    display_name="Ordered camera poses",
                    metadata={
                        "reference_frame": 0,
                        "pose_count": len(image_paths) - 1,
                        "input_order_preserved": True,
                    },
                ),),
                resources=(
                    _SidecarResource(
                        name="raw_pose",
                        path=raw_path,
                        media_type="text/plain",
                        metadata={"format": "upstream-tagged-pose"},
                    ),
                    _SidecarResource(
                        name="parsed_pose",
                        path=json_path,
                        media_type="application/json",
                        metadata={"deterministic_json": True},
                    ),
                ),
            )

        raise AssertionError(f"Unreachable SenseNova node: {node_id}")

    def _call_generate(self, **kwargs: Any) -> Any:
        kwargs["images"] = [str(path) for path in kwargs.get("images", [])]
        with contextlib.redirect_stdout(sys.stderr):
            return self._model.generate(**kwargs)

    @staticmethod
    def _tagged_text_result(
        *,
        raw_text: str,
        run_dir: Path,
        source_handle: str,
        display_name: str,
        schema: str,
        mode: str,
        prompt: str,
        include_json: bool,
    ) -> _DispatchResult:
        resources: list[_SidecarResource] = []
        if include_json:
            envelope_path = run_dir / f"{source_handle}-raw.json"
            envelope = {
                "schema": schema,
                "mode": mode,
                "prompt": prompt,
                "raw_text": raw_text,
                "semantic_parsing": "not_applied",
            }
            envelope_path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            resources.append(_SidecarResource(
                name="raw_json",
                path=envelope_path,
                media_type="application/json",
                metadata={"semantic_parsing": "not_applied"},
            ))
        return _DispatchResult(
            outputs=(_GeneratedArtifact(
                source_handle=source_handle,
                kind="text",
                text=raw_text,
                display_name=display_name,
                metadata={
                    "mode": mode,
                    "format": "upstream-raw-tagged-text",
                },
            ),),
            resources=tuple(resources),
        )

    @staticmethod
    def _image_generation_overrides(
        params: dict[str, Any],
        text_default: float,
        image_default: float,
    ) -> dict[str, Any]:
        return {
            "num_timesteps": _safe_int(params, "num_timesteps", 50, 1, 100),
            "cfg_text_scale": _safe_float(
                params, "cfg_text_scale", text_default, 0.0, 20.0
            ),
            "cfg_img_scale": _safe_float(
                params, "cfg_img_scale", image_default, 0.0, 20.0
            ),
        }

    def _validate_checkpoint_before_imports(self) -> Path:
        checkpoint_dir = self.model_dir / "checkpoint"
        missing = [
            name for name in REQUIRED_CHECKPOINT_FILES
            if not (checkpoint_dir / name).is_file()
            or (checkpoint_dir / name).stat().st_size <= 0
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                "SenseNova weights are incomplete. Download the pinned model from "
                f"the Modly UI into {checkpoint_dir}. Missing: {missing_text}"
            )
        return checkpoint_dir

    @staticmethod
    def _verify_vendored_source() -> None:
        try:
            payload = json.loads(VENDOR_PROVENANCE.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("Vendored SenseNova provenance is missing or invalid") from exc
        upstream = payload.get("upstream", {})
        if upstream.get("commit") != UPSTREAM_COMMIT:
            raise RuntimeError("Vendored SenseNova commit does not match the adapter pin")
        for record in payload.get("files", []):
            relative = record.get("path")
            expected = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise RuntimeError("Vendored SenseNova provenance has an invalid record")
            path = (VENDOR_ROOT / relative).resolve()
            if VENDOR_ROOT.resolve() not in path.parents or not path.is_file():
                raise RuntimeError(f"Vendored SenseNova file is missing: {relative}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Vendored SenseNova file hash mismatch: {relative}")

    @staticmethod
    def _configure_offline_environment(checkpoint_dir: Path) -> None:
        offline = {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(checkpoint_dir / ".cache"),
            "TORCH_HOME": str(checkpoint_dir / ".torch-cache"),
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
        os.environ.update(offline)
        tunable = "glibc.rtld.optional_static_tls=262144"
        current = os.environ.get("GLIBC_TUNABLES", "")
        if tunable not in current.split(":"):
            os.environ["GLIBC_TUNABLES"] = ":".join(
                value for value in (current, tunable) if value
            )

    @staticmethod
    def _normalized_node_id(value: str) -> str:
        return str(value or "").rsplit("/", 1)[-1]

    def _request_root(self, request: _GenerationTask) -> Path:
        root = request.outputs_dir or self.outputs_dir
        if root is None:
            raise ValueError("SenseNova generation requires an outputs directory")
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _create_request_dir(self, request: _GenerationTask, node_id: str) -> Path:
        root = self._request_root(request)
        for _ in range(16):
            candidate = root / f"sensenova-{node_id}-{uuid.uuid4().hex}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("Unable to allocate a collision-safe SenseNova output directory")

    @staticmethod
    def _materialize_images(
        request: _GenerationTask,
        run_dir: Path,
    ) -> list[tuple[str, Path]]:
        if not request.images:
            return []
        if len(request.images) != len(request.image_names):
            raise ValueError("SenseNova image payloads and names must have equal length")
        from PIL import Image, ImageOps

        input_dir = run_dir / ".inputs"
        input_dir.mkdir()
        records: list[tuple[str, Path]] = []
        for index, (name, payload) in enumerate(
            zip(request.image_names, request.images), start=1
        ):
            try:
                with Image.open(io.BytesIO(payload)) as decoded:
                    normalized = ImageOps.exif_transpose(decoded).convert("RGB")
                    path = input_dir / f"{index:02d}-{name}.png"
                    normalized.save(path, format="PNG")
            except Exception as exc:
                raise ValueError(
                    f"Named image '{name}' at position {index} is not a valid image"
                ) from exc
            records.append((name, path))
        return records

    @staticmethod
    def _require_no_images(records: Sequence[tuple[str, Path]]) -> None:
        if records:
            raise ValueError("open-image-generation does not accept image inputs")

    @staticmethod
    def _require_unique_ports(
        records: Sequence[tuple[str, Path]],
        expected_names: Sequence[str],
    ) -> list[Path]:
        grouped: dict[str, list[Path]] = {}
        for name, path in records:
            grouped.setdefault(name, []).append(path)
        if set(grouped) != set(expected_names):
            raise ValueError(
                "Expected named image ports: " + ", ".join(expected_names)
            )
        for name in expected_names:
            if len(grouped[name]) != 1:
                raise ValueError(f"Named image port '{name}' requires exactly one image")
        return [grouped[name][0] for name in expected_names]

    @staticmethod
    def _require_repeatable_port(
        records: Sequence[tuple[str, Path]],
        name: str,
        minimum: int,
        maximum: int,
    ) -> list[Path]:
        if any(port_name != name for port_name, _ in records):
            raise ValueError(f"Only the ordered repeatable image port '{name}' is accepted")
        paths = [path for _, path in records]
        if len(paths) < minimum or len(paths) > maximum:
            raise ValueError(
                f"Named image port '{name}' requires between {minimum} and "
                f"{maximum} ordered images; received {len(paths)}"
            )
        return paths

    @staticmethod
    def _required_prompt(request: _GenerationTask, params: dict[str, Any]) -> str:
        value = request.prompt if request.prompt is not None else params.get("prompt")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A non-empty prompt is required")
        return value.strip()

    @staticmethod
    def _required_string_param(
        params: dict[str, Any],
        key: str,
        label: str,
    ) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_string_param(
        params: dict[str, Any],
        key: str,
        default: str,
    ) -> str:
        value = params.get(key, default)
        if not isinstance(value, str):
            raise TypeError(f"Parameter '{key}' must be a string")
        return value.strip()

    @staticmethod
    def _save_image(
        value: Any,
        path: Path,
        *,
        color_mode: Optional[str] = None,
    ) -> Path:
        if isinstance(value, dict):
            value = value.get("image")
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None or not callable(getattr(value, "save", None)):
            raise RuntimeError("SenseNova did not return a serializable image")
        if color_mode is not None:
            converter = getattr(value, "convert", None)
            if not callable(converter):
                raise RuntimeError("SenseNova image does not support color conversion")
            value = converter(color_mode)
        value.save(path, format="PNG")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("SenseNova PNG serialization produced an empty file")
        return path.resolve()

    @staticmethod
    def _require_text(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("text")
        if isinstance(value, list):
            value = value[0] if value else None
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("SenseNova did not return non-empty text")
        return value

    @classmethod
    def _require_image_and_text(cls, value: Any) -> tuple[Any, str]:
        if not isinstance(value, dict):
            raise RuntimeError(
                "SenseNova caption_generate did not return image/text intermediates"
            )
        image = value.get("image")
        if image is None or not callable(getattr(image, "save", None)):
            raise RuntimeError("SenseNova caption_generate did not return an image")
        return image, cls._require_text(value.get("text"))

    @staticmethod
    def _require_image_optional_text(value: Any) -> tuple[Any, Optional[str]]:
        if not isinstance(value, dict):
            raise RuntimeError(
                "SenseNova reasoning image mode did not return intermediates"
            )
        image = value.get("image")
        if image is None or not callable(getattr(image, "save", None)):
            raise RuntimeError("SenseNova reasoning image mode did not return an image")
        text = value.get("text")
        if text is None:
            return image, None
        if not isinstance(text, str):
            raise RuntimeError("SenseNova reasoning output must be text")
        return image, text if text.strip() else None

    @staticmethod
    def _validate_pointmaps(path: Path, expected_views: int) -> tuple[list[int], str]:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("SenseNova reconstruct_3d did not write pointmaps.npy")
        import numpy as np

        try:
            pointmaps = np.load(path, allow_pickle=False, mmap_mode="r")
        except Exception as exc:
            raise RuntimeError("SenseNova pointmaps.npy is invalid") from exc
        if (
            pointmaps.ndim != 4
            or pointmaps.shape[0] != expected_views
            or pointmaps.shape[-1] != 3
        ):
            raise RuntimeError(
                "SenseNova pointmaps must have shape [views, height, width, 3]"
            )
        return [int(value) for value in pointmaps.shape], str(pointmaps.dtype)

    @staticmethod
    def _verify_glb_points(path: Path) -> int:
        if not path.is_file() or path.stat().st_size < 20:
            raise RuntimeError("SenseNova reconstruct_3d did not write a valid GLB")
        data = path.read_bytes()
        if data[:4] != b"glTF":
            raise RuntimeError("SenseNova reconstruction output is not a GLB file")
        version, declared_length = struct.unpack_from("<II", data, 4)
        if version != 2 or declared_length != len(data):
            raise RuntimeError("SenseNova reconstruction GLB header is invalid")
        offset = 12
        document: Optional[dict[str, Any]] = None
        while offset + 8 <= len(data):
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset:offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                document = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\0"))
                break
        if document is None:
            raise RuntimeError("SenseNova reconstruction GLB has no JSON chunk")
        modes = [
            primitive.get("mode", 4)
            for mesh in document.get("meshes", [])
            for primitive in mesh.get("primitives", [])
        ]
        if not modes or any(mode != 0 for mode in modes):
            raise RuntimeError(
                "SenseNova reconstruction GLB must contain only POINTS primitives"
            )
        return len(modes)

    @staticmethod
    def _validate_pose_result(value: Any, expected_count: int) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise RuntimeError(
                "SenseNova camera-pose text did not match the official tagged format"
            )
        rotations = value.get("rotation")
        translations = value.get("translation")
        if not isinstance(rotations, list) or not isinstance(translations, list):
            raise RuntimeError("SenseNova camera-pose parser returned an invalid object")
        if len(rotations) != expected_count or len(translations) != expected_count:
            raise RuntimeError(
                "SenseNova camera-pose count does not match the ordered input frames: "
                f"expected {expected_count}, got rotations={len(rotations)}, "
                f"translations={len(translations)}"
            )
        if any(not isinstance(item, list) or len(item) != 4 for item in rotations):
            raise RuntimeError("SenseNova camera-pose rotations must be quaternions")
        if any(not isinstance(item, list) or len(item) != 3 for item in translations):
            raise RuntimeError("SenseNova camera-pose translations must be 3D vectors")
        return {"rotation": rotations, "translation": translations}

