from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import struct
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import generator as adapter


ROOT = Path(__file__).resolve().parents[1]

# Independent fixtures derived from OpenSenseNova/SenseNova-Vision commit
# beea1f771b8192c597085c0e668f4430d2089d8d. These hashes intentionally do
# not derive from generator.py, so adapter prompt drift cannot update both
# sides of an assertion. Values hash the exact UTF-8 prompt passed upstream.
PINNED_PROMPT_SHA256 = {
    "depth": "7265cf0110344bb5740f44d19b4f6264d665dcd1c00f2e46c015515223faa405",
    "normal": "c28a7549775a2253625c5cfc1204668cbbbe3b7eab37157f4c5402c805de00fd",
    "binary_csv": "8d4222a2e345c2eb3518ac2d657714d0e1564c92ca0423bf1df3b0a5c741c10c",
    "binary_tagged": "cf7c3dfd1ef087e3d7d02510dd0541343bd4d1920f096008774632ddd774ba78",
    "panoptic_coco": "d5eeb1ea3ef4bf709eeb03273385b4ab3bc99162b90d2078b15ca2a5dfb273c3",
    "bbox": "176e233fbba44503a8b38cdec43c4fba13858e15dca06b5ff978843609847ee5",
    "point": "5981718ab1dc29150a1ee5cd29779f060c58c18d971c64741baa46c179dd0c56",
    "ocr": "bf1f77205a5dafceeaf5ddd4046b4022ebbc9c37e5d8b8203a681f7ef0f1e87a",
    "keypoint_human": "598659c71a0c222ddd9a1535108c38b7d5a6674c2fef52620765dbfd85183316",
    "keypoint_animal": "12516d03ad8e077e85f1ce8500c3df5a010df7e7d8437f954e7117e4ae7de789",
    "keypoint_mixed": "7ac4439d692e927a7a91eb555576ffd58d3c2a57dc35f50ed12cc0ad4c3288c2",
    "visual_prompt_segmentation": (
        "baaa6be19abdba1eef19268a9db8628871f0eaa18aaafb00db79298e9e127870"
    ),
    "captioned_segmentation": "c40c07bc5fc936c5cf8107da6d8ab486e815d43b23cd6bd2b0b16cb9ce8d8474",
    "recon3d": "84a835c4e1076cabaf88a938bfee4794743052a3d642455a8376b7bd073d96e9",
    "camera_pose_3_images": "415866d6a57aefda47867f54f031524016eec9ec3abb82421745062d86ccd252",
}
PINNED_COCO_TUPLE_SHA256 = (
    "ec1a167c4659d761436ab4739b17bbcb0a990a28eb40f26818aebb8b6fb8a584"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(stream, format="PNG")
    return stream.getvalue()


def write_png(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (4, 3), color).save(path, format="PNG")
    return path


def write_points_glb(path: Path, mode: int = 0) -> None:
    binary = struct.pack("<fff", 0.0, 0.0, 0.0)
    while len(binary) % 4:
        binary += b"\0"
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": mode}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "accessors": [{
            "bufferView": 0,
            "componentType": 5126,
            "count": 1,
            "type": "VEC3",
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
        }],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    payload = bytearray(struct.pack("<4sII", b"glTF", 2, total))
    payload.extend(struct.pack("<II", len(json_chunk), 0x4E4F534A))
    payload.extend(json_chunk)
    payload.extend(struct.pack("<II", len(binary), 0x004E4942))
    payload.extend(binary)
    path.write_bytes(payload)


def load_official_pose_parser():
    path = (
        ROOT
        / "vendor/opensensenova_sensenova_vision/inference/utils_3d/camera_pose_parser.py"
    )
    spec = importlib.util.spec_from_file_location("sensenova_official_pose_parser", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.resolve_pose_string


def pose_text(frame_count: int) -> str:
    frame = (
        "<frame><quat>[0,0,0,1000]</quat>"
        "<offset>[1000,0,0]</offset><scale>100</scale></frame>"
    )
    return frame * frame_count


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.camera_vit_transform = object()
        self.structured_text = "<bbox>[10,20,30,40]</bbox> untouched"
        self.reasoning_text: str | None = "upstream reasoning text"
        self.glb_mode = 0

    @staticmethod
    def _colors(paths: list[str]) -> list[tuple[int, int, int]]:
        colors = []
        for path in paths:
            with Image.open(path) as image:
                colors.append(tuple(image.convert("RGB").getpixel((0, 0))))
        return colors

    def generate(self, **kwargs: Any) -> Any:
        call = dict(kwargs)
        call["input_colors"] = self._colors(call.get("images", []))
        self.calls.append(call)
        mode = kwargs["mode"]
        if mode == "caption_generate":
            return {
                "image": Image.new("RGB", (5, 4), (90, 80, 70)),
                "text": "<p>cat<color>(1,2,3)</color></p>",
            }
        if mode in {"think_generate", "think_edit"}:
            return {
                "image": Image.new("RGB", (5, 4), (21, 43, 65)),
                "text": self.reasoning_text,
            }
        if mode in {
            "understanding",
            "think_understanding",
            "dense_detection",
            "dense_OCR",
        }:
            image_count = kwargs["question"].count("<image>")
            if "relative pose" in kwargs["question"]:
                return pose_text(image_count - 1)
            return self.structured_text
        return Image.new("RGB", (5, 4), (12, 34, 56))

    def reconstruct_3d(self, **kwargs: Any) -> dict[str, Any]:
        call = dict(kwargs)
        call["method"] = "reconstruct_3d"
        call["input_colors"] = self._colors(call["images"])
        self.calls.append(call)
        points = np.zeros((len(call["images"]), 2, 3, 3), dtype=np.float32)
        np.save(call["raw_output"], points)
        write_points_glb(Path(call["glb_output"]), mode=self.glb_mode)
        return {"pts3d": points, "scene": object(), "text": None}


def make_generator(tmp_path: Path) -> tuple[adapter.SenseNovaVisionGenerator, FakeModel]:
    model = FakeModel()
    generator = adapter.SenseNovaVisionGenerator(
        tmp_path / "models/shared",
        tmp_path / "outputs",
    )
    generator._model = model
    generator._pose_parser = load_official_pose_parser()
    generator.model_id = "sensenova-vision-7b-mot/test"
    return generator, model


def invoke(
    generator: adapter.SenseNovaVisionGenerator,
    node_id: str,
    *,
    image_bytes: bytes = b"",
    params: dict[str, Any] | None = None,
    progress_cb=None,
    cancel_event=None,
) -> tuple[Path, Path]:
    generator.node_id = node_id
    root = Path(generator.outputs_dir)
    before = set(root.iterdir()) if root.is_dir() else set()
    result = generator.generate(
        image_bytes,
        params or {},
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )
    created = set(root.iterdir()) - before
    assert len(created) == 1
    run_dir = created.pop()
    assert run_dir.is_dir()
    return result, run_dir


def test_generator_module_is_import_light() -> None:
    tree = ast.parse((ROOT / "generator.py").read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "open3d",
        "transformers",
        "accelerate",
    })


def test_missing_checkpoint_fails_before_any_heavy_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = adapter.SenseNovaVisionGenerator(tmp_path / "models/shared", tmp_path / "out")
    imports: list[str] = []

    def forbidden_import(name: str):
        imports.append(name)
        raise AssertionError(name)

    monkeypatch.setattr(adapter.importlib, "import_module", forbidden_import)
    with pytest.raises(FileNotFoundError) as caught:
        generator.load()
    assert "Modly UI" in str(caught.value)
    assert "checkpoint/ema.safetensors" not in str(caught.value)
    assert imports == []


def test_open_generation_and_edit_use_direct_contract_and_official_modes(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    progress: list[tuple[int, str]] = []
    first, _run_dir = invoke(
        generator,
        "open-image-generation",
        params={"prompt": "A city street scene.", "seed": 7, "num_timesteps": 40},
        progress_cb=lambda value, label: progress.append((value, label)),
    )
    assert isinstance(first, Path) and first.is_file()
    assert model.calls[-1]["mode"] == "generate"
    assert model.calls[-1]["question"] == "A city street scene."
    assert model.calls[-1]["images"] == []
    assert model.calls[-1]["noise_seed"] == 7
    assert model.calls[-1]["num_timesteps"] == 40
    assert [value for value, _ in progress] == [0, 10, 20, 90, 100]

    edited, _run_dir = invoke(
        generator,
        "image-to-image-vision",
        image_bytes=png_bytes((200, 1, 2)),
        params={"prompt": "Turn the apples in the picture green."},
    )
    assert isinstance(edited, Path) and edited.is_file()
    assert model.calls[-1]["mode"] == "edit"
    assert model.calls[-1]["question"] == (
        "<image> Turn the apples in the picture green."
    )
    assert model.calls[-1]["input_colors"] == [(200, 1, 2)]
    assert model.calls[-1]["cfg_img_scale"] == 2.0


def test_visual_prompt_uses_primary_scene_plus_picker_image_in_canonical_order(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    prompt_path = write_png(tmp_path / "prompt.png", (1, 200, 2))
    result, _run_dir = invoke(
        generator,
        "visual-prompt-segmentation",
        image_bytes=png_bytes((200, 1, 2)),
        params={"prompt_image_path": str(prompt_path)},
    )
    assert isinstance(result, Path) and result.name == "segmentation.png"
    call = model.calls[-1]
    assert call["mode"] == "dense_perception"
    assert call["question"] == adapter.VISUAL_PROMPT_SEGMENTATION_PROMPT
    assert call["input_colors"] == [(200, 1, 2), (1, 200, 2)]


def test_captioned_segmentation_returns_image_and_keeps_text_sidecar(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    result, run_dir = invoke(
        generator,
        "captioned-segmentation",
        image_bytes=png_bytes((3, 4, 5)),
    )
    assert isinstance(result, Path) and result.is_file()
    assert (run_dir / "caption.txt").read_text(encoding="utf-8") == (
        "<p>cat<color>(1,2,3)</color></p>"
    )
    call = model.calls[-1]
    assert call["mode"] == "caption_generate"
    assert call["return_intermediate_outputs"] is True
    assert call["question"] == (
        "<image> Please briefly describe the contents of the image. Please respond "
        "with interleaved segmentation masks for the corresponding parts of the answer."
    )


def test_structured_vision_returns_text_artifact_and_keeps_json_sidecar(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    result, run_dir = invoke(
        generator,
        "structured-vision",
        image_bytes=png_bytes((6, 7, 8)),
        params={
            "prompt": "Return boxes with upstream tags.",
            "include_json_resource": True,
        },
    )
    assert isinstance(result, Path) and result.is_file()
    assert result.parent == run_dir.resolve()
    assert result.name == "raw_text.txt"
    assert result.read_text(encoding="utf-8") == model.structured_text
    envelope = json.loads((run_dir / "raw_text-raw.json").read_text(encoding="utf-8"))
    assert envelope["raw_text"] == model.structured_text
    assert envelope["semantic_parsing"] == "not_applied"
    assert model.calls[-1]["mode"] == "understanding"


def test_reconstruction_preserves_picker_order_and_returns_points_glb_with_npy_sidecar(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    view_2 = write_png(tmp_path / "view-2.png", (0, 2, 0))
    view_3 = write_png(tmp_path / "view-3.png", (0, 0, 3))
    result, run_dir = invoke(
        generator,
        "multi-view-reconstruction",
        image_bytes=png_bytes((1, 0, 0)),
        params={
            "view_2_image_path": str(view_2),
            "view_3_image_path": str(view_3),
            "mask_edge": True,
            "mask_black_bg": False,
            "mask_white_bg": True,
        },
    )
    assert isinstance(result, Path) and result.name == "pointcloud.glb"
    assert adapter.SenseNovaVisionGenerator._verify_glb_points(result) == 1
    pointmaps = run_dir / "pointmaps.npy"
    assert np.load(pointmaps, allow_pickle=False).shape == (3, 2, 3, 3)
    call = model.calls[-1]
    assert call["method"] == "reconstruct_3d"
    assert call["input_colors"] == [(1, 0, 0), (0, 2, 0), (0, 0, 3)]
    assert call["prompt"] == adapter.RECON3D_PROMPT
    assert call["postprocess_predictions"] is True
    assert call["mask_white_bg"] is True


def test_reconstruction_rejects_non_points_glb(tmp_path: Path) -> None:
    generator, model = make_generator(tmp_path)
    model.glb_mode = 4
    with pytest.raises(RuntimeError, match="POINTS"):
        invoke(
            generator,
            "multi-view-reconstruction",
            image_bytes=png_bytes((1, 2, 3)),
        )


def test_camera_pose_returns_json_text_artifact_and_keeps_pose_sidecars(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    view_2 = write_png(tmp_path / "pose-view-2.png", (0, 20, 0))
    view_3 = write_png(tmp_path / "pose-view-3.png", (0, 0, 30))
    result, run_dir = invoke(
        generator,
        "camera-pose",
        image_bytes=png_bytes((10, 0, 0)),
        params={
            "view_2_image_path": str(view_2),
            "view_3_image_path": str(view_3),
        },
    )
    assert isinstance(result, Path) and result.is_file()
    assert result.parent == run_dir.resolve()
    assert result.name == "poses.txt"
    call = model.calls[-1]
    assert call["mode"] == "understanding"
    assert call["question"] == "<image><image><image> " + adapter.CAMERA_POSE_PROMPT
    assert call["input_colors"] == [(10, 0, 0), (0, 20, 0), (0, 0, 30)]
    assert call["vit_transform"] is model.camera_vit_transform
    text = result.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed == {
        "rotation": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "translation": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    assert text == json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    assert (run_dir / "raw-pose.txt").is_file()
    assert json.loads((run_dir / "camera-poses.json").read_text(encoding="utf-8")) == parsed


def test_camera_pose_fails_when_official_parse_fails(tmp_path: Path) -> None:
    generator, model = make_generator(tmp_path)
    model.generate = lambda **kwargs: "not a tagged pose"
    generator._pose_parser = lambda text: None
    view_2 = write_png(tmp_path / "bad-pose-view.png", (4, 5, 6))
    with pytest.raises(RuntimeError, match="official tagged format"):
        invoke(
            generator,
            "camera-pose",
            image_bytes=png_bytes((1, 2, 3)),
            params={"view_2_image_path": str(view_2)},
        )


def test_supplemental_image_validation_requires_declared_inputs_and_rejects_gaps(
    tmp_path: Path,
) -> None:
    generator, _model = make_generator(tmp_path)
    with pytest.raises(ValueError, match="prompt_image_path"):
        invoke(
            generator,
            "visual-prompt-segmentation",
            image_bytes=png_bytes((1, 2, 3)),
        )
    with pytest.raises(ValueError, match="view_2_image_path"):
        invoke(
            generator,
            "camera-pose",
            image_bytes=png_bytes((1, 2, 3)),
        )

    view_3 = write_png(tmp_path / "gap-view-3.png", (4, 5, 6))
    with pytest.raises(ValueError, match="sequential without gaps"):
        invoke(
            generator,
            "multi-view-reconstruction",
            image_bytes=png_bytes((1, 2, 3)),
            params={"view_3_image_path": str(view_3)},
        )


def test_cancellation_and_collision_safe_output_directories(tmp_path: Path) -> None:
    generator, model = make_generator(tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    generator.node_id = "open-image-generation"
    with pytest.raises(adapter.GenerationCancelled):
        generator.generate(
            b"",
            {"prompt": "A test image."},
            cancel_event=cancelled,
        )
    assert model.calls == []

    first, first_dir = invoke(
        generator,
        "open-image-generation",
        params={"prompt": "First image."},
    )
    second, second_dir = invoke(
        generator,
        "open-image-generation",
        params={"prompt": "Second image."},
    )
    assert first_dir != second_dir
    assert isinstance(first, Path) and first.is_file()
    assert isinstance(second, Path) and second.is_file()


def test_reconstruction_and_camera_prompts_match_vendored_upstream_constants() -> None:
    source = (
        ROOT
        / "vendor/opensensenova_sensenova_vision/inference/sensenova_vision.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "RECON3D_PROMPT",
                "CAMERA_POSE_PROMPT",
            }:
                constants[target.id] = ast.literal_eval(node.value)
    assert adapter.RECON3D_PROMPT == constants["RECON3D_PROMPT"]
    assert adapter.CAMERA_POSE_PROMPT == constants["CAMERA_POSE_PROMPT"]


def test_reasoning_modes_keep_optional_text_as_sidecar_and_primary_text_is_artifact(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    generated, generated_dir = invoke(
        generator,
        "open-image-generation",
        params={"prompt": "Create a diagram.", "reasoning_mode": True},
    )
    assert isinstance(generated, Path) and generated.is_file()
    assert (generated_dir / "reasoning.txt").read_text(encoding="utf-8") == (
        "upstream reasoning text"
    )
    assert model.calls[-1]["mode"] == "think_generate"
    assert model.calls[-1]["return_intermediate_outputs"] is True

    edited, edited_dir = invoke(
        generator,
        "image-to-image-vision",
        image_bytes=png_bytes((7, 8, 9)),
        params={"prompt": "Make the sky darker.", "reasoning_mode": True},
    )
    assert isinstance(edited, Path) and edited.is_file()
    assert (edited_dir / "reasoning.txt").is_file()
    assert model.calls[-1]["mode"] == "think_edit"

    understood, understood_dir = invoke(
        generator,
        "structured-vision",
        image_bytes=png_bytes((1, 2, 3)),
        params={
            "prompt": "Explain the scene.",
            "reasoning_mode": True,
            "include_json_resource": True,
        },
    )
    assert isinstance(understood, Path) and understood.is_file()
    assert understood.parent == understood_dir.resolve()
    assert understood.name == "raw_text.txt"
    assert understood.read_text(encoding="utf-8") == model.structured_text
    assert model.calls[-1]["mode"] == "think_understanding"
    envelope = json.loads(
        (understood_dir / "raw_text-raw.json").read_text(encoding="utf-8")
    )
    assert envelope["mode"] == "think_understanding"


@pytest.mark.parametrize(
    ("node_id", "prompt_fixture", "filename", "pil_mode"),
    [
        ("relative-depth-estimation", "depth", "depth.png", "L"),
        ("surface-normal-estimation", "normal", "normals.png", "RGB"),
    ],
)
def test_dense_geometry_uses_official_prompts_and_expected_png_encoding(
    tmp_path: Path,
    node_id: str,
    prompt_fixture: str,
    filename: str,
    pil_mode: str,
) -> None:
    generator, model = make_generator(tmp_path)
    result, _run_dir = invoke(
        generator,
        node_id,
        image_bytes=png_bytes((31, 32, 33)),
    )
    assert isinstance(result, Path) and result.name == filename
    with Image.open(result) as image:
        assert image.mode == pil_mode
    call = model.calls[-1]
    assert call["mode"] == "dense_perception"
    assert sha256_text(call["question"]) == PINNED_PROMPT_SHA256[prompt_fixture]
    assert call["num_timesteps"] == 50
    assert call["cfg_text_scale"] == 4.0
    assert call["cfg_img_scale"] == 1.0


def test_binary_segmentation_formats_categories_and_requires_query(tmp_path: Path) -> None:
    generator, model = make_generator(tmp_path)
    result, _run_dir = invoke(
        generator,
        "binary-segmentation",
        image_bytes=png_bytes((41, 42, 43)),
        params={"query": "person furthest right, red car"},
    )
    assert isinstance(result, Path) and result.name == "segmentation.png"
    assert model.calls[-1]["mode"] == "dense_perception"
    assert model.calls[-1]["question"] == (
        "<image> Can you segment the image based on the following categories: "
        "<p>person furthest right</p>, <p>red car</p>? Please output the binary "
        "segmentation masks."
    )
    with pytest.raises(ValueError, match="must be a non-empty string"):
        invoke(
            generator,
            "binary-segmentation",
            image_bytes=png_bytes((1, 2, 3)),
            params={"query": "  "},
        )


def test_category_formatting_preserves_tags_and_keypoint_template_selection() -> None:
    tagged = "<p>Person Left</p>, <p>dog</p>"
    assert adapter._format_categories(tagged) == tagged
    assert adapter._format_categories("person, dog") == "<p>person</p>, <p>dog</p>"
    assert "COCO human keypoints" in adapter._keypoint_prompt("person, dog")
    assert "nose, left eye, right eye" in adapter._keypoint_prompt("person")
    assert "root of tail" in adapter._keypoint_prompt("dog")
    with pytest.raises(ValueError, match="explicit categories"):
        adapter._format_categories("")


def test_panoptic_empty_categories_use_official_coco_order_and_keep_caption_sidecar(
    tmp_path: Path,
) -> None:
    generator, model = make_generator(tmp_path)
    result, run_dir = invoke(
        generator,
        "panoptic-segmentation",
        image_bytes=png_bytes((51, 52, 53)),
        params={"categories": ""},
    )
    assert isinstance(result, Path) and result.is_file()
    assert (run_dir / "caption.txt").read_text(encoding="utf-8") == (
        "<p>cat<color>(1,2,3)</color></p>"
    )
    call = model.calls[-1]
    assert call["mode"] == "caption_generate"
    assert call["return_intermediate_outputs"] is True
    prompt = call["question"]
    assert prompt.startswith(
        "<image> Can you segment the image based on the following categories: "
        "<p>person</p>, <p>bicycle</p>, <p>car</p>"
    )
    category_prompt = prompt.split(" Please find all instances", 1)[0]
    assert len(adapter._extract_category_names(category_prompt)) == len(
        adapter.COCO_CATEGORY_NAMES
    )
    assert "<p>door</p>" in prompt and "<p>door-stuff</p>" not in prompt
    assert "<p>tree</p>" in prompt and "<p>tree-merged</p>" not in prompt


@pytest.mark.parametrize(
    ("node_id", "categories", "mode", "sidecar_stem", "prompt_fragment"),
    [
        (
            "bounding-box-detection",
            "person, dog",
            "dense_detection",
            "detections",
            "bounding box coordinates in <bbox> format",
        ),
        (
            "point-detection",
            "person, dog",
            "dense_detection",
            "detections",
            "pixel coordinates defining the object point location",
        ),
        (
            "keypoint-detection",
            "person, dog",
            "dense_detection",
            "keypoints",
            "COCO human keypoints",
        ),
        (
            "ocr",
            None,
            "dense_OCR",
            "ocr_text",
            "Perform word-level text detection and recognition",
        ),
    ],
)
def test_detection_and_ocr_return_text_artifacts_and_keep_raw_json_sidecar(
    tmp_path: Path,
    node_id: str,
    categories: str | None,
    mode: str,
    sidecar_stem: str,
    prompt_fragment: str,
) -> None:
    generator, model = make_generator(tmp_path)
    params: dict[str, Any] = {"include_json_resource": True}
    if categories is not None:
        params["categories"] = categories
    result, run_dir = invoke(
        generator,
        node_id,
        image_bytes=png_bytes((61, 62, 63)),
        params=params,
    )
    assert isinstance(result, Path) and result.is_file()
    assert result.parent == run_dir.resolve()
    assert result.name == f"{sidecar_stem}.txt"
    assert result.read_text(encoding="utf-8") == model.structured_text
    call = model.calls[-1]
    assert call["mode"] == mode
    assert prompt_fragment in call["question"]
    if categories is not None:
        assert "<p>person</p>, <p>dog</p>" in call["question"]
    envelope = json.loads(
        (run_dir / f"{sidecar_stem}-raw.json").read_text(encoding="utf-8")
    )
    assert envelope["raw_text"] == model.structured_text
    assert envelope["semantic_parsing"] == "not_applied"
    assert envelope["mode"] == mode


@pytest.mark.parametrize(
    "node_id",
    ["bounding-box-detection", "point-detection", "keypoint-detection"],
)
def test_category_detection_nodes_reject_empty_categories(
    tmp_path: Path,
    node_id: str,
) -> None:
    generator, _model = make_generator(tmp_path)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        invoke(
            generator,
            node_id,
            image_bytes=png_bytes((1, 2, 3)),
            params={"categories": ""},
        )


def test_manifest_node_set_and_generator_dispatch_set_remain_identical() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert adapter.NODE_IDS == frozenset(node["id"] for node in manifest["nodes"])


def test_all_pinned_upstream_prompt_builders_match_independent_hash_fixtures() -> None:
    runtime_prompts = {
        "depth": adapter._model_question(adapter.DEPTH_PROMPT, 1),
        "normal": adapter._model_question(adapter.NORMAL_PROMPT, 1),
        "binary_csv": adapter._model_question(
            adapter._binary_seg_prompt("person furthest right, red car"), 1
        ),
        "binary_tagged": adapter._model_question(
            adapter._binary_seg_prompt("<p>Person Left</p>, <p>dog</p>"), 1
        ),
        "panoptic_coco": adapter._model_question(adapter._panoptic_prompt(""), 1),
        "bbox": adapter._model_question(
            adapter._category_prompt(
                adapter.BBOX_DETECTION_PROMPT_TEMPLATE,
                "person, dog",
            ),
            1,
        ),
        "point": adapter._model_question(
            adapter._category_prompt(
                adapter.POINT_DETECTION_PROMPT_TEMPLATE,
                "person, dog",
            ),
            1,
        ),
        "ocr": adapter._model_question(adapter.OCR_PROMPT, 1),
        "keypoint_human": adapter._model_question(adapter._keypoint_prompt("person"), 1),
        "keypoint_animal": adapter._model_question(adapter._keypoint_prompt("dog"), 1),
        "keypoint_mixed": adapter._model_question(
            adapter._keypoint_prompt("person, dog"), 1
        ),
        "visual_prompt_segmentation": adapter.VISUAL_PROMPT_SEGMENTATION_PROMPT,
        "captioned_segmentation": adapter._model_question(
            adapter.CAPTIONED_SEGMENTATION_PROMPT, 1
        ),
        "recon3d": adapter.RECON3D_PROMPT,
        "camera_pose_3_images": adapter._model_question(adapter.CAMERA_POSE_PROMPT, 3),
    }
    assert set(runtime_prompts) == set(PINNED_PROMPT_SHA256)
    assert {
        name: sha256_text(prompt) for name, prompt in runtime_prompts.items()
    } == PINNED_PROMPT_SHA256

    serialized_categories = json.dumps(
        list(adapter.COCO_CATEGORY_NAMES),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(adapter.COCO_CATEGORY_NAMES) == 133
    assert sha256_text(serialized_categories) == PINNED_COCO_TUPLE_SHA256


def test_reasoning_image_without_text_returns_only_primary_image(tmp_path: Path) -> None:
    generator, model = make_generator(tmp_path)
    model.reasoning_text = None
    result, run_dir = invoke(
        generator,
        "open-image-generation",
        params={
            "prompt": "Create a diagram without an intermediate explanation.",
            "reasoning_mode": True,
        },
    )
    assert isinstance(result, Path) and result.is_file()
    assert not (run_dir / "reasoning.txt").exists()
    assert model.calls[-1]["mode"] == "think_generate"


def test_primary_text_artifact_uses_collision_safe_source_handle_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    occupied = run_dir / "result.txt"
    occupied.write_text("existing payload", encoding="utf-8")
    dispatch = adapter._DispatchResult(outputs=(adapter._GeneratedArtifact(
        source_handle="result",
        kind="text",
        text="fresh payload",
    ),))

    result = adapter.SenseNovaVisionGenerator._finalize_primary_output(dispatch, run_dir)

    assert result == (run_dir / "result-2.txt").resolve()
    assert result.read_text(encoding="utf-8") == "fresh payload"
    assert occupied.read_text(encoding="utf-8") == "existing payload"



def test_primary_and_sidecar_paths_must_stay_inside_the_run_directory(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("payload", encoding="utf-8")

    bad_primary = adapter._DispatchResult(outputs=(adapter._GeneratedArtifact(
        source_handle="result",
        kind="text",
        path=outside,
    ),))
    with pytest.raises(RuntimeError, match="outside the SenseNova run directory"):
        adapter.SenseNovaVisionGenerator._finalize_primary_output(bad_primary, run_dir)

    inside = run_dir / "image.png"
    inside.write_bytes(b"payload")
    bad_sidecar = adapter._DispatchResult(
        outputs=(adapter._GeneratedArtifact(
            source_handle="image",
            kind="image",
            path=inside,
        ),),
        resources=(adapter._SidecarResource(name="outside", path=outside),),
    )
    with pytest.raises(RuntimeError, match="outside the SenseNova run directory"):
        adapter.SenseNovaVisionGenerator._finalize_primary_output(bad_sidecar, run_dir)
