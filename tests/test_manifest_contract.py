from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
REVISION = "79548fcc5b954598799b9317f8d3ec5e347d5c0e"
EXPECTED_FILES = {
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
}
EXPECTED_NODE_IDS = {
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
}


def by_id() -> dict[str, dict]:
    return {node["id"]: node for node in MANIFEST["nodes"]}


def param(node: dict, param_id: str) -> dict:
    return next(value for value in node["params_schema"] if value["id"] == param_id)


def test_manifest_identity_and_authorship() -> None:
    assert MANIFEST["id"] == "sensenova-vision-7b-mot"
    assert MANIFEST["kind"] == MANIFEST["type"] == "model"
    assert MANIFEST["version"] == "0.2.1"
    assert MANIFEST["author"] == "DrHepa"
    assert MANIFEST["source"] == (
        "https://github.com/DrHepa/modly-sensenova-vision-7b-mot-extension"
    )
    assert MANIFEST["generator_class"] == "SenseNovaVisionGenerator"
    assert MANIFEST["setup"]["downloads_weights"] is False


def test_manifest_uses_only_established_model_io_fields() -> None:
    assert "io_contract" not in MANIFEST
    for node in MANIFEST["nodes"]:
        assert "io_contract" not in node
        assert "inputs" not in node
        assert "outputs" not in node
        assert node["input"] in {"none", "image"}
        assert node["output"] in {"image", "text", "mesh"}

    serialized = json.dumps(MANIFEST, sort_keys=True).casefold()
    assert "named-v1" not in serialized


def test_all_fifteen_nodes_share_one_exact_weight_and_process_owner() -> None:
    nodes = MANIFEST["nodes"]
    assert len(nodes) == 15
    assert {node["id"] for node in nodes} == EXPECTED_NODE_IDS
    for node in nodes:
        assert node["weight_owner_id"] == "shared"
        assert node["process_owner_id"] == "shared"
        assert node["hf_repo"] == "sensenova/SenseNova-Vision-7B-MoT"
        assert node["download_check"] == "checkpoint/ema.safetensors"
        assert len(node["hf_downloads"]) == 1
        descriptor = node["hf_downloads"][0]
        assert descriptor["repo_id"] == "sensenova/SenseNova-Vision-7B-MoT"
        assert descriptor["revision"] == REVISION
        assert descriptor["target_subdir"] == "checkpoint"
        assert {item["path"] for item in descriptor["files"]} == EXPECTED_FILES


def test_published_weight_hashes_are_used_for_both_safetensors() -> None:
    files = {
        item["path"]: item
        for item in MANIFEST["nodes"][0]["hf_downloads"][0]["files"]
    }
    assert files["ae.safetensors"]["sha256"] == (
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"
    )
    assert files["ema.safetensors"]["sha256"] == (
        "96f29abd98791288c5a24087322e964bb9bcabfc2f185ece71543f827bc2b11e"
    )


def test_primary_input_and_output_types_match_every_capability() -> None:
    nodes = by_id()
    assert nodes["open-image-generation"]["input"] == "none"
    for node_id, node in nodes.items():
        if node_id != "open-image-generation":
            assert node["input"] == "image"

    expected_outputs = {
        "open-image-generation": "image",
        "image-to-image-vision": "image",
        "visual-prompt-segmentation": "image",
        "captioned-segmentation": "image",
        "structured-vision": "text",
        "multi-view-reconstruction": "mesh",
        "camera-pose": "text",
        "relative-depth-estimation": "image",
        "surface-normal-estimation": "image",
        "binary-segmentation": "image",
        "panoptic-segmentation": "image",
        "bounding-box-detection": "text",
        "point-detection": "text",
        "keypoint-detection": "text",
        "ocr": "text",
    }
    assert {node_id: node["output"] for node_id, node in nodes.items()} == expected_outputs
    assert nodes["multi-view-reconstruction"]["output_formats"] == ["glb", "npy"]


def _assert_image_picker(value: dict, *, required: bool) -> None:
    assert value["type"] == "string"
    assert value["default"] == ""
    assert value["required"] is required
    assert value["pickerIntent"] == "image"
    assert value["filters"] == [{
        "name": "Image files",
        "extensions": ["png", "jpg", "jpeg", "webp"],
    }]


def test_visual_prompt_uses_primary_scene_and_prompt_image_picker() -> None:
    node = by_id()["visual-prompt-segmentation"]
    picker = param(node, "prompt_image_path")
    _assert_image_picker(picker, required=True)
    assert "second image" in picker["tooltip"]
    assert "primary image is the scene" in picker["tooltip"]


def test_ordered_multi_image_nodes_use_primary_input_plus_picker_params() -> None:
    nodes = by_id()
    expected_ids = [f"view_{index}_image_path" for index in range(2, 11)]
    for node_id in ("multi-view-reconstruction", "camera-pose"):
        node = nodes[node_id]
        actual = [
            value["id"]
            for value in node["params_schema"]
            if value["id"].startswith("view_")
        ]
        assert actual == expected_ids
        for index, param_id in enumerate(expected_ids, start=2):
            _assert_image_picker(
                param(node, param_id),
                required=node_id == "camera-pose" and index == 2,
            )
    assert "connected primary image" in param(
        nodes["multi-view-reconstruction"], "view_2_image_path"
    )["tooltip"]
    assert "Frame 1 is the connected primary image" in param(
        nodes["camera-pose"], "view_2_image_path"
    )["tooltip"]


def test_reasoning_and_category_defaults_are_literal_and_compatible() -> None:
    nodes = by_id()
    for node_id in (
        "open-image-generation",
        "image-to-image-vision",
        "structured-vision",
    ):
        assert param(nodes[node_id], "reasoning_mode")["default"] is False
        assert param(nodes[node_id], "reasoning_mode")["type"] == "boolean"
    assert param(nodes["binary-segmentation"], "query")["required"] is True
    assert param(nodes["binary-segmentation"], "query")["default"] == ""
    assert param(nodes["panoptic-segmentation"], "categories")["required"] is False
    assert param(nodes["panoptic-segmentation"], "categories")["default"] == ""
    for node_id in (
        "bounding-box-detection",
        "point-detection",
        "keypoint-detection",
    ):
        assert param(nodes[node_id], "categories")["required"] is True
        assert param(nodes[node_id], "categories")["default"] == ""
    for node in nodes.values():
        assert param(node, "seed")["default"] == 42


def test_image_task_generation_defaults_match_runtime_parameters() -> None:
    nodes = by_id()
    for node_id in (
        "open-image-generation",
        "image-to-image-vision",
        "visual-prompt-segmentation",
        "captioned-segmentation",
        "relative-depth-estimation",
        "surface-normal-estimation",
        "binary-segmentation",
        "panoptic-segmentation",
    ):
        node = nodes[node_id]
        assert param(node, "num_timesteps")["default"] == 50
        assert param(node, "cfg_text_scale")["default"] == 4.0
        assert param(node, "cfg_img_scale")["default"] == (
            2.0 if node_id == "image-to-image-vision" else 1.0
        )
