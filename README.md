# SenseNova Vision 7B MoT for Modly

An independent Modly model extension maintained by **DrHepa** for the pinned
[OpenSenseNova SenseNova-Vision](https://github.com/OpenSenseNova/SenseNova-Vision)
runtime and
[`sensenova/SenseNova-Vision-7B-MoT`](https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT).



## Installation

1. Open **Models/Extensions → Install from GitHub** in Modly and install this repository.
2. Run **Setup/Repair**. It creates a fresh extension-local environment, installs pinned dependencies through the normal pip route, performs offline readiness checks, and commits the environment only after setup reports ready.
3. Open the Modly Models UI and download the SenseNova model weights separately.
4. Reload the extension registry, then add a node to a workflow.

No manual FlashAttention wheel provisioning is required for the normal route. The audited Linux ARM64 wheel is an automatic fallback only for its matching selector lane. Setup may download Python packages or that fallback dependency, but it never downloads model weights.

If an older installation shows only the repository-folder name, no metadata, or zero nodes, reinstall from the latest `main` and run Setup/Repair again. Linking or copying a local repository does not prepare its Python environment.

## Usage

Version `0.2.2` exposes all 15 capabilities through Modly's established model-node interface. `open-image-generation` receives one required connected text value; every image-input node receives one connected primary image. Additional ordered images use ordered absolute-path parameters. Every primary result is an existing artifact inside a collision-safe run directory: PNG for image nodes, GLB for reconstruction, and UTF-8 `.txt` served as `text/plain` for text nodes. Optional and diagnostic results remain beside the primary artifact as sidecars.

### Nodes

| Node ID | Primary input and supplemental images | Task parameters | Primary output and sidecars |
| --- | --- | --- | --- |
| `open-image-generation` | required connected Modly Text node; transported as `params.prompt` | `seed=42`; `num_timesteps=50`; `cfg_text_scale=4.0`; `cfg_img_scale=1.0`; `reasoning_mode=false` | `image.png`; optional `reasoning.txt` |
| `image-to-image-vision` | connected source image | `prompt` required; `seed=42`; `num_timesteps=50`; `cfg_text_scale=4.0`; `cfg_img_scale=2.0`; `reasoning_mode=false` | `image.png`; optional `reasoning.txt` |
| `visual-prompt-segmentation` | connected scene plus required `prompt_image_path` image-path parameter | `seed=42`; `num_timesteps=50`; text/image guidance `4.0`/`1.0` | `segmentation.png` |
| `captioned-segmentation` | connected image | `seed=42`; `num_timesteps=50`; text/image guidance `4.0`/`1.0` | `segmentation.png`; `caption.txt` sidecar |
| `structured-vision` | connected image | `prompt` required; `seed=42`; `reasoning_mode=false`; `include_json_resource=true` | `raw_text.txt` (`text/plain`); optional `raw_text-raw.json` |
| `relative-depth-estimation` | connected image | `seed=42`; `num_timesteps=50`; text/image guidance `4.0`/`1.0` | `depth.png`, 8-bit grayscale relative depth |
| `surface-normal-estimation` | connected image | `seed=42`; `num_timesteps=50`; text/image guidance `4.0`/`1.0` | `normals.png`, RGB normal visualization |
| `binary-segmentation` | connected image | `query` required; `seed=42`; `num_timesteps=50`; text/image guidance `4.0`/`1.0` | `segmentation.png` |
| `panoptic-segmentation` | connected image | `categories=""` uses official COCO order; `seed=42`; `num_timesteps=50`; text/image guidance `4.0`/`1.0` | `segmentation.png`; `caption.txt` sidecar |
| `bounding-box-detection` | connected image | `categories` required; `seed=42`; `include_json_resource=true` | `detections.txt` (`text/plain`); optional `detections-raw.json` |
| `point-detection` | connected image | `categories` required; `seed=42`; `include_json_resource=true` | `detections.txt` (`text/plain`); optional `detections-raw.json` |
| `keypoint-detection` | connected image | `categories` required; `seed=42`; `include_json_resource=true` | `keypoints.txt` (`text/plain`); optional `keypoints-raw.json` |
| `ocr` | connected image | `seed=42`; `include_json_resource=true` | `ocr_text.txt` (`text/plain`); optional `ocr_text-raw.json` |
| `multi-view-reconstruction` | connected view 1; optional ordered `view_2_image_path` through `view_10_image_path` image-path parameters | `seed=42`; `num_timesteps=50`; `mask_edge=true`; black/white background masks `false` | `pointcloud.glb` with POINTS primitives; `pointmaps.npy` sidecar |
| `camera-pose` | connected frame 1; required `view_2_image_path`; optional ordered frame 3-10 image-path parameters | `seed=42` | `poses.txt` deterministic UTF-8 JSON (`text/plain`); `raw-pose.txt` and `camera-poses.json` sidecars |

Every image-generation override keeps the existing defaults: seed `42`, 50
steps, text guidance `4.0`, and image guidance `1.0` except image editing, which
uses the official `2.0` default. Seed `-1` leaves the current random state
unchanged.

### Open image generation workflow

1. Add a Modly **Text** node and enter the generation instruction.
2. Connect its text output to `open-image-generation`.
3. Set the optional seed, diffusion, guidance, and reasoning controls.
4. Run the workflow and consume the returned `image.png`.

The connected value is the only prompt control. Modly transports it in
`params.prompt`; the extension deliberately exposes no duplicate **Prompt**
parameter on this node.


### Multi-image ordering

- **Visual prompt segmentation:** connect the scene as the normal image input, then enter the absolute visual-prompt image path in `prompt_image_path`.
- **Multi-view reconstruction:** connect view 1, then enter absolute paths in `view_2_image_path`, `view_3_image_path`, and later parameters in capture order.
- **Camera pose:** connect frame 1 and enter the absolute frame 2 path in `view_2_image_path`. Frames 3-10 are optional.
- Do not leave gaps. For example, view 4 cannot be set while view 3 is empty.

On official Modly v0.4.1 these fields are ordinary string parameters. Enter image-file paths manually; their accessory Browse button opens a directory selector, not an image-file picker.

## Parameters

The manifest exposes only parameters consumed by the selected upstream mode.
Common image parameters retain the defaults documented in the node table.

### Category and query rules

Comma-separated categories such as `person, red car` are formatted as
`<p>person</p>, <p>red car</p>`. Already tagged input is preserved verbatim.
Binary segmentation requires a non-empty category, reference, or reasoning
query. Bounding-box, point, and keypoint detection require categories.

Panoptic segmentation accepts an empty category string. Empty input expands to
all 133 names from upstream `COCO_CATEGORIES` in the exact official order,
including the upstream removal of `-stuff`, `-other`, and `-merged` suffixes.
Keypoint detection selects the official COCO-human, AP-10K-animal, or mixed
prompt based on the requested category names.


## Outputs

Image capabilities return an existing absolute `.png` path. Reconstruction returns an existing absolute `pointcloud.glb` path whose primitives are validated as POINTS. Each text capability writes the model's non-empty response to a collision-safe UTF-8 `.txt` artifact inside the run directory and returns that existing path; Modly serves these primaries as `text/plain`. The wrapper does not return raw Python strings, custom transport dictionaries, or invented semantic schemas. Secondary image/text results and diagnostic resources remain as validated sidecars in the same run directory.

### Official modes

- Standard image generation/editing: `generate` / `edit`
- Reasoning image generation/editing: `think_generate` / `think_edit`
- Structured vision: `understanding` or `think_understanding`
- Depth, normals, binary segmentation, and visual prompting: `dense_perception`
- Captioned and panoptic segmentation: `caption_generate`
- Bounding boxes, points, and keypoints: `dense_detection`
- OCR: `dense_OCR`
- Camera pose: `understanding` with the official camera transform
- Multi-view reconstruction: `SenseNovaVisionModel.reconstruct_3d`

Detection, keypoint, OCR, and structured-vision text is written exactly as the model produces it to the primary UTF-8 `.txt` artifact. Optional JSON sidecars contain the raw text and prompt and explicitly state that semantic parsing was not applied.


## Requirements and compatibility

### Modly upstream v0.4.1 input contract

Compatibility targets official Modly tag `v0.4.1`, commit `d2d100dd4364ad17f398e55a4a9ce529313352da`. That host recognizes a model as text-input when the node declares `"input": "text"`. It copies the connected value into `params.prompt` and `params.text`, routes the request through the established generation endpoint with a synthetic 1x1 PNG, and invokes the unchanged generator signature. This adapter ignores that placeholder and calls upstream open generation with `images=[]`.

### Platform scope

SenseNova-Vision-7B-MoT requires a CUDA-capable GPU and the vendored runtime currently uses BF16 model weights. The setup code has contract coverage for CPython 3.11/3.12 on Windows/Linux and x86_64/aarch64, but package resolution is not equivalent to native validation.

| Component | Normal route | Exact verified fallback |
| --- | --- | --- |
| Torch/Torchvision | pinned `2.7.0` / `0.22.0`; CUDA index selected from reported CUDA 11.8, 12.6, or 12.8, otherwise normal PyPI | none |
| Open3D | `open3d==0.19.0` from pip | audited local Linux aarch64 CPython 3.12 wheel |
| FlashAttention | upstream-pinned `flash-attn==2.6.3` pip build | audited `2.7.3` wheel only for Linux aarch64, CPython 3.12, the cu128 selector (reported CUDA 12.8 or newer), and SM121; native validation remains CUDA 12.8 only |

The only historical live lane is Linux aarch64, CPython 3.12, Torch `2.7.0+cu128`, CUDA 12.8, SM121. Windows and the other declared setup lanes remain unvalidated until tested on equivalent native hardware.

The exact vendored runtime source closure is hashed in `vendor/opensensenova_sensenova_vision/PROVENANCE.json`. Setup does not clone source code or model repositories.

## Setup

Modly/Electron invokes the setup with one JSON argument:

```text
python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":0,"cuda_version":0,"accelerator":"unknown","platform":"windows","arch":"x86_64"}'
```

Only `python_exe` and `ext_dir` are required. Legacy positional arguments remain supported. Setup accepts CPython 3.11 or 3.12 and normalizes Windows/Linux plus x86_64/aarch64 paths from the supplied interpreter; optional host metadata is checked when present but incomplete CUDA/SM metadata is not treated as a hardware rejection.

Setup/Repair is transactional:

1. Preserve an existing extension `venv` with an atomic same-filesystem rename.
2. Build the replacement at the final `<extension>/venv` path.
3. Install pinned dependencies and run offline readiness checks.
4. Write ready status and the success log.
5. Commit the replacement and remove the old backup.

Any install, smoke, status, or logging failure restores the previous environment. A later Setup/Repair also recovers an interrupted transaction. A failed first installation removes its partial `venv` instead of presenting it as ready.

The normal FlashAttention route is `flash-attn==2.6.3 --no-build-isolation --no-deps` with pip. The audited `2.7.3` Linux aarch64 CPython 3.12 wheel is only a cu128-selector (reported CUDA 12.8 or newer)/SM121 fallback, validated natively only on CUDA 12.8, sourced locally when present or from the checksum-pinned v0.2.2 GitHub Release asset. Open3D likewise uses normal pip except for its exact audited Linux ARM64 wheel when available.

The executable `setup.py` is authoritative. The documentary `manifest.json` setup metadata still contains the superseded wheel-only policy; Modly does not consume that metadata to execute installation, and it requires a separate manifest cleanup.

The offline smoke imports Open3D, Torch, torchvision compiled operators, FlashAttention, and the vendored SenseNova runtime. CUDA kernels are exercised only when CUDA is actually available during setup; successful preparation alone is not a generation or hardware-support claim.

Setup downloads Python dependencies but never model weights, never calls model-loading download APIs, and never uses `apt`, `sudo`, or host package installation. Human-readable output goes to stderr. Durable evidence is written to `.modly/setup/logs/setup.log` and `.modly/setup/setup-status.json`.

## Model download in Modly

After Setup/Repair succeeds, download model weights through the Modly Models UI. Setup never downloads them.

Each node declares `sensenova/SenseNova-Vision-7B-MoT` and checks `checkpoint/ema.safetensors`. The generator loads only from the exact `model_dir` supplied by Modly, forces Hugging Face and Transformers offline mode, and never falls back to a global cache.

On hosts that implement the extension-specific shared-owner fields, all nodes may resolve to one shared checkpoint. Official Modly v0.4.1 does not consume `weight_owner_id`, `process_owner_id`, or `hf_downloads`; on that host, do not assume that downloading one node automatically prepares every other node.

## Validation status

### Source and setup validation

Commit `8e5f045` passed source-only AST/import checks, all generated smoke-script variants, transactional rollback and crash-recovery scenarios, main success/failure ordering tests, and a fresh adversarial review. These checks used mocks and temporary directories; they did not install packages, access weights, or prove native execution on Windows.

The older repository pytest suite is not currently all green: the latest offline run reported **61 passed, 6 failed**. Those six failures assert the superseded host-locked/wheel-only setup behavior. They must not be hidden or reported as a fully passing current suite.

### Live Modly validation

The unchanged v0.2.2 generator/runtime previously completed **15/15 live node generations** on Linux aarch64, CPython 3.12, Torch `2.7.0+cu128`, CUDA 12.8, and SM121. The current transactional setup path has not yet repeated a clean installation plus the complete live matrix. Windows, Linux x86_64, CPython 3.11, other CUDA/Torch combinations, and other GPU architectures remain unvalidated.

## Troubleshooting

- **Repository-folder name, no metadata, or zero nodes:** Modly is showing its incomplete/corrupted extension fallback. Inspect the setup log, then reinstall from the latest `main` if the copy predates commit `8e5f045`, and run Setup/Repair again.
- **Setup failure:** inspect `.modly/setup/logs/setup.log` and `.modly/setup/setup-status.json`. A failed Repair restores the previous working `venv`; a failed clean installation remains incomplete until Setup/Repair succeeds.
- **FlashAttention build failure:** the normal route builds pinned `flash-attn==2.6.3` through pip. Read the complete compiler/package error in the setup log; no universal Windows or x86_64 wheel is bundled.
- **Exact Linux ARM64 fallback:** CPython 3.12 + the cu128 selector (reported CUDA 12.8 or newer) + SM121 may use the audited `flash-attn==2.7.3` wheel automatically after the normal route fails. Do not manually substitute an unrelated wheel.
- **Missing weights:** use the Models UI and verify `checkpoint/ema.safetensors` in the model directory supplied for that node. The generator never falls back to a network cache.
- **Missing reasoning sidecar:** `reasoning.txt` appears only when upstream returns non-empty intermediate text.
- **Tagged text is unexpected:** open the primary `.txt` artifact or raw JSON sidecar; the extension intentionally preserves upstream output.
- **Multi-image validation error:** enter absolute image paths in order without gaps, starting at view or frame 2.

## Limitations

- Model weights are CC BY-NC 4.0 and restricted to non-commercial use.
- Relative depth is not metric depth. Brighter pixels mean nearer and darker pixels mean farther; output is normalized to grayscale values 0-255.
- The normal-map PNG is an RGB direction visualization, not a serialized floating-point array.
- Tagged detections, keypoints, OCR, captions, and structured responses are preserved rather than converted into an invented semantic schema.
- Multi-view `pointmaps.npy` contains XYZ coordinates shaped `[views, height, width, 3]`; it is not a scalar depth map.
- Reasoning text exists only when the upstream think mode returns non-empty intermediate text.
- Official Modly v0.4.1 does not define a `boolean` parameter control. The current boolean fields may render as integer inputs; UI parity requires a future manifest-only correction.
- Official Modly v0.4.1 ignores `pickerIntent: "image"` on string parameters. Supplemental image paths must be entered manually.
- Official Modly v0.4.1 does not consume `weight_owner_id`, `process_owner_id`, or `hf_downloads`; shared-owner behavior is host-specific.
- The current transactional setup revision has not completed a fresh 15-node live matrix. Do not present Windows or any additional hardware lane as validated.

## Credits

- Modly extension wrapper and maintenance: **DrHepa**
- Vendored runtime source: **OpenSenseNova/SenseNova-Vision** by OpenSenseNova / SenseTime, pinned to commit `beea1f771b8192c597085c0e668f4430d2089d8d`
- Model repository and weights: **sensenova/SenseNova-Vision-7B-MoT**, pinned to revision `79548fcc5b954598799b9317f8d3ec5e347d5c0e`
- Modly workflow/runtime contract: Lightning Pixel
- Open3D and FlashAttention retain their respective upstream authorships


## Licenses

The root [LICENSE](LICENSE) is the standard MIT License for the independently authored Modly wrapper/adaptation only: copyright (c) 2026 DrHepa. It does **not** relicense or grant rights in the vendored SenseNova source, SenseNova model weights, Open3D, FlashAttention, or any other third-party component.

- Wrapper/adaptation: MIT
- Vendored OpenSenseNova/SenseNova-Vision source: Apache License 2.0; retained unchanged under `vendor/opensensenova_sensenova_vision/LICENSE`
- SenseNova-Vision-7B-MoT weights: CC BY-NC 4.0, non-commercial; downloaded separately through the Models UI and not distributed under the wrapper MIT license
- Open3D wheel: MIT
- FlashAttention wheel/Release asset: BSD 3-Clause

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before use or redistribution. Third-party license terms and notices remain controlling for their respective materials.
