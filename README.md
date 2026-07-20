# SenseNova Vision 7B MoT for Modly

An independent Modly model extension maintained by **DrHepa** for the pinned
[OpenSenseNova SenseNova-Vision](https://github.com/OpenSenseNova/SenseNova-Vision)
runtime and
[`sensenova/SenseNova-Vision-7B-MoT`](https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT).


## Installation

1. Open **Models/Extensions → Install from GitHub** in Modly and install this repository.
2. From the matching GitHub Release, download `flash_attn-2.7.3-cp312-cp312-linux_aarch64.whl`, verify SHA-256 `8b7c2138cce6ff36f13051d3d83ac688ed0f997c629e5c1e140c120b552a9fc6`, and provision it at `wheels/flash_attn/flash_attn-2.7.3-cp312-cp312-linux_aarch64.whl` inside the installed extension. GitHub Release assets are used because this verified 406 MB binary exceeds the normal Git repository file limit.
3. Run **Setup/Repair**. It creates or repairs the extension-local environment, validates the exact local wheel, installs dependencies, and runs offline readiness checks. It does **not** download model weights.
4. Open the Modly Models UI and download the pinned SenseNova model assets.
5. Reload the extension registry, then add any of the 15 nodes below to a workflow.

Local development copies require the same Release-asset provisioning and explicit Setup/Repair step; linking or copying the repository does not prepare its Python environment.

## Usage

Version `0.2.2` exposes all 15 capabilities through Modly's established model-node interface. `open-image-generation` receives one required connected text value; every image-input node receives one connected primary image. Additional ordered images use the declared image-picker parameters. Every primary result is an existing artifact inside a collision-safe run directory: PNG for image nodes, GLB for reconstruction, and UTF-8 `.txt` served as `text/plain` for text nodes. Optional and diagnostic results remain beside the primary artifact as sidecars.

### Nodes

| Node ID | Primary input and supplemental images | Task parameters | Primary output and sidecars |
| --- | --- | --- | --- |
| `open-image-generation` | required connected Modly Text node; transported as `params.prompt` | `seed=42`; `num_timesteps=50`; `cfg_text_scale=4.0`; `cfg_img_scale=1.0`; `reasoning_mode=false` | `image.png`; optional `reasoning.txt` |
| `image-to-image-vision` | connected source image | `prompt` required; `seed=42`; `num_timesteps=50`; `cfg_text_scale=4.0`; `cfg_img_scale=2.0`; `reasoning_mode=false` | `image.png`; optional `reasoning.txt` |
| `visual-prompt-segmentation` | connected scene plus required `prompt_image_path` image picker | `seed=42`; `num_timesteps=50`; text/image guidance `4.0`/`1.0` | `segmentation.png` |
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
| `multi-view-reconstruction` | connected view 1; optional ordered `view_2_image_path` through `view_10_image_path` image pickers | `seed=42`; `num_timesteps=50`; `mask_edge=true`; black/white background masks `false` | `pointcloud.glb` with POINTS primitives; `pointmaps.npy` sidecar |
| `camera-pose` | connected frame 1; required `view_2_image_path`; optional ordered frame 3-10 image pickers | `seed=42` | `poses.txt` deterministic UTF-8 JSON (`text/plain`); `raw-pose.txt` and `camera-poses.json` sidecars |

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

- **Visual prompt segmentation:** connect the scene as the normal image input,
  then select the visual prompt image in `prompt_image_path`.
- **Multi-view reconstruction:** connect view 1, then fill
  `view_2_image_path`, `view_3_image_path`, and later pickers in capture order.
- **Camera pose:** connect frame 1 and select frame 2 in the required
  `view_2_image_path` picker. Frames 3-10 are optional.
- Do not leave gaps. For example, view 4 cannot be set while view 3 is empty.

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

Compatibility is pinned to official Modly tag `v0.4.1`, commit
`d2d100dd4364ad17f398e55a4a9ce529313352da`. That host recognizes a model as
text-input only when the node manifest declares `"input": "text"`. It copies
the connected text into `params.prompt` (and `params.text`), routes the request
through the established image generation endpoint with a synthetic 1x1 PNG,
and then invokes the unchanged generator signature. This adapter accepts both
that placeholder and the legacy empty byte payload but materializes neither as
a SenseNova model image; open generation always calls upstream with
`images=[]`.


SenseNova-Vision-7B-MoT requires a CUDA GPU with BF16 support. The extension is
currently verified only for the concrete lane below and should not be presented
as validated on other operating systems, Python ABIs, CUDA versions, or GPU
architectures without equivalent hardware evidence.

### Reproducible pins

- Runtime lane: Linux aarch64, CPython 3.12, Torch `2.7.0+cu128`, CUDA 12.8,
  SM121
- Open3D: audited local `0.19.0` wheel
- FlashAttention official upstream reference: `2.6.3`
- FlashAttention selected compatibility adaptation: audited local `2.7.3`
  wheel

The exact runtime source closure is vendored and hashed in
`vendor/opensensenova_sensenova_vision/PROVENANCE.json`. Setup does not clone
source code.

## Setup

Modly/Electron invokes:

```text
python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":121,"cuda_version":128}'
```

Legacy positional arguments are also supported. The verified local setup lane is
**unblocked**. Setup creates or repairs only the extension-local `venv`, installs
pinned dependencies, verifies local wheel
hashes and metadata, runs `pip check`, and performs an offline import plus CUDA
kernel smoke. It never downloads model weights, builds FlashAttention from
source, or uses `apt`, `sudo`, or host package installation.

The selected wheel is
`wheels/flash_attn/flash_attn-2.7.3-cp312-cp312-linux_aarch64.whl`:

- Size: `406023844` bytes
- Verified with Python 3.12.3, Torch `2.7.0+cu128`, CUDA 12.8, SM121, and NVIDIA
  driver 580.159.03
- `flash_attn_varlen_func` passed BF16 CUDA causal and noncausal checks

No FlashAttention source build is permitted.

Setup fails closed if the wheel, selector evidence, size, hash, or package metadata changes. Each published extension version must attach the exact wheel as a GitHub Release asset; it is not part of the source archive and must be provisioned at the documented local path before Setup/Repair. Setup never downloads a replacement, builds from source, or silently substitutes another binary.

Both stdout and stderr are shown by Modly's setup UI. Setup reserves stdout,
sends human-readable progress/errors to stderr, and persists evidence at
`.modly/setup/logs/setup.log` and `.modly/setup/setup-status.json`.

## Model download in Modly

After setup succeeds, use the Modly Models UI to download the weights. Every
node resolves to the same canonical owner and exact asset plan:

- Raw manifest owner: `shared` (scoped by Modly to
  `sensenova-vision-7b-mot/shared`). Unlike the standard
  `models/<extension-id>/<node-id>/` layout, every node in this fork resolves to
  that one canonical shared owner and does not duplicate the checkpoint.
- Target: `<Modly models>/sensenova-vision-7b-mot/shared/checkpoint/`
- Readiness check: `checkpoint/ema.safetensors`
- Repository: `sensenova/SenseNova-Vision-7B-MoT`

Generation forces Hugging Face and Transformers offline mode and loads only the
`model_dir` supplied by Modly.

## Validation status

### Offline source suite

The weight-free source contract suite passes without network or model downloads:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -p no:cacheprovider -q
```

Current result: **67 passed**. Coverage includes all 15 dispatches, exact official prompt builders and modes, shared weight/process ownership, category formatting and COCO ordering, supplemental image ordering, UTF-8 text artifacts, sidecar retention, progress/cancellation, pose parsing, POINTS GLB/NPY reconstruction, setup commands, and wheel hashes.

### Live Modly validation

**15/15 nodes passed live generation in Modly** on the verified Linux aarch64/NVIDIA lane documented above (CPython 3.12, Torch `2.7.0+cu128`, CUDA 12.8, SM121). This is the only live platform claim. Windows, macOS, Linux x86_64, other Python/CUDA/Torch combinations, and other GPU architectures remain unvalidated.

## Troubleshooting

- **Missing weights:** use the Models UI and verify the shared owner's
  `checkpoint/ema.safetensors`; `load()` never falls back to a network cache.
- **Setup failure:** inspect `.modly/setup/logs/setup.log` and
  `.modly/setup/setup-status.json`, then use Repair.
- **Missing FlashAttention Release asset:** download the exact wheel from the matching GitHub Release, verify the documented SHA-256, provision it at `wheels/flash_attn/`, and rerun Repair. Do not substitute another build.
- **Unsupported wheel lane:** do not build FlashAttention from source or install
  a different wheel silently; the setup status explains the failed selector.
- **Missing reasoning sidecar:** `reasoning.txt` appears only when the upstream
  think-image mode returns non-empty intermediate text.
- **Tagged text is unexpected:** open the primary `.txt` artifact or its optional raw JSON sidecar; the extension preserves upstream tags and deliberately does not invent a semantic parser.
- **Multi-image validation error:** fill image pickers in order without gaps,
  starting at view or frame 2.

## Limitations

- Model weights are CC BY-NC 4.0 and restricted to non-commercial use.
- Relative depth is not metric depth. Brighter pixels mean nearer and darker
  pixels mean farther; output is normalized to grayscale values 0-255.
- The normal-map PNG is an RGB direction visualization, not a serialized array
  of floating-point normal vectors.
- Tagged detections, keypoints, OCR, captions, and structured responses are not
  converted into an invented semantic schema.
- Multi-view `pointmaps.npy` contains XYZ coordinates shaped
  `[views, height, width, 3]`; it is not a scalar depth map.
- Reasoning-mode text is only the optional intermediate text returned by the
  upstream model. Standard mode remains the default.
- Live validation is limited to the documented Linux/NVIDIA lane. Runtime, dependency, manifest, or node changes require the 15-node live matrix to be repeated.

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
