# Third-Party Notices

## License scope

The root `LICENSE` applies only to the independently authored Modly wrapper/adaptation, copyright (c) 2026 DrHepa. It does not apply to, relicense, or replace the terms for the vendored OpenSenseNova source, SenseNova model weights, Open3D wheel, FlashAttention wheel, or other third-party materials. Those components remain under the separate licenses and attributions below.

## SenseNova-Vision source code

This repository is an independent Modly wrapper and is not affiliated with,
endorsed by, or sponsored by SenseTime or OpenSenseNova.

The runtime closure under `vendor/opensensenova_sensenova_vision/` is copied
without modification from:

- Project: OpenSenseNova/SenseNova-Vision
- Repository: <https://github.com/OpenSenseNova/SenseNova-Vision>
- Commit: `beea1f771b8192c597085c0e668f4430d2089d8d`
- License: Apache License 2.0

The complete upstream license is retained at
`vendor/opensensenova_sensenova_vision/LICENSE`. File selection and SHA-256
evidence are recorded in that directory's `PROVENANCE.json`.

## SenseNova-Vision-7B-MoT model weights

Model weights are **not distributed by this repository** and are never
downloaded by `setup.py`. Modly's model UI downloads the exact allowlisted
files from `sensenova/SenseNova-Vision-7B-MoT` at revision
`79548fcc5b954598799b9317f8d3ec5e347d5c0e` into the shared `checkpoint/`
directory.

The local extension manifest records the model weights as **CC BY-NC 4.0** and non-commercial. That license is separate from, and is not replaced by, this wrapper's MIT license. Review the model repository's retained terms before downloading or using the weights.

## Open3D wheel

The repository redistributes an Open3D 0.19.0 CPython 3.12 Linux aarch64 wheel
built from Open3D `v0.19.0`. Open3D is licensed under the MIT License. The wheel
contains its upstream license, and its complete build selectors, size, and
SHA-256 are recorded in `open3d-wheelhouse.manifest.json`.

## FlashAttention

A public extension release must distribute `flash_attn-2.7.3-cp312-cp312-linux_aarch64.whl` as a GitHub Release asset. After download it is provisioned locally at `wheels/flash_attn/flash_attn-2.7.3-cp312-cp312-linux_aarch64.whl` (`406023844` bytes, SHA-256
`8b7c2138cce6ff36f13051d3d83ac688ed0f997c629e5c1e140c120b552a9fc6`).
FlashAttention is licensed under the BSD 3-Clause License. Its embedded
license and author attribution were copied without modification to:

- `licenses/flash-attn/LICENSE`
- `licenses/flash-attn/AUTHORS`

The official pinned SenseNova source environment references
`flash-attn==2.6.3`. This wrapper preserves that fact as upstream provenance,
but selects `flash-attn==2.7.3` as an explicit compatibility adaptation for the
verified Linux aarch64, CPython 3.12, Torch `2.7.0+cu128`, CUDA 12.8, SM121
lane. The selected wheel was imported from its extracted contents—not from the
separately installed FlashAttention 2.8.3—and
`flash_attn_varlen_func` passed BF16 CUDA causal and noncausal shape and
finite-output checks.

The wheel is approximately 406 MB and therefore is distributed as a custom GitHub Release asset, not as part of the Git source archive. The published asset must preserve the exact filename, byte size, SHA-256, BSD-3-Clause license, and author attribution recorded above. Users provision that verified asset at the documented local wheel path before Setup/Repair; setup does not download a substitute wheel or build FlashAttention from source.
