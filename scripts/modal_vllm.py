"""Serve an open-weight model on Modal with vLLM behind an OpenAI-compatible endpoint for `--provider modal`.

Setup, once per Modal workspace (the key is what the benchmark sends as `Authorization: Bearer $MODAL_API_KEY`):

    modal secret create world-adapt-vllm VLLM_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    modal deploy scripts/modal_vllm.py

Set `MODAL_BASE_URL` to the printed URL plus `/v1`, `MODAL_API_KEY` to the secret's value, and use the preset's
`served_name` as the model. Deploying starts no GPU; the first request does. The first cold start downloads the
weights into a Modal Volume; later cold starts load them from it. A container stops after `SCALEDOWN_SECONDS`
without requests, so an idle deployment costs only Volume storage. `modal app stop <app name>` removes it.
"""

from __future__ import annotations

import os
import subprocess

import modal

# Checkpoints are pinned so a redeploy serves the benchmarked weights. Parsers follow the model card.
PRESETS = {
    "qwen3.6-35b-a3b": {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "served_name": "qwen3.6-35b-a3b",
        "gpu": "H200",  # BF16 weights are ~70 GB; the hybrid attention keeps the 131k-token KV cache small.
        "tool_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
    },
}

PRESET_NAME = os.environ.get("WORLD_ADAPT_VLLM_PRESET", "qwen3.6-35b-a3b")
PRESET = PRESETS[PRESET_NAME]
VLLM_VERSION = "0.31.0"
MAX_MODEL_LEN = 131072  # 80k input plus a 32,768-token reply, with headroom.
SCALEDOWN_SECONDS = 5 * 60
PORT = 8000
MINUTES = 60

app = modal.App(f"world-adapt-vllm-{PRESET_NAME}")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "TRITON_CACHE_DIR": "/root/.cache/vllm/triton_cache"})
)

# Weights, compiled graphs and JIT kernels persist across cold starts.
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name("world-adapt-hf-cache", create_if_missing=True),
    "/root/.cache/vllm": modal.Volume.from_name("world-adapt-vllm-cache", create_if_missing=True),
    "/root/.cache/flashinfer": modal.Volume.from_name("world-adapt-flashinfer-cache", create_if_missing=True),
}


# vLLM checks the bearer key itself (it reads VLLM_API_KEY from the secret), so Modal's proxy auth is off.
@app.server(
    image=image,
    gpu=PRESET["gpu"],
    volumes=volumes,
    secrets=[modal.Secret.from_name("world-adapt-vllm", required_keys=["VLLM_API_KEY"])],
    port=PORT,
    unauthenticated=True,
    startup_timeout=30 * MINUTES,
    min_containers=0,
    max_containers=1,
    target_concurrency=32,
    scaledown_window=SCALEDOWN_SECONDS,
)
class Server:
    @modal.enter()
    def start(self) -> None:
        cmd = [
            "vllm", "serve", PRESET["model"],
            "--revision", PRESET["revision"],
            "--served-model-name", PRESET["served_name"],
            "--host", "0.0.0.0", "--port", str(PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", "64",
            "--gpu-memory-utilization", "0.92",
            "--enable-prefix-caching",
            "--async-scheduling",
            "--enable-auto-tool-choice", "--tool-call-parser", PRESET["tool_parser"],
            "--reasoning-parser", PRESET["reasoning_parser"],
            "--uvicorn-log-level", "warning",
        ]
        print("starting:", " ".join(cmd), flush=True)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
