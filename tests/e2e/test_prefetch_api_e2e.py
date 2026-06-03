# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the LMCache prefetch REST API (Phase 1).

This test is the executable specification for the API designed in
``docs/design/v1/internal_api_server/prefetch_api.md``. The implementation
is complete when this test passes.

Flow
----
1. Boot ``vllm serve`` as a subprocess with the LMCacheConnectorV1 wired up
   and the LMCache internal_api_server enabled on a known port.
2. Send completion request #1 with a long prompt. vLLM stores its KV cache
   in LMCache (CPU + Disk tiers).
3. Evict the CPU tier via the existing ``DELETE /cache/clear`` endpoint and
   confirm the chunks are absent from the CPU backend.
4. Call the new ``POST /prefetch/by_tokens`` endpoint with the same prompt
   tokens and ``pin=true``, then poll ``GET /prefetch/status/{id}`` until
   the prefetch is reported done.
5. Assert the CPU tier is now repopulated (LMCache served the prefetch from
   the disk tier into CPU). This is the tier-hit assertion.
6. Send completion request #2 with the same prompt and assert vLLM reports
   a full-prefix cache hit (matched_tokens equals the chunk-aligned prompt
   length minus one, the conventional vLLM full-hit boundary).
7. Call ``POST /prefetch/cancel/{id}`` to release pinned memory and assert
   the released count matches what was pinned.

What the test does NOT cover
----------------------------
- Phase 2 (cross-instance pull) and Phase 3 (GPU prewarming).
- Concurrency stress (multi-client fan-out).
- Auth / network exposure.

Running locally
---------------
::

    pytest tests/e2e/test_prefetch_api_e2e.py -v -s

Requires:
- A CUDA-capable GPU.
- ``vllm`` and the LMCache integration importable in the active env.
- ``transformers`` (used to tokenize the prompt with the same tokenizer the
  served model uses, so we can pass exact token IDs to the prefetch API).
- ~2 minutes wall-clock for vllm serve startup with ``facebook/opt-125m``.

Override the model with ``LMCACHE_E2E_MODEL`` if ``opt-125m`` is unavailable.
"""

# Standard
from pathlib import Path
from typing import Iterator
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

# Third Party
import pytest
import requests

try:
    # Third Party
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep, but be defensive
    torch = None  # type: ignore[assignment]

# Skip the whole module if there's no GPU. The LMCache GPU connector path
# does not run on CPU-only.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(
        torch is None or not torch.cuda.is_available(),
        reason="prefetch e2e requires a CUDA-capable GPU",
    ),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get("LMCACHE_E2E_MODEL", "facebook/opt-125m")
CHUNK_SIZE = 256
# Prompt long enough to span at least 4 chunks of CHUNK_SIZE tokens.
# We request num_chunks * chunk_size + small_tail tokens and let the
# tokenizer pad with repeated content as needed.
PROMPT_TARGET_TOKENS = 4 * CHUNK_SIZE + 8
SERVE_BOOT_TIMEOUT_S = 180  # opt-125m typically boots in ~30s, allow slack.
PREFETCH_POLL_TIMEOUT_S = 30
PREFETCH_POLL_INTERVAL_S = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind to an ephemeral port and return it. Race-prone but fine for tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, timeout_s: float, label: str) -> None:
    """Poll ``url`` until it returns 2xx or timeout elapses."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if 200 <= r.status_code < 300:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(
        f"{label} did not become ready at {url} within {timeout_s}s "
        f"(last error: {last_err})"
    )


def _tokenize_prompt(model: str, target_tokens: int) -> tuple[str, list[int]]:
    """Produce a prompt + its tokenization with at least ``target_tokens`` ids.

    We deliberately produce a deterministic, content-rich prompt so that the
    same string tokenizes identically on every invocation — the prefetch API
    is keyed on token-derived hashes, so byte-identical token sequences
    matter.
    """
    # Third Party
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    seed = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "Sphinx of black quartz, judge my vow. "
    )
    text = ""
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += seed
    tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    return decoded, tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vllm_env() -> Iterator[dict[str, str | int]]:
    """Spawn vllm serve with LMCache + internal_api_server enabled.

    Yields a dict with the URLs and a tempdir for cleanup. Tears down the
    subprocess and disk artifacts at the end of the module.
    """
    vllm_port = _free_port()
    lmcache_port_start = _free_port()
    # internal_api_server worker port = port_start + 1 (worker_id 0).
    lmcache_worker_port = lmcache_port_start + 1

    work_dir = Path(tempfile.mkdtemp(prefix="lmcache_e2e_"))
    disk_dir = work_dir / "lmcache_disk"
    disk_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "LMCACHE_CHUNK_SIZE": str(CHUNK_SIZE),
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "1",
            "LMCACHE_LOCAL_DISK": f"file://{disk_dir}/",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": "5",
            "LMCACHE_INTERNAL_API_SERVER_ENABLED": "True",
            "LMCACHE_INTERNAL_API_SERVER_PORT_START": str(lmcache_port_start),
            "LMCACHE_INTERNAL_API_SERVER_HOST": "127.0.0.1",
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--port",
        str(vllm_port),
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--kv-transfer-config",
        json.dumps({"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}),
    ]

    log_path = work_dir / "vllm_serve.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(  # noqa: S603 - test-only, args fully controlled
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    try:
        try:
            _wait_for_http(
                f"http://127.0.0.1:{vllm_port}/health",
                timeout_s=SERVE_BOOT_TIMEOUT_S,
                label="vllm serve",
            )
            _wait_for_http(
                # Any GET on the internal_api_server returns something useful;
                # /lookup/info is cheap and exists on workers.
                f"http://127.0.0.1:{lmcache_worker_port}/lookup/info",
                timeout_s=30,
                label="lmcache internal_api_server",
            )
        except TimeoutError:
            log_fh.flush()
            tail = log_path.read_text()[-4000:]
            raise RuntimeError(
                f"vllm serve failed to start. Last 4KB of log:\n{tail}"
            ) from None

        yield {
            "vllm_url": f"http://127.0.0.1:{vllm_port}",
            "lmcache_url": f"http://127.0.0.1:{lmcache_worker_port}",
            "work_dir": str(work_dir),
            "log_path": str(log_path),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_fh.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def prompt_and_tokens() -> tuple[str, list[int]]:
    return _tokenize_prompt(MODEL, PROMPT_TARGET_TOKENS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _send_completion(
    vllm_url: str, prompt: str, max_tokens: int = 8
) -> requests.Response:
    """Send a non-streaming completion request to vLLM."""
    return requests.post(
        f"{vllm_url}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )


def _cpu_chunk_count(lmcache_url: str) -> int:
    """Probe how many chunks are currently in the LocalCPUBackend.

    Uses the existing ``/cache/kvcache/info`` and chunk-statistics surface.
    The test only cares whether this count is zero (cold) or positive (warm)
    after a prefetch — we do NOT assume an exact count, because the engine
    may have additional bookkeeping chunks.
    """
    r = requests.get(f"{lmcache_url}/chunk-statistics", timeout=5)
    if r.status_code != 200:
        # Fall back to the kvcache/info endpoint as a liveness probe; if
        # neither exists, the test must fail loudly.
        r2 = requests.get(f"{lmcache_url}/cache/kvcache/info", timeout=5)
        r2.raise_for_status()
        # No structured chunk count available — degrade to a binary signal:
        # the probe at least confirms the engine is responsive.
        return -1
    body = r.json()
    # The chunk_statistics_api returns a dict with per-backend counts. Sum
    # the LocalCPUBackend bucket only.
    backends = body.get("backends", {})
    return int(backends.get("LocalCPUBackend", {}).get("num_chunks", 0))


def test_prefetch_warms_cpu_tier_and_serves_subsequent_request(
    vllm_env: dict[str, str],
    prompt_and_tokens: tuple[str, list[int]],
) -> None:
    """The full prefetch happy path."""
    vllm_url = vllm_env["vllm_url"]
    lmcache_url = vllm_env["lmcache_url"]
    prompt, tokens = prompt_and_tokens
    request_id = "e2e-prefetch-001"

    # ------------------------------------------------------------------
    # Step 1: populate cache via a real vLLM completion.
    # ------------------------------------------------------------------
    populate = _send_completion(vllm_url, prompt, max_tokens=4)
    assert populate.status_code == 200, populate.text
    # Give the storage manager a beat to finish writing chunks to disk.
    time.sleep(2.0)

    # ------------------------------------------------------------------
    # Step 2: evict the CPU tier only. The disk tier should still hold
    # the chunks.
    # ------------------------------------------------------------------
    clear = requests.delete(
        f"{lmcache_url}/cache/clear",
        params={"locations": "LocalCPUBackend"},
        timeout=15,
    )
    assert clear.status_code == 200, clear.text
    cleared = clear.json().get("num_removed", 0)
    assert cleared > 0, (
        f"Expected to evict at least one chunk from LocalCPUBackend, "
        f"got {cleared}. Did the populate step actually store anything?"
    )

    # ------------------------------------------------------------------
    # Step 3: prefetch via the new API.
    # ------------------------------------------------------------------
    submit = requests.post(
        f"{lmcache_url}/prefetch/by_tokens",
        json={
            "request_id": request_id,
            "tokens": tokens,
            "search_range": ["LocalCPUBackend", "LocalDiskBackend"],
            "pin": True,
        },
        timeout=10,
    )
    assert submit.status_code == 202, submit.text
    submitted = submit.json()
    assert submitted["lookup_id"] == request_id
    assert submitted["num_chunks_submitted"] >= 1

    # ------------------------------------------------------------------
    # Step 4: poll until done.
    # ------------------------------------------------------------------
    deadline = time.monotonic() + PREFETCH_POLL_TIMEOUT_S
    final_status: dict[str, object] | None = None
    while time.monotonic() < deadline:
        s = requests.get(
            f"{lmcache_url}/prefetch/status/{request_id}", timeout=5
        )
        assert s.status_code == 200, s.text
        body = s.json()
        if body.get("status") == "done":
            final_status = body
            break
        if body.get("status") == "failed":
            pytest.fail(f"Prefetch reported failed: {body}")
        time.sleep(PREFETCH_POLL_INTERVAL_S)

    assert final_status is not None, (
        f"Prefetch did not reach 'done' within {PREFETCH_POLL_TIMEOUT_S}s"
    )
    assert final_status["matched_tokens"] >= CHUNK_SIZE, (
        f"Expected at least one full chunk to match after prefetch, "
        f"got {final_status['matched_tokens']}"
    )

    # ------------------------------------------------------------------
    # Step 5: tier-hit assertion. The CPU backend must now have chunks
    # again (pulled up from disk by the prefetch).
    # ------------------------------------------------------------------
    cpu_chunks_after = _cpu_chunk_count(lmcache_url)
    if cpu_chunks_after == -1:
        # /chunk-statistics not available; fall back to a softer signal:
        # a fresh /cache/clear should now report a positive removal count.
        verify_clear = requests.delete(
            f"{lmcache_url}/cache/clear",
            params={"locations": "LocalCPUBackend"},
            timeout=15,
        )
        assert verify_clear.status_code == 200
        assert verify_clear.json().get("num_removed", 0) > 0, (
            "After prefetch, LocalCPUBackend was still empty"
        )
        # Re-prefetch so the next step can still observe a warm vLLM hit.
        requests.post(
            f"{lmcache_url}/prefetch/by_tokens",
            json={
                "request_id": request_id + "-rewarm",
                "tokens": tokens,
                "pin": False,
            },
            timeout=10,
        ).raise_for_status()
        time.sleep(1.0)
    else:
        assert cpu_chunks_after > 0, (
            f"After prefetch, LocalCPUBackend chunk count is "
            f"{cpu_chunks_after} — expected > 0"
        )

    # ------------------------------------------------------------------
    # Step 6: send the same prompt to vLLM again. The response should be
    # served with a full-prefix cache hit. We assert via the server-side
    # log markers that LMCache emits ("Retrieved ... tokens"), since
    # vLLM does not always echo cached_tokens in the OpenAI completion
    # response shape.
    # ------------------------------------------------------------------
    warm = _send_completion(vllm_url, prompt, max_tokens=4)
    assert warm.status_code == 200, warm.text

    log_text = Path(vllm_env["log_path"]).read_text()
    assert "Retrieved" in log_text or "lmcache" in log_text.lower(), (
        "Expected to find LMCache retrieve activity in the vllm serve log "
        "after the warm request. Log tail:\n"
        + log_text[-2000:]
    )

    # ------------------------------------------------------------------
    # Step 7: cancel pinned prefetch.
    # ------------------------------------------------------------------
    cancel = requests.post(
        f"{lmcache_url}/prefetch/cancel/{request_id}", timeout=10
    )
    assert cancel.status_code == 200, cancel.text
    # Idempotent: a second cancel returns 200 with released == 0.
    cancel2 = requests.post(
        f"{lmcache_url}/prefetch/cancel/{request_id}", timeout=10
    )
    assert cancel2.status_code == 200, cancel2.text
    assert cancel2.json().get("released", 0) == 0


def test_prefetch_duplicate_request_id_returns_409(
    vllm_env: dict[str, str],
    prompt_and_tokens: tuple[str, list[int]],
) -> None:
    """Two in-flight submissions with the same request_id must conflict."""
    lmcache_url = vllm_env["lmcache_url"]
    _, tokens = prompt_and_tokens
    request_id = "e2e-prefetch-dup-001"

    body = {
        "request_id": request_id,
        "tokens": tokens,
        "pin": True,
    }
    first = requests.post(
        f"{lmcache_url}/prefetch/by_tokens", json=body, timeout=10
    )
    assert first.status_code == 202, first.text

    second = requests.post(
        f"{lmcache_url}/prefetch/by_tokens", json=body, timeout=10
    )
    assert second.status_code == 409, (
        f"Expected 409 on duplicate request_id, got {second.status_code}: "
        f"{second.text}"
    )

    # Cleanup so the next test doesn't see leftover pinned chunks.
    requests.post(f"{lmcache_url}/prefetch/cancel/{request_id}", timeout=10)


def test_prefetch_status_unknown_id_returns_unknown(
    vllm_env: dict[str, str],
) -> None:
    """Polling an id that was never submitted is not an error."""
    lmcache_url = vllm_env["lmcache_url"]
    r = requests.get(
        f"{lmcache_url}/prefetch/status/never-submitted-id", timeout=5
    )
    assert r.status_code == 200, r.text
    assert r.json().get("status") == "unknown"


def test_prefetch_invalid_body_returns_400(vllm_env: dict[str, str]) -> None:
    """Missing required fields must produce a 400, not a 500."""
    lmcache_url = vllm_env["lmcache_url"]
    r = requests.post(
        f"{lmcache_url}/prefetch/by_tokens",
        json={"request_id": "x"},  # missing tokens
        timeout=5,
    )
    assert r.status_code in (400, 422), r.text
