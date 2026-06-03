# Prefetch REST API

## Overview

This document proposes a REST API that lets an external orchestrator
(router, agent, scheduler) tell LMCache to **prefetch** KV-cache chunks for a
request that has not yet arrived at vLLM. The goal is to hide cache-miss
latency: by the time vLLM admits the request and calls
`engine.retrieve(...)`, the chunks are already staged in a host-resident
LMCache tier (or pinned in CPU memory), so the retrieve is a fast in-process
copy instead of a disk or remote pull.

The endpoints live alongside the existing
[`internal_api_server`](../../../../lmcache/v1/internal_api_server/) and reuse
its FastAPI scaffold, lifecycle, and per-rank port/socket conventions.

## Scope

**In scope (Phase 1):** staging-only prefetch. Pull KV chunks for a list of
tokens (or precomputed block hashes) into LMCache's CPU/disk tiers, optionally
pinned, and let the caller poll completion. This is what
`LMCacheEngine.async_lookup_and_prefetch` already does internally; the new
API is a thin HTTP front for that primitive.

**In scope (Phase 2):** cross-instance pull. Issue a `MoveMsg` from the
cluster controller so instance A pulls chunks from instance B over the
existing P2P transfer channel, driven by an HTTP call.

**Out of scope (Phase 3, requires vLLM changes):** pushing prefetched chunks
all the way into vLLM's GPU KV cache before the request is admitted. The
GPU-side write requires a `slot_mapping` that only exists once vLLM's
scheduler has admitted the request and assigned blocks. We document the
boundary here and defer the upstream work.

## Why an HTTP API in LMCache

A natural alternative is to keep the orchestrator-to-LMCache hint path in-band
through vLLM (the router talks to vLLM, vLLM tells LMCache). LMCache already
exposes such an in-band path: the vLLM adapter's
`maybe_submit_lookup_request` (in
[`vllm_multi_process_adapter.py:531`](../../../../lmcache/integration/vllm/vllm_multi_process_adapter.py))
calls into the same async lookup+prefetch primitive on every incoming
request.

That in-band path is sufficient when prefetch should fire **at admission
time**. It is insufficient when:

- The router knows the request is coming **before** it forwards it to vLLM
  (e.g. agentic workflows where the next turn's prefix is computable from the
  current turn's output, or PD-disaggregated routers that pick a decode
  replica seconds before dispatching).
- The orchestrator wants to **warm a specific replica's local tier** by
  pulling from a peer (Phase 2), independent of vLLM scheduling.
- The orchestrator wants to **pin** chunks for a known-imminent request so a
  concurrent eviction does not undo the prefetch.

In all three cases the trigger is upstream of vLLM and LMCache needs an
externally addressable endpoint. The existing `internal_api_server` is the
right home: it already runs per worker, already wraps engine methods, and
already has a precedent endpoint (`/cache/load-fs-chunks`) that does a bulk
prefetch into the hot cache.

## Architecture

### Existing substrate

| Capability | Module | Notes |
|---|---|---|
| Per-engine FastAPI server | [`internal_api_server/api_server.py`](../../../../lmcache/v1/internal_api_server/api_server.py) | One server per scheduler/worker, port `internal_api_server_port_start + 1 + worker_id` or unix socket |
| Auto-discovered router registration | [`api_registry.py`](../../../../lmcache/v1/internal_api_server/api_registry.py) | Drops any `router = APIRouter()` under `vllm/`, `common/`, `controller/` into the app |
| Engine async lookup + prefetch | `LMCacheEngine.async_lookup_and_prefetch` ([`cache_engine.py:1254`](../../../../lmcache/v1/cache_engine.py)) | Non-blocking; runs on storage manager loop; supports `pin` |
| Cross-tier prefetch with prefix-continuity stop | `StorageManager.async_lookup_and_prefetch` ([`storage_backend/storage_manager.py:639`](../../../../lmcache/v1/storage_backend/storage_manager.py)) | Tier-by-tier, stops at first gap |
| Event tracking for in-flight prefetch | `EventManager` (`EventType.LOADING`) | `engine.event_manager.get_event_status(...)` returns `pending`/`done`/`failed` |
| Aborted-prefetch cleanup | `LMCacheEngine.cleanup_memory_objs` ([`cache_engine.py:1298`](../../../../lmcache/v1/cache_engine.py)) | Releases pinned memory objects when caller cancels |
| Cross-instance KV move | `MoveMsg` ([`cache_controller/message.py:615`](../../../../lmcache/v1/cache_controller/message.py)) and `KVController.move` ([`controllers/kv_controller.py:135`](../../../../lmcache/v1/cache_controller/controllers/kv_controller.py)) | Controller-orchestrated pull from peer over P2P |
| Bulk prefetch precedent | [`vllm/load_fs_chunks_api.py`](../../../../lmcache/v1/internal_api_server/vllm/load_fs_chunks_api.py) | Mirrors the request/response style we will follow |

### What's new

A single new router file at
`lmcache/v1/internal_api_server/vllm/prefetch_api.py`, auto-registered by
the existing `APIRegistry`. No new server, no new lifecycle, no config
changes for Phase 1.

For Phase 2, a controller-side HTTP front is needed; the
[`cache_controller/frontend/`](../../../../lmcache/v1/cache_controller/frontend/)
directory is currently static-only and would gain its first FastAPI app.
Phase 2 design lives in [`controller_prefetch_api.md`](controller_prefetch_api.md)
(to be written when Phase 1 lands).

## Phase 1: Staging-only Prefetch (per-engine)

### Endpoints

All endpoints live on the per-worker `internal_api_server`. The orchestrator
addresses a specific scheduler or worker; cluster-wide fan-out is the
orchestrator's responsibility in Phase 1.

#### `POST /prefetch/by_tokens`

Submit a prefetch by raw token IDs. LMCache hashes them with its configured
`token_database` to produce chunk keys.

**Request body:**

```json
{
  "request_id": "req-abc123",
  "tokens": [101, 234, 567, ...],
  "search_range": ["LocalCPUBackend", "LocalDiskBackend"],
  "pin": true,
  "request_configs": null
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `request_id` | string | yes | Used as `lookup_id`. Must be unique while the prefetch is in flight |
| `tokens` | list[int] | yes | Full prefix to prefetch |
| `search_range` | list[str] | no | Defaults to engine's `retrieve_locations` |
| `pin` | bool | no | Default `false`. When `true`, chunks are pinned until cancel/retrieve |
| `request_configs` | dict | no | Forwarded to token database (e.g. cache_salt) |

**Response (202 Accepted):**

```json
{
  "status": "accepted",
  "lookup_id": "req-abc123",
  "num_chunks_submitted": 7
}
```

#### `POST /prefetch/by_hashes`

Same semantics, but the caller supplies precomputed chunk hashes plus the
cumulative chunk lengths. This is the lighter-wire path for routers that
already maintain prefix hashes (e.g. for their own affinity routing).

**Request body:**

```json
{
  "request_id": "req-abc123",
  "hashes": [12345, 67890, ...],
  "offsets": [0, 256, 512, ...],
  "search_range": ["LocalCPUBackend"],
  "pin": false
}
```

#### `GET /prefetch/status/{lookup_id}`

Poll completion. Wraps `engine.event_manager.get_event_status(EventType.LOADING, lookup_id)`.

**Response:**

```json
{
  "lookup_id": "req-abc123",
  "status": "done",
  "matched_tokens": 1536,
  "matched_chunks": 6
}
```

`status` is one of `pending`, `done`, `failed`, `unknown`. `matched_*`
fields are populated once `status == "done"` from the prefetch result.

#### `POST /prefetch/cancel/{lookup_id}`

Cancel a pinned prefetch and release memory. Calls
`engine.cleanup_memory_objs(lookup_id)`. Idempotent — calling on an unknown
or already-cleaned lookup returns success with `released: 0`.

**Response:**

```json
{
  "lookup_id": "req-abc123",
  "released": 6
}
```

### Adapter contract

The router file reads `request.app.state.lmcache_adapter` (set by
`InternalAPIServer.__init__`) and pulls the engine via `lmcache_engine =
adapter.lmcache_engine`. If `lmcache_engine is None` (scheduler with no engine
attached, or engine not yet initialized), endpoints return **503** with the
existing error envelope used by `cache_api.py` and `load_fs_chunks_api.py`.

No new methods on `LMCacheManager` or `LMCacheEngine` are required for
Phase 1. The endpoints call `engine.async_lookup_and_prefetch(...)`,
`engine.event_manager.get_event_status(...)`, and
`engine.cleanup_memory_objs(...)` directly.

### Concurrency and idempotency

- Submitting a prefetch with the same `request_id` while a prior submission
  is still `pending` returns **409 Conflict**. (Mirrors the
  `_pending_lookups` guard in `vllm_multi_process_adapter.maybe_submit_lookup_request`.)
- Submitting after `done` is allowed and re-submits — useful when the prior
  prefetch was consumed and the chunks were evicted.
- `cancel` is safe to call from any state.
- The endpoint does not block the FastAPI event loop: the engine call is
  fire-and-forget (`asyncio.run_coroutine_threadsafe` against the storage
  manager loop), and `status` polling is a dict lookup.

### Error handling

| Condition | HTTP | Body |
|---|---|---|
| Engine not configured | 503 | `{"error": "...unavailable", "message": "..."}` |
| Missing required field | 400 | `{"error": "Invalid request", "message": "..."}` |
| `request_id` already in flight | 409 | `{"error": "Already in flight", "message": "..."}` |
| Internal exception | 500 | `{"error": "...", "message": str(e), "exception_type": "..."}` |

Same envelope shape as `vllm/cache_api.py` and `vllm/load_fs_chunks_api.py`,
so existing client tooling parses identically.

### Observability

- Reuse existing `EventManager` counters; no new metrics.
- A new Prometheus counter `lmcache_prefetch_api_requests_total{endpoint,status}`
  is added to the standard `internal_api_server` metrics surface (the
  existing `metrics_api.py` already exposes them).
- Each endpoint logs at INFO with `lookup_id` and `num_chunks_submitted`.

## Phase 2: Cross-instance Pull (controller-level)

The cluster controller already exposes `KVController.move(MoveMsg)` over its
ZMQ control plane. To make this externally callable, add a small FastAPI app
to `cache_controller/frontend/` (currently static-only) with:

- `POST /controller/prefetch/from_peer` — body: `{target_instance, source_instance, tokens|hashes, pin?}`. Constructs a `MoveMsg` and awaits the controller's response.
- `GET /controller/prefetch/status/{worker_event_id}` — wraps `KVController.check_finish`.

Phase 2 is **additive** to Phase 1 and does not change Phase 1 endpoints.
Detailed contracts are deferred until Phase 1 lands.

## Phase 3: GPU Prewarming (deferred, vLLM-side)

The boundary: `engine.retrieve(...)` requires `kvcaches` (vLLM's GPU
tensors) and a `slot_mapping` (which physical blocks to write into). Both
are scheduler-owned. A "prefetch into GPU" path requires either:

1. A vLLM hook that lets LMCache speculatively reserve blocks for a
   not-yet-admitted `request_id`, then performs the GPU write when those
   blocks materialize, or
2. A separate "shadow" GPU buffer in LMCache that the connector consumes on
   admission — large memory cost.

Either is an upstream vLLM change. We document the constraint and stop here.

## Compatibility Matrix

| Capability | LMCache alone | Phase 1 | Phase 2 | Phase 3 (vLLM coop) |
|---|---|---|---|---|
| Warm CPU/disk tier for future request | ✅ | ✅ via API | ✅ | ✅ |
| Pin chunks against eviction | ✅ | ✅ | ✅ | ✅ |
| Pull from peer LMCache instance | ✅ (controller) | — | ✅ via API | ✅ |
| Prewarm vLLM GPU KV cache | ❌ | ❌ | ❌ | ✅ |

## Implementation Plan

A single PR for Phase 1, scoped to:

1. New file `lmcache/v1/internal_api_server/vllm/prefetch_api.py` with the
   four endpoints, modeled after `cache_api.py` and `load_fs_chunks_api.py`.
2. Pydantic request/response models co-located in the same file (matching
   the `LoadFSChunksRequest`/`LoadFSChunksResponse` precedent).
3. Tests under `tests/v1/internal_api_server/test_prefetch_api.py` covering:
   - Submit → poll → done happy path
   - Submit duplicate `request_id` returns 409
   - Cancel of pinned prefetch releases memory (verify via storage-manager state)
   - 503 when engine unattached
   - Both `by_tokens` and `by_hashes` produce identical chunk keys for
     equivalent inputs
4. One new Prometheus counter wired through `metrics_api.py`.
5. README snippet at the top of the new file with `curl` examples (matches
   `cache_api.py` style).

No changes to `LMCacheEngine`, `StorageManager`, or any config schema.

Phase 2 and Phase 3 are tracked as follow-up issues, not part of the
Phase 1 PR.

## Executable Specification (E2E Test)

The end-to-end test at
[`tests/e2e/test_prefetch_api_e2e.py`](../../../../tests/e2e/test_prefetch_api_e2e.py)
is the implementation's pass/fail contract. It boots a real `vllm serve`
subprocess with the LMCache connector and the internal_api_server enabled,
and exercises:

1. **Populate** — a real completion request to vLLM stores KV chunks in
   the CPU + Disk tiers.
2. **Evict CPU** — `DELETE /cache/clear?locations=LocalCPUBackend` removes
   the CPU copies; the disk copies remain.
3. **Prefetch** — `POST /prefetch/by_tokens` with `pin=true`, then
   `GET /prefetch/status/{id}` polled until `status == "done"`.
4. **Tier-hit assertion** — verifies the LocalCPUBackend now contains
   chunks again (pulled up from disk by the prefetch). This is the
   load-bearing assertion for "did the API actually do anything?"
5. **Warm vLLM hit** — a second completion with the same prompt hits the
   warm tier; LMCache retrieve activity is observable in the vLLM serve
   log.
6. **Cancel** — `POST /prefetch/cancel/{id}` releases the pinned chunks;
   a second call returns `released: 0` (idempotent).

Plus three error-path tests:

- Duplicate `request_id` while in flight → **409**.
- Status of an unknown id → `200` with `{"status": "unknown"}`.
- Body missing required fields → **400** or **422**.

The implementation is complete when these four tests pass on a
GPU-equipped runner. Until then, every test should fail at the call to
the not-yet-existing `/prefetch/...` endpoint.

## Open Questions

1. **Auth / network exposure.** `internal_api_server` is "internal" by
   naming convention but is reachable on a TCP port by default. If a router
   in another pod will hit this endpoint, do we want a shared-secret header,
   or do we require unix-socket-only deployment for the prefetch surface?
   Suggested default: same posture as the rest of `internal_api_server` —
   document the trust boundary, defer auth to a separate PR if needed.
2. **Tokens vs hashes wire format.** Routers that already maintain prefix
   hashes will prefer `by_hashes`; routers that only see decoded text will
   prefer `by_tokens`. Both are cheap to support; ship both.
3. **Backpressure.** Should the API reject submissions when the storage
   manager has more than N in-flight prefetches? The engine already queues;
   exposing a cap is a follow-up.
