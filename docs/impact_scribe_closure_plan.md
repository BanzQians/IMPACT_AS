# IMPACT-Scribe Closure Plan

## Purpose

This document is the direct handoff for the remaining work needed to close the
IMPACT-Scribe method loop.

Use it together with:

- `docs/impact_scribe_complete_idea.md`
- `docs/impact_scribe_implementation_spec.md`

This file is the source of truth for the next engineering steps after the
baseline research prototype.

---

## Current Status

The repository already contains a working baseline prototype for:

- temporal boundary scribble interaction
- per-view scribble state and proposal persistence
- checkpoint-backed local refinement with heuristic fallback
- lightweight query suggestion across boundary, label, and state buckets
- constrained global decode from accepted scribble anchors
- correction-driven online cost, utility, and confidence learning
- correction-session logging and sidecar persistence
- synthetic scribble generation
- tiny local refiner training entry

The runtime method loop is now closed through Step 3. The main remaining
closure is concentrated in the offline/evaluation layer:

1. evaluation and interaction-budget reporting
2. downstream export for background global-model training

---

## Existing Entry Points

Another AI should start from these files and functions.

### Live interaction

- `ui/action_window.py`
  - `_run_scribble_local_refiner`
  - `_accept_last_scribble_proposal`
  - `_build_query_candidates`
  - `_suggest_next_query`
  - `_begin_correction_session`
  - `_commit_correction_session`

- `ui/timeline.py`
  - scribble interaction and proposal rendering

### Core logic

- `core/local_boundary_refiner.py`
  - current runtime contract for local window refinement

- `core/structured_decode.py`
  - current local patch-style update helpers

- `core/query_planner.py`
  - current deterministic heuristic planner contract

- `core/action_corrections.py`
  - grouped correction-session summaries

- `core/procedure_trace.py`
  - PSR/ASR conflict signals

### Offline training and data generation

- `tools/simulate_scribbles.py`
  - synthetic scribble generation from native annotations

- `tools/train_local_refiner.py`
  - tiny local refiner training and checkpoint writing

---

## Invariants

Do not break these behaviors while closing the method loop:

- existing action segmentation annotation must remain usable without a trained checkpoint
- PSR/ASR/ASD review behavior must keep working
- hover preview must remain low-latency
- the app must keep a heuristic fallback when learned components are unavailable
- all new online logic must run on cached features, not RGB re-encoding
- correction-session logging and sidecar save/load must remain backward-compatible

---

## Step 1. Learned Local Refiner in the Live Path

### Goal

Replace the runtime-only heuristic local refiner with a checkpoint-backed tiny
model while preserving the current `LocalRefinerInput` and
`LocalRefinerOutput` contract.

### Why this matters

Without this step, the tool still behaves like a GUI plus heuristic helper.
The paper claim requires the local refiner to be a learned model conditioned on
scribble masks and local features.

### Files to change

- `core/local_boundary_refiner.py`
- `ui/action_window.py`
- optionally `feature_defaults.json` if a default checkpoint path is introduced

### Current implementation note

The repository now supports three checkpoint discovery paths in the live GUI:

- `_algo_cfg["scribble_local_refiner"]["checkpoint"]`
- environment variables:
  - `IMPACT_SCRIBE_LOCAL_REFINER_CKPT`
  - `IMPACT_SCRIBE_LOCAL_REFINER`
- auto-discovery of `*local_refiner*.pt` or `*local_refiner*.pth` inside the
  active features directory

If another AI extends this further, it should preserve this resolution order
instead of replacing it.

### Required implementation

#### 1. Add a checkpoint-backed refiner class

In `core/local_boundary_refiner.py`, add a second implementation next to
`HeuristicLocalBoundaryRefiner`.

Recommended class:

- `CheckpointLocalBoundaryRefiner`

Required behavior:

- load the checkpoint written by `tools/train_local_refiner.py`
- reconstruct the tiny model shape from checkpoint metadata
- accept `LocalRefinerInput`
- map boundary logits to frame probabilities
- map left/right class logits back to label strings
- emit a `LocalRefinerOutput` with:
  - `boundary_probs`
  - `boundary_frame`
  - `left_label`
  - `right_label`
  - `confidence`
  - `extras`

Required checkpoint assumptions:

- checkpoint keys already include:
  - `state_dict`
  - `label_to_idx`
  - `idx_to_label`
  - `window_radius`
  - `input_dim`
  - `hidden_dim`
  - `dropout`

#### 2. Keep the live fallback chain simple

In `ui/action_window.py`:

- add a lightweight loader such as `_make_scribble_local_refiner()`
- if a valid checkpoint exists, use `CheckpointLocalBoundaryRefiner`
- otherwise use `HeuristicLocalBoundaryRefiner`

Do not hard-require PyTorch just to launch the GUI when no checkpoint is used.

#### 3. Preserve the current runtime contract

Do not rewrite the scribble UI call path.

Keep:

- `_run_scribble_local_refiner`
- proposal rendering
- proposal acceptance

Only change the implementation behind `self._scribble_local_refiner`.

### Acceptance criteria

- GUI starts without a checkpoint
- GUI starts with a trained checkpoint
- the same scribble episode can produce:
  - heuristic output when no checkpoint is configured
  - learned output when the checkpoint is configured
- no Qt code leaks into `core/local_boundary_refiner.py`

### Minimal validation

- `python -m py_compile app.py ui\\action_window.py core\\local_boundary_refiner.py`
- run one GUI session without checkpoint
- run one GUI session with checkpoint and confirm proposal generation still works

---

## Step 2. Replace Local Patching with Constrained Global Decode

Status: implemented in the current repository.

### Goal

Upgrade proposal acceptance from local label overwrite to whole-sequence
constrained decoding with hard anchors and lightweight soft penalties.

### Why this matters

Without this step, confirmed scribbles only patch a local segment. The method
claim requires local confirmation to influence the global action hypothesis.

### Files to change

- `core/structured_decode.py`
- `ui/action_window.py`
- optionally `core/procedure_trace.py` consumers in `ui/action_window.py`

### Required implementation

#### 1. Keep the current `ConfirmedWindow` contract

Do not remove:

- `ConfirmedWindow`
- `SoftConstraint`

Extend them only if needed with extra metadata.

#### 2. Add a real decode function

Recommended new function:

- `decode_frame_labels_with_constraints(...)`

Minimum required inputs:

- current frame-label map or framewise label scores
- confirmed hard windows
- soft constraints
- candidate label vocabulary for the active sequence

Minimum required outputs:

- decoded frame-label map
- optional decode diagnostics such as:
  - hard-constraint violations before and after
  - soft-penalty totals
  - changed span count

#### 3. Use a lightweight objective

The first decode version does not need a CRF or transformer.

A valid first implementation is:

- base per-frame score from current labels or imported label candidates
- hard anchors from confirmed scribble windows
- soft penalties for:
  - PSR/ASR conflict regions
  - multiview disagreement regions
  - abrupt label flips outside the confirmed window

A framewise dynamic program or Viterbi-style decoder is preferred.

#### 4. Wire it into scribble acceptance

In `_accept_last_scribble_proposal`:

- keep building a `ConfirmedWindow`
- stop applying only `apply_confirmed_windows_to_frame_labels`
- instead call the new constrained decode
- write back the decoded result span-by-span

#### 5. Reuse existing signals

Use current repository signals rather than inventing a second system:

- `_psr_trace_conflicts`
- `_query_multiview_disagreement`
- current per-frame label store
- existing segment-cut logic

### Acceptance criteria

- accepted scribble can change labels beyond the exact local overwrite range
- hard anchors are never violated after decode
- PSR/ASR conflict regions can influence the decode score
- multiview disagreement can influence the decode score when sync is active

### Minimal validation

- unit-style smoke test on a synthetic frame-label sequence
- GUI accept path still works
- no regression in proposal acceptance without PSR or multiview

### Landed implementation notes

- `core/structured_decode.py` now provides
  `decode_frame_labels_with_constraints(...)`
- `ui/action_window.py:_accept_last_scribble_proposal` now decodes the active
  view sequence instead of only patching the local span
- hard anchors come from `ConfirmedWindow`
- soft penalties currently use:
  - `state_conflict_region`
  - `multiview_preferred_label`
- decoded labels are written back span-by-span and trim cuts are reconciled
  while preserving the accepted boundary

---

## Step 3. Correction-Driven Learning

Status: implemented in the current repository.

### Goal

Turn correction sessions and operation logs into lightweight learned updates for:

- query utility
- human cost
- confidence calibration

### Why this matters

Without this step, the system logs corrections but does not learn from them in
the sense claimed by the method contribution.

### Files to change

- `core/query_planner.py`
- `ui/action_window.py`
- `core/action_corrections.py`
- optionally add:
  - `tools/train_query_models.py`
  - `tools/export_correction_dataset.py`

### Required implementation

#### 1. Define correction-derived training targets

At minimum, derive these targets from committed correction sessions:

- accepted vs rejected proposal
- number of edit steps
- whether a second correction happened nearby soon after
- how many frames changed after acceptance
- whether the accepted result reduced anchor/state conflicts

These can be extracted from:

- correction-session summaries
- sidecar `correction_history`
- operation logs when available

#### 2. Add lightweight learned cost and utility layers

The first version can be a simple regressor or online statistics model.

Recommended additions to `core/query_planner.py`:

- a small feature-extraction helper for `QueryCandidate`
- a light `QueryCostModel`
- a light `QueryUtilityModel`

Allowed first implementations:

- running averages by query type and bucket
- linear model fit offline and loaded at runtime
- simple online-updated scalar tables

Do not jump directly to a heavy ranking stack.

#### 3. Add confidence calibration

Calibrate proposal confidence from observed outcomes.

A valid first implementation is:

- keep acceptance and override counts by confidence bucket
- convert raw proposal confidence into calibrated confidence

This can live in `ui/action_window.py` first if needed, then be moved to core.

#### 4. Preserve prototype memory

The existing prototype update path is already useful.

Keep:

- `_update_label_prototype`

But do not treat it as sufficient for correction learning.

### Acceptance criteria

- query ranking changes after enough correction history is accumulated
- estimated cost is no longer a fixed constant per query family
- proposal confidence reflects observed acceptance behavior
- behavior still works when no history is available

### Minimal validation

- save a sidecar with correction history
- reload the same annotation and confirm learned priors restore
- verify query choice can change after synthetic or real correction history is replayed

### Landed implementation notes

- `core/query_planner.py` now includes:
  - `QueryCostModel`
  - `QueryUtilityModel`
  - `ProposalConfidenceCalibrator`
- `ui/action_window.py` now applies learned cost and utility in
  `_apply_query_learning_to_candidate`
- committed and discarded correction sessions are replayed into the online
  models through `_on_correction_session_finalized`
- proposal accept/refine feedback is stored in `CorrectionBuffer.history` and
  used for confidence calibration
- sidecar `correction_history` reload now rebuilds learned priors automatically

---

## Step 4. Optional but Strongly Recommended Evaluation Closure

Status: implemented in the current repository at the tooling level.

This is not part of the minimum runtime loop, but it is required to support the
paper framing.

### Add an interactive evaluation entry

Recommended new tool:

- `tools/eval_interactive_scribble.py`

Minimum metrics:

- boundary quality vs interaction count
- acceptance rate
- override rate
- second-correction rate
- average estimated cost vs actual steps

### Add downstream supervision export

Recommended new tool:

- `tools/export_confirmed_windows.py`

Purpose:

- convert accepted scribble windows into dense or pseudo-dense training targets
- support later training of a background global model

Without this, the downstream-learning claim remains incomplete.

### Landed implementation notes

- `tools/eval_interactive_scribble.py` now evaluates `_scribble.json` histories with:
  - boundary F1 vs interaction count
  - acceptance rate
  - override rate
  - second-correction rate
  - estimated cost vs actual steps
- `tools/export_confirmed_windows.py` now exports accepted correction windows as
  dense local supervision with:
  - absolute and relative frame bounds
  - boundary metadata
  - local dense labels
  - local segment lists
- `tools/impact_scribe_io.py` centralizes annotation/sidecar parsing for these
  offline tools

---

## Current Data Contracts Another AI Should Reuse

### Local refiner input contract

Current runtime code already provides:

- `window_start`
- `window_end`
- `boundary_energy`
- `scribble_channels`
- `left_candidates`
- `right_candidates`
- metadata such as view index and scribble kinds

Do not redesign this contract unless a hard blocker appears.

### Correction-session contract

Current correction summaries already preserve:

- correction kind
- metadata
- started and committed timestamps
- step count
- committed records
- changed flag

Treat this as the seed dataset for correction-driven learning.

### Sidecar persistence

The current `_scribble.json` sidecar should remain backward-compatible and
should continue to store:

- scribble items
- latest proposal
- latest query decision
- correction history

If new learned state is added, extend the sidecar rather than replacing it.

---

## Suggested Order of Execution

Another AI should implement the remaining closure in this order:

1. learned local refiner live integration
2. constrained global decode
3. correction-driven query and confidence learning
4. evaluation and downstream export

This order keeps the live interaction path stable while progressively upgrading
the method claim.

---

## Definition of Done

The live runtime loop is reasonably closed when all of the following are true:

- the GUI can use a trained local scribble refiner at runtime
- accepted scribble proposals trigger a true constrained global decode
- correction history changes future query ranking or cost estimation
- confidence is calibrated from observed interaction outcomes

The broader paper/evaluation closure additionally requires:

- accepted corrections can be exported for downstream model training
- an evaluation script exists for interaction-budget metrics

At the current state, the repository has:

- the online runtime loop implemented through constrained decode and
  correction-driven query learning
- an offline export path for confirmed windows
- an offline interaction-budget evaluation script

The remaining work is now primarily experimental rather than infrastructural:

- running larger real-data studies
- training background global models from exported windows
- reporting paper-scale ablations and human studies
