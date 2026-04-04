# IMPACT-Scribe Implementation Spec

## Purpose

This document turns the research idea in `docs/impact_scribe_complete_idea.md` into an ordered coding plan.

The intent is to let another AI or engineer implement the method incrementally without destabilizing the current annotation tool.

Status note:

- this document mainly describes the baseline prototype phases that are now largely landed
- the current method-closure status and offline evaluation/export handoff live in `docs/impact_scribe_closure_plan.md`
- another AI should read both files together before continuing implementation

The implementation strategy is:

- add non-invasive core abstractions first
- keep the live interaction loop lightweight
- integrate into the existing UI in small steps
- preserve current Action Segmentation and PSR/ASR behavior while new pieces are added


---

## Phase 0. Guardrails

Before adding new interaction behavior, preserve these invariants:

- `hover preview` must keep working
- existing `extra_mode` manual segmentation must keep working
- existing `assisted review` must keep working
- existing PSR/ASR timeline and review behavior must keep working
- all new online logic must be fast enough for desktop interaction

Do not start by rewriting `ActionWindow` or `TimelineArea` wholesale.

---

## Phase 1. Core data abstractions

Create these files first:

- `core/temporal_scribble.py`
- `core/query_planner.py`
- `core/local_boundary_refiner.py`
- `core/structured_decode.py`

These files should be pure logic modules.
They should not import Qt.

### 1.1 `core/temporal_scribble.py`

Responsibilities:

- represent temporal scribbles as intervals
- merge and clamp intervals
- build mask channels inside a local window

Required types:

- `ScribbleKind`
- `TemporalScribble`
- `TemporalScribbleSet`

Required functions:

- `normalize_interval`
- `merge_intervals`
- `build_scribble_channels`

Acceptance check:

- can build `uncertain`, `left`, `right` masks for a window without any UI dependency

### 1.2 `core/query_planner.py`

Responsibilities:

- represent boundary / label / state queries
- support two-stage query selection
- compute lightweight linear query utility

Required types:

- `QueryType`
- `QueryCandidate`
- `QueryPlannerWeights`
- `QueryDecision`

Required functions:

- `score_candidate`
- `choose_query_type`
- `choose_query`

Acceptance check:

- given a list of candidates with score terms and cost, the planner returns one decision deterministically

### 1.3 `core/local_boundary_refiner.py`

Responsibilities:

- define the input and output contract for the local window model
- provide a tiny baseline implementation for integration testing

Required types:

- `LocalRefinerInput`
- `LocalRefinerOutput`
- `BaseLocalBoundaryRefiner`
- `HeuristicLocalBoundaryRefiner`

Acceptance check:

- baseline refiner can accept scribble channels plus boundary energy and output a boundary distribution and side labels

### 1.4 `core/structured_decode.py`

Responsibilities:

- define hard anchors and soft constraints
- provide a first lightweight patch-based sequence updater

Required types:

- `ConfirmedWindow`
- `SoftConstraint`

Required functions:

- `apply_confirmed_windows_to_frame_labels`
- `count_anchor_violations`

Acceptance check:

- a confirmed local edit can be propagated into a frame-label map without touching Qt or the global model

---

## Phase 2. Timeline scribble UI

Primary file:

- `ui/timeline.py`

Goal:

- add the new interaction primitive without breaking existing drag / split / assisted interactions

### 2.1 Add non-destructive scribble state to rows

Add row-level state for:

- current scribble mode
- current draft stroke
- committed scribbles and archived markers for the visible view
- current local proposal

Recommended APIs on `CombinedTimelineRow` and/or `TimelineArea`:

- `set_scribble_mode(enabled: bool)`
- `set_scribble_items(items)`
- `clear_scribble_items()`
- `set_scribble_proposal(proposal)`

Recommended signal shape:

- `scribbleEditedDetailed(payload: object)`
- `scribbleActivated(payload: object)`
- `scribbleProposalAdjusted(payload: object)`
- `scribbleRemoved(payload: object)`

### 2.2 Reuse existing hover-preview flow

Do not replace hover preview.

Use it like this:

- hover gives temporal inspection
- drag while in scribble mode creates freehand temporal supervision

### 2.3 Rendering

Add a separate overlay layer for:

- active uncertain scribble stroke
- archived proposal / accepted markers
- local refiner proposal line if available

Keep the visible user path single-stroke by default:

- the user draws one freehand uncertain scribble
- follow-up passes around the same proposal can be interpreted as left/right support implicitly
- explicit brush switching is not required in the primary workflow

Acceptance checks:

- can toggle scribble mode on a row
- can draw a freehand uncertain stroke and see it rendered
- existing segment selection still works when scribble mode is off

---

## Phase 3. `ActionWindow` integration

Primary file:

- `ui/action_window.py`

Goal:

- connect the new timeline scribble interaction to the existing interaction pipeline

### 3.1 New runtime state

Add minimal new runtime members:

- per-view scribble store
- active scribble mode flag
- active scribble session id
- current local refiner instance
- current query planner instance
- current structured decoder state

Suggested names:

- `_scribble_items_by_view`
- `_scribble_mode_enabled`
- `_active_scribble_session_id`
- `_local_refiner`
- `_query_planner`

### 3.2 Connect timeline signal

When a scribble is committed:

1. create a `TemporalScribble`
2. add it to the active-view scribble set
3. begin a correction session if needed
4. invoke local refinement
5. render proposal overlays

### 3.3 Keep mode hierarchy simple

Recommended first version:

- `uncertain scribble` only in the default scribble mode
- `left/right` enabled only after refine is requested

Do not merge scribble mode into `extra_mode` on day one.
Keep them separate so rollback is easy.

Acceptance checks:

- can draw a scribble in the current view
- local proposal appears
- correction session metadata captures scribble information

---

## Phase 4. Local refiner baseline hookup

Primary files:

- `core/local_boundary_refiner.py`
- `ui/action_window.py`

Goal:

- integrate a working baseline before any learned tiny model exists

### 4.1 Baseline logic

For the first version:

- use boundary energy within the uncertain interval to get a peak
- use prelabel / assisted candidates for left/right label guesses
- derive confidence from:
  - boundary-energy contrast
  - candidate margin
  - interval width

### 4.2 Window extraction

Reuse existing feature-path logic from `ActionWindow`:

- `_resolve_features_dir_for_snap`
- `_ensure_boundary_snap_cache`
- `_boundary_energy_at_frame`
- segment embedding utilities

Acceptance checks:

- after drawing an uncertain scribble, the system returns:
  - one boundary proposal
  - left label guess
  - right label guess
  - confidence score

---

## Phase 5. Query planner integration

Primary files:

- `core/query_planner.py`
- `ui/action_window.py`

Goal:

- upgrade the current assisted-review heuristic queue into a two-stage planner

### 5.1 Candidate builders

Build candidates from three sources:

- boundary candidates from action segments and boundary energy
- label candidates from current segments and top-k suggestions
- state candidates from PSR/ASR trace conflicts

### 5.2 Two-stage selection

Stage 1:

- choose `boundary`, `label`, or `state`

Stage 2:

- choose the best candidate region inside that type

### 5.3 Keep the first version linear

Do not start with a learned ranking network.
Start with:

- weighted terms
- cost normalization
- deterministic tie-breaking

Acceptance checks:

- planner can choose a state repair when state conflict dominates
- planner can choose a boundary scribble when boundary uncertainty dominates
- planner can choose a label query when candidate disagreement dominates

---

## Phase 6. PSR/ASR integration into the mainline

Primary files:

- `ui/action_window.py`
- `core/procedure_trace.py`
- `core/structured_decode.py`

Goal:

- make PSR/ASR a structured layer inside the main action-annotation workflow

### 6.1 Query use

Feed PSR/ASR conflict severity into query planning as `state_conflict`.

### 6.2 Decode use

Convert:

- confirmed local action anchors
- PSR/ASR conflicts

into hard anchors and soft penalties.

### 6.3 UI use

When state repair is selected:

- focus the relevant component or time region
- show the action-state inconsistency as a first-class correction target

Acceptance checks:

- a PSR/ASR trace conflict can produce a query candidate
- confirmed action corrections can reduce state conflict count

---

## Phase 7. Structured global patching

Primary files:

- `core/structured_decode.py`
- `ui/action_window.py`

Goal:

- propagate local confirmed edits to the current frame-label hypothesis

### 7.1 First implementation

Keep the first version lightweight:

- patch inside the confirmed window
- treat boundary as hard anchor
- treat left/right labels as local hard labels near the boundary
- count consistency violations

### 7.2 Later implementation

Replace patching with constrained DP / Viterbi when ready.

Acceptance checks:

- local confirmation updates the current frame label map
- anchor violations can be counted for diagnostics

---

## Phase 8. Logging and correction learning

Primary files:

- `core/action_corrections.py`
- `utils/op_logger.py`
- `ui/action_window.py`

Goal:

- make correction traces useful for later learning

### 8.1 Add scribble metadata to sessions

Each committed correction should capture:

- view id
- scribble kind
- interval width
- proposed boundary
- accepted boundary
- proposal confidence
- chosen query type
- whether a second correction was needed

### 8.2 Add query metrics

Record:

- acceptance rate
- override rate
- second-correction rate
- estimated local gain
- estimated conflict reduction

Acceptance checks:

- a committed scribble correction produces structured metadata in session history and logs

---

## Phase 9. Training utilities

Primary files:

- `tools/simulate_scribbles.py`
- `tools/train_local_refiner.py`

Goal:

- support the first learning loop outside the GUI

### 9.1 Synthetic scribbles

Generate:

- uncertain intervals near GT boundaries
- left/right side intervals
- noisy, offset, and broken intervals
- false intervals near wrong candidates

### 9.2 Tiny head training

Train only the local refiner first.
Do not start by retraining the full background segmentation model.

Acceptance checks:

- can generate synthetic scribbles from one annotation file
- can train a tiny local head on cached features plus synthetic scribble channels

---

## Phase 10. Evaluation protocol

Goal:

- evaluate as an interactive human-centered method rather than a static model only

Required outputs:

- budget curves
- synthetic scribble evaluation
- real human scribble evaluation
- acceptance / override / second-correction metrics
- equal-time downstream model comparison

---

## Recommended Coding Order

If implementing sequentially, use this exact order:

1. add core abstraction files
2. add timeline scribble rendering and interval capture
3. connect scribble commits into `ActionWindow`
4. hook up the heuristic local refiner baseline
5. introduce the two-stage query planner
6. feed PSR/ASR conflicts into query planning
7. add structured local-to-global patching
8. enrich correction logs and sessions
9. add synthetic scribble tooling
10. train the tiny local head
11. replace heuristic pieces incrementally

---

## Minimal Acceptance Checklist

At the end of the first milestone, the tool should be able to:

- display a scribble overlay on the timeline
- store `uncertain scribble` intervals per view
- propose one boundary and side labels from a local window
- create boundary / label / state query candidates
- choose one query with a deterministic planner
- patch the current action hypothesis with confirmed local edits
- log the correction in a structured way

If all of the above work, the method has crossed from idea into a real research prototype.
