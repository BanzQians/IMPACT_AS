# IMPACT-Scribe (IMPACT_AS)

IMPACT-Scribe is a PyQt5 desktop system for interactive temporal action segmentation and assembly-state review. Its main loop is:

1. start from a coarse baseline,
2. ask the next most valuable query,
3. refine locally with temporal boundary scribbles,
4. propagate the correction globally,
5. learn from the correction history.

The intended paper-facing story is a lightweight human-in-the-loop system that stays responsive online while improving over time through correction-driven adaptation.

## Mainline Contributions

- `Temporal boundary scribble` as the primary local supervision primitive
- `Query planning` to decide which boundary / label / state query is worth asking next
- `Structured decode` to propagate accepted local corrections into a coherent global sequence
- `Correction-driven adaptation` so the local refiner, planner, and confidence estimates improve with use

## Quick Start

### Conda environment

Create the dedicated environment in-repo:

```bash
conda env create -f environment.yml -p ./.conda/envs/impact_as
```

Then launch with the bundled wrapper:

```bash
./run.sh
```

`run.sh` uses `./.conda/envs/impact_as`, prepares local runtime directories, and performs a small GUI preflight check before starting the Qt application.

If you prefer to launch Python directly:

```bash
./.conda/envs/impact_as/bin/python app.py
```

Optional operation logging:

```bash
./run.sh --oplog
```

If you only need the minimum pip package list, it remains available in `requirements.txt`.

## Core Capabilities

### Interactive Review

- Multi-view video sessions
- Coarse and fine action timelines
- Boundary scribble interaction for local refinement
- Query-driven review via `Suggest Query`
- Structured decoding with anchor/state consistency
- Assembly-state conflict analysis integrated into review

### Learning Loop

- Immediate: label prototype updates and confidence calibration
- Periodic: lightweight LoRA-style local-refiner adaptation on accumulated corrections
- Offline: confirmed-window export and lightweight global-model training

### Query Learning

The planner includes a lightweight `TrainableQueryUtilityModel` that learns from real correction outcomes using the same query signals already exposed by the system: uncertainty, disagreement, multiview conflict, state conflict, propagation gain, history, energy, and human cost.

### Escape Labels

Three reserved labels, `Unknown`, `Other`, and `Background`, remain available in the label palette. They are excluded from query scoring and handled specially in structured decode so uncertain spans can be marked without destabilizing the learning loop.

## Typical Workflow

1. Open a video or session.
2. Import an existing segment file, or optionally bootstrap a coarse external prelabel.
3. Use `Suggest Query` to jump to the highest-value review target.
4. Accept or refine boundary proposals with `Boundary Scribble`.
5. Resolve label and state suggestions when surfaced by the planner.
6. Let the system accumulate correction history for periodic local adaptation.
7. Export confirmed windows and train a lightweight global model offline if needed.

## Current Study-Facing GUI Flow

For the current user-study and paper-facing workflow, the intended path is:

- `Study: Scribble Only`: coarse-only, boundary-only annotation without planner guidance
- `Study: Scribble + Planner`: the same coarse boundary workflow, but with planner-assisted `Next Boundary`

In these study modes, the UI is intentionally narrowed:

- `Fine` mode is hidden
- `Interaction` mode switching is hidden
- the workflow is fixed to coarse-grained boundary scribble annotation
- the left label browser uses study-friendly titles such as `Recommended Labels` and `Labels`

The detailed operator-facing study protocol is documented in [docs/human_study_protocol_zh.md](docs/human_study_protocol_zh.md) and [docs/human_study_protocol_en.md](docs/human_study_protocol_en.md).

## Boundary Scribble Semantics

The current boundary-scribble interaction follows a single-gesture design:

- scribble in blank space: propose a new boundary
- narrow scribble centered on an existing boundary: propose removing that boundary and merging adjacent segments
- broader scribble crossing an existing boundary: refine that boundary

`Accept Boundary` reuses the same accept path for these cases.

On a blank canvas, the timeline is treated as a dense temporal partition rather than isolated patches:

- the first accepted boundary fills only the immediately preceding unlabeled span
- the next accepted boundary fills the next adjacent unlabeled span
- accepted blank-canvas corrections should not leave temporal gaps behind the current frontier

Selection is also biased toward editing stability:

- clicking an existing segment or marker prefers selection
- dragging past a threshold starts a new scribble
- right click keeps the delete behavior for existing annotations and markers

## Label Assistance

Label assistance is part of the correction-driven loop, but it is not a separate headline contribution.

- the left label panel shows recommended coarse labels for the current selected segment or gap
- for blank-fill and merge proposals, recommended labels can be changed before `Accept Boundary`
- candidate ranking uses prototype memory, imported candidates, optional text priors, and runtime confusion memory
- calibrated auto-assign is used conservatively; manual override remains available before acceptance

For the current study workflow, the main path is still boundary-first. Label assistance is there to reduce burden, not to replace the boundary-scribble contribution.

## Logging And Reproducibility

- operation logging is available via `python app.py --oplog` or `./run.sh --oplog`
- GUI logging can also be enabled from the application settings
- logs are written as `*.ops.log.csv` next to saved annotations
- scribble sidecars and accepted-correction exports support later adaptation and evaluation

## Project Structure

- [app.py](app.py): application entry point
- [run.sh](run.sh): local launcher for the in-repo Conda environment
- [ui/](ui): GUI windows, timeline widgets, dialogs
- [core/](core): planner, local refiner, structured decode, background adaptation, and state helpers
- [tools/](tools): training, export, evaluation, and optional baseline helpers
- [tests/](tests): learning-module tests
- [docs/](docs): design documents and user-study materials

## Main Offline Tools

- [tools/train_local_refiner.py](tools/train_local_refiner.py): local boundary model training
- [tools/train_query_model.py](tools/train_query_model.py): offline query-utility fitting from correction history
- [tools/export_real_correction_adaptation.py](tools/export_real_correction_adaptation.py): export accepted boundary corrections as adaptation data
- [tools/export_confirmed_windows.py](tools/export_confirmed_windows.py): export accepted local windows
- [tools/train_global_model_from_windows.py](tools/train_global_model_from_windows.py): train a lightweight global model from confirmed windows
- [tools/eval_interactive_scribble.py](tools/eval_interactive_scribble.py): interaction-budget, cost-budget, and time-budget evaluation

## Optional Components

- `ASOT` is retained as an optional external prelabel initializer. It is not required for the main IMPACT-Scribe loop.
- `SigLIP2` text-bank support is optional. The default text-bank backend now falls back automatically when the external model is unavailable.
- [tools/distill_global_model.py](tools/distill_global_model.py) remains available as an advanced offline route, but it is not the primary paper-facing offline path.

## Outputs

- Annotation JSON
- Scribble sidecar JSON
- Confirmed-window exports
- Adapted local-refiner checkpoints
- Query-model snapshots
- Evaluation JSON reports

## License

See `LICENSE`.
